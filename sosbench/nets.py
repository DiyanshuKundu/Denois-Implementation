"""Networks and diffusion machinery (PyTorch).

UNet       : ADM-style U-Net (Dhariwal & Nichol) — 4 resolution levels with channel
             multipliers (1,2,2,2), base width 64, optional timestep embedding.
Diffusion  : linear-schedule DDPM (T=200), eps-prediction, classifier-free-guidance
             training (condition dropout 0.2), DDIM / DDPM samplers with an optional
             plug-and-play data-consistency hook every K steps.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- normalisation
SOS_MID, SOS_HALF = 1525.0, 125.0          # [1400,1650] -> [-1,1]
B_SCALE = 20.0                             # ns


def sos_norm(c):   return (c - SOS_MID) / SOS_HALF
def sos_denorm(x): return x * SOS_HALF + SOS_MID


# --------------------------------------------------------------------------- U-Net
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def _gn(ch):
    return nn.GroupNorm(min(32, ch // 4), ch)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout):
        super().__init__()
        self.n1 = _gn(cin); self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout) if tdim else None
        self.n2 = _gn(cout); self.drop = nn.Dropout(dropout); self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        if self.t is not None:
            h = h + self.t(F.silu(temb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.n = _gn(ch); self.qkv = nn.Conv2d(ch, 3 * ch, 1); self.o = nn.Conv2d(ch, ch, 1); self.h = heads
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)

    def forward(self, x, temb=None):
        b, c, hh, ww = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, self.h, c // self.h, hh * ww).unbind(1)
        a = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.o(a.transpose(-1, -2).reshape(b, c, hh, ww))


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, base=64, mults=(1, 2, 2, 2), nres=2, attn_levels=(3,),
                 use_t=True, dropout=0.1):
        super().__init__()
        self.use_t = use_t
        tdim = 4 * base if use_t else 0
        if use_t:
            self.temb = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.base = base
        self.inp = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down = nn.ModuleList(); chans = [base]; ch = base
        for li, m in enumerate(mults):
            for _ in range(nres):
                blk = nn.ModuleList([ResBlock(ch, base * m, tdim, dropout)]); ch = base * m
                if li in attn_levels: blk.append(Attn(ch))
                self.down.append(blk); chans.append(ch)
            if li < len(mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)])); chans.append(ch)
        self.mid = nn.ModuleList([ResBlock(ch, ch, tdim, dropout), Attn(ch), ResBlock(ch, ch, tdim, dropout)])
        self.up = nn.ModuleList()
        for li, m in reversed(list(enumerate(mults))):
            for _ in range(nres + 1):
                blk = nn.ModuleList([ResBlock(ch + chans.pop(), base * m, tdim, dropout)]); ch = base * m
                if li in attn_levels: blk.append(Attn(ch))
                self.up.append(blk)
            if li > 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(ch, ch, 3, padding=1)]))
        self.out = nn.Sequential(_gn(ch), nn.SiLU(), nn.Conv2d(ch, out_ch, 3, padding=1))

    def forward(self, x, t=None):
        temb = self.temb(timestep_embedding(t, self.base)) if self.use_t else None
        h = self.inp(x); hs = [h]
        for blk in self.down:
            if isinstance(blk[0], ResBlock):
                h = blk[0](h, temb)
                for extra in blk[1:]: h = extra(h)
            else:
                h = blk[0](h)
            hs.append(h)
        h = self.mid[0](h, temb); h = self.mid[1](h); h = self.mid[2](h, temb)
        for blk in self.up:
            if isinstance(blk[0], ResBlock):
                h = blk[0](torch.cat([h, hs.pop()], 1), temb)
                for extra in blk[1:]: h = extra(h)
            else:
                h = blk[1](blk[0](h))
        return self.out(h)


# --------------------------------------------------------------------------- diffusion
class Diffusion:
    """x-space: (nb, 1, H, W).  Model input = cat[x_t, cond * keep, keep-flag]."""
    def __init__(self, T=200, device='cpu'):
        self.T = T
        scale = 1000.0 / T                                  # keep the classic 1e-4..0.02 @ T=1000 range
        betas = torch.linspace(1e-4 * scale, 0.02 * scale, T, dtype=torch.float64)
        ac = torch.cumprod(1 - betas, 0)
        self.betas = betas.float().to(device)
        self.ac = ac.float().to(device)                    # alpha_bar_t, t = 0..T-1
        self.device = device

    def to(self, device):
        self.betas = self.betas.to(device); self.ac = self.ac.to(device); self.device = device
        return self

    @staticmethod
    def model_in(x_t, cond, keep):
        if cond is None:
            return x_t
        return torch.cat([x_t, cond * keep[:, None, None, None], keep[:, None, None, None].expand_as(x_t)], 1)

    def loss(self, model, x0, cond, p_uncond=0.2):
        nb = x0.shape[0]
        t = torch.randint(0, self.T, (nb,), device=x0.device)
        noise = torch.randn_like(x0)
        a = self.ac[t][:, None, None, None]
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise
        keep = (torch.rand(nb, device=x0.device) >= p_uncond).float() if cond is not None else None
        eps = model(self.model_in(x_t, cond, keep), t)
        return F.mse_loss(eps, noise)

    def eps(self, model, x_t, t_int, cond, guidance=0.0):
        nb = x_t.shape[0]
        t = torch.full((nb,), t_int, device=x_t.device, dtype=torch.long)
        if cond is None:
            return model(x_t, t)
        e_c = model(self.model_in(x_t, cond, torch.ones(nb, device=x_t.device)), t)
        if guidance == 0:
            return e_c
        e_u = model(self.model_in(x_t, cond, torch.zeros(nb, device=x_t.device)), t)
        return (1 + guidance) * e_c - guidance * e_u

    @torch.no_grad()
    def ddim(self, model, cond, x_init=None, t0=None, shape=None, dc=None, K=5, guidance=0.0, seed=None):
        """DDIM (eta=0) from noise level t0 (x_init noised to t0, or pure noise if None).
        dc(x0_hat) -> x0_hat' is applied every K denoising steps and at the end."""
        g = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        t0 = self.T - 1 if t0 is None else min(t0, self.T - 1)
        if x_init is None:
            x = torch.randn(shape, device=self.device, generator=g)
        else:
            a = self.ac[t0]
            x = a.sqrt() * x_init + (1 - a).sqrt() * torch.randn(x_init.shape, device=self.device, generator=g)
        x0 = x
        for i, t in enumerate(range(t0, -1, -1)):
            a = self.ac[t]
            e = self.eps(model, x, t, cond, guidance)
            x0 = ((x - (1 - a).sqrt() * e) / a.sqrt()).clamp(-1.5, 1.5)
            if dc is not None and ((i + 1) % K == 0 or t == 0):
                x0 = dc(x0)
                e = (x - a.sqrt() * x0) / (1 - a).sqrt().clamp_min(1e-8)
            if t > 0:
                ap = self.ac[t - 1]
                x = ap.sqrt() * x0 + (1 - ap).sqrt() * e
        return x0

    @torch.no_grad()
    def ddpm(self, model, cond, shape, dc=None, K=1, guidance=0.0, seed=None):
        """Ancestral DDPM sampling from pure noise (used for the diffusion refiner M_D
        and for the DOLCE baseline with dc every step)."""
        g = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        x = torch.randn(shape, device=self.device, generator=g)
        x0 = x
        for i, t in enumerate(range(self.T - 1, -1, -1)):
            a = self.ac[t]; beta = self.betas[t]
            e = self.eps(model, x, t, cond, guidance)
            x0 = ((x - (1 - a).sqrt() * e) / a.sqrt()).clamp(-1.5, 1.5)
            if dc is not None and ((i + 1) % K == 0 or t == 0):
                x0 = dc(x0)
            if t > 0:
                ap = self.ac[t - 1]
                mean = (ap.sqrt() * beta / (1 - a)) * x0 + ((1 - beta).sqrt() * (1 - ap) / (1 - a)) * x
                var = beta * (1 - ap) / (1 - a)
                x = mean + var.sqrt() * torch.randn(x.shape, device=self.device, generator=g)
        return x0


def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6
