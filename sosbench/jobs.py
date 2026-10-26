"""Run shell command queues on separate GPUs from a notebook and follow their logs."""
import os, subprocess, sys, time


def start_queue(cmds, gpu, log_path, workdir='.'):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
               PYTHONPATH=os.path.abspath(workdir) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    script = ' && '.join(f'echo ">>> {c}" && {c}' for c in cmds)
    f = open(log_path, 'w')
    # start_new_session: interrupting a notebook cell must not kill the background jobs
    p = subprocess.Popen(['bash', '-c', script], env=env, stdout=f, stderr=subprocess.STDOUT, cwd=workdir,
                         start_new_session=True)
    p._log = log_path
    return p


def _tail(path, n=3):
    try:
        with open(path) as f:
            return [l.rstrip() for l in f.readlines()[-n:]]
    except FileNotFoundError:
        return []


def wait_all(procs, every=60, tail=2):
    """Block until every queue finishes, printing the last log lines periodically."""
    t0 = time.time()
    while True:
        alive = [p for p in procs if p.poll() is None]
        print(f'--- {time.strftime("%H:%M:%S")}  elapsed {(time.time() - t0) / 60:.1f} min  running {len(alive)}/{len(procs)}')
        for p in procs:
            for l in _tail(p._log, tail): print(f'   [{os.path.basename(p._log)}] {l}')
        sys.stdout.flush()
        if not alive: break
        time.sleep(every)
    bad = [p._log for p in procs if p.returncode != 0]
    if bad:
        for b in bad:
            print(f'===== {b} (last 40 lines) =====')
            print('\n'.join(_tail(b, 40)))
        raise RuntimeError(f'queue(s) failed: {bad}')
    print(f'all queues finished in {(time.time() - t0) / 60:.1f} min')
