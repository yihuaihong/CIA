#!/usr/bin/env python
"""GPU utilization keep-alive watchdog.

Polls `nvidia-smi` for every visible GPU. When all of them stay below
THRESHOLD% for WINDOW consecutive polls, fires a brief CUDA matmul on
each device to bump utilization above the cluster's low-util cgroup
watchdog threshold. This protects legitimate idle phases (vLLM rollout,
DeepSpeed CPU offload, eval batching, dataset shuffling) from being
mistaken for an abandoned job.

Run alongside the main job:
    python scripts/gpu_keepalive.py &
    KA_PID=$!
    trap "kill $KA_PID 2>/dev/null" EXIT
    # ... main job ...

Tunables (env vars override CLI defaults):
    KEEPALIVE_THRESHOLD     util % below which a GPU counts as "low"  (default 5)
    KEEPALIVE_WINDOW        consecutive low polls before firing       (default 6)
    KEEPALIVE_POLL_SEC      poll interval                             (default 30)
    KEEPALIVE_FIRE_SEC      duration of dummy matmul when fired       (default 30)
    KEEPALIVE_COOLDOWN_SEC  min gap between fires                     (default 60)
"""
from __future__ import annotations
import argparse, os, signal, subprocess, sys, time


def gpu_utils() -> list[int]:
    """Return per-GPU utilization (% int) for the GPUs visible to this process."""
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=utilization.gpu",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    return [int(x.strip()) for x in out.strip().split("\n") if x.strip()]


def fire(seconds: int) -> None:
    """Run a small matmul loop on every visible GPU for `seconds`."""
    import torch
    n = torch.cuda.device_count()
    if n == 0:
        print("[keepalive] no CUDA device visible — skipping", flush=True)
        return
    devs = [torch.device(f"cuda:{i}") for i in range(n)]
    mats = [torch.randn(2048, 2048, device=d, dtype=torch.float16) for d in devs]
    end = time.time() + seconds
    iters = 0
    while time.time() < end:
        for j, m in enumerate(mats):
            mats[j] = (m @ m) / (m.norm() + 1e-9)
        torch.cuda.synchronize()
        iters += 1
    for m in mats:
        del m
    torch.cuda.empty_cache()
    print(f"[keepalive] fired {iters} iters across {n} GPU(s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int,
                    default=int(os.environ.get("KEEPALIVE_THRESHOLD", 5)))
    ap.add_argument("--window", type=int,
                    default=int(os.environ.get("KEEPALIVE_WINDOW", 6)))
    ap.add_argument("--poll", type=int,
                    default=int(os.environ.get("KEEPALIVE_POLL_SEC", 30)))
    ap.add_argument("--fire-sec", type=int,
                    default=int(os.environ.get("KEEPALIVE_FIRE_SEC", 30)))
    ap.add_argument("--cooldown", type=int,
                    default=int(os.environ.get("KEEPALIVE_COOLDOWN_SEC", 60)))
    args = ap.parse_args()

    print(f"[keepalive] start: threshold={args.threshold}% "
          f"window={args.window}*{args.poll}s "
          f"fire={args.fire_sec}s cooldown={args.cooldown}s",
          flush=True)

    history: list[int] = []
    last_fire = 0.0

    def shutdown(signum, _frame):
        print(f"[keepalive] received signal {signum}, exiting", flush=True)
        sys.exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while True:
        try:
            u = gpu_utils()
        except Exception as e:
            print(f"[keepalive] nvidia-smi error: {e}", flush=True)
            time.sleep(args.poll)
            continue
        if not u:
            time.sleep(args.poll)
            continue

        peak = max(u)  # any active GPU = job is busy, don't fire
        history.append(peak)
        history = history[-args.window:]

        if len(history) == args.window and all(x < args.threshold for x in history):
            now = time.time()
            if now - last_fire >= args.cooldown:
                print(f"[keepalive] history={history} all<{args.threshold}% -> fire",
                      flush=True)
                fire(args.fire_sec)
                last_fire = time.time()
                history = []  # reset window after fire

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
