"""GPU check of the training speed-ups (run on the CUDA machine before a long run):

  python -m tests.check_speedups                 # CBN and per-label generators, 62 labels (EMNIST-like)
  python -m tests.check_speedups --steps 400 --device cuda:1

Every mode starts from the same weights, optimizer state and noise generator and sees the same batches.
  old      plain Adam, every op launched one by one (the code before)
  fused    fused Adam, op by op
  graph    fused Adam + the step replayed as one CUDA graph (what runs now)
Expected: fused vs graph identical and fused vs fused-again identical (max diff 0), old vs fused small
float differences (fused Adam rounds differently). Also printed: the GPU's own time per step (graph
replay alone), the same with cuDNN free to pick non-deterministic kernels, and other processes on the GPU
(a busy GPU makes every step slow). A capture failure prints a line and falls back to op by op.
"""
import argparse
import copy
import time

import torch

from nets import DCGANDiscriminator
from secfl.cbn_gan import CBNGenerator, ClientCBNGAN, DCGANTemplate, PerLabelGenerator
from tensor_loader import TensorLoader


def run(G, D, mode, x, y, steps, device):
    config = dict(gen_noise_dim=128, gen_local_epochs=1, cuda_graph=mode == 'graph', fused_adam=mode != 'old')
    gan = ClientCBNGAN(copy.deepcopy(G), copy.deepcopy(D), config, device, seed=1)
    loader = TensorLoader.from_tensors(x, y, 64, shuffle=True)
    loader.sampler.generator.manual_seed(2)
    per_epoch = len(y) // 64
    gan.epochs = max(1, steps // per_epoch)
    torch.cuda.synchronize(device)
    t = time.perf_counter()
    gan.train(loader)
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t
    return gan, dt / (gan.epochs * per_epoch)


def diff(a, b):
    return max((pa - pb).abs().max().item() for pa, pb in
               zip(list(a.G.parameters()) + list(a.D.parameters()), list(b.G.parameters()) + list(b.D.parameters())))


def gpu_ms_per_replay(gan, device, n=50):
    """Pure GPU time of one captured step (no Python, no data copies)."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    for _ in range(n):
        gan._graph.replay()
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) / n


def others_on_gpu():
    import subprocess
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=10).stdout.strip()
        return out or '(none visible)'
    except Exception as e:
        return f'(nvidia-smi not available: {e})'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--labels', type=int, default=62)
    a = ap.parse_args()
    print(f'GPU: {torch.cuda.get_device_name(a.device)}   torch {torch.__version__}')
    print(f'processes on the GPU now:\n{others_on_gpu()}\n')
    g = torch.Generator().manual_seed(0)
    n = 64 * 50 + 17                                                  # a partial last batch too
    x = torch.randint(0, 256, (n, 1, 32, 32), dtype=torch.uint8, generator=g)
    y = torch.randint(0, a.labels, (n,), generator=g)
    torch.manual_seed(0)
    gens = {'cbn': CBNGenerator(a.labels, 128),
            'per-label': PerLabelGenerator(a.labels, DCGANTemplate(128, 3, (64, 32, 16)))}
    D = DCGANDiscriminator(a.labels, img_size=32, channels=3)
    ok = True
    for name, G in gens.items():
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False    # as the runs
        res = {m: run(G, D, m, x, y, a.steps, a.device) for m in ('old', 'fused', 'graph')}
        res['fused_again'] = run(G, D, 'fused', x, y, a.steps, a.device)
        gan = res['graph'][0]
        captured = gan._graph is not None
        same = diff(res['fused'][0], gan)
        repeat = diff(res['fused'][0], res['fused_again'][0])
        print(f"{name:>9}: ms/step  old {res['old'][1] * 1e3:.1f} | fused {res['fused'][1] * 1e3:.1f} | "
              f"graph {res['graph'][1] * 1e3:.1f} (captured: {captured})")
        if captured:
            print(f"{'':>9}  GPU alone per step (graph replay): {gpu_ms_per_replay(gan, a.device):.1f} ms")
        torch.backends.cudnn.deterministic = False                    # informational: cuDNN free choice
        fast = run(G, D, 'graph', x, y, a.steps, a.device)
        if fast[0]._graph is not None:
            print(f"{'':>9}  same, cuDNN not deterministic: {fast[1] * 1e3:.1f} ms/step, "
                  f"GPU alone {gpu_ms_per_replay(fast[0], a.device):.1f} ms")
        print(f"{'':>9}  max|fused-graph| = {same:.2e}   max|fused-fused again| = {repeat:.2e}   "
              f"max|old-fused| = {diff(res['old'][0], res['fused'][0]):.2e}")
        ok &= captured and same == 0 and repeat == 0
    print('OK: graph replay gives the same weights' if ok else
          'CHECK: graph not captured or not identical; send me this output')


if __name__ == '__main__':
    main()
