"""Clients in worker processes instead of threads (same results, no GIL).

A training step of these small GANs is launch-bound: one CPU core per client is the limit, and threads
share one core through the GIL. Here every worker process owns a group of clients for the whole run:
their GAN (G, D, optimizers, noise generator) and their images (pre-decoded, memory-mapped from a file,
so no shared memory is needed; docker's /dev/shm is small). The main process keeps a ClientProxy with
the methods the round loop uses (train, update, load_global, _ref, state_dict), so the protocol code
does not change. Messages are torch.save bytes over a pipe (no tensor sharing either).
"""
import io
import os
import threading
import traceback

import torch
import torch.multiprocessing as mp


def _dumps(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _loads(b):
    return torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)


def _cpu(obj):
    """Every tensor in a (nested) state to the CPU."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu(v) for v in obj)
    return obj


def _serve(conn, device, flags, threads):
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags
    torch.set_num_threads(threads)
    from secfl.cbn_gan import ClientCBNGAN
    from tensor_loader import TensorLoader
    gans, loaders = {}, {}
    while True:
        msg = conn.recv_bytes()
        if not msg:
            break
        cmd, cid, arg = _loads(msg)
        try:
            if cmd == 'add':
                G, D, config, state, guide, weight, data, bs, epochs = arg
                x, y = torch.load(data, mmap=True, weights_only=True)
                loaders[cid] = TensorLoader.from_tensors(x, y, bs, shuffle=True)
                gan = ClientCBNGAN(G, D, config, device)
                if guide is not None:
                    gan.set_guide(guide, weight)
                gan.load_state_dict(state)
                gan.epochs = epochs
                if state.get('shuffle') is not None:
                    loaders[cid].sampler.generator.set_state(state['shuffle'])
                gans[cid] = gan
                out = None
            elif cmd == 'train':
                counts = gans[cid].train(loaders[cid])
                out = (counts, gans[cid].update() if counts else None)
            elif cmd == 'load_global':
                gans[cid].load_global(*arg)
                out = None
            elif cmd == 'ref_rows':
                out = gans[cid]._ref[1]
            elif cmd == 'state':
                out = dict(_cpu(gans[cid].state_dict()), shuffle=loaders[cid].sampler.generator.get_state())
            else:
                raise ValueError(cmd)
            conn.send_bytes(_dumps(('ok', out)))
        except Exception:
            conn.send_bytes(_dumps(('err', traceback.format_exc())))


class ClientPool:
    """Worker processes, each with its own device and a fixed group of clients."""

    def __init__(self, local, clients, procs, scratch):
        """local: secure_cbn's {cid: dict(gan=ClientCBNGAN, loader=TensorLoader, shuffle=...)}."""
        ctx = mp.get_context('spawn')
        devices = sorted({str(local[c.id]['gan'].device) for c in clients})
        per_dev = max(1, -(-procs // len(devices)))                  # ceil: 3 clients on 2 GPUs -> 2+1, not 1+1
        threads = max(1, (os.cpu_count() or 1) // max(procs, 1))
        flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        self.workers, self.where, seen = {}, {}, {}
        for c in clients:                                            # clients of a device, round-robin
            dev = str(local[c.id]['gan'].device)
            key = (dev, seen.setdefault(dev, 0) % per_dev)
            seen[dev] += 1
            if key not in self.workers:
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_serve, args=(child, dev, flags, threads), daemon=True)
                p.start()
                self.workers[key] = (parent, p, threading.Lock())
            self.where[c.id] = key
        os.makedirs(scratch, exist_ok=True)
        try:
            self._hand_over(local, clients, scratch)
        except Exception:
            self.close()
            raise

    def _hand_over(self, local, clients, scratch):
        for c in clients:                                            # hand each client over, once
            v, gan = local[c.id], local[c.id]['gan']
            data = os.path.join(scratch, f'client_{c.id}.pt')
            torch.save((v['loader'].dataset.x, v['loader'].dataset.y), data)
            guide = None if gan.guide is None else _copy_cpu(gan.guide)
            state = dict(_cpu(gan.state_dict()), shuffle=getattr(v['shuffle'], 'generator', None).get_state()
                         if getattr(v['shuffle'], 'generator', None) is not None else None)
            self.call(c.id, 'add', (_copy_cpu(gan.G), _copy_cpu(gan.D), gan.config, state, guide,
                                    gan.guide_weight, data, v['loader'].batch_size, gan.epochs))

    def call(self, cid, cmd, arg=None):
        conn, proc, lock = self.workers[self.where[cid]]
        with lock:
            conn.send_bytes(_dumps((cmd, cid, arg)))
            try:
                status, out = _loads(conn.recv_bytes())
            except (EOFError, OSError):
                proc.join(timeout=1)
                raise RuntimeError(f'client worker for client {cid} died (exit code {proc.exitcode})') from None
        if status != 'ok':
            raise RuntimeError(f'client {cid} failed in its worker process:\n{out}')
        return out

    def proxy(self, cid, gan):
        return ClientProxy(self, cid, gan)

    def close(self):
        for conn, proc, lock in self.workers.values():
            try:
                with lock:
                    conn.send_bytes(b'')
            except (BrokenPipeError, OSError):
                pass
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()


def _copy_cpu(module):
    import copy
    return copy.deepcopy(module).cpu()


class ClientProxy:
    """Stands in for a ClientCBNGAN that lives in a worker process."""

    def __init__(self, pool, cid, gan):
        self._pool, self._cid = pool, cid
        self.device, self.epochs, self.guide = gan.device, gan.epochs, gan.guide
        self._update = None

    def train(self, loader=None):
        counts, self._update = self._pool.call(self._cid, 'train')
        return counts

    def update(self):
        return self._update

    def load_global(self, trunk, own_rows):
        self._pool.call(self._cid, 'load_global', (trunk, own_rows))

    @property
    def _ref(self):
        return None, self._pool.call(self._cid, 'ref_rows')

    def state_dict(self):
        return self._pool.call(self._cid, 'state')
