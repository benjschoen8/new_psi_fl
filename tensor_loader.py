"""Pre-decoded image loader: runs the (deterministic) torchvision transforms once, keeps the images as
uint8 (one channel if grey), and serves batches by indexing. Bit-identical to the DataLoader it replaces:
same values (checked when built), same shuffle order for the same generator (RandomSampler), same data_hash. The old loader decoded every image every epoch on one CPU core
(~4k images/s: EMNIST 5 epochs x 698k images = ~15 min per round)."""
import math
import os
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler


def _decode(u8):
    x = u8.float().div_(255).sub_(.5).div_(.5)                  # = ToTensor + Normalize(0.5, 0.5)
    return x.expand(-1, 3, -1, -1).contiguous() if x.size(1) == 1 else x    # contiguous: same kernels


def _base_seed(generator):
    """Every DataLoader iteration draws one worker base seed from its generator (global RNG if None);
    drawn here too, so every later random number in the run stays the same."""
    torch.empty((), dtype=torch.int64).random_(generator=generator)


def _order(n, generator):
    """The DataLoader's shuffle order: RandomSampler itself, so the generator advances exactly as it
    would (it draws an extra permutation per epoch in recent torch)."""
    return torch.tensor(list(RandomSampler(range(n), generator=generator)), dtype=torch.long)


class TensorData(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return _decode(self.x[i:i + 1])[0], int(self.y[i])


class TensorLoader:
    def __init__(self, loader, shuffle):
        workers, rng = min(8, os.cpu_count() or 1), torch.get_rng_state()
        try:
            xs, ys = self._read(loader.dataset, workers)
        except (RuntimeError, AttributeError, TypeError, OSError):    # e.g. transforms not picklable
            xs, ys = self._read(loader.dataset, 0)
        finally:
            torch.set_rng_state(rng)                             # building must not move the global RNG
        if len({x.size(1) for x in xs}) > 1:                     # mixed grey / colour: keep 3 channels
            xs = [x.expand(-1, 3, -1, -1).contiguous() for x in xs]
        self.dataset = TensorData(torch.cat(xs), torch.cat(ys))
        self.batch_size = loader.batch_size or 64
        self.sampler = SimpleNamespace(generator=torch.Generator()) if shuffle else None

    @classmethod
    def from_tensors(cls, x, y, batch_size, shuffle):
        """Rebuild from the stored uint8 images and labels (client worker processes)."""
        self = cls.__new__(cls)
        self.dataset, self.batch_size = TensorData(x, y), batch_size
        self.sampler = SimpleNamespace(generator=torch.Generator()) if shuffle else None
        return self

    @staticmethod
    def _read(dataset, workers):
        xs, ys = [], []
        for x, y in DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=workers):
            u8 = torch.round((x * .5 + .5) * 255).clamp_(0, 255).to(torch.uint8)
            if u8.size(1) == 3 and torch.equal(u8[:, 0], u8[:, 1]) and torch.equal(u8[:, 0], u8[:, 2]):
                u8 = u8[:, :1].contiguous()
            if not torch.equal(_decode(u8), x):
                raise ValueError('images are not 8-bit after the transforms; keep the DataLoader')
            xs.append(u8)
            ys.append(torch.as_tensor(y))
        return xs, ys

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def _batches(self, order, bs):
        d = self.dataset
        for s in range(0, len(order), bs):
            j = order[s:s + bs]
            yield _decode(d.x[j]), d.y[j]

    def __iter__(self):
        n = len(self.dataset)
        _base_seed(None)                                         # a DataLoader(generator=None) draws it
        order = _order(n, self.sampler.generator) if self.sampler else torch.arange(n)
        return self._batches(order, self.batch_size)

    def ordered(self, bs=1024):
        """Dataset order (data_hash)."""
        _base_seed(None)
        return self._batches(torch.arange(len(self.dataset)), bs)

    def shuffled(self, generator, bs=None):
        """Own shuffle, like DataLoader(dataset, shuffle=True, generator=generator)."""
        _base_seed(generator)                                    # DataLoader(..., generator=g): from g
        return self._batches(_order(len(self.dataset), generator), bs or self.batch_size)
