"""Train global classifier from group generators and a chosen relation table."""
import torch
from torch.utils.data import DataLoader, TensorDataset


def augment(x, shift=0.25, flip=True, noise=0.05):
    """Per-image random shift (up to shift/2 of the side, reflected border), horizontal flip and pixel
    noise, on the GPU in one grid_sample: breaks per-generator fingerprints of the synthetic images
    (config global_augment; flips suit photos, not digits or letters: global_augment_flip)."""
    n = len(x)
    sx = torch.where(torch.rand(n, device=x.device) < .5, -1., 1.) if flip else torch.ones(n, device=x.device)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 1, 1] = sx, 1.
    theta[:, :, 2] = (torch.rand(n, 2, device=x.device) * 2 - 1) * shift
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, padding_mode='reflection', align_corners=False)
    return (x + noise * torch.randn_like(x)).clamp_(-1, 1)


class GlobalClassifierTrainer:
    def __init__(self, model_factory, config=None, device='cpu'):
        self.factory, self.config, self.device = model_factory, config or {}, device
        self.model = self.optimizer = None
        self._mapping = None

    def __call__(self, generators, mapping):
        config = self.config
        ids = {gid for labels in mapping.values() for gid in labels.values()}
        if ids != set(range(len(ids))) or not ids:
            raise ValueError('Global training requires nonempty contiguous global IDs')
        samples = config.get('global_samples_per_class', 1)
        epochs = config.get('global_model_epochs', 1)
        if samples < 1 or epochs < 1:
            raise ValueError('Global training samples and epochs must be positive')
        if self.model is None or mapping != self._mapping:
            import copy
            self._mapping = copy.deepcopy(mapping)
            self.model = self.factory(len(ids)).to(self.device)
            self.optimizer = getattr(torch.optim, config.get('global_model_optim', 'Adam'))(
                self.model.parameters(), lr=config.get('global_model_optim_lr', 1e-3))
        def synthetic():
            xs, ys = [], []
            for group, labels in mapping.items():
                generator = generators[group].to(self.device).eval()
                for local, global_id in labels.items():
                    z = torch.randn(samples, config.get('gen_noise_dim', 128), device=self.device)
                    y = torch.full((samples,), local, dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        xs.append(generator(z, y).cpu())
                    ys.append(torch.full((samples,), global_id, dtype=torch.long))
            return DataLoader(TensorDataset(torch.cat(xs), torch.cat(ys)),
                              batch_size=config.get('batch_size', 64), shuffle=True)

        loader = synthetic()
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        aug = config.get('global_augment', False)
        for epoch in range(epochs):
            if epoch and config.get('global_resample', False):        # fresh synthetic images every epoch
                loader = synthetic()
            for images, labels in loader:
                self.optimizer.zero_grad()
                images = images.to(self.device)
                if aug:
                    images = augment(images, flip=config.get('global_augment_flip', True))
                output = self.model(images)
                logits = output[1] if isinstance(output, tuple) else output
                loss = criterion(logits, labels.to(self.device))
                loss.backward()
                self.optimizer.step()
        return self.model

