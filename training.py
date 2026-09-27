"""Train global classifier from group generators and a chosen relation table."""
import torch
from torch.utils.data import DataLoader, TensorDataset


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
        xs, ys = [], []
        for group, labels in mapping.items():
            generator = generators[group].to(self.device).eval()
            for local, global_id in labels.items():
                z = torch.randn(samples, config.get('gen_noise_dim', 128), device=self.device)
                y = torch.full((samples,), local, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    xs.append(generator(z, y).cpu())
                ys.append(torch.full((samples,), global_id, dtype=torch.long))
        loader = DataLoader(TensorDataset(torch.cat(xs), torch.cat(ys)),
                            batch_size=config.get('batch_size', 64), shuffle=True)
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(epochs):
            for images, labels in loader:
                self.optimizer.zero_grad()
                output = self.model(images.to(self.device))
                logits = output[1] if isinstance(output, tuple) else output
                loss = criterion(logits, labels.to(self.device))
                loss.backward()
                self.optimizer.step()
        return self.model

