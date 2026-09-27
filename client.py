"""Independent GeFL participant. All incoming/outgoing state is copied."""
import copy
import torch
from clustering import local_basis
from contracts import ClientUpdate, GANState, clone_state


class Client:
    def __init__(self, client_id, model, generator, discriminator, train_loader,
                 num_classes, config=None, device='cpu', pretrained=False):
        config = config or {}
        self.id, self.device = client_id, device
        self.model = model.to(device)
        self.generator, self.discriminator = generator.to(device), discriminator.to(device)
        self.train_loader, self.num_samples = train_loader, len(train_loader.dataset)
        if not self.num_samples:
            raise ValueError('Client training data cannot be empty')
        self.local_num_classes = num_classes
        self.noise_dim = config.get('gen_noise_dim', 128)
        self.local_epochs = config.get('local_epochs', 1)
        self.gen_local_epochs = config.get('gen_local_epochs', 5)
        if self.local_epochs < 1 or self.gen_local_epochs < 1:
            raise ValueError('Local classifier and generator epochs must be positive')
        self.aid_by_gen = config.get('aid_by_gen', False)
        self.gen_sample_ratio = config.get('gen_sample_ratio', 1.)
        self.local_loss_fn = torch.nn.CrossEntropyLoss()
        self.adv_criterion = torch.nn.BCEWithLogitsLoss()
        self.local_optimizer = getattr(torch.optim, config.get('local_optim', 'Adam'))(
            self.model.parameters(), lr=config.get('local_lr', 1e-3))
        gan_args = dict(lr=config.get('gen_lr', 2e-4),
                        betas=(config.get('gan_beta1', .5), config.get('gan_beta2', .999)))
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), **gan_args)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), **gan_args)
        self.trained = pretrained

    def basis(self, budget=20):
        return local_basis(self.train_loader, budget)

    def train(self, round_index, train_classifier=True):
        stats = self.train_generator()
        if train_classifier:
            stats['classifier_loss'] = self.train_target_model()
        self.trained = True
        return stats

    def receive(self, state: GANState):
        self.generator.load_state_dict(state.generator)
        self.discriminator.load_state_dict(state.discriminator)

    def export(self, group):
        if not self.trained:
            raise ValueError('Train classifier and generator before aggregation')
        return ClientUpdate(self.id, group, self.num_samples, GANState(
            clone_state(self.generator.state_dict()), clone_state(self.discriminator.state_dict())))

    def classifier_snapshot(self):
        return copy.deepcopy(self.model).cpu().eval()

    def train_target_model(self):
        self.model.to(self.device).train()
        self.generator.eval()
        losses = []
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.local_optimizer.zero_grad()
                result = self.model(x)
                logits = result[1] if isinstance(result, tuple) else result
                loss = self.local_loss_fn(logits, y)
                if self.aid_by_gen:
                    x_gen, y_gen = self.sample_generated(max(1, int(len(x) * self.gen_sample_ratio)))
                    result = self.model(x_gen)
                    logits = result[1] if isinstance(result, tuple) else result
                    loss = loss + self.local_loss_fn(logits, y_gen)
                loss.backward()
                self.local_optimizer.step()
                losses.append(loss.item())
        return sum(losses) / max(1, len(losses))

    def train_generator(self):
        self.generator.train()
        self.discriminator.train()
        g_total = d_total = steps = 0
        for _ in range(self.gen_local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                n = len(x)
                real = torch.ones(n, 1, device=self.device)
                fake = torch.zeros(n, 1, device=self.device)
                self.d_optimizer.zero_grad()
                d_real = self.adv_criterion(self.discriminator(x, y).view(-1, 1), real)
                z = torch.randn(n, self.noise_dim, device=self.device)
                generated = self.generator(z, y).detach()
                d_fake = self.adv_criterion(self.discriminator(generated, y).view(-1, 1), fake)
                d_loss = .5 * (d_real + d_fake)
                d_loss.backward()
                self.d_optimizer.step()
                self.g_optimizer.zero_grad()
                z = torch.randn(n, self.noise_dim, device=self.device)
                generated = self.generator(z, y)
                g_loss = self.adv_criterion(self.discriminator(generated, y).view(-1, 1), real)
                g_loss.backward()
                self.g_optimizer.step()
                g_total += g_loss.item()
                d_total += d_loss.item()
                steps += 1
        return dict(g_loss=g_total / max(1, steps), d_loss=d_total / max(1, steps))

    def sample_generated(self, n):
        labels = torch.randint(self.local_num_classes, (n,), device=self.device)
        z = torch.randn(n, self.noise_dim, device=self.device)
        with torch.no_grad():
            return self.generator(z, labels), labels

