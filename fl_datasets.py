import os
import json
import random
import torch
import numpy as np
from collections import Counter, defaultdict
from torchvision import datasets, transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
from dirichlet_noniid import partition_data, partition_data_noniid_label, partition_data_quantity_skew, partition_data_quantity_skew_equalSize

class Global_Dataset(Dataset):
    def __init__(self, dataset, mapping):
        self.dataset = dataset
        self.mapping = mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, local_label = self.dataset[idx]
        global_label = self.mapping[int(local_label)]
        return img, global_label

class LabelPermutedDataset(Dataset):
    def __init__(self, original_dataset, mapping_dict):
        self.dataset = original_dataset
        self.mapping_dict = mapping_dict
        
        if hasattr(self.dataset, 'targets'):
            self.targets = [self.mapping_dict[int(y)] for y in self.dataset.targets]
        elif hasattr(self.dataset, 'labels'):
            self.labels = [self.mapping_dict[int(y)] for y in self.dataset.labels]
        else:
            self.targets = [self.mapping_dict[int(y)] for _, y in self.dataset.samples]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, original_label = self.dataset[idx]
        permuted_label = self.mapping_dict[int(original_label)]
        return img, permuted_label

class ClassSubsetDataset(Dataset):
    """A client's view of a dataset restricted to its own classes: labels renumbered 0..k-1 in the
    order of `classes` (global label ids); `classes` holds their names (setup.label_names reads it)."""
    def __init__(self, base, classes, names):
        self.base, self.remap = base, {int(c): i for i, c in enumerate(classes)}
        self.classes = [names[int(c)] for c in classes]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        return img, self.remap[int(y)]


def partition_class_subsets(train_labels, test_labels, n_clients, lo, hi, seed, min_holders=2):
    """Every client draws k in [lo, hi] classes; classes go to the least-covered ones first (random
    ties), so coverage is even; redrawn until every class has >= min_holders clients (if possible).
    A class's train / test samples are split evenly at random among its holders.
    Returns (classes per client (sorted global ids), train idcs, test idcs)."""
    train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
    C, rng = int(max(train_labels.max(), test_labels.max())) + 1, np.random.default_rng(seed)
    for _ in range(1000):
        ks = rng.integers(lo, hi + 1, n_clients)
        cover, own = np.zeros(C, int), []
        for k in ks:
            order = np.lexsort((rng.random(C), cover))                  # least covered first, random ties
            mine = np.sort(order[:k])
            cover[mine] += 1
            own.append(mine)
        if cover.min() >= min(min_holders, ks.sum() // C):
            break
    tr, te = {i: [] for i in range(n_clients)}, {i: [] for i in range(n_clients)}
    for c in range(C):
        holders = [i for i in range(n_clients) if c in own[i]]
        for labels, out in ((train_labels, tr), (test_labels, te)):
            idx = rng.permutation(np.flatnonzero(labels == c))
            for h, part in zip(holders, np.array_split(idx, len(holders))):
                out[h] += part.tolist()
    return [m.tolist() for m in own], tr, te


def get_split_cache_path(DATA_ROOT, dataset_name, alpha, total_clients, num_new_clients, seed):
    cache_dir = os.path.join(DATA_ROOT, "splits")
    os.makedirs(cache_dir, exist_ok=True)

    # 把dirichlet alpha的.去掉改成p (e.g. 0.1 -> 0p1)
    alpha_str = f"{alpha:.1f}".replace(".", "p")

    filename = f"{dataset_name}_C{total_clients}_New{num_new_clients}_alpha{alpha_str}_seed{seed}.json"
    return os.path.join(cache_dir, filename)

# ==========================================
# Get Label Counts
# ==========================================
def get_label_counts(dataset, indices):
    """
    label count of subset
    """
    if hasattr(dataset, 'targets'):
        all_labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        all_labels = np.array(dataset.labels)
    else:
        all_labels = np.array([y for x, y in dataset.samples])
    
    subset_labels = all_labels[indices]

    counter = Counter(subset_labels)
    
    return ", ".join([f"{k}:{v}" for k, v in sorted(counter.items())])

# ==========================================
# Dataset Transform
# ==========================================
def get_transforms(name):
    if name in ['MNIST', 'FashionMNIST', 'USPS']:
        return transforms.Compose([
            transforms.Resize((32, 32)),                 
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))

            # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])

    elif name == 'EMNIST':
        return transforms.Compose([
            transforms.Resize((32, 32)),                 
            transforms.Grayscale(num_output_channels=3),
            # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(1, 2)),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    elif name in ['CIFAR10', 'CIFAR100']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    return None

# ==========================================
# Reading Raw Datasets
# ==========================================
def get_raw_dataset_transform(name, root, train=True):
    transform = get_transforms(name)
    
    if name == 'MNIST':
        return datasets.MNIST(root, train=train, download=True, transform=transform)
    
    elif name == 'FashionMNIST':
        return datasets.FashionMNIST(root, train=train, download=True, transform=transform)
    
    elif name == 'EMNIST':
        # return datasets.EMNIST(root, split='balanced', train=train, download=False, transform=transform)
        # return datasets.EMNIST(root, split='bymerge', train=train, download=False, transform=transform)
        return datasets.EMNIST(root, split='byclass', train=train, download=True, transform=transform)
    
    elif name == 'CIFAR10':
        return datasets.CIFAR10(root, train=train, download=True, transform=transform)
    
    elif name == 'CIFAR100':
        return datasets.CIFAR100(root, train=train, download=True, transform=transform)

    elif name == 'USPS':
        return datasets.USPS(root, train=train, download=True, transform=transform)
    
# ==========================================
# Get Readable Class Names
# ==========================================
def get_readable_class_names(name, root='./data/raw'):
    if name in ['MNIST', 'USPS']:
        return [str(i) for i in range(10)]

    elif name == 'FashionMNIST':
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
    elif name == 'EMNIST':
        # d = datasets.EMNIST(root, split='balanced', train=True, download=False)
        # d = datasets.EMNIST(root, split='bymerge', train=True, download=True)
        d = datasets.EMNIST(root, split='byclass', train=True, download=True)
        return d.classes
        
    elif name == 'CIFAR10':
        # CIFAR10 in-build classes: ['airplane', 'automobile', 'bird', ...]
        d = datasets.CIFAR10(root, train=True, download=True)
        return d.classes
    
    elif name == 'CIFAR100':
        # CIFAR100 in-build classes: ['apple', 'aquarium_fish', ...]
        d = datasets.CIFAR100(root, train=True, download=True)
        return d.classes        

# ==========================================
# Loading Datasets
# ==========================================
def load_partitioned_datasets(args, DATA_ROOT, **exp_conf):
    # configs
    dataset_configs = {
        'MNIST': (args.num_train_mnist + args.num_new_clients) if args.num_train_mnist > 0 else 0,
        'FashionMNIST': (args.num_train_fashionmnist + args.num_new_clients) if args.num_train_fashionmnist > 0 else 0,
        'EMNIST': (args.num_train_emnist + args.num_new_clients) if args.num_train_emnist > 0 else 0,
        'CIFAR10': (args.num_train_cifar10 + args.num_new_clients) if args.num_train_cifar10 > 0 else 0,
        'CIFAR100': (args.num_train_cifar100 + args.num_new_clients) if args.num_train_cifar100 > 0 else 0,
        'USPS': (args.num_train_usps + args.num_new_clients) if args.num_train_usps > 0 else 0
    }
    batch_size = exp_conf.get('batch_size', 64)
    dirichlet_alpha = exp_conf.get('dirichlet_alpha', 0.1)
    public_ratio = exp_conf.get('public_data_ratio', 1.0)

    # Partition Datasets
    print(f"{'='*100}")
    print(f"Loading Datasets with Non-IID Split (Dirichlet distribution, Alpha={dirichlet_alpha})")
    print(f"{'='*100}")

    all_client_data_loaders = {}
    server_train_loaders = {}
    server_test_loaders = {}
    usps_label_mapping = None

    for d_name, n_clients in dataset_configs.items():
        if n_clients == 0:
            continue
        
        # --- Readable Class Names (e.g. dog, cat ...) ---
        class_names = get_readable_class_names(d_name, root=DATA_ROOT)
        print(f"[-] Readable Class Names ({len(class_names)} classes): {class_names}\n")

        print(f"\n>>> Processing {d_name} ({n_clients} Clients)")   

        # Getting Datasets
        train_dataset = get_raw_dataset_transform(d_name, DATA_ROOT, train=True)
        test_dataset = get_raw_dataset_transform(d_name, DATA_ROOT, train=False)

        if d_name == 'USPS':
            if usps_label_mapping is None:
                original_labels = list(range(10))
                shuffled_labels = original_labels.copy()
                random.shuffle(shuffled_labels)
                usps_label_mapping = {ori: shf for ori, shf in zip(original_labels, shuffled_labels)}
                print(f"[!] USPS Label Shuffled Mapping: {usps_label_mapping}")
            
            train_dataset = LabelPermutedDataset(train_dataset, usps_label_mapping)
            test_dataset = LabelPermutedDataset(test_dataset, usps_label_mapping)

        # Getting Labels
        # 拿這個 dataset 底下每一筆資料的 label
        if hasattr(train_dataset, 'targets'): 
            train_labels = train_dataset.targets
        elif hasattr(train_dataset, 'labels'): 
            train_labels = train_dataset.labels
        else: 
            train_labels = []
            for path, label in train_dataset.samples:
                train_labels.append(label)

        if hasattr(test_dataset, 'targets'): 
            test_labels = test_dataset.targets
        elif hasattr(test_dataset, 'labels'): 
            test_labels = test_dataset.labels
        else: 
            test_labels = []
            for path, label in test_dataset.samples:
                test_labels.append(label)
                
        train_labels_for_split = train_labels
        test_labels_for_split  = test_labels

        # Dirichlet Non-IID Partition
        cache_path = get_split_cache_path(
            DATA_ROOT,
            d_name,
            dirichlet_alpha,
            n_clients,
            args.num_new_clients,
            args.seed 
        )

        subsets = getattr(args, 'class_subsets', None)
        if subsets:                                                    # own classes per client (no cache:
            lo, hi = map(int, subsets.split(','))                      # deterministic from the seed)
            own, train_idcs, test_idcs = partition_class_subsets(train_labels, test_labels, n_clients, lo, hi,
                                                                 args.seed)
            names = list(train_dataset.classes)
            client_loaders = []
            for i in range(n_clients):
                print(f"{i:<6} | {len(train_idcs[i]):<6} | {len(test_idcs[i]):<6} | {len(own[i])} classes: "
                      f"{[names[c] for c in own[i]]}")
                client_loaders.append({
                    'train': DataLoader(Subset(ClassSubsetDataset(train_dataset, own[i], names), train_idcs[i]),
                                        batch_size=batch_size, shuffle=True, num_workers=0),
                    'test': DataLoader(Subset(ClassSubsetDataset(test_dataset, own[i], names), test_idcs[i]),
                                       batch_size=batch_size, shuffle=False, num_workers=0)})
            all_client_data_loaders[d_name] = client_loaders
            continue
        if args.noniid_partition in ["noniid_label", "quantity_skew", "quantity_skew_equalSize"]:
            cache_dir = os.path.dirname(cache_path)
            cache_name = f"{d_name}_C{n_clients}_New{args.num_new_clients}_{args.noniid_partition}_alpha{str(dirichlet_alpha).replace('.', 'p')}_seed{args.seed}.json"
            cache_path = os.path.join(cache_dir, cache_name)

        if os.path.exists(cache_path):
            print(f"Found existing split for {d_name}, loading from {cache_path}")
            with open(cache_path, "r") as f:
                cached = json.load(f)

            # json 會把 key 變成字串所以要轉回 int
            train_idcs = {int(k): v for k, v in cached["train"].items()}
            test_idcs  = {int(k): v for k, v in cached["test"].items()}
        else:
            print(f"No existing split for {d_name}, generating new partition...")
            if args.noniid_partition == "dirichlet":
                train_idcs, test_idcs = partition_data(train_labels_for_split, test_labels_for_split, alpha=dirichlet_alpha, total_clients=n_clients, num_new_clients=args.num_new_clients)
            elif args.noniid_partition == "noniid_label":
                train_idcs, test_idcs = partition_data_noniid_label(train_labels_for_split, test_labels_for_split, alpha=dirichlet_alpha, total_clients=n_clients, num_new_clients=args.num_new_clients)
            elif args.noniid_partition == "quantity_skew":
                train_idcs, test_idcs = partition_data_quantity_skew(train_labels_for_split, test_labels_for_split, alpha=dirichlet_alpha, total_clients=n_clients, num_new_clients=args.num_new_clients)
            elif args.noniid_partition == "quantity_skew_equalSize":
                train_idcs, test_idcs = partition_data_quantity_skew_equalSize(train_labels_for_split, test_labels_for_split, alpha=dirichlet_alpha, total_clients=n_clients, num_new_clients=args.num_new_clients)

            # 把用這次參數切的資料集存在 json
            to_save = {
                "train": train_idcs,
                "test": test_idcs,
                "meta": {
                    "dataset": d_name,
                    "alpha": dirichlet_alpha,
                    "total_clients": n_clients,
                    "num_new_clients": args.num_new_clients,
                    "seed": args.seed,
                }
            }
            with open(cache_path, "w") as f:
                json.dump(to_save, f, indent=2)
            print(f"Saved split for {d_name} to {cache_path}")

        # Dataloader
        client_loaders = []
        print(f"{'Client':<6} | {'Train':<6} | {'Test':<6} | {'Sample'}")
        print("-" * 120)

        for i in range(n_clients):
            train_subset = Subset(train_dataset, train_idcs[i])
            test_subset = Subset(test_dataset, test_idcs[i])

            train_cnt = len(train_subset)
            test_cnt = len(test_subset)
            train_info_str = get_label_counts(train_dataset, train_idcs[i])
            test_info_str  = get_label_counts(test_dataset,  test_idcs[i])

            print(f"{i:<6} | {train_cnt:<6} | {test_cnt:<6} | Train: [{train_info_str}]")
            print(f"{'':<6} | {'':<6} | {'':<6} | Test : [{test_info_str}]")
            print("-" * 60) 
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            client_loaders.append({
                'train': train_loader,
                'test': test_loader
            })

        all_client_data_loaders[d_name] = client_loaders

        global_samples_per_class = exp_conf.get('global_samples_per_class', 1)

        class_indices = defaultdict(list)
        for idx in range(len(train_dataset)):
            lbl = int(train_labels_for_split[idx])
            class_indices[lbl].append(idx)
            
        selected_indices = []
        for lbl, idcs in class_indices.items():
            if len(idcs) >= global_samples_per_class:
                selected_indices.extend(random.sample(idcs, global_samples_per_class))
            else:
                selected_indices.extend(random.choices(idcs, k=global_samples_per_class))
                
        public_train_dataset = Subset(train_dataset, selected_indices)

        server_train_loaders[d_name] = DataLoader(
            #public_train_dataset, 
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,  
            num_workers=0
        )

        server_test_loaders[d_name] = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )

    print(f"{'='*100}\n")

    return all_client_data_loaders, server_train_loaders, server_test_loaders

