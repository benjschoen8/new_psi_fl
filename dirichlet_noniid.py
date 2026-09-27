import numpy as np

# ==========================================
# Mixed Partition (Non-IID for Train, IID for New)
# ==========================================
def partition_data(train_labels, test_labels, alpha, total_clients, num_new_clients):
    """
    先平均其中一份給new client, 剩下的用non-IID給訓練client
    """
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    num_train_clients = total_clients - num_new_clients
    classes = np.unique(train_labels)
    
    train_client_idcs = {k: [] for k in range(total_clients)}
    test_client_idcs = {k: [] for k in range(total_clients)}
    
    for k in classes:
        # 找出該類別所有 index
        train_idx_k = np.where(train_labels == k)[0]
        test_idx_k = np.where(test_labels == k)[0]
        
        np.random.shuffle(train_idx_k)
        np.random.shuffle(test_idx_k)
        
        # new client (IID)
        # 按 client 比例分配資料, e.g. 10個client有2個是new那就拿20%的資料給他們平分
        total_samples_train = len(train_idx_k)
        total_samples_test = len(test_idx_k)

        iid_ratio = num_new_clients / total_clients
        n_iid_train = int(total_samples_train * iid_ratio)
        n_iid_test = int(total_samples_test * iid_ratio)

        # IID pool
        train_idx_iid = train_idx_k[:n_iid_train]
        test_idx_iid = test_idx_k[:n_iid_test]
        
        # Non-IID pool
        train_idx_noniid = train_idx_k[n_iid_train:]
        test_idx_noniid = test_idx_k[n_iid_test:]
        
        # ---------------------------------------------
        # New Client
        # ---------------------------------------------
        if num_new_clients > 0:
            train_splits_iid = np.array_split(train_idx_iid, num_new_clients)
            test_splits_iid = np.array_split(test_idx_iid, num_new_clients)
            
            for i in range(num_new_clients):
                # new client id = last training client id+1
                cid = num_train_clients + i 
                train_client_idcs[cid] += train_splits_iid[i].tolist()
                test_client_idcs[cid] += test_splits_iid[i].tolist()

        # ---------------------------------------------
        # Training Clients (Dirichlet Non-IID)
        # ---------------------------------------------
        if num_train_clients > 0:
            # Dirichlet Proportions
            proportions = np.random.dirichlet(np.repeat(alpha, num_train_clients))
            
            # Normalize logic same as before
            proportions = np.array([p * (len(idx_j) < len(train_idx_noniid) / num_train_clients) 
                                    for p, idx_j in zip(proportions, [[]]*num_train_clients)])
            proportions_norm = proportions / proportions.sum()
            
            # Split Points
            train_split_pts = (np.cumsum(proportions_norm) * len(train_idx_noniid)).astype(int)[:-1]
            test_split_pts = (np.cumsum(proportions_norm) * len(test_idx_noniid)).astype(int)[:-1]
            
            train_splits_noniid = np.split(train_idx_noniid, train_split_pts)
            test_splits_noniid = np.split(test_idx_noniid, test_split_pts)
            
            for i in range(num_train_clients):
                cid = i 
                
                tr_idxs = train_splits_noniid[i].tolist()
                te_idxs = test_splits_noniid[i].tolist()
                
                if len(tr_idxs) > 0 and len(te_idxs) > 0:
                    train_client_idcs[cid] += tr_idxs
                    test_client_idcs[cid] += te_idxs
                else:
                    pass 

    return train_client_idcs, test_client_idcs


def partition_data_noniid_label(train_labels, test_labels, alpha, total_clients, num_new_clients):
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    num_train_clients = total_clients - num_new_clients
    classes = np.unique(train_labels)

    train_client_idcs = {k: [] for k in range(total_clients)}
    test_client_idcs = {k: [] for k in range(total_clients)}

    for k in classes:
        train_idx_k = np.where(train_labels == k)[0]
        test_idx_k = np.where(test_labels == k)[0]

        np.random.shuffle(train_idx_k)
        np.random.shuffle(test_idx_k)

        iid_ratio = num_new_clients / total_clients
        n_iid_train = int(len(train_idx_k) * iid_ratio)
        n_iid_test = int(len(test_idx_k) * iid_ratio)

        train_idx_iid = train_idx_k[:n_iid_train]
        test_idx_iid = test_idx_k[:n_iid_test]
        train_idx_noniid = train_idx_k[n_iid_train:]
        test_idx_noniid = test_idx_k[n_iid_test:]

        if num_new_clients > 0:
            train_splits_iid = np.array_split(train_idx_iid, num_new_clients)
            test_splits_iid = np.array_split(test_idx_iid, num_new_clients)

            for i in range(num_new_clients):
                cid = num_train_clients + i
                train_client_idcs[cid] += train_splits_iid[i].tolist()
                test_client_idcs[cid] += test_splits_iid[i].tolist()

        if num_train_clients > 0:
            proportions = np.random.dirichlet(np.repeat(alpha, num_train_clients))
            # owner_threshold = 1.0 / num_train_clients
            owner_threshold = 0.05
            owners = np.where(proportions >= owner_threshold)[0]

            if len(owners) == 0:
                owners = np.array([int(np.argmax(proportions))])

            train_splits = np.array_split(train_idx_noniid, len(owners))
            test_splits = np.array_split(test_idx_noniid, len(owners))

            for cid, train_split, test_split in zip(owners, train_splits, test_splits):
                train_client_idcs[int(cid)] += train_split.tolist()
                test_client_idcs[int(cid)] += test_split.tolist()

    for cid in range(total_clients):
        np.random.shuffle(train_client_idcs[cid])
        np.random.shuffle(test_client_idcs[cid])

    return train_client_idcs, test_client_idcs


def partition_data_quantity_skew(train_labels, test_labels, alpha, total_clients, num_new_clients):
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    num_train_clients = total_clients - num_new_clients
    classes = np.unique(train_labels)
    
    train_client_idcs = {k: [] for k in range(total_clients)}
    test_client_idcs = {k: [] for k in range(total_clients)}

    min_samples_per_class = 10
    
    for k in classes:
        train_idx_k = np.where(train_labels == k)[0]
        test_idx_k = np.where(test_labels == k)[0]
        
        np.random.shuffle(train_idx_k)
        np.random.shuffle(test_idx_k)
        
        total_samples_train = len(train_idx_k)
        total_samples_test = len(test_idx_k)

        iid_ratio = num_new_clients / total_clients
        n_iid_train = int(total_samples_train * iid_ratio)
        n_iid_test = int(total_samples_test * iid_ratio)

        train_idx_iid = train_idx_k[:n_iid_train]
        test_idx_iid = test_idx_k[:n_iid_test]
        
        train_idx_noniid = train_idx_k[n_iid_train:]
        test_idx_noniid = test_idx_k[n_iid_test:]
        
        if num_new_clients > 0:
            train_splits_iid = np.array_split(train_idx_iid, num_new_clients)
            test_splits_iid = np.array_split(test_idx_iid, num_new_clients)
            
            for i in range(num_new_clients):
                cid = num_train_clients + i 
                train_client_idcs[cid] += train_splits_iid[i].tolist()
                test_client_idcs[cid] += test_splits_iid[i].tolist()

        if num_train_clients > 0:
            proportions = np.random.dirichlet(np.repeat(alpha, num_train_clients))

            train_min = min(min_samples_per_class, len(train_idx_noniid) // num_train_clients)
            test_min = min(min_samples_per_class, len(test_idx_noniid) // num_train_clients)

            train_min_ratio = train_min / len(train_idx_noniid)
            test_min_ratio = test_min / len(test_idx_noniid)

            train_proportions = train_min_ratio + (1.0 - train_min_ratio * num_train_clients) * proportions
            test_proportions = test_min_ratio + (1.0 - test_min_ratio * num_train_clients) * proportions

            train_split_pts = (np.cumsum(train_proportions) * len(train_idx_noniid)).astype(int)[:-1]
            test_split_pts = (np.cumsum(test_proportions) * len(test_idx_noniid)).astype(int)[:-1]
            
            train_splits_noniid = np.split(train_idx_noniid, train_split_pts)
            test_splits_noniid = np.split(test_idx_noniid, test_split_pts)
            
            for i in range(num_train_clients):
                cid = i 
                train_client_idcs[cid] += train_splits_noniid[i].tolist()
                test_client_idcs[cid] += test_splits_noniid[i].tolist()

    return train_client_idcs, test_client_idcs


def allocate_equal_client_size(class_indices, alpha, num_clients, min_samples_per_class, client_probs, client_order):
    num_classes = len(class_indices)
    total_samples = sum(len(indices) for indices in class_indices)
    samples_per_client = total_samples // num_clients
    usable_total = samples_per_client * num_clients

    excess = total_samples - usable_total

    while excess > 0:
        class_id = max(range(num_classes), key=lambda j: len(class_indices[j]))
        class_indices[class_id] = class_indices[class_id][:-1]
        excess -= 1

    counts = np.full(
        (num_clients, num_classes),
        min_samples_per_class,
        dtype=int
    )

    remaining_class_counts = np.array([
        len(class_indices[class_id]) - min_samples_per_class * num_clients
        for class_id in range(num_classes)
    ])

    extra_per_client = samples_per_client - min_samples_per_class * num_classes

    for cid in client_order[:-1]:
        remaining_quota = extra_per_client

        while remaining_quota > 0:
            available = remaining_class_counts > 0
            probabilities = client_probs[cid] * available
            probabilities = probabilities / probabilities.sum()

            sampled_counts = np.random.multinomial(
                remaining_quota,
                probabilities
            )

            sampled_counts = np.minimum(
                sampled_counts,
                remaining_class_counts
            )

            counts[cid] += sampled_counts
            remaining_class_counts -= sampled_counts
            remaining_quota -= sampled_counts.sum()

    last_cid = client_order[-1]
    counts[last_cid] += remaining_class_counts

    client_idcs = {cid: [] for cid in range(num_clients)}

    for class_id in range(num_classes):
        indices = class_indices[class_id]
        np.random.shuffle(indices)

        start = 0

        for cid in range(num_clients):
            count = counts[cid, class_id]
            client_idcs[cid] += indices[start:start + count].tolist()
            start += count

    return client_idcs


def partition_data_quantity_skew_equalSize(train_labels, test_labels, alpha, total_clients, num_new_clients):
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    num_train_clients = total_clients - num_new_clients
    classes = np.unique(train_labels)
    num_classes = len(classes)

    train_client_idcs = {k: [] for k in range(total_clients)}
    test_client_idcs = {k: [] for k in range(total_clients)}

    train_noniid_by_class = []
    test_noniid_by_class = []

    min_samples_per_class = 10

    for k in classes:
        train_idx_k = np.where(train_labels == k)[0]
        test_idx_k = np.where(test_labels == k)[0]

        np.random.shuffle(train_idx_k)
        np.random.shuffle(test_idx_k)

        iid_ratio = num_new_clients / total_clients
        n_iid_train = int(len(train_idx_k) * iid_ratio)
        n_iid_test = int(len(test_idx_k) * iid_ratio)

        train_idx_iid = train_idx_k[:n_iid_train]
        test_idx_iid = test_idx_k[:n_iid_test]

        train_idx_noniid = train_idx_k[n_iid_train:]
        test_idx_noniid = test_idx_k[n_iid_test:]

        train_noniid_by_class.append(train_idx_noniid)
        test_noniid_by_class.append(test_idx_noniid)

        if num_new_clients > 0:
            train_splits_iid = np.array_split(
                train_idx_iid,
                num_new_clients
            )

            test_splits_iid = np.array_split(
                test_idx_iid,
                num_new_clients
            )

            for i in range(num_new_clients):
                cid = num_train_clients + i
                train_client_idcs[cid] += train_splits_iid[i].tolist()
                test_client_idcs[cid] += test_splits_iid[i].tolist()

    if num_train_clients > 0:
        client_probs = np.random.dirichlet(
            np.repeat(alpha, num_classes),
            size=num_train_clients
        )

        client_order = np.random.permutation(num_train_clients)

        train_allocations = allocate_equal_client_size(
            class_indices=train_noniid_by_class,
            alpha=alpha,
            num_clients=num_train_clients,
            min_samples_per_class=min_samples_per_class,
            client_probs=client_probs,
            client_order=client_order
        )

        test_allocations = allocate_equal_client_size(
            class_indices=test_noniid_by_class,
            alpha=alpha,
            num_clients=num_train_clients,
            min_samples_per_class=min_samples_per_class,
            client_probs=client_probs,
            client_order=client_order
        )

        for cid in range(num_train_clients):
            train_client_idcs[cid] += train_allocations[cid]
            test_client_idcs[cid] += test_allocations[cid]

    for cid in range(total_clients):
        np.random.shuffle(train_client_idcs[cid])
        np.random.shuffle(test_client_idcs[cid])

    return train_client_idcs, test_client_idcs
