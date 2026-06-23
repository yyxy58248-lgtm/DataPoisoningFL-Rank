import torch

def distribute_batches_equally(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.
    Compatible with both 2-value (data, target) and 3-value (data, target, group_ids) formats.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, batch in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers
        
        # 检查batch的长度，兼容2个或3个返回值
        if len(batch) == 2:
            data, target = batch
            distributed_dataset[worker_idx].append((data, target))
        elif len(batch) == 3:
            data, target, group_ids = batch
            distributed_dataset[worker_idx].append((data, target, group_ids))
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

    return distributed_dataset
