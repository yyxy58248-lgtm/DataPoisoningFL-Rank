from .label_replacement import apply_class_label_replacement
from .client_utils import log_client_data_statistics

def poison_data(logger, distributed_dataset, num_workers, poisoned_worker_ids, replacement_method):
    """
    Poison worker data

    :param logger: logger
    :type logger: loguru.logger
    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    :param num_workers: number of workers
    :type num_workers: int
    :param poisoned_worker_ids: list of worker IDs to poison
    :type poisoned_worker_ids: list(int)
    :param replacement_method: the class label replacement method
    :type replacement_method: function
    """
    poisoned_dataset = []

    # 修复：从所有客户端收集标签，而不是只从第一个
    all_labels = set()
    for worker_data in distributed_dataset:
        labels = worker_data[1]
        all_labels.update(set(labels))
    class_labels = list(all_labels)

    logger.info("Poisoning data for workers: {}".format(str(poisoned_worker_ids)))

    for worker_idx in range(num_workers):
        if worker_idx in poisoned_worker_ids:
            poisoned_dataset.append(apply_class_label_replacement(distributed_dataset[worker_idx][0], distributed_dataset[worker_idx][1], replacement_method))
        else:
            poisoned_dataset.append(distributed_dataset[worker_idx])

    log_client_data_statistics(logger, class_labels, poisoned_dataset)

    return poisoned_dataset
