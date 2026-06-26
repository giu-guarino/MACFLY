import wandb
import torch


def write_mean_summaries(metrics, mode="train"):
    logs = {}
    for key in metrics:
        logs["%s_%s/%s" % (mode, "average", key)] = metrics[key]
    wandb.log(logs)


def write_class_summaries(metrics, cls_names, mode="eval"):
    unique_labels, metrics = metrics
    unique_labels = unique_labels.astype("uint16")
    logs = {}
    for key in metrics:
        for i, val in zip(unique_labels, metrics[key]):
            cl = cls_names[i]
            logs["%s_%s_%s/%s" % (mode, key, "cls", cl)] = val
    wandb.log(logs)


def save_model(net, path, local_device_ids):
    if len(local_device_ids) > 1:
        torch.save(net.module.state_dict(), path)
    else:
        torch.save(net.state_dict(), path)
