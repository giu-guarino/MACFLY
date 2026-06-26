import numpy as np
import torch
import sklearn
import pathlib
import tempfile
import sklearn
import torch
from tqdm import tqdm
import numpy as np
import copy
import rasterio as rio
from rasterio import merge
from pathlib import Path
from tqdm import tqdm
import resource
from src.utils.torch_utils import get_device
from src.utils.tensor_utils import resize_logits_to_labels
from src.utils.tensor_utils import resize_logits_to_labels

# --------------------------------------------------------
# Taken partly from https://github.com/michaeltrs/DeepSatModels
# Repository own by Michail Tarasiou
# Modified to avoid concatenating labels, preds and losses
# --------------------------------------------------------

# TODO: add unk_cls


def get_prediction_splits(cm):
    n_classes = len(cm)
    diag = np.diagonal(cm)
    rowsum = cm.sum(axis=1)
    colsum = cm.sum(axis=0)
    TP = (diag).astype(np.float32)
    FN = (rowsum - diag).astype(np.float32)
    FP = (colsum - diag).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        IOU = diag / (rowsum + colsum - diag)
        macro_IOU = diag.sum() / (rowsum.sum() + colsum.sum() - diag.sum())

    num_total = []
    num_correct = []
    for class_ in range(n_classes):
        idx = np.sum(cm[class_, :])
        is_correct = cm[class_, class_]
        num_total.append(idx)
        num_correct.append(is_correct)
    num_total = np.array(num_total).astype(np.float32)
    num_correct = np.array(num_correct)

    return TP, FP, FN, num_correct, num_total, IOU, macro_IOU


def get_split(cm, n_classes):
    num_total = []
    num_correct = []
    for class_ in range(n_classes):
        idx = np.sum(cm[:, class_])
        num_total.append(idx)
        num_correct.append(cm[class_, class_])
    num_total = np.array(num_total)
    num_correct = np.array(num_correct)
    return num_correct, num_total


def get_metrics_from_splits(TP, FP, FN, num_correct, num_total):
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
    if (type(precision) in [np.float32, np.float64]) and (precision + recall == 0.0):
        F1 = 0.0
    else:
        F1 = 2 * precision * recall / (precision + recall)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = num_correct / num_total
    return acc, precision, recall, F1


def nan_mean(v):
    return v[~np.isnan(v)].mean()


def get_classification_metrics(cm, unk_masks=None):
    # if unk_masks is not None:
    #     predicted = predicted[unk_masks]
    #     labels = labels[unk_masks]
    TP, FP, FN, num_correct, num_total, IOU, micro_IOU = get_prediction_splits(
        cm
    )  #  , per_class)
    micro_acc, micro_precision, micro_recall, micro_F1 = get_metrics_from_splits(
        TP.sum(), FP.sum(), FN.sum(), num_correct.sum(), num_total.sum()
    )
    macro_IOU = IOU[~np.isnan(IOU)].mean()
    acc, precision, recall, F1 = get_metrics_from_splits(
        TP, FP, FN, num_correct, num_total
    )
    macro_acc = nan_mean(acc)
    macro_precision = nan_mean(precision)
    macro_recall = nan_mean(recall)
    macro_F1 = nan_mean(F1)
    acc = np.nan_to_num(acc, copy=True, nan=0.0)
    precision = np.nan_to_num(precision, copy=True, nan=0.0)
    recall = np.nan_to_num(recall, copy=True, nan=0.0)
    F1 = np.nan_to_num(F1, copy=True, nan=0.0)
    IOU = np.nan_to_num(IOU, copy=True, nan=0.0)
    return {
        "class": [acc, precision, recall, F1, IOU],
        "micro": [micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU],
        "macro": [macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU],
    }


def get_per_class_loss(losses, labels, n_classes, unk_masks=None):
    if unk_masks is not None:
        losses = losses[unk_masks]
        labels = labels[unk_masks]
    # unique_labels = np.unique(labels)
    class_loss = np.zeros(n_classes)
    weights = np.zeros(n_classes)
    for label in range(n_classes):
        idx = labels == label
        weights[label] = idx.sum()
        class_loss[label] = losses[idx].sum()  # mean()
    return class_loss, weights


def get_unique_labels(cm):
    """return the unique predicted labels"""
    un_cls = []
    for class_ in range(len(cm)):
        if cm[:, class_].sum() != 0:
            un_cls.append(class_)
    return np.asarray(un_cls)


def evaluate(net, evalloader, loss_fn, config, loss_input_fn, device, verbose=False):
    num_classes = config["MODEL"]["num_classes"]
    sum_cm = np.zeros((num_classes, num_classes))
    sum_losses = np.zeros(num_classes)
    sum_weights = np.zeros(num_classes)
    net.eval()
    with torch.no_grad():
        for step, sample in enumerate(evalloader):
            ground_truth = loss_input_fn(sample, device)
            target, mask = ground_truth
            logits = net(sample["inputs"].to(device))[
                0
            ]  # always the first argument in the tuple
            logits = logits.permute(0, 2, 3, 1)
            logits = resize_logits_to_labels(logits, target)
            _, predicted = torch.max(logits.data, -1)
            loss = loss_fn["all"](logits, ground_truth)
            if mask is not None:
                predicted = predicted.view(-1)[mask.view(-1)].cpu().numpy()
                labels = target.view(-1)[mask.view(-1)].cpu().numpy()
            else:
                predicted = predicted.view(-1).cpu().numpy()
                labels = target.view(-1).cpu().numpy()
            cm = sklearn.metrics.confusion_matrix(
                labels, predicted, labels=range(num_classes)
            ).astype("float32")
            sum_cm = sum_cm + cm
            loss, weights = get_per_class_loss(
                loss.view(-1).cpu().detach().numpy(), labels, num_classes
            )
            sum_losses += loss
            sum_weights += weights
        un_cls = get_unique_labels(sum_cm)

    eval_metrics = get_classification_metrics(
        sum_cm,
        unk_masks=None,
    )

    micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics[
        "micro"
    ]
    macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics[
        "macro"
    ]
    class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics[
        "class"
    ]

    with np.errstate(divide="ignore", invalid="ignore"):
        wslosses = np.nan_to_num(sum_losses / sum_weights, nan=0.0)
    weigths = sum_weights / np.nansum(sum_weights)
    micro_loss = np.nansum(wslosses * weigths)
    macro_loss = np.nanmean(wslosses)
    if verbose:
        print(
            "Eval : Mean (micro) Evaluation metrics (micro/macro), loss: %.7f,iou: %.4f/%.4f,"
            "accuracy: %.4f/%.4f, precision: %.4f/%.4f, recall: %.4f/%.4f,"
            "F1: %.4f/%.4f, unique pred labels: %s"
            % (
                micro_loss,
                micro_IOU,
                macro_IOU,
                micro_acc,
                macro_acc,
                micro_precision,
                macro_precision,
                micro_recall,
                macro_recall,
                micro_F1,
                macro_F1,
                un_cls,
            )
        )

    return (
        un_cls,
        {
            "macro": {
                "Loss": macro_loss,  # In the original code, was same as micro_loss
                "Accuracy": macro_acc,
                "Precision": macro_precision,
                "Recall": macro_recall,
                "F1": macro_F1,
                "IOU": macro_IOU,
            },
            "micro": {
                "Loss": micro_loss,  # weighted loss according to classes
                "Accuracy": micro_acc,
                "Precision": micro_precision,
                "Recall": micro_recall,
                "F1": micro_F1,
                "IOU": micro_IOU,
            },
            "class": {
                "Loss": wslosses,
                "Accuracy": class_acc,
                "Precision": class_precision,
                "Recall": class_recall,
                "F1": class_F1,
                "IOU": class_IOU,
            },
        },
    )


# TODO: think to deal with blocks
def logits_to_pred(input_path, output_path):
    "input path: to raster (.tif) file"

    with rio.open(input_path) as src:
        data = src.read()
        profile = src.profile
    data = np.expand_dims(np.argmax(data, 0), 0)
    profile["count"] = 1
    profile["dtype"] = "uint8"

    with rio.open(output_path, "w", **profile) as dst:
        dst.write(data)


def infere(
    net, loader, config, output_folder=None, device_ids=[0], return_logits=False
):
    output_folder = Path(output_folder)
    device = get_device(device_ids, allow_cpu=False)
    num_classes = config["MODEL"]["num_classes"]
    net.eval()
    if output_folder is None:
        pred_lst = []
        label_lst = []
        opt_lst = []
    else:
        output_folder.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for step, sample in enumerate(tqdm(loader)):
            logits = net(sample["inputs"].to(device))[0]

            logits = logits.permute(0, 2, 3, 1)
            # logits = resize_logits_to_labels(logits, torch.ones((4,64,64,1))) #  TODO: for DAFormer, make something properly
            if not return_logits:
                _, predicted = torch.max(logits.data, -1)
            else:
                predicted = logits
            predicted = predicted.cpu().numpy()

            for i, meta in enumerate(sample["meta"]):
                path = sample["id"][i]  # identifier before
                meta_pred_lab = copy.deepcopy(meta)

                if not return_logits:
                    array = np.expand_dims(predicted[i], 0)
                    meta_pred_lab["count"] = 1
                else:
                    array = np.transpose(predicted[i], (2, 0, 1))
                    meta_pred_lab["count"] = 10  # num_classes in GLC
                    meta_pred_lab["dtype"] = "float32"
                if output_folder is not None:
                    with rio.open(
                        Path(output_folder / path), "w", **meta_pred_lab
                    ) as dst:
                        dst.write(array)


# TODO: add relative paths support + temp files usage if possible


def create_mosaic(inputs, tmp_dir, level=0):
    # Was at first a recursive function, but it causes errors    limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)

    limit = 90000  # limit - 50, to be able to open enough files
    src_datasets = []
    temp_path = tmp_dir / f"partial_mosaic_{level:03d}.tif"
    err = []

    if isinstance(inputs, (str, pathlib.PosixPath)):
        paths = list(Path(inputs).glob("*.tif"))
        if len(paths) > limit:
            # temp_path = tmp_dir / f"partial_mosaic_{level:03d}.tif"

            create_mosaic(
                paths[:limit], tmp_dir, level + 1  # , temp_path
            )  # create partial mosaic, use output_path and will be override later
            paths = paths[limit:]
            # output_path = tmp_dir / f"partial_mosaic_{level:03d}.tif"
            # if output_path not in paths:
            #     paths.append(output_path)  # partial mosaic file added
            return create_mosaic(paths, tmp_dir, level + 2)

        else:
            for path in paths:
                src = rio.open(path)
                if src.count != 10:
                    err.append(path)
                src_datasets.append(src)
    else:
        if len(inputs) > limit:
            temp_path = tmp_dir / f"partial_mosaic_{level:03d}.tif"
            create_mosaic(inputs[:limit], tmp_dir, level + 1)
            paths = inputs[limit:]
            # output_path = tmp_dir / f"partial_mosaic_{level:03d}.tif"
            # if output_path not in paths:
            #    paths.append(output_path)  # partial mosaic file added
            return create_mosaic(paths, tmp_dir, level + 2)
        else:
            for path in inputs:
                src = rio.open(path)
                # src.meta.nodata =  #
                src_datasets.append(src)

    # Workaround to do a mean for ovelapped areas
    # sum_mosaic, out_transform = merge.merge(src_datasets, method="sum", nodata=-1)
    mosaic, out_transform = merge.merge(src_datasets, method="sum", nodata=-1)
    # count_mosaic, _ = merge.merge(src_datasets, method=rio.merge.copy_count, nodata=-1)
    # mosaic = np.divide(
    #     sum_mosaic, count_mosaic, out=np.zeros_like(sum_mosaic), where=count_mosaic != 0
    # )

    out_meta = src_datasets[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",  # look for COG
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
        }
    )
    for src in src_datasets:
        src.close()
    with rio.open(temp_path, "w", **out_meta) as dst:
        dst.write(mosaic)
        # else:
    #     return mosaic

    # Reference counting out of scope for datasets closure
