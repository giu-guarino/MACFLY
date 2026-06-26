import numpy as np
import torch
from torch.autograd import grad


# Domain Critic
def gradient_penalty(critic, h_s, h_t, device):
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(
        preds,
        interpolates,
        grad_outputs=torch.ones_like(preds),
        retain_graph=True,
        create_graph=True,
    )[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


# MAE finetuning
def init_wb(pretrain_net, finetune_net):
    """initialise the weights and biases of the model to finetune"""
    not_init = []
    for k in pretrain_net.state_dict().keys():
        if k in finetune_net.state_dict().keys():
            param = pretrain_net.state_dict()[k]
            f_param = finetune_net.state_dict()[k]
            if (
                not param.data.shape and f_param.data.shape == param.data
            ):  # if scalar tensor
                f_param.data = param.data
            elif f_param.data[:].shape == param[:].data[:].shape:
                f_param.data[:] = param[:].data[:]
            else:
                not_init.append(k)
    if len(not_init) != 0:
        print(f"{not_init} were not initialised")


def update_ema_state_dict(student, teacher, alpha, abs_iter):
    alpha_teacher = min(1 - 1 / (abs_iter + 1), alpha)
    keys = (
        student.state_dict().keys()
    )  # same keys for the teacher as the model are identical
    for k in keys:
        ema_param = teacher.state_dict()[k]
        param = student.state_dict()[k]
        if not param.data.shape:  # if scalar tensor
            ema_param.data = (
                alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
            )
        else:
            ema_param.data[:] = (
                alpha_teacher * ema_param[:].data[:]
                + (1 - alpha_teacher) * param[:].data[:]
            )


def pt_fn(logits):
    """
    For each pixel, return a one-hot vector where the 1 value
    indicates the class with the highest confidence
    logits.shape = (N,H,W,C)
    """
    C = logits.shape[3]
    pseudo_label = torch.argmax(logits, dim=3)
    pt = torch.nn.functional.one_hot(pseudo_label, num_classes=C)
    return pt, pseudo_label


def qt_fn(target_logits, source_logits, source_gt, tau):
    """
    qt gives and idea of the model confidence
    softmax required as the logits are unormalised
    logits.shape = (N,H,W,C)
    """
    N, H, W, C = target_logits.shape
    max_class_values, target_pred = torch.max(
        torch.nn.functional.softmax(target_logits, dim=3), dim=3
    )
    qt = torch.where(max_class_values > tau, 1.0, 0.0)

    qt = torch.sum(qt, dim=(1, 2)) / (H * W)  # (B)
    _qt = torch.ones((N, H, W, C))

    for i, qt_i in enumerate(
        qt
    ):  # qt_i ?? -> check why this quirk (function cas modified and not use after probably)
        _qt[i] = qt[i]
    return qt, _qt


def qt_fn_sp(target_logits, source_logits, source_gt, tau):
    """
    qt gives and idea of the model confidence
    softmax required as the logits are unormalised
    logits.shape = (N,H,W,C)
    """
    N, H, W, C = target_logits.shape
    max_class_values, target_pred = torch.max(
        torch.nn.functional.softmax(target_logits, dim=3), dim=3
    )
    qt = torch.where(max_class_values > tau, 1.0, 0.0)

    # Spatial criterion
    # (NxHxW) -> NOT  one-hot labels
    source_pred = generate_pseudo_label(source_logits)
    correct_source_pred = source_pred == source_gt
    same_pred = source_pred == target_pred
    criterion = correct_source_pred * same_pred
    qt = qt * criterion

    qt = torch.sum(qt, dim=(1, 2)) / (H * W)  # (B)
    _qt = torch.ones((N, H, W, C))

    for i, qt_i in enumerate(
        qt
    ):  # qt_i ?? -> check why this quirk (function cas modified and not use after probably)
        _qt[i] = qt[i]
    return qt, _qt


# TODO: seems that tau2 is hard-coded
def qt_fn_hard_sp(target_logits, source_logits, source_gt, tau):
    """
    qt gives and idea of the model confidence
    softmax required as the logits are unormalised
    logits.shape = (N,H,W,C)
    gives hard labels
    """
    N, H, W, C = target_logits.shape
    max_class_values, target_pred = torch.max(
        torch.nn.functional.softmax(target_logits, dim=3), dim=3
    )
    qt = torch.where(max_class_values > tau, 1.0, 0.0)

    # Spatial criterion
    # (NxHxW) -> NOT  one-hot labels
    source_pred = generate_pseudo_label(source_logits)
    correct_source_pred = source_pred == source_gt
    same_pred = source_pred == target_pred
    criterion = correct_source_pred * same_pred
    qt = qt * criterion

    qt = torch.sum(qt, dim=(1, 2)) / (H * W)  # (B)
    # qt = torch.where(qt>0.9,1.,0.) # tau_hard = 0.8
    _qt = torch.ones((N, H, W, C))

    for i in range(len(qt)):
        _qt[i] = 1.0 if qt[i] > 0.85 else 0.0
    return qt, _qt


# TODO: seems that tau2 is hard-coded
def qt_fn_hard(target_logits, source_logits, source_gt, tau):
    """
    qt gives and idea of the model confidence
    softmax required as the logits are unormalised
    logits.shape = (N,H,W,C)
    gives hard labels
    """
    N, H, W, C = target_logits.shape
    max_class_values, target_pred = torch.max(
        torch.nn.functional.softmax(target_logits, dim=3), dim=3
    )
    qt = torch.where(max_class_values > tau, 1.0, 0.0)

    # Spatial criterion
    # (NxHxW) -> NOT  one-hot labels
    # source_pred = generate_pseudo_label(source_logits)
    # correct_source_pred = source_pred == source_gt
    # same_pred = source_pred == target_pred
    # criterion = correct_source_pred * same_pred
    # qt = qt * criterion

    qt = torch.sum(qt, dim=(1, 2)) / (H * W)  # (B)
    _qt = torch.ones((N, H, W, C))

    for i in range(len(qt)):
        _qt[i] = 1.0 if qt[i] > 0.9 else 0.0
    return qt, _qt


def generate_pseudo_label(target_logits):
    _, pseudo_label = torch.max(target_logits.data, -1)
    return pseudo_label


def prob_cls(source_logits, source_gt, tau_cls, ncls=10):
    """
    gives the mean prob by class when a pred is correct
    """
    max_class_values = torch.max(
        torch.nn.functional.softmax(source_logits, dim=3), dim=3
    ).values  # (NxHxW)
    source_pred = generate_pseudo_label(source_logits)  # (BxHxW)
    correct_source_pred = source_pred == source_gt  # (BxHxW) nécessaire?
    for i in range(ncls):
        cl = source_pred == i  # (BxHxW)
        correct_cl = cl * correct_source_pred
        tau_cl = torch.sum(max_class_values * (correct_cl))
        tau_cls["%d" % (i)]["tau"] += tau_cl
        tau_cls["%d" % (i)]["nb_pixels"] += torch.sum(correct_cl)
    return tau_cls


def mean_dict(tau_cls, ncls=10):
    mean = np.zeros(ncls)
    for i in range(ncls):
        if tau_cls["%d" % (i)]["tau"] != 0:
            mean[i] = tau_cls["%d" % (i)]["tau"] / tau_cls["%d" % (i)]["nb_pixels"]
    return mean


# def evaluate_pseudo_labels_quality(gt, pred, qt):
#     """
#     Return the mean of the micro F1-score for valid
#     amples (for which qt=1)
#     """
#     gt = torch.split(gt.detach().cpu(), 1, dim=0)  # split in samples
#     pred = torch.split(pred.detach().cpu(), 1, dim=0)
#     f1 = [
#         f1_score(p.flatten(), g.flatten(), labels=list(range(10)), average="micro")
#         for (p, g) in zip(list(compress(gt, qt)), list(compress(pred, qt)))
#     ]
#     return np.mean(np.array(f1))
