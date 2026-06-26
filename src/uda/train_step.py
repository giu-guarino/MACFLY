from abc import ABC, abstractmethod
import torch
import numpy as np
from sklearn.metrics import f1_score
from .utils import (
    update_ema_state_dict,
    pt_fn,
    qt_fn,
    qt_fn_hard,
    qt_fn_hard_sp,
    qt_fn_sp,
)
from ..build_train import TRAIN_STEP
import torch.nn as nn
from src.uda.dacs_transforms import get_class_masks, strong_transform
from einops import rearrange
import random
from src.utils.tensor_utils import resize_logits_to_labels, input_shape
from src.uda.utils import set_requires_grad, gradient_penalty
import os
import matplotlib.pyplot as plt


def save_rgb_from_batch(
    batch, step, save_dir="debug_images", batch_idx=0, img_idx=0, prefix="stage"
):
    """
    batch: numpy array or torch tensor, shape [B, N, H, W, C] (ex: [2, 14, 64, 64, 7])
    step: nom ou numéro d'étape/transformation
    save_dir: dossier de sortie
    batch_idx: index du batch à sauvegarder
    img_idx: index de l'image dans le batch à sauvegarder
    prefix: préfixe du fichier
    """
    os.makedirs(save_dir, exist_ok=True)
    label = 0
    if hasattr(batch, "detach"):
        if len(batch.shape) == 5:
            img = (
                batch[batch_idx, img_idx, :, :, :3].detach().cpu().numpy()
            )  # [64, 64, 3]
        elif len(batch.shape) == 4 and batch.shape[-1] == 1:

            img = batch[batch_idx, :, :, 0].detach().cpu().numpy()  # [64, 64, 3]
            label = 1
        else:
            assert batch.shape[1] == 84
    else:
        if len(batch.shape) == 5:
            img = batch[batch_idx, img_idx, :, :, :3]  # [64, 64, 3]
        elif len(batch.shape) == 4 and batch.shape[-1] == 1:
            img = batch[batch_idx, :, :, 0].detach().cpu().numpy()  # [64, 64, 3]
        else:
            assert batch.shape[1] == 84
    if label == 0:
        plt.imsave(
            os.path.join(save_dir, f"{prefix}_{step}_b{batch_idx}_i{img_idx}.png"), img
        )
    else:
        plt.imsave(os.path.join(save_dir, f"{prefix}_{step}_b{batch_idx}.png"), img)


def normalise_logs(logs):
    """input: {"key":np.array}"""
    normalise_logs = {}
    for key, val in logs.items():
        if len(val) == 2:
            normalise_logs[key] = val[0] / val[1]
        else:
            normalise_logs[key] = val[0]
    return normalise_logs


class TrainStepBase(ABC):
    def __init__(self, net, teacher, optimizer, scaler, loss_input_fn, alpha):
        self.net = net
        self.teacher = teacher
        self.optimizer = optimizer
        self.scaler = scaler
        self.loss_input_fn = loss_input_fn
        self.alpha = alpha
        self.device = next(net.parameters()).device
        print(self.device)

    @abstractmethod
    def __call__(self, sample, abs_step, logs, use_amp):
        raise NotImplementedError()


@TRAIN_STEP.register_module()
class GRL_train_step(TrainStepBase):
    """Gradient Reversal Layer train step,
    some metrics lines can be uncomment if you have access to the GT"""

    def __init__(self, net, teacher, optimizer, scaler, loss_input_fn, alpha):
        super().__init__(net, teacher, optimizer, scaler, loss_input_fn, alpha)

    def __call__(self, sample, abs_step, logs, use_amp):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            data = {}
            source_sample, target_sample = sample["source"], sample["target"]
            B, _, H, W, _ = source_sample["inputs"].shape
            self.optimizer.zero_grad()
            # LS computation
            source_logits, source_adv = self.net(
                source_sample["inputs"].to(self.device)
            )
            source_logits = source_logits.permute(0, 2, 3, 1)
            source_ground_truth = self.loss_input_fn(source_sample, self.device)
            if len(source_ground_truth) == 2:
                labels, unk_masks = source_ground_truth
            else:
                labels = source_ground_truth
                unk_masks = None
            labels = labels.squeeze(dim=3).long()
            C = source_logits.shape[3]  # number of classes
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=C)
            # TODO: use the pytorch CrossEntropyLoss
            # CrossEntropyLoss
            LS = torch.mean(
                -torch.sum(
                    torch.mul(
                        one_hot_labels,
                        torch.nn.functional.log_softmax(source_logits, dim=3),
                    ),
                    dim=3,
                )
            )
            # l_adv computation
            student_pred, target_adv = self.net(target_sample["inputs"].to(self.device))
            student_pred = student_pred.permute(0, 2, 3, 1)
            student_pred = torch.nn.functional.log_softmax(student_pred, dim=3)

            y_adv = (
                torch.cat(
                    [torch.ones(source_adv.shape[0]), torch.zeros(target_adv.shape[0])]
                )
                .to(self.device)
                .long()
            )
            loss_adv = torch.nn.CrossEntropyLoss()(
                torch.cat([source_adv, target_adv]), y_adv
            )

        if logs is not None:
            epoch_LS = LS.cpu().detach().numpy()
            epoch_adv = loss_adv.cpu().detach().numpy()
            logs.setdefault("epoch_LS", np.zeros(2))
            logs["epoch_LS"] += np.array([epoch_LS, B])
            logs.setdefault("epoch_adv", np.zeros(2))
            logs["epoch_adv"] += np.array([epoch_adv, B])

        loss = LS + loss_adv
        self.scaler.scale(loss).backward()
        # To Clip gradient may be useful
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # update du teacher
        update_ema_state_dict(
            self.net, self.teacher, self.alpha, abs_step
        )  # vérifier abs_iter
        # voir si cpu().detach().numpy()
        data["source_logits"] = source_logits
        data["labels"] = labels
        data["unk_masks"] = unk_masks
        data["loss"] = loss
        return data


# TODO: prefer a wrapper rather than doing several cases
@TRAIN_STEP.register_module()
class DAFormer_train_step(TrainStepBase):
    """Original DAFormer train step"""

    def __init__(self, net, teacher, optimizer, scaler, loss_input_fn, alpha, tau):
        super().__init__(net, teacher, optimizer, scaler, loss_input_fn, alpha)
        self.net_name = self.net.__class__.__name__
        self.tau = tau

    def __call__(self, sample, abs_step, logs, use_amp):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            data = {}
            source_sample, target_sample = sample["source"], sample["target"]
            B, H, W, CH, *rest = input_shape(source_sample["inputs"])  # CH=num_channels
            T = rest[0] if rest else None

            if "UNet3D" in type(self.net).__name__:
                B, C, T, H, W = source_sample["inputs"].shape

            self.optimizer.zero_grad()
            # LS computation
            source_logits = self.net(source_sample["inputs"].to(self.device))
            source_logits = source_logits.permute(0, 2, 3, 1)
            source_ground_truth = self.loss_input_fn(source_sample, self.device)
            if len(source_ground_truth) == 2:
                labels, unk_masks = source_ground_truth
            else:
                labels = source_ground_truth
                unk_masks = None
            labels = labels.squeeze(dim=3).long()
            source_logits = resize_logits_to_labels(source_logits, labels)
            C = source_logits.shape[3]  # number of classes
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=C)
            # Classic cross-entropy
            LS = torch.mean(
                -torch.sum(
                    torch.mul(
                        one_hot_labels,
                        torch.nn.functional.log_softmax(source_logits, dim=3),
                    ),
                    dim=3,
                )
            )
            # LT computation
            # torch.no_grad() not mandatory -> teacher parameters are not taken into
            # account by the optimizer
            with torch.no_grad():
                target_logits = self.teacher(target_sample["inputs"].to(self.device))
                target_logits = target_logits.permute(0, 2, 3, 1)
                target_logits = resize_logits_to_labels(target_logits, labels)
                _, pseudo_label = pt_fn(target_logits)
                qt_values, qt = qt_fn(target_logits, source_logits, labels, self.tau)
                qt = qt.to(self.device)

            mix_masks = get_class_masks(labels)
            mixed_img, mixed_lbl = [None] * B, [None] * B
            strong_parameters = {
                "mix": None,
                "color_jitter": random.uniform(0, 1),
                "color_jitter_s": 0.2,
                "color_jitter_p": 0.2,
                "blur": random.uniform(0, 1),
            }
            source_aug_img, target_aug_img = source_sample["inputs"].to(
                self.device
            ), target_sample["inputs"].to(self.device)
            source_aug_lbl = source_sample["labels"].to(self.device)
            # May be useless
            if "TSViT" in type(self.net).__name__:
                source_embeddings = source_aug_img[..., 6:7]
                source_aug_img, target_aug_img = (
                    source_aug_img[..., 0:6],
                    target_aug_img[..., 0:6],
                )
                source_aug_img = rearrange(source_aug_img, "b t h w c -> b (t c) h w")
                target_aug_img = rearrange(target_aug_img, "b t h w c -> b (t c) h w")
            elif "UNet3D" in type(self.net).__name__:
                source_aug_img = rearrange(source_aug_img, "b c t h w  -> b (t c) h w")
                target_aug_img = rearrange(target_aug_img, "b c t h w -> b (t c) h w")

            assert len(source_aug_lbl.shape) == 4
            assert source_aug_lbl.shape[3] == 1
            source_aug_lbl = source_aug_lbl.squeeze(3)
            pseudo_weight = qt.permute(0, 3, 1, 2)
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=self.device)

            for i in range(B):
                strong_parameters["mix"] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((source_aug_img[i], target_aug_img[i])),
                    target=torch.stack((source_aug_lbl[i], pseudo_label[i])),
                )
                _, pseudo_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])),
                )
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)
            if "TSViT" in type(self.net).__name__:
                mixed_img = rearrange(mixed_img, "b (t c) h w  -> b t h w c", t=T)
                mixed_img = torch.cat((mixed_img, source_embeddings), dim=-1)
            elif "UNet3D" in type(self.net).__name__:
                mixed_img = rearrange(mixed_img, "b (t c) h w  -> b c t h w", t=T)
            student_pred = self.net(mixed_img)
            student_pred = student_pred.permute(0, 2, 3, 1)
            student_pred = resize_logits_to_labels(student_pred, labels)
            student_pred = torch.nn.functional.log_softmax(student_pred, dim=3)
            mixed_lbl = mixed_lbl.permute(0, 2, 3, 1)
            mixed_lbl = torch.repeat_interleave(mixed_lbl, C, -1)
            pseudo_weight = pseudo_weight.permute(0, 2, 3, 1)
            LT = torch.mean(
                -torch.sum(
                    torch.mul(pseudo_weight, torch.mul(student_pred, mixed_lbl)), dim=3
                )
            )

        # Uncomment if you have access to the GT
        # weights = qt[:, :, :, 0].flatten().detach().cpu()
        # weight = torch.sum(qt_values).detach().cpu().numpy()
        # target_ground_truth = self.loss_input_fn(target_sample, self.device)
        # if len(target_ground_truth) == 2:
        #     target_labels, target_unk_masks = target_ground_truth
        # else:
        #     target_labels = target_ground_truth

        # if weight > 0:
        #     f1_valid = f1_score(
        #         target_labels.flatten().cpu(),
        #         pseudo_label.flatten().cpu(),
        #         average="micro",
        #         sample_weight=weights,
        #     )
        #     if logs is not None:
        #         epoch_f1_valid = weight * f1_valid
        #         logs.setdefault("epoch_f1_valid", np.zeros(2))
        #         logs["epoch_f1_valid"] += np.array([epoch_f1_valid, weight])
        #         epoch_valid_samples = (
        #             torch.sum(torch.where(qt_values > 0.0, 1.0, 0.0))
        #             .detach()
        #             .cpu()
        #             .item()
        #         )
        #         logs.setdefault("epoch_valid_samples", np.zeros(2))
        #         logs["epoch_valid_samples"] += np.array([epoch_valid_samples, B])
        #         epoch_valid_pixels = torch.sum(qt_values * H * W).detach().cpu().item()
        #         logs.setdefault("epoch_valid_pixels", np.zeros(2))
        #         logs["epoch_valid_pixels"] += np.array([epoch_valid_pixels, B * H * W])

        # f1_global = f1_score(
        #     target_labels.flatten().detach().cpu(),
        #     pseudo_label.flatten().detach().cpu(),
        #     average="micro",
        # )

        if logs is not None:
            epoch_LS = LS.cpu().detach().numpy()
            epoch_LT = LT.cpu().detach().numpy()
            # Uncomment if you have access to the GT
            # epoch_f1_global = f1_global
            # logs.setdefault("epoch_f1_global", np.zeros(2))
            # logs["epoch_f1_global"] += np.array([epoch_f1_global, 1])
            logs.setdefault("epoch_LS", np.zeros(2))
            logs["epoch_LS"] += np.array([epoch_LS, B])
            logs.setdefault("epoch_LT", np.zeros(2))
            logs["epoch_LT"] += np.array([epoch_LT, B])
        # Lambda = abs_step / num_steps
        # loss = (1 - Lambda) * (LS + loss_adv) + LT * Lambda
        loss = LS + LT
        self.scaler.scale(loss).backward()
        # To Clip gradient may be useful
        self.scaler.step(self.optimizer)
        self.scaler.update()

        update_ema_state_dict(self.net, self.teacher, self.alpha, abs_step)
        data["source_logits"] = source_logits
        data["labels"] = labels
        data["unk_masks"] = unk_masks
        data["loss"] = loss
        return data


# --------------------------------------------------------------------------------------
#                        Supplementary training steps
# --------------------------------------------------------------------------------------


# TODO: Make a warmup parameter and get rid of the current hard-coded one
@TRAIN_STEP.register_module()
class GRL_warm_hard_sp_dis_train_step(TrainStepBase):
    """Train step with:
    Gradient Reversal Layer,
    warm-up period,
    hard labels,
    spatial criterion (source and target data alignement),
    disentanglement
    """

    def __init__(self, net, teacher, optimizer, scaler, loss_input_fn, alpha, tau):
        super().__init__(net, teacher, optimizer, scaler, loss_input_fn, alpha)
        self.tau = tau
        # TODO: Add warmup as a parameter in the config file
        self.warmup = 50 * 1125

    def __call__(self, sample, abs_step, logs, use_amp):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            data = {}
            loss = 0
            source_sample, target_sample = sample["source"], sample["target"]
            B, _, H, W, _ = source_sample["inputs"].shape
            self.optimizer.zero_grad()
            # LS computation
            source_logits, (
                source_adv,
                source_inv_dis,
                source_spc_dis,
                source_spc_dom,
            ) = self.net(source_sample["inputs"].to(self.device))
            source_logits = source_logits.permute(0, 2, 3, 1)
            source_ground_truth = self.loss_input_fn(source_sample, self.device)
            if len(source_ground_truth) == 2:
                labels, unk_masks = source_ground_truth
            else:
                labels = source_ground_truth
                unk_masks = None
            labels = labels.squeeze(dim=3).long()
            C = source_logits.shape[3]  # number of classes
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=C)
            LS = torch.mean(
                -torch.sum(
                    torch.mul(
                        one_hot_labels,
                        torch.nn.functional.log_softmax(source_logits, dim=3),
                    ),
                    dim=3,
                )
            )
            loss += LS
            student_pred, (
                target_adv,
                target_inv_dis,
                target_spc_dis,
                target_spc_dom,
            ) = self.net(target_sample["inputs"].to(self.device))
            student_pred = student_pred.permute(0, 2, 3, 1)
            student_pred = torch.nn.functional.log_softmax(student_pred, dim=3)

            with torch.no_grad():
                target_logits, _ = self.teacher(target_sample["inputs"].to(self.device))
                target_logits = target_logits.permute(0, 2, 3, 1)
                pt, pseudo_label = pt_fn(target_logits)
                qt_values, qt = qt_fn_hard_sp(
                    target_logits, source_logits, labels, self.tau
                )
                qt = qt.to(self.device)

            if abs_step > self.warmup:
                LT = torch.mean(
                    -torch.sum(torch.mul(qt, torch.mul(student_pred, pt)), dim=3)
                )
                loss += LT

            y_adv = (
                torch.cat(
                    [torch.ones(source_adv.shape[0]), torch.zeros(target_adv.shape[0])]
                )
                .to(self.device)
                .long()
            )
            loss_adv = torch.nn.CrossEntropyLoss()(
                torch.cat([source_adv, target_adv]), y_adv
            )
            loss += loss_adv

            def dot_product(inv, spc):
                inv = nn.functional.normalize(inv)
                spc = nn.functional.normalize(spc)
                prod = torch.sum(inv * spc, dim=1)
                prod = torch.mean(torch.abs(prod))
                return prod

            ortho_loss = dot_product(source_inv_dis, source_spc_dis) + dot_product(
                target_inv_dis, target_spc_dis
            )
            loss += ortho_loss

            # domain
            dom = (
                torch.cat(
                    [
                        torch.ones(source_spc_dom.shape[0]),
                        torch.zeros(target_spc_dom.shape[0]),
                    ]
                )
                .to(self.device)
                .long()
            )
            dom_loss = torch.nn.CrossEntropyLoss()(
                torch.cat([source_spc_dom, target_spc_dom]), dom
            )
            loss += dom_loss  # 0.1 *

        weights = qt[:, :, :, 0].flatten().cpu().detach()
        qt_values_thresh = torch.where(qt_values > 0.85, 1.0, 0.0)
        weight = torch.sum(qt_values_thresh).detach().cpu().numpy()
        target_ground_truth = self.loss_input_fn(target_sample, self.device)
        if len(target_ground_truth) == 2:
            target_labels, target_unk_masks = target_ground_truth
        else:
            target_labels = target_ground_truth

        if weight > 0:
            f1_valid = f1_score(
                target_labels.flatten().cpu(),
                pseudo_label.flatten().cpu(),
                average="micro",
                sample_weight=weights,
            )
            if logs is not None:
                epoch_f1_valid = weight * f1_valid
                logs.setdefault("epoch_f1_valid", np.zeros(2))
                logs["epoch_f1_valid"] += np.array([epoch_f1_valid, weight])

        if logs is not None:
            epoch_valid_samples = torch.sum(qt_values_thresh).detach().cpu().item()
            logs.setdefault("epoch_valid_samples", np.zeros(2))
            logs["epoch_valid_samples"] += np.array([epoch_valid_samples, B])
            epoch_valid_pixels = torch.sum(qt_values * H * W).detach().cpu().item()
            logs.setdefault("epoch_valid_pixels", np.zeros(2))
            logs["epoch_valid_pixels"] += np.array([epoch_valid_pixels, B * H * W])

        f1_global = f1_score(
            target_labels.flatten().detach().cpu(),
            pseudo_label.flatten().detach().cpu(),
            average="micro",
        )
        if logs is not None:
            epoch_LS = LS.cpu().detach().numpy()
            epoch_adv = loss_adv.cpu().detach().numpy()
            if abs_step > self.warmup:
                epoch_LT = LT.cpu().detach().numpy()
                logs.setdefault("epoch_LT", np.zeros(2))
                logs["epoch_LT"] += np.array([epoch_LT, B])
            epoch_f1_global = f1_global
            logs.setdefault("epoch_f1_global", np.zeros(2))
            logs["epoch_f1_global"] += np.array([epoch_f1_global, 1])

            logs.setdefault("epoch_LS", np.zeros(2))
            logs["epoch_LS"] += np.array([epoch_LS, B])
            logs.setdefault("epoch_adv", np.zeros(2))
            logs["epoch_adv"] += np.array([epoch_adv, B])

            logs.setdefault("epoch_lortho", np.zeros(2))
            logs["epoch_lortho"] += np.array([ortho_loss.cpu().detach().numpy(), B])
            logs.setdefault("epoch_ldom", np.zeros(2))
            logs["epoch_ldom"] += np.array([dom_loss.cpu().detach().numpy(), B])

        self.scaler.scale(loss).backward()
        # To Clip gradient may be useful
        self.scaler.step(self.optimizer)
        self.scaler.update()

        update_ema_state_dict(self.net, self.teacher, self.alpha, abs_step)
        data["source_logits"] = source_logits
        data["labels"] = labels
        data["unk_masks"] = unk_masks
        data["loss"] = loss
        return data


# DOMAIN CRITIC
@TRAIN_STEP.register_module()
class DC_GRL_warm_hard_sp_train_step(TrainStepBase):
    """Train step with:
    Domain critic,
    Gradient Reversal Layer,
    warm-up period,
    hard labels,
    spatial criterion
    """

    def __init__(
        self,
        net,
        teacher,
        optimizer,
        scaler,
        loss_input_fn,
        alpha,
        tau,
        critic,
        critic_optim,
    ):
        super().__init__(net, teacher, optimizer, scaler, loss_input_fn, alpha)
        self.tau = tau
        # TODO: Add warmup as a parameter in the config file
        self.warmup = 50 * 1125
        self.critic = critic
        self.critic_optim = critic_optim

    def __call__(self, sample, abs_step, logs, use_amp):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            data = {}
            loss = 0

            set_requires_grad(self.net, requires_grad=True)
            set_requires_grad(self.critic, requires_grad=False)

            source_sample, target_sample = sample["source"], sample["target"]
            B, _, H, W, _ = source_sample["inputs"].shape
            self.optimizer.zero_grad()
            # LS computation
            source_logits, (source_tokens, source_adv) = self.net(
                source_sample["inputs"].to(self.device)
            )
            source_logits = source_logits.permute(0, 2, 3, 1)
            source_ground_truth = self.loss_input_fn(source_sample, self.device)
            if len(source_ground_truth) == 2:
                labels, unk_masks = source_ground_truth
            else:
                labels = source_ground_truth
                unk_masks = None
            labels = labels.squeeze(dim=3).long()
            C = source_logits.shape[3]  # number of classes
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=C)
            LS = torch.mean(
                -torch.sum(
                    torch.mul(
                        one_hot_labels,
                        torch.nn.functional.log_softmax(source_logits, dim=3),
                    ),
                    dim=3,
                )
            )
            loss = loss + LS
            student_pred, (target_tokens, target_adv) = self.net(
                target_sample["inputs"].to(self.device)
            )
            student_pred = student_pred.permute(0, 2, 3, 1)
            student_pred = torch.nn.functional.log_softmax(student_pred, dim=3)

            with torch.no_grad():
                target_logits, (_, _) = self.teacher(
                    target_sample["inputs"].to(self.device)
                )
                target_logits = target_logits.permute(0, 2, 3, 1)
                pt, pseudo_label = pt_fn(target_logits)
                qt_values, qt = qt_fn_hard_sp(
                    target_logits, source_logits, labels, self.tau
                )
                qt = qt.to(self.device)

            if abs_step > self.warmup:
                LT = torch.mean(
                    -torch.sum(torch.mul(qt, torch.mul(student_pred, pt)), dim=3)
                )
                loss = loss + LT

            y_adv = (
                torch.cat(
                    [torch.ones(source_adv.shape[0]), torch.zeros(target_adv.shape[0])]
                )
                .to(self.device)
                .long()
            )
            loss_adv = torch.nn.CrossEntropyLoss()(
                torch.cat([source_adv, target_adv]), y_adv
            )

            loss = loss + loss_adv

            # DOMAIN CRITIC
            wasserstein_distance = (
                self.critic(source_tokens).mean() - self.critic(target_tokens).mean()
            )
            dc_loss = 0.1 * wasserstein_distance
            loss = loss + dc_loss
            self.scaler.scale(loss).backward()

            # TRAIN DOMAIN CRITIC
            set_requires_grad(self.net, requires_grad=False)
            set_requires_grad(self.critic, requires_grad=True)
            with torch.no_grad():
                _, (h_s, _) = self.net(source_sample["inputs"].to(self.device))
                _, (h_t, _) = self.net(target_sample["inputs"].to(self.device))

            ITER_DC = 10
            GP_PARAM = 10
            for _ in range(ITER_DC):
                gp = gradient_penalty(self.critic, h_s, h_t, self.device)

                critic_s = self.critic(h_s)
                critic_t = self.critic(h_t)
                wasserstein_distance = critic_s.mean() - critic_t.mean()

                critic_cost = -wasserstein_distance + GP_PARAM * gp

                self.critic_optim.zero_grad()
                self.scaler.scale(critic_cost).backward()
                self.scaler.step(self.critic_optim)

        weights = qt[:, :, :, 0].flatten().cpu().detach()
        qt_values_thresh = torch.where(qt_values > 0.85, 1.0, 0.0)
        weight = torch.sum(qt_values_thresh).detach().cpu().numpy()
        target_ground_truth = self.loss_input_fn(target_sample, self.device)
        if len(target_ground_truth) == 2:
            target_labels, target_unk_masks = target_ground_truth
        else:
            target_labels = target_ground_truth

        if weight > 0:
            f1_valid = f1_score(
                target_labels.flatten().cpu(),
                pseudo_label.flatten().cpu(),
                average="micro",
                sample_weight=weights,
            )
            if logs is not None:
                epoch_f1_valid = weight * f1_valid
                logs.setdefault("epoch_f1_valid", np.zeros(2))
                logs["epoch_f1_valid"] += np.array([epoch_f1_valid, weight])

        if logs is not None:
            epoch_valid_samples = torch.sum(qt_values_thresh).detach().cpu().item()
            logs.setdefault("epoch_valid_samples", np.zeros(2))
            logs["epoch_valid_samples"] += np.array([epoch_valid_samples, B])
            epoch_valid_pixels = torch.sum(qt_values * H * W).cpu().item()
            logs.setdefault("epoch_valid_pixels", np.zeros(2))
            logs["epoch_valid_pixels"] += np.array([epoch_valid_pixels, B * H * W])

        f1_global = f1_score(
            target_labels.flatten().detach().cpu(),
            pseudo_label.flatten().detach().cpu(),
            average="micro",
        )
        if logs is not None:
            epoch_LS = LS.cpu().detach().numpy()
            epoch_adv = loss_adv.cpu().detach().numpy()
            if abs_step > self.warmup:
                epoch_LT = LT.cpu().detach().numpy()
                logs.setdefault("epoch_LT", np.zeros(2))
                logs["epoch_LT"] += np.array([epoch_LT, B])
            epoch_f1_global = f1_global
            logs.setdefault("epoch_f1_global", np.zeros(2))
            logs["epoch_f1_global"] += np.array([epoch_f1_global, 1])

            logs.setdefault("epoch_LS", np.zeros(2))
            logs["epoch_LS"] += np.array([epoch_LS, B])
            logs.setdefault("epoch_adv", np.zeros(2))
            logs["epoch_adv"] += np.array([epoch_adv, B])

            logs.setdefault("epoch_ldc", np.zeros(2))
            logs["epoch_ldc"] += np.array([dc_loss.cpu().detach().numpy(), B])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        update_ema_state_dict(self.net, self.teacher, self.alpha, abs_step)
        data["source_logits"] = source_logits
        data["labels"] = labels
        data["unk_masks"] = unk_masks
        data["loss"] = loss
        return data
