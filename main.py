import os
import copy
import argparse
import logging.config
from pathlib import Path

import numpy as np
import pandas as pd
import random
import yaml

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.metrics.loss_functions import MaskedCrossEntropyLoss, MaskedTverskyLoss
from models import MACFLY
from utils import SyncAugment
import uda_test

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RCS and class weight utilities
# ---------------------------------------------------------------------------

def compute_rcs_weights(dataset, num_classes, temperature=0.01):
    """Compute Rare Class Sampling weights for each patch in the dataset.

    Parameters
    ----------
    dataset : Dataset
        Source dataset with ``labels`` field in each sample.
    num_classes : int
        Total number of land-cover classes.
    temperature : float
        Softmax temperature; lower values concentrate mass on rare classes.

    Returns
    -------
    torch.Tensor
        Per-patch importance weights of shape ``(len(dataset),)``.
    """
    counts = torch.zeros(num_classes)
    for sample in dataset:
        y = sample["labels"].reshape(-1)
        for c in range(num_classes):
            counts[c] += (y == c).sum()

    freq = counts / counts.sum()
    logits = (1.0 - freq) / temperature
    logits[counts == 0] = -1e9
    P_class = torch.softmax(logits, dim=0)

    print("RCS class probabilities:",
          {c: f"{P_class[c]:.4f}" for c in range(num_classes)})

    patch_weights = []
    for sample in dataset:
        y = sample["labels"].reshape(-1)
        classes_in_patch = y.unique()
        w = sum(P_class[c].item() for c in classes_in_patch if c < num_classes)
        patch_weights.append(w)

    return torch.tensor(patch_weights)


def compute_class_weights(dataset, num_classes, device):
    """Compute inverse-frequency class weights for the Tversky loss.

    Parameters
    ----------
    dataset : Dataset
        Source dataset with ``labels`` field in each sample.
    num_classes : int
        Total number of land-cover classes.
    device : torch.device
        Target device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Normalised class weights of shape ``(num_classes,)``.
    """
    counts = torch.zeros(num_classes)
    for sample in dataset:
        y = sample["labels"].reshape(-1)
        for c in range(num_classes):
            counts[c] += (y == c).sum()

    print("Per-class pixel counts:", counts)

    weights = torch.zeros(num_classes)
    present = counts > 0
    weights[present] = 1.0 / counts[present]
    weights = weights / weights[present].sum()

    print("Class weights:", weights)
    return weights.to(device)


# ---------------------------------------------------------------------------
# Domain alignment losses
# ---------------------------------------------------------------------------

def coral_loss(xs: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    """Deep CORAL loss: Frobenius norm between source and target covariances.

    Parameters
    ----------
    xs, xt : torch.Tensor
        Feature matrices of shape ``(N, D)``.

    Returns
    -------
    torch.Tensor
        Scalar CORAL loss.
    """
    assert xs.ndim == 2 and xt.ndim == 2

    xs = xs - xs.mean(dim=0, keepdim=True)
    xt = xt - xt.mean(dim=0, keepdim=True)

    cov_s = (xs.T @ xs) / (xs.shape[0] - 1)
    cov_t = (xt.T @ xt) / (xt.shape[0] - 1)

    return torch.mean((cov_s - cov_t) ** 2)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SingleYearPatchDataset(Dataset):
    """Dataset of pre-extracted 64x64 Landsat SITS patches for a single year.

    Parameters
    ----------
    city_root : str
        Root directory containing per-year sub-directories.
    year : int
        Acquisition year to load.
    with_gt : bool
        Whether to load GLC_FCS30D ground-truth labels.
    """

    def __init__(self, city_root: str, year: int, with_gt: bool):
        year_dir = os.path.join(city_root, str(year))

        self.x = np.load(
            os.path.join(year_dir, "landsat_patches.npy"),
            mmap_mode="r",
        )
        self.with_gt = with_gt

        if with_gt:
            self.y = np.load(
                os.path.join(year_dir, "glc_patches.npy"),
                mmap_mode="r",
            )
            assert len(self.x) == len(self.y), "Mismatch between image and label patches."

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {"inputs": torch.from_numpy(self.x[idx].copy()).float()}
        if self.with_gt:
            sample["labels"] = torch.from_numpy(self.y[idx].copy()).long()
        return sample


class PairedYearUDADataset(Dataset):
    """Pairs source (labelled) and target (unlabelled) patch datasets for UDA.

    Parameters
    ----------
    source_ds : Dataset
        Labelled source-year dataset.
    target_ds : Dataset
        Unlabelled target-year dataset.
    aligned : bool
        If ``True``, source and target patches are matched by index (same
        spatial location). If ``False``, target patches are sampled randomly.
    transform : callable, optional
        Synchronised spatial augmentation applied to both domains.
    """

    def __init__(self, source_ds, target_ds, aligned=True, transform=None):
        self.source = source_ds
        self.target = target_ds
        self.aligned = aligned
        self.transform = transform

        if aligned:
            assert len(self.source) == len(self.target)

        self.length = len(self.source)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        src = self.source[idx]
        tgt_idx = idx if self.aligned else random.randint(0, len(self.target) - 1)
        tgt = self.target[tgt_idx]

        src_x = src["inputs"]
        tgt_x = tgt["inputs"]
        src_y = src.get("labels", None)

        if self.transform is not None:
            src_x, tgt_x, src_y = self.transform(src_x, tgt_x, src_y)

        out = {
            "source": {"inputs": src_x},
            "target": {"inputs": tgt_x},
        }
        if src_y is not None:
            out["source"]["labels"] = src_y

        return out


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

def load_model_config(config_path: str) -> dict:
    """Load a YAML model configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_uda(
    model,
    teacher,
    optimizer,
    dataloader,
    loss_fn,
    device,
    start_epoch: int,
    num_epochs: int,
    loss_log: dict,
    use_axial_coral: bool = True,
    use_global_coral: bool = False,
    use_tversky: bool = True,
    source_ds=None,
):
    """Train the student model for one chunk of epochs under UDA.

    The teacher model is updated via exponential moving average (EMA) of
    the student weights at the end of each batch.

    Parameters
    ----------
    model : nn.Module
        Student segmentation model.
    teacher : nn.Module
        EMA teacher model (weights are updated in-place).
    optimizer : torch.optim.Optimizer
        Optimiser attached to the student model.
    dataloader : DataLoader
        Paired source/target dataloader.
    loss_fn : callable
        Masked cross-entropy loss for the source domain.
    device : torch.device
        Training device.
    start_epoch : int
        Index of the first epoch in this chunk (used for logging).
    num_epochs : int
        Number of epochs to run in this chunk.
    loss_log : dict
        Dictionary accumulating per-epoch losses (modified in-place).
    use_axial_coral : bool
        Whether to apply the axial CORAL alignment loss.
    use_global_coral : bool
        Whether to apply the global CORAL alignment loss.
    use_tversky : bool
        Whether to add the masked Tversky loss to the objective.
    source_ds : Dataset, optional
        Source dataset used to compute class weights for the Tversky loss.

    Returns
    -------
    teacher : nn.Module
    model : nn.Module
    """
    model.train()
    teacher.eval()

    class_weights = compute_class_weights(source_ds, num_classes=10, device=device)

    loss_tversky = MaskedTverskyLoss(
        alpha=0.3,
        beta=0.7,
        class_weights=class_weights,
    )

    pbar = tqdm(range(start_epoch, start_epoch + num_epochs), desc="Training")

    for epoch in pbar:
        epoch_L_seg = 0.0
        epoch_L_coral = 0.0
        epoch_L_tversky = 0.0
        epoch_L_total = 0.0
        den = 0

        for batch in dataloader:
            src = batch["source"]
            tgt = batch["target"]

            x_s = src["inputs"].to(device)
            y_s = src["labels"].to(device)
            x_t = tgt["inputs"].to(device)

            with autocast(dtype=torch.float16):

                # Forward pass
                out_s, feat_s = model(x_s)
                out_t, feat_t = model(x_t)

                # Segmentation loss (source only)
                L_seg = loss_fn(out_s, y_s)

                # Domain alignment loss
                L_coral = torch.tensor(0.0, device=device)

                if use_global_coral:
                    feat_s_global = feat_s.mean(dim=(1, 2))  # [B*C, D]
                    feat_t_global = feat_t.mean(dim=(1, 2))
                    L_coral = L_coral + coral_loss(feat_s_global, feat_t_global)

                if use_axial_coral:
                    feat_s_H = feat_s.mean(dim=1).reshape(-1, feat_s.shape[-1])
                    feat_t_H = feat_t.mean(dim=1).reshape(-1, feat_t.shape[-1])
                    feat_s_W = feat_s.mean(dim=2).reshape(-1, feat_s.shape[-1])
                    feat_t_W = feat_t.mean(dim=2).reshape(-1, feat_t.shape[-1])
                    L_coral = L_coral + coral_loss(feat_s_H, feat_t_H) \
                                      + coral_loss(feat_s_W, feat_t_W)

                # Tversky loss (source only)
                L_tversky = loss_tversky(out_s, y_s) if use_tversky \
                    else torch.tensor(0.0, device=device)

                # Combined objective
                loss = L_seg + 0.5 * L_tversky + 0.1 * L_coral

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA teacher update
            with torch.no_grad():
                t_state = teacher.state_dict()
                s_state = model.state_dict()
                for k in t_state:
                    if "classifier" in k:
                        continue
                    if t_state[k].shape != s_state[k].shape:
                        continue
                    if not torch.is_floating_point(t_state[k]):
                        continue
                    t_state[k].mul_(0.99).add_(0.01 * s_state[k])
                teacher.load_state_dict(t_state)

            epoch_L_seg += L_seg.item()
            epoch_L_coral += L_coral.item()
            epoch_L_tversky += L_tversky.item()
            epoch_L_total += loss.item()
            den += 1

        loss_log["epoch"].append(epoch)
        loss_log["L_seg"].append(epoch_L_seg / den)
        loss_log["L_coral"].append(epoch_L_coral / den)
        loss_log["L_tversky"].append(epoch_L_tversky / den)
        loss_log["L_total"].append(epoch_L_total / den)

        pbar.set_postfix({
            "Loss": f"{epoch_L_total / den:.4f}",
            "CORAL": f"{epoch_L_coral / den:.4f}",
        })

    return teacher, model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MACFLY: Cross-year UDA training script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument("-c", "--city", type=str, required=True,
                          help="City name (must match a sub-directory in ds_dir).")
    required.add_argument("-yr", "--years", type=int, nargs=2, required=True,
                          metavar=("SOURCE_YEAR", "TARGET_YEAR"),
                          help="Source and target acquisition years.")

    parser.add_argument("-n_gpu", "--gpu_number", type=str, default="0",
                        help="CUDA device index.")
    parser.add_argument("-e", "--epochs", type=int, default=500,
                        help="Total number of training epochs.")
    parser.add_argument("-o", "--out_dir", type=str, default="Results",
                        help="Output directory for checkpoints and predictions.")
    parser.add_argument("-ds", "--ds_dir", type=str, default="./",
                        help="Root directory containing the dataset.")
    parser.add_argument("--axial_coral", action="store_true", default=True,
                        help="Use axial CORAL alignment loss.")
    parser.add_argument("--global_coral", action="store_true", default=False,
                        help="Use global CORAL alignment loss.")
    parser.add_argument("--no_tversky", action="store_true", default=False,
                        help="Disable the Tversky loss term.")
    return parser.parse_args()


def find_reference_tiff(data_root: str, city: str, year: int) -> Path:
    """Locate the GeoTIFF used as spatial reference for output prediction."""
    base_dir = Path(data_root) / "data_input" / city / str(year) / "000001"
    date_code = f"{year}0115"
    matching = list(base_dir.glob(f"*{date_code}"))
    if len(matching) != 1:
        raise RuntimeError(
            f"Expected exactly one directory matching '*{date_code}' "
            f"under {base_dir}, found {len(matching)}."
        )
    landsat_dir = matching[0]
    tif_path = landsat_dir / f"{landsat_dir.name}_B1_B2_B3_B4_B5_B7.tif"
    if not tif_path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {tif_path}")
    return tif_path


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number
    device = "cuda" if torch.cuda.is_available() else "cpu"

    source_year, target_year = args.years
    data_root = args.ds_dir
    city_root = os.path.join(data_root, "data_output", args.city)

    results_dir = os.path.join(args.out_dir, args.city)
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Datasets and dataloader
    # ------------------------------------------------------------------
    source_ds = SingleYearPatchDataset(city_root, year=source_year, with_gt=True)
    target_ds = SingleYearPatchDataset(city_root, year=target_year, with_gt=False)

    sync_transform = SyncAugment(
        angles=[0, 90, 180, 270],
        p_rotate=0.5,
        p_hflip=0.5,
        p_vflip=0.5,
    )

    uda_ds = PairedYearUDADataset(
        source_ds, target_ds, aligned=True, transform=sync_transform
    )

    rcs_weights = compute_rcs_weights(source_ds, num_classes=10, temperature=0.01)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=rcs_weights,
        num_samples=len(rcs_weights),
        replacement=True,
    )
    dataloader = DataLoader(
        uda_ds,
        batch_size=4,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model and optimiser
    # ------------------------------------------------------------------
    config_file = "./src/uda/configs/_base_/models/TSViTSW.yaml"
    model_config = load_model_config(config_file)

    model = MACFLY(model_config).to(device)
    teacher = copy.deepcopy(model).to(device).eval()

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
    loss_fn = MaskedCrossEntropyLoss()

    loss_log = {"epoch": [], "L_seg": [], "L_coral": [],
                "L_tversky": [], "L_total": []}

    # ------------------------------------------------------------------
    # Training loop (chunked checkpointing)
    # ------------------------------------------------------------------
    chunk_size = 500
    current_epoch = 0

    print(f"Training on city: {args.city} | "
          f"{source_year} -> {target_year} | device: {device}")

    while current_epoch < args.epochs:
        epochs_this_chunk = min(chunk_size, args.epochs - current_epoch)
        print(f"\n=== Epochs {current_epoch} -> "
              f"{current_epoch + epochs_this_chunk - 1} ===")

        teacher, model = train_uda(
            model=model,
            teacher=teacher,
            optimizer=optimizer,
            dataloader=dataloader,
            loss_fn=loss_fn,
            device=device,
            start_epoch=current_epoch,
            num_epochs=epochs_this_chunk,
            loss_log=loss_log,
            use_axial_coral=args.axial_coral,
            use_global_coral=args.global_coral,
            use_tversky=not args.no_tversky,
            source_ds=source_ds,
        )

        # Checkpoint
        ckpt_dir = os.path.join(
            results_dir, "checkpoints", f"{source_year}_{target_year}"
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(
            ckpt_dir,
            f"ckpt_epoch_{current_epoch + epochs_this_chunk}.pth",
        )
        torch.save(
            {
                "epoch": current_epoch + epochs_this_chunk,
                "model": model.state_dict(),
                "teacher": teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            ckpt_path,
        )
        print(f"Checkpoint saved: {ckpt_path}")

        current_epoch += epochs_this_chunk

        # Inference and georeferenced prediction
        print("Saving prediction...")
        full_map = uda_test.inference_teacher_timeseries(
            teacher=teacher,
            year_dir=os.path.join(
                data_root, "data_input", args.city, str(target_year), "000001"
            ),
            device=device,
            num_classes=10,
            patch_size=64,
            overlap=16,
        )

        tif_path = find_reference_tiff(data_root, args.city, target_year)

        pred_dir = os.path.join(results_dir, f"{source_year}_{target_year}")
        os.makedirs(pred_dir, exist_ok=True)

        uda_test.save_georeferenced_tiff(
            prediction=full_map,
            reference_raster_path=tif_path,
            out_path=os.path.join(
                pred_dir,
                f"prediction_{source_year}_{target_year}_"
                f"epoch{current_epoch}.tif",
            ),
            crop_border=0,
        )

    # Save loss history
    loss_df = pd.DataFrame(loss_log)
    loss_path = os.path.join(results_dir, f"loss_{source_year}_{target_year}.csv")
    loss_df.to_csv(loss_path, index=False)
    print(f"Loss history saved to {loss_path}")


if __name__ == "__main__":
    main()