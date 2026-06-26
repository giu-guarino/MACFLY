import torch
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import os
import logging

from datetime import datetime

logger = logging.getLogger(__name__)


def extract_doy_from_folder(folder_name):
    """
    folder_name: es. LANDSAT45-PLANETARY_000001_GAPFILL_19950404
    """
    date_str = folder_name[-8:]  # YYYYMMDD
    dt = datetime.strptime(date_str, "%Y%m%d")
    return dt.timetuple().tm_yday  # 1..365

def list_date_folders(year_dir):
    dates = []
    for d in os.listdir(year_dir):
        full_path = os.path.join(year_dir, d)
        if os.path.isdir(full_path):
            try:
                datetime.strptime(d[-8:], "%Y%m%d")
                dates.append(d)
            except ValueError:
                pass

    dates = sorted(dates)

    # Controllo esplicito ordine
    for i in range(len(dates) - 1):
        assert dates[i] < dates[i + 1], "Date folders not sorted correctly"

    return [os.path.join(year_dir, d) for d in dates]

def read_landsat_timeseries(year_dir):
    date_folders = list_date_folders(year_dir)

    images = []
    doys = []

    shape_ref = None
    crs_ref = None
    transform_ref = None

    for d in date_folders:
        folder_name = os.path.basename(d)
        doy = extract_doy_from_folder(folder_name)
        doys.append(doy)

        tif_files = [f for f in os.listdir(d) if f.endswith(".tif")]
        assert len(tif_files) == 1, f"Expected one tif in {d}"

        tif_path = os.path.join(d, tif_files[0])

        with rasterio.open(tif_path) as src:
            img = src.read()  # (C, H, W)

            if shape_ref is None:
                shape_ref = img.shape
                crs_ref = src.crs
                transform_ref = src.transform
            else:
                assert img.shape == shape_ref
                assert src.crs == crs_ref
                assert src.transform == transform_ref

        images.append(img)

    # (T, C, H, W)
    ts = np.stack(images, axis=0)

    # ---------- TIME CHANNEL ----------
    T, C, H, W = ts.shape

    doys = np.array(doys, dtype=np.float32)           # (T,)
    doys_norm = doys / 365.0                           # [0,1]

    # (T, 1, H, W)
    time_channel = doys_norm[:, None, None, None]
    time_channel = np.broadcast_to(time_channel, (T, 1, H, W))

    # concatena come ultimo canale
    ts = np.concatenate([ts, time_channel], axis=1)   # (T, C+1, H, W)

    return ts



def percentile_normalize(ts, pmin=2, pmax=98):
    """
       ts: (T, C+1, H, W)
       ultimo canale = tempo (NON normalizzato)
       """

    spectral = ts[:, :-1]  # (T, C, H, W)
    time = ts[:, -1:]  # (T, 1, H, W)

    flat = spectral.reshape(-1)
    lo, hi = np.percentile(flat, [pmin, pmax])

    spectral = np.clip(spectral, lo, hi)
    spectral = (spectral - lo) / (hi - lo + 1e-6)

    ts_norm = np.concatenate([spectral, time], axis=1)
    return ts_norm.astype(np.float32)

def sliding_window_coords_full(H, W, patch_size, overlap):
    stride = patch_size - overlap

    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))

    # forza ultima patch sul bordo se necessario
    if ys[-1] != H - patch_size:
        ys.append(H - patch_size)
    if xs[-1] != W - patch_size:
        xs.append(W - patch_size)

    for y in ys:
        for x in xs:
            yield y, x


def compute_valid_region(y, x, H, W, patch_size, border_crop):
    y0, y1 = 0, patch_size
    x0, x1 = 0, patch_size

    if y > 0:
        y0 = border_crop
    if y + patch_size < H:
        y1 = patch_size - border_crop
    if x > 0:
        x0 = border_crop
    if x + patch_size < W:
        x1 = patch_size - border_crop

    return y0, y1, x0, x1



def inference_teacher_timeseries(
    teacher,
    year_dir,
    device,
    num_classes,
    patch_size=64,
    overlap=16,
    batch_size=4,
):
    # ---- load & normalize ----
    ts = read_landsat_timeseries(year_dir)  # (T,C,H,W)
    ts = percentile_normalize(ts)

    print(f"Min = {np.min(ts)}, Max = {np.max(ts)}")

    T, C, H, W = ts.shape

    #border_crop = overlap // 2
    border_crop = 0

    teacher.eval()
    teacher.to(device)

    full_map = np.zeros((H, W), dtype=np.uint8)

    patch_buffer = []
    coord_buffer = []

    with torch.no_grad():
        for (y, x) in sliding_window_coords_full(H, W, patch_size, overlap):
            patch = ts[:, :, y:y+patch_size, x:x+patch_size]
            patch_buffer.append(patch)
            coord_buffer.append((y, x))

            if len(patch_buffer) == batch_size:
                _process_batch_full(
                    teacher,
                    patch_buffer,
                    coord_buffer,
                    full_map,
                    device,
                    patch_size,
                    border_crop,
                    H,
                    W,
                )
                patch_buffer.clear()
                coord_buffer.clear()

        if patch_buffer:
            _process_batch_full(
                teacher,
                patch_buffer,
                coord_buffer,
                full_map,
                device,
                patch_size,
                border_crop,
                H,
                W,
            )

    final_map = full_map[
        border_crop : H - border_crop,
        border_crop : W - border_crop
    ]

    return final_map



def _process_batch_full(
    teacher,
    patch_buffer,
    coord_buffer,
    full_map,
    device,
    patch_size,
    border_crop,
    H,
    W,
):
    x = torch.from_numpy(np.stack(patch_buffer)).float().to(device)

    logits, _ = teacher(x)  # (B,C,64,64)
    preds = torch.argmax(logits, dim=1).cpu().numpy()

    for i, (y, x0) in enumerate(coord_buffer):
        pred = preds[i]

        y1, y2, x1, x2 = compute_valid_region(
            y, x0, H, W, patch_size, border_crop
        )

        fy1 = y + y1
        fy2 = y + y2
        fx1 = x0 + x1
        fx2 = x0 + x2

        full_map[fy1:fy2, fx1:fx2] = pred[y1:y2, x1:x2]


def save_georeferenced_tiff(
        prediction,
        reference_raster_path,
        out_path,
        crop_border=0,
):

    with rasterio.open(reference_raster_path) as ref:
        crs = ref.crs
        transform = ref.transform
        dtype = prediction.dtype

        if crop_border > 0:
            H, W = prediction.shape

            # crop array
            prediction = prediction[
                crop_border: H - crop_border,
                crop_border: W - crop_border
            ]

            # aggiorna transform
            transform = transform * Affine.translation(
                crop_border,
                crop_border
            )

        height, width = prediction.shape

        profile = ref.profile.copy()
        profile.update(
            {
                "height": height,
                "width": width,
                "count": 1,
                "dtype": dtype,
                "transform": transform,
                "crs": crs,
            }
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(prediction, 1)
