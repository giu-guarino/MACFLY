import pandas as pd
from pathlib import Path
import rasterio as rio
from tqdm import tqdm
import numpy as np
import json


def create_stats_file(fpath, outpath=None, ncls=10):
    """Create a csv file where each line gives the number of pixel for each class"""
    fpath = Path(fpath)
    df = pd.read_csv(fpath, header=None, names=["path"])
    for i in range(ncls):
        df["%d" % (i)] = 0
    with tqdm(total=len(df.index)) as pbar:
        for index, row in tqdm(df.iterrows()):
            p = Path(fpath.parent, row["path"])
            src = rio.open(p)
            img = src.read(1)
            for i in range(ncls):
                df.at[index, "%d" % (i)] = np.sum(np.where(img == i, 1, 0))
            src.close()
            pbar.update(1)
    if outpath is not None:
        df.to_csv(outpath, index=False)  # header=False
    else:
        return df


def gt_stability(source_gtpath, target_gtpath, ncls=10):
    """
    outputs:
        df: pixelwise stats
        global_stats: gives the ratio of the unchanged labels
    """
    source_gtpath = Path(source_gtpath)
    target_gtpath = Path(target_gtpath)

    df = pd.read_csv(source_gtpath, header=None, names=["path"])
    df_target = pd.read_csv(target_gtpath, header=None, names=["path"])
    df = pd.merge(
        df, df_target, how="inner", on="path"
    )  # if the paths are not the same
    for i in range(ncls):
        df["source %d" % (i)] = 0
        df["stable %d" % (i)] = 0
    df["stability"] = 0
    with tqdm(total=len(df.index), position=0, leave=True) as pbar:
        for index, row in tqdm(df.iterrows()):
            source_p = Path(source_gtpath.parent, row["path"])
            source_src = rio.open(source_p)
            source_img = source_src.read(1)

            target_p = Path(target_gtpath.parent, row["path"])
            target_src = rio.open(target_p)
            target_img = target_src.read(1)
            for i in range(ncls):
                eq_cl = source_img == i
                stability = source_img == target_img
                df.at[index, "source %d" % (i)] = np.sum(eq_cl)
                df.at[index, "stable %d" % (i)] = np.sum(eq_cl * stability)
                df.at[index, "stability"] = np.sum(stability)
            source_src.close()
            target_src.close()
            pbar.update(1)

    s = df.sum(numeric_only=True)
    global_stats = pd.Series()
    for i in range(ncls):
        if s["source %d" % (i)] != 0:
            global_stats["ratio %d" % (i)] = s["stable %d" % (i)] / s["source %d" % (i)]
        else:
            global_stats["ratio %d" % (i)] = -1.0
    global_stats["stability"] = s["stability"] / df.shape[0] / 64 / 64
    global_stats.round(decimals=3)
    return df, global_stats


def get_sample_class_stats(sample_path, CLS_dict):
    src = rio.open(sample_path)
    labels = src.read(1).flatten()
    values = CLS_dict["values"]
    sample_class_stats = {"%s" % v: np.sum(np.where(labels == v, 1, 0)) for v in values}
    return sample_class_stats


def save_class_stats(out_json_path, sample_class_stats):
    with open(Path(out_json_path), "w") as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop("file")
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, "sample_class_stats_dict.json"), "w") as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, "samples_with_class.json"), "w") as of:
        json.dump(samples_with_class, of, indent=2)
