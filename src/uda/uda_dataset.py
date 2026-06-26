import numpy as np
import torch
import pandas as pd
from src.data.builder import DATASETS
import warnings


def get_overall_class_stats(path):
    df = pd.read_csv(path)
    header = df.columns
    overall_class_stats = {}
    for i in range(len(df)):
        s = dict(zip(header, df.iloc[i]))
        del s["path"]
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v for k, v in sorted(overall_class_stats.items(), key=lambda item: item[1])
    }

    return overall_class_stats


def get_rcs_class_probs(overall_class_stats, temperature):
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)
    return list(overall_class_stats.keys()), freq.numpy()


@DATASETS.register_module()
class UDADataset(object):
    def __init__(self, source, target, cfg, DATASET_INFO):
        """cfg -> dataset config"""
        self.source = source
        self.target = target
        rcs_cfg = cfg["RCS"]
        self.rcs_enabled = rcs_cfg["enabled"]

        if self.rcs_enabled:
            path = DATASET_INFO[cfg["source"]["dataset"]]["paths_stats"]

        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg["class_temp"]
            # self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg["min_pixels"]
            self.paired = rcs_cfg["source_target_paired"]
            if not self.paired:
                self.target_order = np.arange(self.source.__len__())
                np.random.shuffle(self.target_order)

            self.overall_class_stats = get_overall_class_stats(path)
            df = pd.read_csv(path)
            # TODO: Warning, source and target can be different, add an assert or do differently
            import copy

            self.landsat_paths = copy.deepcopy(self.source.landsat_paths)
            pref = df.iloc[0]["path"].split("_")[0]

            def convert_paths(s_landsat, pref):
                return "_".join([pref] + s_landsat.split("_")[1:])

            self.lulc_paths = self.landsat_paths
            self.lulc_paths.rename(columns={"landsat_paths": "path"}, inplace=True)
            self.lulc_paths["path"] = self.lulc_paths["path"].apply(
                convert_paths, args=(pref,)
            )
            self.samples_with_class_n = pd.merge(
                self.lulc_paths, df, on="path", how="left"
            )

            self.samples_with_class = {}
            for c in self.overall_class_stats:
                self.samples_with_class[c] = []
            for idx, row in self.samples_with_class_n.iterrows():
                for c in self.overall_class_stats:
                    if row.loc[str(c)] > self.rcs_min_pixels:
                        self.samples_with_class[c].append(idx)
            # Check there is at least one sample for each class
            empty = []
            for k, v in self.samples_with_class.items():

                if len(v) == 0:
                    empty.append(k)
            if len(empty) > 0:
                warnings.warn(f"no images for classes {empty}")
                for v in empty:
                    del self.overall_class_stats[v]
            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                self.overall_class_stats, self.rcs_class_temp
            )
            print("RCS")
            print(
                f"\tnumber of items by class: {[len(v) for k,v in self.samples_with_class.items()]}"
            )
            print(
                f"\tclasses: {self.rcs_classes}\n\tclass probabilities: {list(self.rcs_classprob)}"
            )

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        idx = np.random.choice(self.samples_with_class[c])
        # i1 = self.file_to_idx[f1]
        s1 = self.source.__getitem__(idx)
        # No crop right now, so the following can be commented
        #  if self.rcs_min_crop_ratio > 0:
        #      for j in range(10):
        #          n_class = torch.sum(s1["gt_semantic_seg"].data == c)
        #          # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
        #          if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
        #              break
        #          # Sample a new random crop from source image i1.
        #          # Please note, that self.source.__getitem__(idx) applies the
        #          # preprocessing pipeline to the loaded image, which includes
        #          # RandomCrop, and results in a new crop of the image.
        #          s1 = self.source[i1]
        #  i2 = np.random.choice(range(len(self.target)))
        if self.paired:
            s2 = self.target.__getitem__(idx)
            assert s1["ids"] == s2["ids"], "patches are not well matched"
        else:
            s2 = self.target.__getitem__(self.target_order[idx])

        return {"source": s1, "target": s2}

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            s1 = self.source.__getitem__(idx)
            s2 = self.target.__getitem__(idx)
            assert s1["id"] == s2["id"], "patches are not well matched"
            return {"source": s1, "target": s2}

    def __len__(self):
        length = min(len(self.source), len(self.target))
        return length
