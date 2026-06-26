import sys
#from .build_train import TRAIN_STEP
#from src.uda.train_step import GRL_train_step

GLAD_CLS = {
    "labels": [
        "True desert",
        "Semi-arid",
        "Dense short vegetation",
        "Tree cover",
        "Wetlands",
        "Open surface water",
        "Snow/ice",
        "Cropland",
        "Built-up",
        "Ocean",
    ],
    "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "colors": [
        "#FEFECC",
        "#FAFAC3",
        "#C0C02F",
        "#609C60",
        "#BFC0C0",
        "#1964EB",
        "#ffffff",
        "#ff7d00",
        "#64dcdc",
        "#111133",
    ],
}

BANDS_COLORS = {
    "red": "#FF0000",
    "green": "#00FF00",
    "blue": "#0000FF",
    "nir08": "#FF5733",
    "swir16": "#FF33CE",
    "swir22": "#9633FF",
}
__all__ = ["GLAD_CLS", "BANDS_COLORS"]
