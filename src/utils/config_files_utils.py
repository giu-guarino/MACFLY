# -------------------------------------------------------------------------------------
# Code partly from https://gist.github.com/joshbode/569627ced3076931b02f
# -------------------------------------------------------------------------------------


import os
import json
import yaml
from typing import Any, IO
import argparse
import warnings
from ast import literal_eval
from pathlib import Path


def save_yaml(data, file_path):
    file_path = Path(file_path).resolve()
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)


class RecursiveLoader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: RecursiveLoader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(
        os.path.join(loader._root, loader.construct_scalar(node))
    )
    extension = os.path.splitext(filename)[1].lstrip(".")

    with open(filename, "r") as f:
        if extension in ("yaml", "yml"):
            return yaml.load(f, RecursiveLoader)
        elif extension in ("json",):
            return json.load(f)
        else:
            return "".join(f.readlines())


yaml.add_constructor("!include", construct_include, RecursiveLoader)


def read_yaml(yaml_file, loader=RecursiveLoader):
    with open(yaml_file, "r") as config_file:
        yaml_dict = yaml.load(config_file, loader)
    return yaml_dict


def copy_yaml(config_file):
    """
    copies config file to training savedir
    """
    if type(config_file) is str:
        yfile = read_yaml(config_file)
    elif type(config_file) is dict:
        yfile = config_file
    save_name = yfile["CHECKPOINT"]["save_path"] + "/config_file.yaml"
    with open(save_name, "w") as outfile:
        yaml.dump(yfile, outfile, default_flow_style=False)


def get_params_values(args, key, default=None):
    """
    set default to None if a value is required in the config file
    """
    if (key in args) and (args[key] is not None):
        return args[key]
    return default


def get_dataset_info(dataset):
    with open(
        "/home/christopher/DATA/GloUrb/src/data/lulc_datasets/legends.yaml", "r"
    ) as file:
        info = yaml.safe_load(file)
    return info[dataset]


def convert_to_simplest_type(string):
    try:
        return literal_eval(string)
    except:
        return string


def access_dict_value(d, key):
    """Access a nested dictionary value given a dotted string key."""
    keys = key.split(".")
    for k in keys:
        d = d.get(k, {})
        if not isinstance(d, dict):
            return d
    return d


def set_dict_value(d, key, value):
    """Set a nested dictionary value given a dotted string key."""
    keys = key.split(".")
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]  # reference is changed but not the original dict
    d[keys[-1]] = value


class ParseDict(argparse.Action):
    """https://sumit-ghosh.com/posts/parsing-dictionary-key-value-pairs-kwargs-argparse-python/"""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for kv in values:
            key, value = kv.split("=")
            getattr(namespace, self.dest)[key] = value


def add_branch(tree, vector, value):
    """https://stackoverflow.com/questions/30880973/parse-a-dot-seperated-string-into-dictionary-variable"""
    key = vector[0]
    tree[key] = (
        convert_to_simplest_type(value)
        if len(vector) == 1
        else add_branch(tree[key] if key in tree else {}, vector[1:], value)
    )
    return tree


def convert_to_type(v, val):
    if isinstance(v, bool):
        return val == "True"
    else:
        return type(v)(val)


def override_config(parser):
    known_args, unknown_args = parser.parse_known_args()
    assert known_args.config is not None, "specify a config file"
    cfg = read_yaml(known_args.config)
    for arg in unknown_args:
        if arg.startswith("--"):
            parser.add_argument(arg, nargs="?")
    args = parser.parse_args()
    for key, val in vars(args).items():
        if key not in vars(known_args):
            v = access_dict_value(cfg, key)
            if v == {}:
                warnings.warn("arg %s not in config file" % key)
                set_dict_value(cfg, key, convert_to_simplest_type(val))
            else:
                set_dict_value(cfg, key, convert_to_type(v, val))
    return args, cfg
