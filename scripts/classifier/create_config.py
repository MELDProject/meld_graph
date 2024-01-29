import argparse
from copy import deepcopy
import os

import pprint
import black
import collections.abc


def write_config(name, config_dict, save_path):
    """write config.py file from config_dict to save_path"""
    pp = pprint.PrettyPrinter(indent=0)

    fname = os.path.join(save_path, name + ".py")
    # ensure exists
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    # write config file
    config_str = ""

    config_str += "network_parameters = "
    config_str += pp.pformat(config_dict["network_parameters"])
    config_str += "\n"
    config_str += "data_parameters = "
    config_str += pp.pformat(config_dict["data_parameters"])
    config_str += "\n"

    # format with black
    mode = black.FileMode()
    fast = False
    config_str = black.format_file_contents(config_str, fast=fast, mode=mode)

    # save
    with open(fname, "w") as f:
        f.write(config_str)


def load_config(config_file):
    """load config.py file and return config object"""
    import importlib.machinery, importlib.util

    loader = importlib.machinery.SourceFileLoader("config", config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config


def update(d, u):
    """
    Deep update dict d with u.
    NOTE if d contains a subdict that is updated with u, existing entries in d are not removed.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Helper to create multiple experiment configs from base and var config files.
        """
    )
    parser.add_argument("base_config", help="path to base experiment_config.py")
    parser.add_argument("var_config", help="path to var experiment_config.py")
    parser.add_argument("--save_path", help="dir to place configs in", default=".")
    args = parser.parse_args()

    var_config = load_config(args.var_config)
    base_config = load_config(args.base_config)

    configs = []
    # first, create parallel configs
    for cur_var_config in var_config.parallel:
        cur_config = {
            "network_parameters": deepcopy(base_config.network_parameters),
            "data_parameters": deepcopy(base_config.data_parameters),
        }
        cur_config = update(cur_config, cur_var_config)
        configs.append(cur_config)

    if len(configs) == 0:
        # if not parallel configs defined, use base config, but ensure that name is base_name
        cur_config = {
            "network_parameters": deepcopy(base_config.network_parameters),
            "data_parameters": deepcopy(base_config.data_parameters),
        }
        cur_config["network_parameters"]["name"] = var_config.base_name
        configs.append(cur_config)

    # then, create sequential configs
    final_configs = {}
    for config in configs:
        init_weights = None
        for i, cur_var_config in enumerate(var_config.sequential):
            cur_config = deepcopy(config)
            cur_config = update(cur_config, cur_var_config)
            # upate name
            # set name of config to save first - need to put fold in front of the s_X folder
            cur_fold = cur_config["data_parameters"]["fold_n"]
            config_name = cur_config["network_parameters"]["name"] + f"/fold_{cur_fold:02d}" + f"/s_{i}"
            cur_config["network_parameters"]["name"] = cur_config["network_parameters"]["name"] + f"/s_{i}"
            cur_config["network_parameters"]["training_parameters"]["init_weights"] = init_weights
            # update init weights to this experiment for finetuning next experiment
            init_weights = cur_config["network_parameters"]["name"] + f"/fold_{cur_fold:02d}/best_model.pt"
            # add to configs list
            final_configs[config_name] = cur_config

    # write final configs to disk
    # configs will be written to a folder with base_name + parallel_name
    # configs will be named s_i.py

    for n, config in final_configs.items():
        write_config(n, config, save_path=args.save_path)

    print(f"Generated {len(final_configs)} configs in {var_config.base_name}*")
