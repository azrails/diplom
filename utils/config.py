import yaml
import os

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def dump_config(path, config: dict, config_name) -> None:
    """
    Create target dir if need and dump config
    """
    path_to_file = os.path.join(path, config_name)
    os.makedirs(path, exist_ok=True)
    with open(path_to_file, 'w') as f:
        yaml.dump(config, f)

