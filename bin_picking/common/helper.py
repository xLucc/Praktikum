import json
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

# Return the src relative path
# NOTE: This is hard coded, please check if the colcon path vary.
def get_package_root() -> Path:
    return Path(get_package_share_directory('bin_picking')).parents[3] / 'src'

# Load dict from json file.
def load_dict_from_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
