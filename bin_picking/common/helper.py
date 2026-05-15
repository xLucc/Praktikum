import json
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

# Return the workspace src directory dynamically by searching ancestor paths.
def get_package_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'src').is_dir():
            return parent / 'src'
    raise RuntimeError(f'Found no src directory in ancestors of {current}')

def get_project_dir() -> Path:
    return get_package_root() / 'bin_picking'


# Load dict from json file.
def load_dict_from_json(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def write_json_from_dict(data: dict, path: Path) -> bool:

    try:
        json_data = json.dumps(data)
    except Exception as e:
        print("Can't convert the data to json.")
        print(e)
        return False
    
    try:
        with open(path, 'w') as f:
            f.write(json_data)
    except Exception as e:
        print('Error occured while writing the file.')
        print(e)
        return False
    
    return True
