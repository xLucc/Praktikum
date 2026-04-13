from pathlib import Path
from ament_index_python.packages import get_package_share_directory

# Return the src relative path
def get_package_root() -> Path:
    return Path(get_package_share_directory('bin_picking')).parents[3] / 'src'