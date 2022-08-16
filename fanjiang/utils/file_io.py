import os

from iopath.common.file_io import (HTTPURLHandler, OneDrivePathHandler,
                                   PathHandler)
from iopath.common.file_io import PathManager as PathManagerBase

__all__ = ["PathManager", "PathHandler"]


PathManager = PathManagerBase()
"""
This is a detectron2 project-specific PathManager.
We try to stay away from global PathManager in fvcore as it
introduces potential conflicts among other libraries.
"""

PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())


def find_all(input_dir, postfix="", return_dict=False):
    names = []
    for root, _, files in os.walk(input_dir, topdown=False):
        for name in files:
            if name.endswith(postfix):
                names.append(os.path.join(root, name))

    names = sorted(names)

    if return_dict:
        names = {os.path.basename(name): name for name in names}

    return names
