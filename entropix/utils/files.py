"""Files utils."""

import os

__all__ = ('get_input_filepaths')


def get_input_filepaths(dirpath):
    """Return all the files under the specified directory."""
    return [os.path.join(dirpath, filename) for filename in
            os.listdir(dirpath) if not filename.startswith('.')]
