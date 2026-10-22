import os
import random
import time

import h5py


# Modes that require the file to already exist. h5py raises FileNotFoundError
# (a subclass of OSError) for these, which must not be retried.
_READ_MODES = ("r", "r+")


def open_hdf5_file(file_path, mode="r", max_retries=100, sleep_max_seconds=1.0, create_parent=False):
    """Open an HDF5 file with bounded retries for transient OSErrors.

    The retries exist to survive lock contention when several processes work on
    the same file. A missing file is not transient, so it fails immediately
    instead of being retried for the full timeout.

    Args:
        file_path: HDF5 path.
        mode: Mode passed to h5py.File.
        max_retries: Maximum number of open attempts.
        sleep_max_seconds: Max random sleep between retries.
        create_parent: If True, create parent directory before opening.

    Returns:
        Open h5py.File handle.

    Raises:
        FileNotFoundError: if mode requires an existing file and there is none.
        OSError: if opening fails after all retries.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    if mode in _READ_MODES and not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such HDF5 file: {file_path} (mode={mode})")

    if create_parent:
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    last_error = None
    for attempt in range(max_retries):
        try:
            return h5py.File(file_path, mode)
        except OSError as err:
            last_error = err
            if attempt >= max_retries - 1:
                break
            time.sleep(random.random() * sleep_max_seconds)

    raise OSError(
        f"Could not open HDF5 file after {max_retries} retries: {file_path} (mode={mode})"
    ) from last_error
