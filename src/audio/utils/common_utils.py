import os
import time
import math
import random
import logging
import functools
from io import BytesIO

import numpy as np
import pandas as pd
import torch


def define_seed(seed: int = 12) -> None:
    """Fix seed for reproducibility

    Args:
        seed (int, optional): seed value. Defaults to 12.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def wait_for_it(time_left: int) -> None:
    """Wait time in sec. before run the following statement

    Args:
        time_left (int): time in seconds
    """
    t = time_left
    while t > 0:
        print("Time left: {0}".format(t))
        time.sleep(60)
        t = t - 1


def create_logger(
    log_path, console_level: int = logging.ERROR, file_level: int = logging.WARNING
) -> logging.Logger:
    """Create console and file logger

    Args:
        log_path (str): Logs file path
        console_level (int, optional): Console level. Defaults to logging.ERROR.
        file_level (int, optional): File level. Defaults to logging.WARNING.

    Returns:
        logging.Logger: Registered logger
    """
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
    file_handler = logging.FileHandler(log_path, "a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(file_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)

    res_logger = logging.getLogger("mask_logger")
    for hndlr in res_logger.handlers[:]:
        res_logger.removeHandler(hndlr)

    res_logger.addHandler(file_handler)
    res_logger.addHandler(console_handler)
    res_logger.setLevel(logging.DEBUG)

    return res_logger


def majority_voting(
    targets: list[np.ndarray], predicts: list[np.ndarray], samples_info: list[dict]
) -> tuple[list, list[np.ndarray], list[str]]:
    """Window-wise majority voting (or mode) with grouping by file names

    Args:
        targets (list[np.ndarray]): List of targets
        predicts (list[np.ndarray]): List of predicts
        samples_info (list[dict]): List of samples info with corresponding filenames for each item of targets/predicts lists

    Returns:
        tuple[list, list[np.ndarray], list[str]]:
    """
    filenames = []
    for s_info in samples_info:
        filenames.extend(s_info["a_filename"])

    # forming dataframe: targets, predicted class, filenames
    df_dict = {}
    df_dict["targets"] = targets
    df_dict["predicts"] = np.argmax(predicts, axis=1)
    df_dict["filenames"] = filenames

    # group by filenames with mode calculation
    df = (
        pd.DataFrame(df_dict)
        .groupby("filenames", as_index=False)
        .agg(lambda x: pd.Series.mode(x)[0])
    )

    # one-hot encoding grouped predicts
    preds = [
        (np.arange(len(predicts[0])) == i).astype(int) for i in df["predicts"].values
    ]
    return df["targets"].to_list(), preds, df["filenames"].to_list()


def round_math(val: float) -> int:
    """Rounds value. Proposed by *** # TODO

    Args:
        val (float): Value

    Returns:
        int: Rounded value
    """
    modf = math.modf(val)

    if modf[0] >= 0.5:
        res = modf[1] + 1
    else:
        if modf[0] <= -0.5:
            res = modf[1] - 1
        else:
            res = math.ceil(modf[1])

    return int(res)


def array_to_bytes(x: np.ndarray) -> bytes:
    """Converts numpy array to bytes

    Args:
        x (np.ndarray): Numpy array

    Returns:
        bytes: Bytes
    """
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()


def bytes_to_array(b: bytes) -> np.ndarray:
    """Converts bytes to numpy array

    Args:
        b (bytes): Bytes

    Returns:
        np.ndarray: Numpy array
    """
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)
