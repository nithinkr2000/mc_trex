import numpy as np
from numpy.typing import NDArray
from typing import List


def blocked_standard_error(
    data: NDArray[np.float64] | List[float], block_size: int = 1
) -> float:
    """
    Perform a blocked standard error analysis on the data passed as argument.

    Parameters
    ----------

    data:NDArray[np.float64]
        Contains the data to be analyzed in a linear array.

    block_size:int
        Size of the blocks into which the data is divided. It must be less than
        half the length of the data array. Default value is
        block_size = len(data) // 2 - 1

    Returns
    -------
    standard_error : float
        Standard error of the mean for the given block size.

    """
    
    len_dat = len(data)

    if not block_size:
        block_size = len(data) // 2 - 1
    else:
        assert block_size < len_dat // 2, ValueError("Block too large, blockhead!")

    n_blocks = len_dat // block_size
    blocks = np.array_split(data[: n_blocks * block_size], n_blocks)
    block_means = np.array([np.mean(block) for block in blocks])
    standard_error = np.std(block_means, ddof=1) / np.sqrt(n_blocks)

    return standard_error


def running_average(
    data: NDArray[np.float64] | List[float], window: int = 1
) -> NDArray[np.float64]:
    """
    Calculate the rolling mean or running average of an array.
    """
    dat_cumsum = np.cumsum(data)
    return (dat_cumsum[window:] - dat_cumsum[:-window]) / window
