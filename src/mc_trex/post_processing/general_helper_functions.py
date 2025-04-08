import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple


def statistical_inefficiency(
    data: NDArray[np.float64], block_sizes: None | NDArray[np.int64]
) -> NDArray[np.float64]:
    """
    Calculate the statistical inefficiency for data. Can be used to estimtae
    error for the data.

    Parameters
    ----------

    data : NDArray[np.float64]
        Contains the data to be analyzed in a linear array.

    block_sizes : None | NDArray[np.int64]
        Size of the blocks into which the data is divided. The largest value
        must be less than or equal to half the length of the data array.
        block_size <= len(data) // 2

    Returns
    -------

    statistical_inefficiency : NDArray[np.float64]
        Statistical inefficiency, converges to twice the auto-correlation time.
        The auto-correlation time can be used to make an error estimate.

    """

    len_dat = len(data)
    full_var = np.var(data)

    if block_sizes is None:
        block_sizes = np.arange(1, len_dat // 2 - 1)

    statistical_inefficiency = np.zeros(len(block_sizes))

    for idx, block_size in enumerate(block_sizes):
        # Number of blocks
        n_blocks = len_dat // block_size

        # Creating blocks
        blocks = np.array_split(data[: n_blocks * block_size], n_blocks)

        # Calculating the block averages
        block_means = np.apply_along_axis(np.mean, arr=blocks, axis=1)

        # Calculating the inefficiency
        statistical_inefficiency[idx] = (
            block_size * np.var(block_means, ddof=1) / full_var
        )

    return statistical_inefficiency


def running_average(
    data: NDArray[np.float64] | List[float], window: int = 1
) -> NDArray[np.float64]:
    """
    Calculate the rolling mean or running average of an array.
    """
    dat_cumsum = np.cumsum(data)
    return (dat_cumsum[window:] - dat_cumsum[:-window]) / window


def tm_estimation(
    temperatures: NDArray[np.float64],
    fit_T1: NDArray[np.float64],
    fit_T2: NDArray[np.float64],
    ref_name: str,
) -> Tuple[int, np.float64]:
    """
    Function to generate the melting temperature from the melting curve fits.
    The melting curves passed generally correspond to the unfolded states and a
    folded state. The point where they meet would be the melting point for that
    folded state.

    Parameters
    ----------

    temperatures : NDArray[np.float64]
        The temperatures for which the melting curves have been calculated.

    fit_T1 : NDArray[np.float64]
        The first melting curve, calculated for the `temperatures` array.

    fit_T2 : NDArray[np.float64]
        The second melting curve, calculated for the `temperatures` array.

    ref_name : str
        Name of the structure/configuration for which the melting curve has
        been passed. Default is "folded" which implies the natively folded
        structure of the biomolecule.

    Results
    -------

    min_diff_idx : int
        Index of the estimate of the melting temperature from the melting
        curves. It is taken to be the temperature at which the melting
        curves are the closest.

    temperatures[min_diff_idx] : float
        The melting temperature.

    Additionally generates messages in the following scenarios
    1. if the minimum is the first element, the melting temperature is below
       the range of `temperatures`.
    2. if the minimum is the last element, the melting temperature is above the
       the range of `temperatures`.
    3. if the differences between the first elements and the last elements of
       the melting curves are not of opposite signs, then the curves diverge on
       either side.
    """

    # Test whether the melting curves are the same length as the `temperatures`
    # array
    assert len(temperatures) == len(fit_T1) and len(temperatures) == len(fit_T2), (
        "Length mismatch between inputs."
    )

    # Find point of least difference between melting curves.
    min_diff_idx = np.argmin(np.abs(np.subtract(fit_T1, fit_T2)))

    if min_diff_idx == 0:
        print(
            "Minimum distance between melting curves detected at least\
        temperature. Melting curve fit likely to be inaccurate."
        )

    elif min_diff_idx == len(temperatures) - 1:
        print(
            "Minimum distance between melting curves detected at highest\
        temperature. Melting curve fit likely to be inaccurate."
        )

    elif np.sign(fit_T1[0] - fit_T2[0]) * np.sign(fit_T1[-1] - fit_T2[-1]) > 0:
        print(
            "Lowest temperature where melting curves meet is {}. However, \
        melting curves do not follow consistent trend.".format(
                temperatures[min_diff_idx]
            )
        )

    else:
        print(
            ref_name
            + " melting temperature: {}K".format(
                np.round(temperatures[min_diff_idx], decimals=2)
            )
        )

    return min_diff_idx, np.round(temperatures[min_diff_idx], decimals=2)


# def
