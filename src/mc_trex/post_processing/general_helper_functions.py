import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Callable


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
        Sizes of the blocks into which the data is divided. The largest value
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


def jack_knife(
    data: NDArray[np.float64], n_blocks: int, f: Callable = np.mean
) -> NDArray[np.float64]:
    """
    Jack-knife method for error estimation.

    Parameters
    ----------

    data : NDArray[np.float64]
        Contains the data to be analyzed in a linear array.

    n_blocks : int
        Number of blocks to be used in the calculation of the error.

    f : Callable
        Function to be applied to the dataset before application of the
        jack-knife method.

    Returns
    -------

    del_rho : float
        The error estimate from jack-knife.

    """
    len_dat = len(data)
    block_size = len_dat // n_blocks

    blocks = np.array(np.array_split(data[: n_blocks * block_size], n_blocks))

    # Dataset without one block
    blocks_m = np.zeros([n_blocks, (n_blocks - 1) * block_size])
    for idx in range(n_blocks):
        blocks_m[idx] = np.append(blocks[:idx].flatten(), blocks[idx + 1 :].flatten())

    rho_m_bar = np.apply_along_axis(func1d=f, axis=1, arr=blocks_m)
    rho_bar = f(data[: n_blocks * block_size])

    del_rho = np.sqrt((n_blocks - 1) / n_blocks) * np.sqrt(
        np.sum(np.power(np.subtract(rho_m_bar, rho_bar), 2))
    )

    return del_rho


def blocked_bootstrap(
    data: NDArray[np.float64],
    block_size: int,
    confidence: int = 5,
    n_bootstraps: int = 1000,
    f: Callable = np.mean,
) -> Tuple[NDArray[np.float64], List[float]]:
    """
    Perform bootstrap on blocked values.

    Parameters
    ----------

    data : NDArray[np.float64]
        Contains the data to be analyzed in a linear array.

    block_size : int
        Sizes of the blocks into which the data is divided.

    confidence : int
        Confidence level to select values for.

    n_bootstraps : int
        Number of times values should be resampled from the distribution with
        replacement.

    f : Callable
        Function to be applied to the resampled data.


    Returns
    -------

    Tuple[NDArray[np.float64], List[float]]
        The blocked_bootstrap values in an array after the function f has been
        applied to them and the confidence interval for the passed confidence
        level.
    """

    len_dat = len(data)
    n_blocks = len_dat // block_size

    blocks = np.array(np.array_split(data[: n_blocks * block_size], n_blocks))

    blocked_bootstrap = np.zeros(n_bootstraps)

    for resampling_idx in range(n_bootstraps):
        blocks_to_pick = np.random.randint(low=0, high=n_blocks, size=n_blocks)
        resample = blocks[blocks_to_pick].flatten()
        blocked_bootstrap[resampling_idx] = f(resample)

    confidence_intervals = [
        np.percentile(blocked_bootstrap, confidence / 2),
        np.percentile(blocked_bootstrap, 100 - (confidence / 2)),
    ]

    return blocked_bootstrap, confidence_intervals


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

    print_temp : bool
        Set to true to print the melting temperature after estimation. 
        If the estimated temperature is on of the extreme values i.e. it does
        does not lie in the temperature range passed, then the message printed
        cannot be suppressed.

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
        if print_temp:
            print(
                ref_name
                + " melting temperature: {}K".format(
                    np.round(temperatures[min_diff_idx], decimals=2)
                )
            )
        
    return min_diff_idx, np.round(temperatures[min_diff_idx], decimals=2)
