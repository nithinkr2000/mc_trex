import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Callable, Any
from numba import jit
from mc_trex.post_processing.fit_func import sigmoid_melting_curve
from math import comb, sqrt
from statistics import variance, mean
from itertools import combinations

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

@jit(nopython=True)
def jack_knife(
    data: NDArray[np.float64], n_blocks: int, func: str = 'mean'
) -> NDArray[np.float64]:
    """
    Jack-knife method for error estimation.

    Parameters
    ----------

    data : NDArray[np.float64]
        Contains the data to be analyzed in a linear array.

    n_blocks : int
        Number of blocks to be used in the calculation of the error.

    func : str
        Function to be applied to the dataset before application of the
        jack-knife method.

    Returns
    -------

    del_rho : float
        The error estimate from jack-knife.

    """

    len_dat = len(data)
    block_size = len_dat // n_blocks

    # Exclude any elements that cannot form a full block
    clipped_data = np.array(data[: n_blocks * block_size])
    match func:
        case 'mean':
            rho_bar = np.mean(clipped_data)
        case 'var':
            rho_bar = np.std(clipped_data)

    # To hold the sum under root in jack-knife error
    sum_diff_sq = 0

    for i in range(n_blocks):
        # Create the data set excluding the block
        block_m = np.append(
            clipped_data[: i * block_size], clipped_data[(i + 1) * block_size :]
        )

        # Apply function to the data set excluding the block
        match func:
            case 'mean':
                rho_m_bar = np. mean(block_m)
            case 'var':
                rho_m_bar = np.std(block_m)

        sum_diff_sq += np.power(rho_m_bar - rho_bar, 2)

    # Apply the full formula of jack-knife error
    del_rho = np.sqrt((n_blocks - 1) / n_blocks) * np.sqrt(sum_diff_sq)

    return del_rho


def blocked_bootstrap(
    data: NDArray[np.float64],
    block_size: int = 1,
    block_indices: List[int] | None = None,
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

    block_indices: List[int] | None
        The indices of block edges, including the beginning of the first block
        (0) and the ending of the last block (len(data)). If this variable is
        passed, then block_size array is ignored.

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
    if block_indices is None:
        n_blocks = len_dat // block_size
        blocks = np.array(np.array_split(data[: n_blocks * block_size], n_blocks))
    else:
        n_blocks = len(block_indices) - 1
        blocks = np.array(
            [
                data[idx1:idx2]
                for idx1, idx2 in zip(block_indices[:-1], block_indices[1:])
            ],
            dtype=object,
        )

    blocked_bootstrap = np.zeros(n_bootstraps)

    for resampling_idx in range(n_bootstraps):
        blocks_to_pick = np.random.randint(low=0, high=n_blocks, size=n_blocks)
        resample = np.concatenate(blocks[blocks_to_pick])
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
    print_temp: bool,
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
            "Lowest temperature where melting curves meet is {}. However,\
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


def autocorrelation(
    x: NDArray[np.float64], lag: int | None = None
) -> NDArray[np.float64] | float:
    """
    Function to calculate the autocorrelation of a timeseries. If the lag is
    passed, the the autocorrelation for a specific lag is returned. If not,
    then the autocorrelation for the all possible values of lag is returned.

    Parameters
    ----------

    x : NDArray[np.float64]
        The data for which the autocorrelation is to be calculated. Expected
        shape is a linear array containing timeseries data.

    lag : int | None
        The lag for which the autocorrelation is to be calculated. Provide to
        return a single value.

    Returns
    -------
    NDArray[np.float64] | float
        Returns the autocorrelation for all lag times or for a single value of
        the lag depending on whether the argument is provided.

    """

    x_ = np.subtract(x, np.mean(x))
    variance = np.var(x)

    ac = np.correlate(x_, x_, mode="full")
    ac = ac[len(ac) // 2 :] / variance

    if (lag is not None) and (lag < len(ac)):
        return ac[lag]

    else:
        return ac


def residuals(
    params: List[Any],
    T: List[float],
    fracs: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """
    Calculates the residuals for a constrained optimization of melting curve
    fit parameters. It is simply the mean squared error of the melting curves,
    but added for all melting curves as opposed to just one.

    Parameters
    ----------

    params : List[Any]
        The parameter values of the fit.

    T : List[float]
        The temperature values at which the input values of the melting curve
        are known.

    fracs : NDArray[np.float64]
        The fraction of different configurations at each temperature in T.

    weights : NDArray[np.float64]
        The weights to be assigned to the residuals before adding them to
        the data. If None, then equal weights are assigned to add points.

    Returns
    -------

    float
        Mean squared error between the melting curves and input values for all
        melting curves at all temperatures.

    """

    n_confs = fracs.shape[0]
    if weights is None:
        weights = [[1 for _ in range(len(fracs[i]))] for i in range(n_confs)]

    error = 0.0

    for i in range(n_confs):
        Tm = params[3 * i]
        dT = params[3 * i + 1]
        p = params[3 * i + 2]
        fi = sigmoid_melting_curve(T, Tm, dT, p)
        error += np.sum(np.multiply(weights, (fi - fracs[i]) ** 2))

    return error


def penalty(
    params: List[Any], T: List[float], n_confs: int, pen_weight: float = 1000
) -> float:
    """
    Function to calculate the penalty for the sum of melting curves
    exceeding 1.0
    Can be used as a NonlinearConstraint in the scipy.optimize.minimize
    function with the trust-constr option. Necessary step in the constrained
    minimization of the fit parameters for melting curves.

    Parameters
    ----------

    params : List[Any]
        The parameter values of the fit.

    T : List[float]
        The temperature values at which the input values of the melting curve
        are known.

    n_confs : int
        The number of configurations in the system, "unfolded" state included.

    pen_weight : float
        The weight of the penalty being imposed for deviation from total
        probability = 1.0

    Returns
    -------

    float
        The penalty for the sum of melting curves exceeding one.

    """
    f_sum = np.zeros_like(T)
    for i in range(n_confs):
        Tm = params[3 * i]
        dT = params[3 * i + 1]
        p = params[3 * i + 2]

        f_sum += sigmoid_melting_curve(T, Tm, dT, p)

    pen = pen_weight * (f_sum - 1.0)
    return pen


def block_dat(data: NDArray[Any], 
              block_size: int | None = None, 
              block_indices: List[int] | None = None
             ) -> NDArray[Any]:
    """
    Function to block data into chunks of size `block_size` or
    splits the data at the block indices passed.
    
    Parameters
    ----------

    data : NDArray[Any]
        The data to be blocked.

    block_size : int
        The size of the blocks into which the data is to be 
        split.

    block_indices : List[int]
        The indices of the positions where the data is to be 
        split.

    Returns
    -------
    
    NDArray[Any]
        Blocked data.


    It should be noted that either the block_size or the 
    block_indices must be provided.
    
    """

    len_dat = len(data)
    
    if (block_size is not None) and (block_size > 0):
        n_blocks = len_dat // block_size
        blocks = np.array(np.array_split(data[: n_blocks * block_size], n_blocks))
        
    else:
        n_blocks = len(block_indices) - 1
        blocks = np.array(
            [
                data[idx1:idx2]
                for idx1, idx2 in zip(block_indices[:-1], block_indices[1:])
            ],
            dtype=object,
        )
        
    return blocks


def leave_kblocks(blocks: List[NDArray[Any]], k: int, n_sets: int )-> List[NDArray[Any]]:
    """
    Generates new data sets from the blocks of data provided by
    excluding `k` blocks from all the blocks provided. The
    blocks must have the same shape along all but the last axis.

    Parameters
    ----------

    blocks :List[NDArray[Any]]
        Blocks of data from which the new data sets are to be
        generated. It is important to note that the blocks must
        have the same shape along all axes except for the last one.

    k : int
        The number of blocks to leave out.

    n_sets : int
        The number of data sets to be generated. The minimum of this
        and the total number of possible datasets that can be generated
        is chosen as the final number of sets.

    Returns
    -------

    List[NDArray[Any]]
        The new datasets generated by leave `k` out.

    """

    n_blocks = len(blocks)
    n_possible_sets = comb(n_blocks, k)

    if k < 0:
        raise ValueError("Number of blocks excluded cannot be negative.")

    datasets = []
    n_sets = int(np.min([n_sets, n_possible_sets]) )
    if n_sets < n_possible_sets:
        for set_idx in range(n_sets):
            block_idcs = np.random.randint(0, n_blocks, n_blocks - k, dtype=int)

            new_set = np.concatenate([blocks[block_idx] for block_idx in block_idcs], axis=-1)
            datasets.append(new_set)

    elif n_possible_sets <= n_sets:

        for combination in combinations(blocks, n_blocks - k):

            new_set = np.concatenate(combination, axis=-1)
            datasets.append(new_set)

    # Probabilities of configurations at different temperatures for each leave-k dataset
    dataset_fracs = []

    ## Check for values in the order unfolded, folded, misfolded
    for block_sfs in datasets:
        block_fracs = []
        for conf_idx in [-1, 0, 1]:
            block_fracs.append([np.sum(np.equal(temp_sf, conf_idx)) / len(temp_sf) for temp_sf in block_sfs])

        dataset_fracs.append(np.array(block_fracs))

    return dataset_fracs, np.shape(datasets)[-1]
