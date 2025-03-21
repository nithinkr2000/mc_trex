import numpy as np

from typing import List
from numpy.typing import NDArray


def sigmoid_melting_curve(
    T: List[float] | NDArray[np.float64] = [0, 1], Tm: float = 0, dT: float = 0.5
) -> NDArray[np.float64]:
    """
    Sigmoid function for fitting melting curves. Limits at 0 and 1.

    f(x) = 1 / (1 + exp((T - Tm) / dT))
    """

    return 1 / (1 + np.exp(np.subtract(T, Tm) / dT))


def gaussian(
    x: List[float] | NDArray[np.float64] = [0, 1],
    amplitude: float = 0.0,
    mean: float = 0.0,
    stddev: float = 0.5,
) -> NDArray[np.float64]:
    """
    Gaussian function

    f(x) = amplitude * exp((-(x - mean) / sqrt(2)*stddev)^2)
    """

    return amplitude * np.exp(
        -np.power(np.subtract(x, mean) / np.multiply(np.sqrt(2), stddev), 2)
    )


def exponential(
    x: List[float] | NDArray[np.float64] = [0.0, 1.0],
    offset: float = 0.0,
    coef: float = 0.0,
    scal: float = 0.0,
) -> NDArray[np.float64]:
    """
    Exponential function with an offset, scaling factor and coefficient.

    f(x) = offset + scale * exp(coef * x)
    """

    return np.add(offset, np.multiply(scal, np.exp(np.multiply(coef, x))))


def sigmoid_for_dist(
    x: List[float] | NDArray[np.float64] = [0.0, 1.0],
    cut_off: float = 4.0,
    coeff: float = 18.42,
) -> NDArray[np.float64]:
    """
    Calculate the score for distance-cut-off. For the score to go to within exp(-a)
    of the limit (1/0) within a distance difference of d Ang, the coeff should be
    k = 2.3 a/d

    Default is a = 4, d = 0.5

    The cut-off is slightly shifted to allow for the point close to 4.0 to be
    counted as a contact as well. Hence the cut is chosen to be
    cut_off + np.exp(-a) = 0.02

    Parameters
    ----------

    x:List[float]
        The input values of distance.

    cut_off:float
        Actual cut-off to be used for the contact analyses.

    coeff:float
        Coefficient that determines how fast the score falls to 0 or 1 around the cut-off.

    Returns
    -------

    List[float]
        List of floating point values in the range [0, 1] corresponding to the score of each distance passed to the function.

    """

    return 1 / (1 + np.exp(coeff * np.subtract(x, (cut_off + 0.02))))


def sigmoid_for_ang(
    ang: List[float] = [0.0, 1.57],
    up_ang_cut: float = 90.0,
    lo_ang_cut: float = 0.0,
    coeff: float = -1.99,
) -> NDArray[np.float64]:
    """
    Works much like the distance soft-cut, except there are two now, with an upper and lower cut.

    Parameters
    ----------

    ang:List[float]
        Angles for which the score is to be determined (=1 - score within range, =0 - score out of range)

    up_ang_cut:float
        Upper bound of the angle for some analysis.

    lo_ang_cut:float
        Lower bound of the angle for some analysis.

    Returns
    -------

    List[float]
        List of floating point values in the range [0, 1] corresponding to the score of each angle passed to the function.

    """

    sup = 1 / (1 + np.exp(coeff * np.subtract(ang, (lo_ang_cut - 1))))
    sdown = 1 / (1 + np.exp(coeff * np.subtract((up_ang_cut + 1), ang)))

    return (sup + sdown) / 2
