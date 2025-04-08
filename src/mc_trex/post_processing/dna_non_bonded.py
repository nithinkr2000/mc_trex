import numpy as np

from typing import List, Literal, Tuple
from numpy.typing import NDArray

import pandas as pd
from string import capwords

import re

from enum import Enum

from Bio import BiopythonDeprecationWarning, BiopythonWarning
import warnings

with warnings.catch_warnings():
    warnings.simplefilter(category=BiopythonDeprecationWarning, action="ignore")
    warnings.simplefilter(category=BiopythonWarning, action="ignore")
    from MDAnalysis import Universe  # type: ignore
    from MDAnalysis.topology.tables import TABLE_VDWRADII as vdwradii  # type: ignore

from mc_trex.post_processing.fit_func import sigmoid_for_dist, sigmoid_for_ang


class Method(Enum):
    soft = "soft"
    hard = "hard"


species = []
rvdw_vals = []
for line in vdwradii.split(sep="\n"):
    if line.startswith("#") or len(line) == 0:
        continue
    else:
        words = line.split()
        species.append(words[0])
        rvdw_vals.append(float(words[1]))
df_vdwradii = pd.DataFrame(
    {
        "Element": [capwords(i) for i in species],
        "Van der Waals radius": rvdw_vals,
    }
)


def check_distance(
    vec1: NDArray[np.float64] = np.zeros(1),
    vec2: NDArray[np.float64] = np.zeros(1),
    cut_off: float = 3.0,
    method: Literal["soft", "hard"] = "soft",
) -> float | np.bool_:
    """
    This functions checks if the distance between points denoted by vec1 and vec2
    is less than cut_off. The method defines whether to use a hard-cut or a
    soft-cut. The latter becomes useful for dealing with dynamic structures such
    as base pairs, which fluctuate in and out of bonding.

    A soft-cut is performed by calculating a score based on absolute distance
    using a sigmoid function.

    Parameters
    ----------

    vec1, vec2 : NDArray[np.float64]
        Position vectors of the points being checked

    cut_off : float
        Cut-off distance for checking non-bonded interactions

    method : Literal["soft", "hard"]
        Whether to perform a hard cut or a soft cut

    Returns
    -------
    float | bool
        floating point score for soft cut-offs
        1|0 for hard cut-offs

    """

    # Check if passed method is soft or hard
    if isinstance(method, str):
        Method(method)

    d = np.linalg.norm(np.subtract(vec1, vec2))

    if not (cut_off > 0.0):
        print("The cut-off must be greater than 0!")
        return 0

    if method.strip().casefold() == "hard":
        return d < cut_off

    else:
        return sigmoid_for_dist(d, cut_off)


def check_angle(
    vec1: NDArray[np.float64],
    vec2: NDArray[np.float64],
    up_ang_cut: float = 30,
    down_ang_cut: float = 150,
    method: Literal["soft", "hard"] = "soft",
) -> float | bool:
    """
    Cut-off angles in degrees can be passed along with the two vectors.
    Returns True or False depending on whether the angle between the vectors
    lies between the cut-off values.

    Parameters
    ----------

    vec1, vec2 : NDArray[np.float64]
        Position vectors of the points being checked

    up_ang_cut, down_ang_cut : float
        Cut-off angles (upper limit and lower limit) for checking non-bonded
        interactions.

    method : Literal["soft", "hard"]
        Whether to perform a hard cut or a soft cut.

    Returns
    -------
    float | bool
        floating point score for soft cut-offs.
        1|0 for hard cut-offs.

    """
    # Check if passed method is soft or hard
    if not isinstance(method, Method):
        raise ValueError("Not a valid method.")

    ang = np.rad2deg(
        np.arccos(
            np.absolute(
                np.clip(
                    np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    if method.strip().casefold() == "hard":
        if (ang > down_ang_cut) and (ang < up_ang_cut):
            return True
        else:
            return False

    else:
        return sigmoid_for_ang(ang, up_ang_cut, down_ang_cut)


def get_ids_and_masses(u: Universe) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Define the atom names in the purine and pyrimidine six-member rings that are
    involved in stacking. Generate resids, which are unique and write out the
    corresponding atom selection for the atoms.

    The five-member rings in purines are not included and their contribution is
    assumed to be small compared to the contribution of the six-member rings.


    Parameters
    ----------

    u : Universe
        Universe from which the atoms are to be selected or a corresponding
        reference structure (should have the same topology).

    Returns
    -------
    ids : List[float]
        Contains the IDs of the atoms in the rings.

    masses : List[float]
        Contains the masses of the atoms in the rings.

    """

    # Non-hydrogen atoms in the purine (A, G) and pyrimidine (C, U, T) rings
    ref_ring_atoms = {
        "DC": [["N1", "C2", "N3", "C4", "C5", "C6"]],
        "DU": [["N1", "C2", "N3", "C4", "C5", "C6"]],
        "DT": [["N1", "C2", "N3", "C4", "C5", "C6"]],
        "DG": [["N1", "C2", "N3", "C4", "C5", "C6"], ["C4", "C5", "N7", "C8", "N9"]],
        "DA": [["N1", "C2", "N3", "C4", "C5", "C6"], ["C4", "C5", "N7", "C8", "N9"]],
    }

    ids = []
    masses = []

    for resid in ["resid " + str(resid) for resid in u.residues.resids]:
        temp = u.select_atoms(resid)
        atoms = [
            temp.select_atoms("name " + " or name ".join(ring))
            for ring in ref_ring_atoms[re.sub(r"[0-9]+", "", temp.resnames[0])]
        ]

        ids.append([np.subtract(atom_set.ids, 1) for atom_set in atoms])
        masses.append([atom_set.masses for atom_set in atoms])

    return ids, masses


def get_com_normal(
    coords: NDArray[np.float64], masses: List[float]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Given coordinates and masses of a set of atoms, calculate their center of mass.
    Assuming they all lie in one plane, the normal to that plane is also calculated
    using the coordinates of the first three atoms.

    Parameters
    ----------

    coords : NDArray[np.float64]
        Coordinates of the atoms for which the center of mass and normal are to be
        calculated.

    masses : List[float]
        Masses of the atoms for which the coordinates were passed (must be in the
        same order).

    Returns
    -------

    Tuple[NDArray[np.float64], NDArray[np.float64]]
        Center of mass coordinates and normal to the plane.

    """
    com = np.divide(
        np.sum(coords * np.reshape(masses, (len(masses), 1)), axis=0), np.sum(masses)
    )
    r12 = np.subtract(coords[0], coords[1])
    r32 = np.subtract(coords[2], coords[1])
    normal = np.cross(r12, r32)

    if np.linalg.norm(normal) < 1e-8:
        raise ValueError("Degenerate or very small normal vector.")

    normal = normal / np.linalg.norm(normal)

    return com, normal


def pi_stacking(
    frames: NDArray[np.float64],
    ids: List[List[int]],
    masses: List[List[List[float]]],
    method: Literal["soft", "hard"] = "soft",
) -> List[List[List[float]]]:
    """
    Checking for pi-stacking in the passed trajectories. The selection of the
    atoms in the trajectories is done with IDs passed to the function.

    Parameters
    ----------

    frames : NDArray[np.float64]
        Contains the trajectories in the format
        n_frames x n_atoms x 3

    ids : List[List[int]]
        IDs of the atoms involved in stacking (srings).

    masses : List[List[List[float]]]
        Masses of the atoms in the rings (used to calculate the
        center of mass position of the rings)

    Returns
    -------

    pi_stack_mats : List[List[List[float]]]
        For a particular frame and residue (1) temp contains information for
        residue 1 with respect to the other residues.

        pi_stack_status is the upper triangle of the symmetric matrix that says
        whether two residues are stacked (N_residues x N_residues). This is for
        a single frame.

        pi_stack_mat contains the pi_stack_status matrices for all
        the frames.

        The ordering of the rings is (6, 5). This means for contacts between
        A and B, one has the following ordering of stacking interactions returned

        1. A6-B6, A6-B5, A5-B6, A5-B5   if both A and B are purines
        2. A6-B6, A6-B5                 if A is a pyrimidine and B is a purine
        3. A6-B6, A5-B6                 if A is a purine and B is a pyrimidine
        4. A6-B6                        if A and B are pyrimidines

    Checks for pi-stacking, taken from get contacts
    (parameters may vary for DNA)


    Check 1 - distance between centers of rings less than 7 Ang
    Check 2 - normals to the rings are at an angle less than 30 deg
    Check 3 - vector between COMs and normal 1 are at an angle less than 45 deg
    Check 4 - vector between COMs and normal 2 are at an angle less than 45 deg


    Example
    -------

    myu = Universe(top_path, traj_path)
    ids, ms = get_ids_and_massses(myu)
    pi_stack_mats = pi_stacking(myu.trajectory.timeseries(), ids, ms)

    """

    pi_stack_mats = []

    for frame in frames:
        pi_stack_status = []

        for idx1, (masses1, resids1) in enumerate(zip(masses[:-1], ids[:-1])):
            temp = []
            coms1 = []
            norms1 = []

            for resids_, masses_ in zip(resids1, masses1):
                com1, norm1 = get_com_normal(frame[resids_], masses_)

                coms1.append(com1)
                norms1.append(norm1)

            for idx2, (masses2, resids2) in enumerate(
                zip(masses[idx1 + 1 :], ids[idx1 + 1 :])
            ):
                coms2 = []
                norms2 = []

                for resids_, masses_ in zip(resids2, masses2):
                    com2, norm2 = get_com_normal(frame[resids_], masses_)

                    coms2.append(com2)
                    norms2.append(norm2)

                rel_coms = [np.subtract(com1, com2) for com1, com2 in zip(coms1, coms2)]

                for rel_com, com1, com2 in zip(rel_coms, coms1, coms2):
                    check1 = check_distance(com1, com2, cut_off=7.0, method=method)
                    check2 = check_angle(norm1, norm2, 30, 0, method=method)
                    check3 = check_angle(norm1, rel_com, 45, 0, method=method)
                    check4 = check_angle(norm2, rel_com, 45, 0, method=method)

                    if method == "hard":
                        if check1:
                            if check2 and check3 and check4:
                                temp.append(1.0)
                            else:
                                temp.append(0.0)
                        else:
                            temp.append(0.0)

                    else:
                        temp.append(np.mean([check1, check2, check3, check4]))

            pi_stack_status.append(temp)

        pi_stack_mats.append(pi_stack_status)

    return pi_stack_mats


def t_stacking(
    frames: NDArray[np.float64],
    ids: List[List[int]],
    masses: List[List[List[float]]],
    method: Literal["soft", "hard"] = "soft",
) -> List[List[List[float]]]:
    """
    Check for pi-stacking in the passed trajectories. The selection of the atoms
    in the trajectories is done with IDs passed to the function.

    Parameters
    ----------

    frames : NDArray[np.float64]
        Contains the trajectories in the format
        n_frames x n_atoms x 3

    ids : List[List[int]]
        IDs of the atoms involved in stacking (srings).

    masses : List[List[float]]
        Masses of the atoms in the rings (used to calculate the
        center of mass position of the rings)

    Returns
    -------
    t_stack_mats : List[List[List[float]]]
        For a particular frame and residue (1) temp contains information for residue 1
        with respect to the other residues.

        t_stack_status is the upper triangle of the symmetric matrix that says whether
        two residues are stacked (N_residues x N_residues). This is for a single frame.

        t_stack_mats contains the t_stack_status matrices for all the frames.

    The ordering of the rings is (6, 5). This means for contacts between
    A and B, one has the following ordering of stacking interactions returned

        1. A6-B6, A6-B5, A5-B6, A5-B5   if both A and B are purines
        2. A6-B6, A6-B5                 if A is a pyrimidine and B is a purine
        3. A6-B6, A5-B6                 if A is a purine and B is a pyrimidine
        4. A6-B6                        if A and B are pyrimidines

    Checks for t-stacking, taken from get contacts
    (parameters may vary for DNA)

    Check 1 - distance between centers of rings less than 5 Ang
    Check 2 - normals to the rings are at an angle between 60 deg and 90 deg
    Check 3 - vector between COMs and normal 1 are at an angle less than 45 deg
    Check 4 - vector between COMs and normal 2 are at an angle less than 45 deg

    Example
    -------

    myu = Universe(top_path, traj_path)
    ids, ms = get_ids_and_massses(myu)
    t_stack_mats = t_stacking(myu.trajectory.timeseries(), ids, ms)

    """

    t_stack_mats = []

    for frame in frames:
        t_stack_status = []

        for idx1, (masses1, resids1) in enumerate(zip(masses[:-1], ids[:-1])):
            temp = []
            coms1 = []
            norms1 = []

            for resids_, masses_ in zip(resids1, masses1):
                com1, norm1 = get_com_normal(frame[resids_], masses_)

                coms1.append(com1)
                norms1.append(norm1)

            for idx2, (masses2, resids2) in enumerate(
                zip(masses[idx1 + 1 :], ids[idx1 + 1 :])
            ):
                coms2 = []
                norms2 = []

                for resids_, masses_ in zip(resids2, masses2):
                    com2, norm2 = get_com_normal(frame[resids_], masses_)

                    coms2.append(com2)
                    norms2.append(norm2)

                rel_coms = [np.subtract(com1, com2) for com1, com2 in zip(coms1, coms2)]

                for rel_com, com1, com2 in zip(rel_coms, coms1, coms2):
                    check1 = check_distance(com1, com2, cut_off=5.0, method=method)
                    check2 = check_angle(norm1, norm2, 90, 60, method=method)
                    check3 = check_angle(norm1, rel_com, 45, 0, method=method)
                    check4 = check_angle(norm2, rel_com, 45, 0, method=method)

                    if method == "hard":
                        if check1:
                            if check2 and check3 and check4:
                                temp.append(1.0)
                            else:
                                temp.append(0.0)
                        else:
                            temp.append(0.0)

                    else:
                        temp.append(np.mean([check1, check2, check3, check4]))

            t_stack_status.append(temp)

        t_stack_mats.append(t_stack_status)

    return t_stack_mats


def van_der_waals(
    u: Universe,
    rvdw_mat: NDArray[np.float64] | None = None,
    ids: List[int] | None = None,
) -> NDArray[np.bool_]:
    """
    Check for Van der Waals forces in the molecule.

    Parameters
    ----------

    u : Universe
        The universe containing the molecule to be analyzed.

    rvdw_mat : NDArray[np.float64]
        Contains the van der Waals radii for pairs of elements for easy look-up.

    ids : List[int]
        Atom IDs of the atoms to be included in the analysis, if necessary.

    Returns
    -------

    NDArray[np.float64]
        Matrix of shape n_frames x n_atoms x n_atoms that contains whether or not a
        pair of atoms for a van der Waals bond for each frame.

    Criterion used to determine whether there is a van der Waals force or not is

    (distance between atoms) < (sum of their van der Waals radii + 5 Ang)

    """

    rvdw_arr = np.array(
        [
            df_vdwradii[df_vdwradii["Element"] == element][
                "Van der Waals radius"
            ].values[0]
            for element in u.atoms.elements
        ]
    )
    if not rvdw_mat:
        rvdw_mat = rvdw_arr[:, np.newaxis] + rvdw_arr + 0.5
    if ids:
        D = np.linalg.norm(
            np.subtract(
                u.trajectory.timeseries()[:, ids, :][:, :, np.newaxis],
                u.trajectory.timeseries()[:, ids, :][:, np.newaxis, :],
            ),
            axis=3,
        )

    return np.less(D, rvdw_mat)
