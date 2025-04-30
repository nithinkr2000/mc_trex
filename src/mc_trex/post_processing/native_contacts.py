import numpy as np

from tqdm.auto import tqdm

from typing import Any, List, Literal, Tuple
from numpy.typing import NDArray
from enum import Enum

from Bio import BiopythonDeprecationWarning, BiopythonWarning
import warnings

with warnings.catch_warnings():
    warnings.simplefilter(category=BiopythonDeprecationWarning, action="ignore")
    warnings.simplefilter(category=BiopythonWarning, action="ignore")

    from MDAnalysis import Universe  # type: ignore
    from MDAnalysis.analysis import contacts, distances  # type: ignore


class Method(Enum):
    soft = "soft"
    hard = "hard"


def get_frac_natcons(
    sims: List[List[Universe] | NDArray[Universe]] | None = None,
    ref: Universe = None,
    cut_off: float = 3.0,
    ids: List[int] | None = None,
    atom_selection: str = "nucleic",
    nonoh: bool = True,
    skip_neighs: int = 1,
    method: Literal["soft", "hard"] = "soft",
    start: int = 0,
    stop: int = -1,
    step: int = 1,
    verbose: bool = False,
    frames: List[int] | NDArray[np.int64] | None = None,
) -> List[List[contacts.Contacts]]:
    """
    Function to determine the fraction of native contacts distribution of
    the passed trajectory with respect to the passed reference structure.
    Expects MDAnalysis trajectories.

    Parameters
    ----------

    sims : List[List[Universe] | NDArray[Universe]]
        MDAnalysis Universe class objects containing the trajectories to be
        analyzed.

    ref : Universe
        An MDAnalysis Universe class object containing the reference frame
        to be used to generative the native contacts reference.

    cut_off : float
        The cut off distance to be used in determining the fraction of native
        contacts.
    
    ids : List[int] | None
        The list of indices of the residues to consider in the native contacts 
        analysis. If not provided, all residues are considered.

    atom_selection : str
        The atom selection string to be used when determining the distance 
        matrix with respect to which the contacts will be determined. 
        Essentially, only contacts between these atoms will be considered
        in the contacts analysis.
    
    nonoh : bool
        Set to False to exclude hydrogen atoms from contact analyses.
    
    skip_neighs : int
        Number of neighbours to skip in determining the reference contacts.

    method : Literal["soft", "hard"]
        The type of cut-off to be used in determining the fraction of native
        contacts.

    start : int
        Start frame for the analysis

    stop : int
        Stop frame for the analysis

    step : int
        Number of frames to skip for each step from start to stop.

    verbose : bool
        Verbosity.

    frames : List[int] | NDArray[np.int64] | None
        If passed, used in place of start, stop and step.
        For now, it is expected to be less than the shortest trajectory passed
        if multiple trajectories are passed.

    Returns
    -------

    all_native_contacts: List[List[contacts.Contacts]]
        Native contacts as an array.

    """

    # Check if passed method is soft or hard
    if isinstance(method, str):
        Method(method)

    # Select contacts to include from the reference based on
    # cut-off and number of neighbours to skip
    sel_cons = []
    
    if (ids is None) or (len(ids) < 2):
        ids = np.unique(ref.residues.resids)

    for idx1, id1 in enumerate(
        tqdm(
            ids,
            colour="green",
            desc="Setting references and selections based on cut-off",
        )
    ):
        for id2 in ids[idx1 + skip_neighs + 1 :]:
            temp = distances.distance_array(
                ref.select_atoms(
                    "resid "
                    + str(id1)
                    + atom_selection
                ),
                ref.select_atoms(
                    "resid "
                    + str(id2)
                    + atom_selection
                ),
            )
            idcs = np.argwhere(temp < cut_off).tolist()
            for row, col in idcs:
                c1 = "id " + str(ref.select_atoms("resid " + str(id1)).ids[row])
                c2 = " id " + str(ref.select_atoms("resid " + str(id2)).ids[col])
                sel_cons.append((c1, c2))

    sel_contacts = [
        contact
        for contact in sel_cons
        if ((ref.select_atoms(contact[0]).elements[0] != "H"
        and ref.select_atoms(contact[1]).elements[0] != "H")
        or nonoh)
        ]
    ref_contacts = [
        (ref.atoms.select_atoms(i), ref.atoms.select_atoms(j))
        for (i, j) in sel_contacts
    ]

    all_native_contacts = []
    
    # If frames is not None and actually contains an array-like 
    # data-structure with integers, then start, stop and step 
    # are set to None
    min_sim_len = np.min([sim.trajectory.n_frames for sim in sims])
    
    if (frames and 
        all([int(frameidx)==frameidx for frameidx in frames]) and 
        np.all(np.logical_and(np.greater(frames, 0), np.less(frames, min_sim_len)))):
        start = None                
        stop = None
        step = None
    
    else:
        frames = None

    # Using the selection, analyze all the frames of the simulation
    for sim in tqdm(sims, desc="Processing native contacts"):
        native_contacts = []
        for csel, cref in zip(sel_contacts, ref_contacts):
            native_contacts.append(
                contacts.Contacts(
                    sim,
                    select=(csel[0], csel[1]),
                    refgroup=(cref[0], cref[1]),
                    radius=cut_off,
                    method=method + "_cut",
                ).run(start=start, 
                    stop=stop, 
                    step=step, 
                    verbose=verbose, 
                    frames=frames)
            )
        all_native_contacts.append(native_contacts)

    return all_native_contacts


def post_process_natcons(nat_cons: List[contacts.Contacts]) -> Tuple[Any, Any]:
    """
    Calculates the fraction of native contacts from the True-False native contact
    arrays. It gives a time series of length n_frames.

    Parameters
    ----------

    nat_cons : List[contacts.Contacts]
        MDAnalysis.analysis.contacts.Contacts objects for each native contact
        identified in the reference structure.

    """

    yss = []
    for i, nc in enumerate(nat_cons):
        xs, ys = nc.results.timeseries.T
        yss.append(ys)

    try:
        return (xs, np.mean(yss, axis=0))
        
    except (TypeError, ValueError) as ex:
        print(f"An error occurred: {ex}")
        print("Check data format - must be list of MDAnalysis.contacts.Contacts.")
        raise ValueError("Placeholder for error") from ex
