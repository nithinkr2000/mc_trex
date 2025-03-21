import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

from typing import List, Any, Tuple
from numpy.typing import NDArray

from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from MDAnalysis import Universe
    from MDAnalysis.analysis import align
    from MDAnalysisTests.datafiles import COORDINATES_TOPOLOGY, COORDINATES_XYZ

from mc_trex.post_processing import native_contacts


class Method(Enum):
    """Enumeration of available methods in trajectory classification."""

    rmsd = "rmsd"
    native = "native"


class TrajectoryLoader:
    """
    Load trajectories and references using MDAnalysis.

    Attributes
    ----------

    traj_loc : List[str]
        Paths to the trajectories to be analyzed.

    top : str
        Topology for the trajectory to be analyzed.

    top_ref : str
        Topology for the reference structures.

    ref_loc : List[str]
        Paths to the reference structures with respect to which the trajectory
        will be sorted.


    Methods
    -------

    load_trajectory
        Loads trajectory.

    load_reference
        Loads reference structures
    """

    def __init__(
        self,
        traj_loc: List[str] | None = None,
        top: str = COORDINATES_TOPOLOGY,
        ref_loc: List[str] | None = None,
        top_ref: str | None = None,
    ):
        """Initialize trajectory loader."""

        self.traj_loc = traj_loc if traj_loc is not None else [COORDINATES_XYZ]
        self.top = top
        self.ref_loc = ref_loc if ref_loc is not None else [COORDINATES_XYZ]
        self.top_ref = top_ref if top_ref is not None else top

    def load_trajectory(self, **kwargs) -> Universe:
        """
        Wrapper for loading trajectory.

        Parameters
        ----------

        **kwargs
            Arguments for MDAnalysis.Universe. See module docstring for list of
            all possible arguments.

        Returns
        -------

        MDAnalysis.Universe
            Universe containing trajectory to be analyzed.

        """

        return Universe(self.top, self.traj_loc, **kwargs)

    def load_reference(self, **kwargs) -> List[Universe]:
        """
        Wrapper for loading reference structures.

        Parameters
        ----------

        **kwargs
            Arguments for MDAnalysis.Universe. See module docstring for list of
            all possible arguments.

        Returns
        -------

        List[MDAnalysis.Universe]
            List of Universe class instances containing reference trajectories.
        """

        return [Universe(self.top_ref, ref, **kwargs) for ref in self.ref_loc]


class TrajectoryClassifier(TrajectoryLoader, ABC):
    """
    Base class for classifying and sorting trajectory frames based on reference
    structures and determining fractions of each configuration in the
    trajectory.

    Attributes
    ----------

    get_which : int
        The peak to return. For RMSD, the lowest peak corresponds to the
        closest structure. For fraction of native contacts, the highest
        peak corresponds to the closest structure.
        Set to 0 for lowest and 1 for highest.

    Methods
    -------

    get_similarity_metric
        Calculates similarity metrics between trajectory and each reference
        structure passed.

    sort_frames
        Determine cut-off using Gaussian mixture models and sort through frames
        using these cut-offs.
    """

    @property
    @abstractmethod
    def get_which(self):
        """Set get_which depending on the measure."""
        pass

    @abstractmethod
    def get_similarity_metric(
        self,
        traj: Any,
        ref_trajs: List[Any],
        start: int = 0,
        stop: int = -1,
        step: int = 1,
        verbose: bool = False,
        **kwargs,
    ) -> List[Any]:
        """
        Calculate the similarity between passed reference structure(s) and
        trajectory.
        """
        pass

    def bin_frames(
        self, similarity_metric: List[NDArray[np.float64]], **kwargs
    ) -> List[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Perform binning of similarity metric to pass to fit to a Gaussian mixture
        model.

        Parameters
        ----------

        similarity_metric :  List[NDArray[np.float64]]
            Similarity metric between reference structure and trajectory. The
            passed shape can be expected to be n_ref_structs x n_frames

        **kwargs
            Extra key word arguments for numpy.histogram.
            Check module docstring for all possible arguments.

        Note, binning in numpy can be done either by explicitly specifying
        the number of bins or by selecting one of the following methods:

        auto, fd, doane, scott, stone, rice, sturges, sqrt,

        """

        return [np.histogram(similarity, **kwargs) for similarity in similarity_metric]

    def get_cuts(
        self,
        binned_metrics: List[Tuple[NDArray[np.float64], NDArray[np.float64]]],
        n_components: int = 2,
        max_components: int | None = None,
        return_models: bool = False,
        height: Tuple[float, float] | float = 100,
        **kwargs,
    ) -> Tuple[List[List[Any | None]], List[Any]] | List[List[Any | None]]:
        """
        Generate cut off values (lowest cut off) for trajectory with respect to
        each reference structure using Gaussian mixture models.

        Use cut off values to calculate the fraction of time spent in each
        configuration.

        Parameters
        ----------

        binned_metrics : List[NDArray[np.float64]]
            Measures of similarity between trajectory and reference structures.
            Shape n_ref_strucs x n_frames

        n_components : int
            Number of components (Gaussian functions) to be fit to the data.

        max_components : int
            Maximum number of components to check for. If passed, n_components
            is ignored and the function searches for the component number that
            minimizes the BIC.

        return_models : bool
            Return Gaussian mixture or not.

        height : Tuple[float, float] | float
            The peaks heights for the scipy.signal.find_peaks to use.
            One value is interpreted as the minimum and two values are
            interpreted as the minimum and maximum allowed values of the
            peaks allowed.
            Can be used to specify the minimum population a configuration must
            have before it can be considered for further analyses.

        **kwargs
            Extra key word arguments for sklearn.mixture.GaussianMixtureModel.
            Check module docstring for all possible arguments.

        Returns
        -------

        closest_cuts : Tuple[NDArray[np.float64], List[GaussianMixture]]:
            Contains the cut-offs for the trajectory with respect to all the
            reference structures and optionally, the model fit to the histogram.

        """
        if self.get_which not in [0, 1]:
            raise ValueError("Not acceptable value for get_which.")

        all_cuts = []
        all_models = []

        for bin_pops, bin_edges in binned_metrics:
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
            bin_count_dat = np.concatenate(
                [
                    np.repeat(bincenters, binpops)
                    for bincenters, binpops in zip(bin_centers, bin_pops)
                ]
            )

            if max_components:
                n_components_list = np.arange(1, max_components + 1)
                bic_scores = []
                fits = []

                for component_count in n_components_list:
                    gmm = GaussianMixture(
                        n_components=component_count, covariance_type="full"
                    )
                    gmm.fit(bin_count_dat.reshape(-1, 1))

                    fits.append(gmm)
                    bic_scores.append(gmm.bic(bin_count_dat.reshape(-1, 1)))

                n_components = n_components_list[np.argmin(bic_scores)]
                gmm = fits[np.argmin(bic_scores)]

            else:
                gmm = GaussianMixture(n_components=n_components, **kwargs)
                gmm.fit(bin_count_dat.reshape(-1, 1))

            ## Determine the minima from the best fit for the Gaussian Mixture Model

            newbins = np.linspace(
                np.min(bin_count_dat),
                np.max(bin_count_dat),
                (len(bin_count_dat) // 1000 + 1) * 1000,
            )

            # Determine probability distribution function for each point
            logpdf = gmm.score_samples(newbins.reshape(-1, 1))
            pdf = np.exp(logpdf)

            # Find gradient of the PDF
            grad_pdf = np.gradient(pdf, newbins)

            # Find minima by checking where the gradient goes from 0/negative to positive
            min_bin_in = np.where(np.diff(np.sign(grad_pdf)) > 0)[0] + 1

            # Get these points on the measure axis
            minima_measures = newbins[min_bin_in]

            ## Determine peaks to select minima immediately after or immediately before them

            # Calculate the envelope for the histogram to find the peaks
            bin_width = bin_centers[1] - bin_centers[0]
            pdf_per_component = gmm.predict_proba(newbins.reshape(-1, 1))
            profile = (
                np.sum(pdf_per_component, axis=1) * pdf * np.sum(bin_pops) * bin_width
            )

            # Find the peaks
            peak_idcs = find_peaks(profile, height=height)

            # Select the minima after each identified peak
            cuts = (
                [
                    minima_measures[minima_measures < newbins[idx]][0]
                    if np.any(minima_measures > newbins[idx])
                    else None
                    for idx in peak_idcs[0]
                ]
                if self.get_which
                else [
                    minima_measures[minima_measures > newbins[idx]][0]
                    if np.any(minima_measures > newbins[idx])
                    else None
                    for idx in peak_idcs[0]
                ]
            )

            all_cuts.append(cuts)
            all_models.append(gmm)

        return_vals = (all_cuts, all_models) if return_models else all_cuts

        return return_vals

    def sort_frames(
        self,
        similarity_metric: List[NDArray[np.float64]],
        cut_offs: List[float],
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        Using cut-offs, sort frames into different configurations, The sorting
        is done as follows

        1. For each frame, check similarity metric with respect to all
           reference structures
        2. If only one similarity metric is less/greater than the corresponding
           cut-off, then the configuration corresponds that reference structure.
        3. If multiple similarity metrics are less than their corresponding
           cut-offs, then, choose the one closest to the ideal value.
        4. If several references have the same metric value with respect to a
           structure the references were not selected properly. In this case,
           the frame will be marked for manual review.

        Parameters
        ----------

        similarity_metric : List[NDArray[np.float64]]
            Measure of similarity between reference structure and trajectory. The
            passed shape can be expected to be n_ref_structs x n_frames

        cut_offs : List[float]

        Returns
        -------

        frames : List[int]
            List of frame indices

        configurations : List[List[int]]
            Configurations of the systems. For numbers >= 0, the index of the
            reference structure is given.
            If the value is -1, it did not match any of the passed structures.
        """

        if similarity_metric is None or len(similarity_metric) == 0:
            raise ValueError("similarity_metric cannot be empty.")

        if self.get_which == 0:
            # Mask based on where the measure is less than the cut-off
            # Select minimum for each frame (column) after masking
            min_metrics = np.ma.masked_greater(similarity_metric, cut_offs).min(axis=0)
            min_idcs_frame_idcs = np.where(np.equal(similarity_metric, min_metrics))

            # Create an array of -1s of the length of the frame count
            n_frames = np.shape(similarity_metric)[1]
            configurations = np.full(n_frames, -1)

            # Fill result for frames that were identified
            configurations[min_idcs_frame_idcs[1]] = min_idcs_frame_idcs[0]

            # If you need the original frames array sorted
            frames = np.arange(n_frames)

        elif self.get_which == 1:
            # Mask based on where the measure is greater than the cut-off
            # Select maximum for each frame (column) after masking
            max_metrics = np.ma.masked_less(similarity_metric, cut_offs).max(axis=0)
            max_idcs_frame_idcs = np.where(np.equal(similarity_metric, max_metrics))

            # Create an array of -1s of the length of the frame count
            n_frames = np.shape(similarity_metric)[1]
            configurations = np.full(n_frames, -1)

            # Fill result for frames that were identified
            configurations[max_idcs_frame_idcs[1]] = max_idcs_frame_idcs[0]

            # If you need the original frames array sorted
            frames = np.arange(n_frames)

        else:
            raise ValueError("Invalid choice of get_which.")

        return frames, configurations


class RMSDAnalysis(TrajectoryClassifier):
    """
    RMSD based sorting of trajectory frames.
    """

    def get_similarity_metric(
        self,
        traj: Universe,
        ref_trajs: List[Universe],
        start: int = 0,
        stop: int = -1,
        step: int = 1,
        verbose: bool = False,
        **kwargs,
    ) -> List[NDArray[np.float64]]:
        """
        Calculates RMSD of the passed trajectories with respect to the
        reference structures. Generates list of RMSDs of size n_ref x n_frames
        where n_ref is the number of references and n_frames is the number of
        frames in the trajectory.

        Parameters
        ----------

        traj : Universe
            Trajectory to be analyzed.

        ref_trajs : List[Universe]
            List of reference trajectories to be analyzed.

        start : int
            Starting frame of analysis

        stop : int
            Stop frame of analysis

        step : int
            Number of frames to skip per step

        verbose : bool
            Turn on verbosity.

        **kwargs
            Extra key word arguments for MDAnalysis.align.AlignTraj. Check
            module docstring
        **kwargs
            Extra key word arguments for MDAnalysis.align.AlignTraj. Check
            module docstring for all possible arguments.


        Returns
        -------

        List[NDArray[np.float64]]
        List[float]
            List of RMSD values of the trajectory with respect to each
            reference.
        """

        all_rmsds = []

        for ref_traj in ref_trajs:
            aligner = align.AlignTraj(traj, ref_traj, **kwargs).run(
                start=start, stop=stop, step=step, verbose=verbose
            )
            all_rmsds.append(aligner.rmsd)

        return all_rmsds

    @property
    def get_which(self):
        """Extract the first peak."""
        return 0


class NativeContactAnalysis(TrajectoryClassifier):
    """
    Fraction of native contacts based sorting of trajectory frames.
    """

    def get_similarity_metric(
        self,
        traj: Universe,
        ref_trajs: List[Universe],
        start: int = 0,
        stop: int = -1,
        step: int = 1,
        verbose: bool = False,
        **kwargs,
    ) -> List[NDArray[np.float64]]:
        """
        Calculates fraction of native contacts of the passed trajectories with
        respect to the reference structures. Generates list of RMSDs of size
        n_ref x n_frames where n_ref is the number of references and n_frames
        is the number of frames in the trajectory.

        Parameters
        ----------

        traj : Universe
            Trajectory to be analyzed.

        ref_trajs : List[Universe]
            List of reference trajectories to be analyzed.

        start : int
            Starting frame of analysis

        stop : int
            Stop frame of analysis

        step : int
            Number of frames to skip per step

        verbose : bool
            Turn on verbosity.

        **kwargs
            Extra key word arguments for native_contacts.get_frac_natcons.
            Check module docstring for all possible arguments.


        Returns
        -------

        List[float]
            List of RMSD values of the trajectory with respect to each
            reference.
        """

        all_natcons = []
        for ref_traj in ref_trajs:
            temp_natcons = native_contacts.get_frac_natcons(
                sims=[traj],
                ref=ref_traj,
                start=start,
                stop=stop,
                step=step,
                verbose=verbose,
                **kwargs,
            )
            for natcons in temp_natcons:
                all_natcons.append(native_contacts.post_process_natcons(natcons))

        return all_natcons

    @property
    def get_which(self):
        """Extract the first peak."""
        return 1
