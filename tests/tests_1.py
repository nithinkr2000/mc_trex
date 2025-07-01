import numpy as np
import pytest
from unittest.mock import Mock, patch
from mc_trex.post_processing.traj_classification import TrajectoryLoader, RMSDAnalysis, NativeContactAnalysis
import MDAnalysis as mda

from mc_trex.post_processing.dna_non_bonded import (
    get_ids_and_masses,
    Method,
    check_distance,
    check_angle,
    get_com_normal,
    pi_stacking,
    t_stacking,
)
from mc_trex.post_processing.native_contacts import get_frac_natcons, post_process_natcons

class TestDNANonBonded:
    """Tests for the mc_trex.post_processing.dna_non_bonded module."""

    def test_method_enum(self):
        """Test Method enum values."""
        assert Method.soft.value == "soft"
        assert Method.hard.value == "hard"

    def test_check_distance_hard_method(self):
        """Test check_distance function with a hard cut-off."""
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([2.0, 0.0, 0.0])

        # Distance is 2.0, cut_off is 3.0 => should return True
        result = check_distance(vec1, vec2, cut_off=3.0, method="hard")
        assert result is np.True_

        # Distance is 2.0, cut_off is 1.0 => should return False
        result = check_distance(vec1, vec2, cut_off=1.0, method="hard")
        assert result is np.False_

    def test_check_distance_soft_method(self):
        """Test check_distance function with a soft cut-off."""
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([2.0, 0.0, 0.0])

        with patch(
            "mc_trex.post_processing.dna_non_bonded.sigmoid_for_dist"
        ) as mock_sigmoid:
            mock_sigmoid.return_value = 0.7
            result = check_distance(vec1, vec2, cut_off=3.0, method="soft")
            assert result == 0.7
            mock_sigmoid.assert_called_once_with(2.0, 3.0)

    def test_check_distance_invalid_cutoff(self):
        """Test check_distance with an invalid cut-off."""
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([2.0, 0.0, 0.0])

        result = check_distance(vec1, vec2, cut_off=0.0, method="hard")
        assert result == 0

    def test_check_angle_hard_method(self):
        """Test check_angle function with a hard cut-off."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        # Angle is 90 degrees, between 150 and 30 => should return False
        result = check_angle(vec1, vec2, up_ang_cut=30, down_ang_cut=150, method="hard")
        assert result is False

        # Angle is 90 degrees, between 60 and 120 => should return True
        result = check_angle(vec1, vec2, up_ang_cut=120, down_ang_cut=60, method="hard")
        assert result is True

    def test_check_angle_soft_method(self):
        """Test check_angle function with a soft cut-off."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        with patch(
            "mc_trex.post_processing.dna_non_bonded.sigmoid_for_ang"
        ) as mock_sigmoid:
            mock_sigmoid.return_value = 0.5
            result = check_angle(
                vec1, vec2, up_ang_cut=30, down_ang_cut=150, method="soft"
            )
            assert result == 0.5
            mock_sigmoid.assert_called_once_with(90.0, 30, 150)

    def test_check_angle_invalid_method(self):
        """Test check_angle with invalid cut-off method."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        with pytest.raises(ValueError, match="Not a valid method"):
            check_angle(vec1, vec2, method="invalid")

    def test_get_com_normal(self):
        """Test get_com_normal."""

        # Create a simple triangle in the xy-plane
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        masses = [1.0, 1.0, 1.0]

        com, normal = get_com_normal(coords, masses)

        # Check center of mass
        expected_com = np.array([1 / 3, 1 / 3, 0.0])
        np.testing.assert_array_almost_equal(com, expected_com)

        # Check normal (should point in z direction)
        expected_normal = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(np.abs(normal), np.abs(expected_normal))

    def test_get_com_normal_degenerate(self):
        """Test get_com_normal with degenerate coordinates."""
        # Collinear points
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        masses = [1.0, 1.0, 1.0]

        with pytest.raises(ValueError, match="Degenerate or very small normal vector"):
            get_com_normal(coords, masses)

    @patch("mc_trex.post_processing.dna_non_bonded.get_com_normal")
    @patch("mc_trex.post_processing.dna_non_bonded.check_distance")
    @patch("mc_trex.post_processing.dna_non_bonded.check_angle")
    def test_pi_stacking_hard_method(
        self, mock_check_angle, mock_check_distance, mock_get_com_normal
    ):
        """Test pi_stacking function with hard method."""
        # Setup mock returns
        mock_get_com_normal.return_value = (
            np.array([0.0, 0.0, 0.0]),  # com
            np.array([0.0, 0.0, 1.0]),  # normal
        )
        mock_check_distance.return_value = True
        mock_check_angle.return_value = True

        # Create test data
        frames = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])  
        ids = [[[0]], [[1]]]  # 2 residues, 1 ring each, 1 atom each
        masses = [[[1.0]], [[1.0]]]  # corresponding masses

        result = pi_stacking(frames, ids, masses, method="hard")

        assert len(result) == 1  # 1 frame
        assert len(result[0]) == 1  # 1 residue comparison
        assert result[0][0] == [1.0]  # stacking detected

    @patch("mc_trex.post_processing.dna_non_bonded.get_com_normal")
    @patch("mc_trex.post_processing.dna_non_bonded.check_distance")
    @patch("mc_trex.post_processing.dna_non_bonded.check_angle")
    def test_t_stacking_soft_method(
        self, mock_check_angle, mock_check_distance, mock_get_com_normal
    ):
        """Test t_stacking function with a soft cut-off."""
        # Setup mock returns
        mock_get_com_normal.return_value = (
            np.array([0.0, 0.0, 0.0]),  # com
            np.array([0.0, 0.0, 1.0]),  # normal
        )
        mock_check_distance.return_value = 0.8
        mock_check_angle.return_value = 0.6

        # Create test data
        frames = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
        ids = [[[0]], [[1]]]
        masses = [[[1.0]], [[1.0]]]

        result = t_stacking(frames, ids, masses, method="soft")

        assert len(result) == 1
        assert len(result[0]) == 1
        # Should be average of 4 checks: (0.8 + 0.6 + 0.6 + 0.6) / 4 = 0.65
        assert result[0][0] == [0.65]

    def test_get_ids_and_masses(self, test_universe):
        """Test get_ids_and_masses function with a real MDAnalysis Universe."""

        ids, masses = get_ids_and_masses(test_universe)

        assert len(ids) == 7
        assert test_universe.residues.n_residues == 7

        for res_ids, res_masses in zip(ids, masses):
            for ring_ids, ring_masses in zip(res_ids, res_masses):
                assert isinstance(ring_ids, np.ndarray)
                assert isinstance(ring_masses, np.ndarray)
                assert len(ring_ids) > 0
                assert len(ring_masses) > 0


class TestNativeContacts:
    """Test suite for native_contacts.py module."""

    def test_get_frac_natcons_post_process_natcons(self, test_universe):
        """Test get_frac_natcons."""
        
        ref_universe = mda.Universe(
            test_universe.filename,
            test_universe.trajectory.timeseries(start=9551, stop=9552, step=1),
        )

        result = get_frac_natcons(
            sims=[test_universe], ref=ref_universe, cut_off=3.0, method="soft",
            atom_selection=" and nucleic", start=0, stop=-1, step=100
        )
        
        assert len(result) == 1  # 1 reference
        assert isinstance(result[0][0], mda.analysis.contacts.Contacts)

        xs, ys = post_process_natcons(result[0])

        np.testing.assert_array_equal(xs, np.arange(0, 36729, 100))
        np.testing.assert_almost_equal(np.median(ys), 0.8649939291161526, 5)

    def test_post_process_natcons_error(self):
        """Test post_process_natcons with invalid input."""

        with pytest.raises(AttributeError, match=""):
            post_process_natcons("invalid_input")
            
            
class TestTrajectoryClassification:
    """Test suite for traj_classification.py module."""

    def test_trajectory_loader_init(self, test_universe):
        """Test TrajectoryLoader initialization."""
        loader = TrajectoryLoader(
            traj_loc=[test_universe.trajectory.filename],
            top = test_universe.filename,
            ref_loc = [test_universe.trajectory.filename],
            top_ref = test_universe.filename
        )
        
        assert loader.traj_loc == [test_universe.trajectory.filename]
        assert loader.top == test_universe.filename
        assert loader.ref_loc == [test_universe.trajectory.filename]
        assert loader.top_ref == test_universe.filename
        assert loader.traj is None
        assert loader.refs is None
    
    def test_trajectory_loader_no_top_ref(self, test_universe):
        """Test TrajectoryLoader with no separate reference topology."""
        loader = TrajectoryLoader(
            traj_loc=[test_universe.trajectory.filename],
            top=test_universe.filename,
            ref_loc=test_universe.filename
        )
        
        assert loader.top_ref == test_universe.filename
    
    
    def test_load_trajectory_no_traj_loc(self, test_universe):
        """Test load_trajectory with no trajectory location."""
        loader = TrajectoryLoader(top=test_universe.filename)
        
        with pytest.raises(ValueError, match="No trajectory provided"):
            loader.load_trajectory()
    
    def test_load_trajectory_no_top(self, test_universe):
        """Test load_trajectory with no topology."""
        loader = TrajectoryLoader(traj_loc=test_universe.trajectory.filename)
        
        with pytest.raises(ValueError, match="No topology provided"):
            loader.load_trajectory()
    

    def test_load_reference(self, test_universe):
        """Test load_reference method."""
        
        loader = TrajectoryLoader(
            top=test_universe.filename,
            ref_loc=[test_universe.trajectory.filename],
            top_ref=test_universe.filename
        )
        
        result = loader.load_reference()
        
        assert len(result) == 1
        np.testing.assert_array_equal(result[0].residues.resnames, test_universe.residues.resnames)
    
    def test_rmsd_analysis_get_which(self):
        """Test RMSDAnalysis get_which property."""
        analysis = RMSDAnalysis()
        assert analysis.get_which == 0
    
    def test_native_contact_analysis_get_which(self):
        """Test NativeContactAnalysis get_which property."""
        analysis = NativeContactAnalysis()
        assert analysis.get_which == 1
    
    @patch('mc_trex.post_processing.traj_classification.native_contacts.get_frac_natcons')
    @patch('mc_trex.post_processing.traj_classification.native_contacts.post_process_natcons')
    def test_native_contact_analysis_get_similarity_metric(self, mock_post_process, mock_get_frac):
        """Test NativeContactAnalysis get_similarity_metric."""
        # Setup mocks
        mock_get_frac.return_value = [[Mock(), Mock()]]
        mock_post_process.return_value = ([0, 1, 2], [0.5, 0.7, 0.3])
        
        # Create analysis instance
        analysis = NativeContactAnalysis()
        analysis.traj = Mock()
        analysis.refs = [Mock(), Mock()]
        
        result = analysis.get_similarity_metric(post_process=True)
        
        assert len(result) == 2  # 2 references
        # Each result should be the post-processed output
        assert result[0] == ([0, 1, 2], [0.5, 0.7, 0.3])
        assert result[1] == ([0, 1, 2], [0.5, 0.7, 0.3])
    
    def test_bin_frames(self):
        """Test bin_frames method."""
        analysis = RMSDAnalysis()
        
        similarity_metric = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        ]
        
        result = analysis.bin_frames(similarity_metric, bins=3)
        
        assert len(result) == 2  # 2 references
        assert len(result[0]) == 2  # (counts, bin_edges)
        assert len(result[1]) == 2  # (counts, bin_edges)
    
    def test_get_cuts(self, test_universe):
        """Test get_cuts method."""
        analysis = RMSDAnalysis()
        
        # Create binned metrics
        binned_metrics = [
            (np.array([10, 20, 15]), np.array([0.0, 1.0, 2.0, 3.0]))
        ]
        
        result = analysis.get_cuts(binned_metrics, n_components=2)
        
        assert len(result) == 1  # 1 reference
        assert isinstance(result[0], np.ndarray)
    
    def test_sort_frames_get_which_0(self):
        """Test sort_frames method with get_which=0 (minimum)."""
        analysis = RMSDAnalysis()
        
        similarity_metric = np.array([
            [1.0, 3.0, 2.0, 4.0],  # ref 1
            [2.0, 1.0, 3.0, 2.0]   # ref 2
        ])
        cut_offs = np.reshape([2.5, 2.5], (-1, 1))
        
        frames, configurations = analysis.sort_frames(similarity_metric, cut_offs)
        
        np.testing.assert_array_equal(frames, [0, 1, 2, 3])
        
        # Frame 0: min(1.0, 2.0) = 1.0 (ref 0), Frame 1: min(inf, 1.0) = 1.0 (ref 1)
        # Frame 2: min(2.0, inf) = 2.0 (ref 0), Frame 3: min(inf, 2.0) = 2.0 (ref 1)
        
        expected_configs = [0, 1, 0, 1]
        np.testing.assert_array_equal(configurations, expected_configs)
    
    def test_sort_frames_empty_similarity_metric(self):
        """Test sort_frames with empty similarity metric."""
        analysis = RMSDAnalysis()
        
        with pytest.raises(ValueError, match="similarity_metric cannot be empty"):
            analysis.sort_frames([], [])
