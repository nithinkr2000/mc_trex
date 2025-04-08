from mc_trex.post_processing.traj_classification import RMSDAnalysis

import numpy as np


def rmsdcalculations():
    top_loc  = "input_samples/tumuc_gcgcagc.prmtop"
    traj_loc = "input_samples/gcgcagc.nc"
    ref_loc  = "input_samples/gcgcagc_ref.nc"
    
    expected_loc = "output_samples/gcgcagc_rmsd.dat" 
    
    expected_output = []
    with open(expected_loc, "r") as f:
        expected_rmsds = f.readlines()
        expected_rmsds = [float(expected_rmsd.strip('\n')) for expected_rmsd in expected_rmsds]

    rmsdinst = RMSDAnalysis(traj_loc, top_loc, [ref_loc], top_loc)
    traj = rmsdinst.load_trajectory()
    ref = rmsdinst.load_reference()

    rmsds = rmsdinst.get_similarity_metric(traj, ref, select="all")
    
    return rmsds, expected_rmsds


    
   
    
def test_rmsdcalculations():
    rmsds, expected_output = rmsdcalculations()
    assert(np.all(np.equal(rmsds, expected_output))), "RMSDs do not match expected output."

