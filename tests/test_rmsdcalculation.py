from mc_trex.post_processing.traj_classification import RMSDAnalysis

def test_rmsdcalculations(top, traj, ref, top_ref, expected_out):
    rmsdinst = RMSDAnalysis(top, traj, ref, top_ref)
    traj = rmsdinst.load_trajectory()
    ref = rmsdinst.load_reference()

    rmsds = rmsdinst(traj, ref)
    assert(np.all(np.equal(rmsds, expected_output))), "RMSDs do not match expected output."
    
   
   
if __name__ == '__main__':
    top_loc  = "tests/input_samples/tumuc_gcgcagc.prmtop"
    traj_loc = "tests/input_samples/gcgcagc.nc"
    ref_loc  = "tests/input_samples/gcgcagc_ref.nc"
    
    expected_loc = "tests/output_samples/gcgcagc_rmsd.dat" 
    
    expected_output = []
    with open(expected_loc, "r") as f:
        expected_rmsds = f.readlines()
        expected_rmsds = [float(expected_rmsd.strip('\n')) for expected_rmsd in expected_rmsds]
        
    test_rmsdcalculations(top_loc, traj_loc, ref_loc, top_loc, expected_out)
    
        
