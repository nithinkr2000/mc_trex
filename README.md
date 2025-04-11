<p align="center">
  <img src="./assets/logo2.jpg" width="500" height="500">
</p>

# mc_trex

Helper functions for performing analyses on the temperature replica exchange simulations with AMBER. Generic helper functions for classification of trajectories. 

## Installation
Create a new environment on your machine and then do the following
``` bash
mkdir mc_trex
cd mc_trex
python3 -m pip install .
```


## Summary of features

The different methods are set up to be as easy as possible to use, while also allowing for intervention wherever necessary. Currently, the following can be performed by the package

For AMBER replica exchange trajectories
1. Generate thermodynamically continuous trajectories from structurally continuous trajectories
2. Strip residues from AMBER trajectories.
3. Strip residues from AMBER topologies.
4. Read AMBER output and temperature replica exchange log files and generate dataframes.
5. Calculate RMSD or fraction of native contacts analyses with respect to multiple reference structures.
6. Calculate stacking interactions and van der Waals forces in DNA molecules, using a hard or soft cut.
7. Sort trajectory frames into bins based on passed reference structures, including the automatic selection of cut-offs.
8. Determine round-trip times and residence times from AMBER temperature replica exchange log files.
9. Determine heat capacity from AMBER output files.

The following functionalities are coming soon
1. Modules to set up replica exchange simulation input files for estimating the melting temperature.
2. Modules to perform Parallel Tempering + Simulated Annealing with AMBER.
3. Modules to analyze PT + SA simulations.
4. Quick method to perform clustering trajectories using RMSDs

## Usage

```{r,eval=False,tidy=False}
topology = top_path
trajectories = [traj_paths] 
reference_structures = [ref_paths]
reference_topology = ref_top_path
```

The RMSD can be calculated fairly easily as 

```python

from post_processing.traj_classification import RMSDAnalysis

rmsd_analyzer = RMSDAnalysis(trajectories, topology, reference_structures, reference_topology)

trajs = rmsd_analyzer.load_trajectory()
refs = rmsd_analyzer.load_reference()

rmsds = rmsd_analyzer.get_similarity_metric()
```

This can be used to generate cut-offs for RMSD with respect to each structure.

```python

binned_rmsds = rmsd_analyzer.bin_frames(bins=50)

cut_offs = rmsd_analyzer.get_cuts(binned_rmsds, max_components=len(reference_structures)+5, height=1)
```

Finally, to sort the frames based on the RMSDs and the cut-offs determined above,
```python
rmsds, cut_offs

```

Which generates the index of the reference structure if a reference structure was associated with the frame and -1 if it was not associated with any reference structure.




