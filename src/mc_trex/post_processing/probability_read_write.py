
@dataclass
class leave_k_resampled_data():
    """
    Dataclass containing leave-k resampled datasets.
    Used for estimating standard error of the mean.
    
    Attributes
    ----------

    temperatures : NDArray[np.float64] | None
        Temperatures at which the simulations we performed.

    sim_type : str
        Simulation type -("conventional" or "T-REMD")

    frame_size : float
        The step size of each frame in nanoseconds.

    sim_times : NDArray[np.float64]
        Simulation times in the same order as the resampled 
        data (increasing order) in nanoseconds. 

    block_size : int
        The size of the blocks into which the trajectories at 
        different temperatures was divided (assumed constant).

    n_blocks : int | None
        Number of blocks. Same as the size of the leave-k
        dataset for lowest k (last one) divided by the block_size.

    leave_k_datasets : Dict[np.float64, NDArray[np.float64]] | None
        The leave-k datasets in increasing order of simulation
        time considered or decreasing order of k value.
    """

    temperatures : NDArray[np.float64] | None = None
    sim_type : str = "Conventional"
    frame_size : float = 0.4 
    sim_times : float = 0
    block_size : int = 0
    n_blocks : int | None = None
    leave_k_datasets : Dict[np.float64, NDArray[np.float64]] | None = None 


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_leave_k_data_json(data: leave_k_resampled_data, base_filename: str):
    """
    Save leave_k_resampled_data instance to JSON + NPZ files.
    
    Parameters
    ----------
    data : leave_k_resampled_data
        The dataclass instance to save
    base_filename : str
        Base filename (without extension). Will create .json and .npz files
    """
    base_path = Path(base_filename)
    json_path = base_path.with_suffix('.json')
    npz_path = base_path.with_suffix('.npz')
    
    # Prepare data for JSON (exclude numpy arrays)
    json_data = {
        'sim_type': data.sim_type,
        'frame_size': data.frame_size,
        'block_size': data.block_size,
        'n_blocks': data.n_blocks,
    }
    
    # Prepare numpy arrays for NPZ
    numpy_data = {}
    
    if data.temperatures is not None:
        numpy_data['temperatures'] = data.temperatures
        json_data['has_temperatures'] = True
    else:
        json_data['has_temperatures'] = False
    
    # Handle sim_times (could be float or numpy array)
    if isinstance(data.sim_times, np.ndarray):
        numpy_data['sim_times'] = data.sim_times
        json_data['sim_times_is_array'] = True
    else:
        json_data['sim_times'] = data.sim_times
        json_data['sim_times_is_array'] = False
    
    if data.leave_k_datasets is not None:
        json_data['has_leave_k_datasets'] = True
        json_data['leave_k_keys'] = list(data.leave_k_datasets.keys())
        for key, array in data.leave_k_datasets.items():
            numpy_data[f'leave_k_{key}'] = array
    else:
        json_data['has_leave_k_datasets'] = False
    
    # Save JSON metadata
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, cls=NumpyEncoder)
    
    # Save numpy arrays
    if numpy_data:
        np.savez_compressed(npz_path, **numpy_data)
    
    print(f"Data saved to {json_path} and {npz_path}")






def load_leave_k_data_json(base_filename: str) -> leave_k_resampled_data:
    """
    Load leave_k_resampled_data instance from JSON + NPZ files.

    Parameters
    ----------
    base_filename : str
        Base filename (without extension)

    Returns
    -------
    leave_k_resampled_data
        The loaded dataclass instance
    """
    base_path = Path(base_filename)
    json_path = base_path.with_suffix('.json')
    npz_path = base_path.with_suffix('.npz')

    # Load JSON metadata
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # Load numpy arrays if they exist
    numpy_data = {}
    if npz_path.exists():
        numpy_data = dict(np.load(npz_path))

    # Reconstruct the dataclass
    temperatures = numpy_data.get('temperatures') if json_data['has_temperatures'] else None

    # Handle sim_times reconstruction
    if json_data['sim_times_is_array']:
        sim_times = numpy_data['sim_times']
    else:
        sim_times = json_data['sim_times']

    leave_k_datasets = None
    if json_data['has_leave_k_datasets']:
        leave_k_datasets = {}
        for key in json_data['leave_k_keys']:
            leave_k_datasets[key] = numpy_data[f'leave_k_{key}']

    data = leave_k_resampled_data(
        temperatures=temperatures,
        sim_type=json_data['sim_type'],
        frame_size=json_data['frame_size'],
        sim_times=sim_times,
        block_size=json_data['block_size'],
        n_blocks=json_data['n_blocks'],
        leave_k_datasets=leave_k_datasets
    )

    print(f"Data loaded from {json_path} and {npz_path}")
    return data

