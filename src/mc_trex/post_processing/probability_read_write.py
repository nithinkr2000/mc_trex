import numpy as np
from numpy.typing import NDArray
from typing import Dict, List
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class leave_k_resampled_data:
    """
    Dataclass containing leave-k resampled datasets.
    Used for estimating standard error of the mean.

    Attributes
    ----------

    temperatures : NDArray[np.float64] | None
        Temperatures at which the simulations we performed.

    sim_type : str
        Simulation type -("Conventional" or "T-REMD")

    frame_size : float
        The step size of each frame in nanoseconds.

    sim_times : NDArray[np.float64]
        Simulation times in the same order as the resampled
        data (increasing order) in nanoseconds.
        Can be a single float or a NumPy array.

    block_size : int
        The size of the blocks into which the trajectories at
        different temperatures was divided (assumed constant).

    n_blocks : int | None
        Number of blocks. Same as the size of the leave-k
        dataset for lowest k (last one) divided by the block_size.

    leave_k_datasets : Dict[float, NDArray[np.float64]] | None
        The leave-k datasets in increasing order of simulation
        time considered or decreasing order of k value.


    fit_params : Dict[float, NDArray[np.float64]] | None
        The fits to the leave-k datasets in increasing order of
        simulation time (same order as the datasets).

    log : List[datetime]
        Time-stamp(s) when the data-set was last modified/processed.
    """

    temperatures: NDArray[np.float64] | None = None
    sim_type: str = "Conventional"
    frame_size: float = 0.4

    sim_times: NDArray[np.float64] = 0
    block_size: int = 0
    n_blocks: int | None = None

    leave_k_datasets: Dict[float, NDArray[np.float64]] | None = None

    fit_params: Dict[float, NDArray[np.float64]] | None = None

    log: List[datetime] = field(default_factory=list)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types and datetime objects."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):  # Handle datetime objects
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)


def _process_float_key_dict(data_dict_field, prefix, json_to_save, numpy_to_save):
    if data_dict_field is not None:
        # Store keys as strings in JSON
        json_to_save[f"has_{prefix}"] = True
        json_to_save[f"{prefix}_keys"] = [str(k) for k in data_dict_field.keys()]

        # Store arrays in NPZ
        for k, arr in data_dict_field.items():
            numpy_to_save[f"{prefix}_{k}"] = arr
    else:
        json_to_save[f"has_{prefix}"] = False


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
    json_path = base_path.with_suffix(".json")
    npz_path = base_path.with_suffix(".npz")

    data_dict = asdict(data)

    json_to_save = {}
    numpy_to_save = {}

    for field_name, value in data_dict.items():
        if field_name == "temperatures":
            if value is not None:
                numpy_to_save["temperatures"] = value
                json_to_save["has_temperatures"] = True

            else:
                json_to_save["has_temperatures"] = False

        elif field_name == "sim_times":
            if isinstance(value, np.ndarray):
                numpy_to_save["sim_times"] = value
                json_to_save["sim_times_is_array"] = True

            else:
                json_to_save["sim_times"] = value
                json_to_save["sim_times_is_array"] = False

        elif field_name == "leave_k_datasets":
            _process_float_key_dict(value, "leave_k_datasets", json_to_save, numpy_to_save)

        elif field_name == "fit_params":
            _process_float_key_dict(value, "fit_params", json_to_save, numpy_to_save)

        elif field_name == "log":
            # Convert list of datetime objects to list of ISO strings for JSON
            json_to_save["log"] = [dt.isoformat() for dt in value]

        else:
            # For simple types (str, float, int, bool, None), directly add to JSON
            json_to_save[field_name] = value

    # Save JSON metadata
    with open(json_path, "w") as f:
        json.dump(json_to_save, f, indent=2, cls=NumpyEncoder)

    # Save numpy arrays
    if numpy_to_save:
        np.savez_compressed(npz_path, **numpy_to_save)

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
    json_path = base_path.with_suffix(".json")
    npz_path = base_path.with_suffix(".npz")

    # Load JSON metadata
    with open(json_path, "r") as f:
        json_data = json.load(f)

    # Load numpy arrays if they exist
    numpy_data = {}
    if npz_path.exists():
        numpy_data = dict(np.load(npz_path, allow_pickle=True)) # allow_pickle might be needed for old files

    # Reconstruct the dataclass arguments
    kwargs = {}

    # Handle temperatures
    kwargs["temperatures"] = numpy_data.get("temperatures") if json_data.get("has_temperatures", False) else None

    # Handle sim_type, frame_size, block_size, n_blocks (simple types)
    kwargs["sim_type"] = json_data.get("sim_type", "Conventional")
    kwargs["frame_size"] = json_data.get("frame_size", 0.4)
    kwargs["block_size"] = json_data.get("block_size", 0)
    kwargs["n_blocks"] = json_data.get("n_blocks") # Can be None

    # Handle sim_times reconstruction
    if json_data.get("sim_times_is_array", False):
        kwargs["sim_times"] = numpy_data.get("sim_times")
    else:
        kwargs["sim_times"] = json_data.get("sim_times", 0)
    
    # Helper to reconstruct dicts with float keys
    def _reconstruct_float_key_dict(json_prefix, numpy_prefix):
        if json_data.get(f"has_{json_prefix}", False):
            reconstructed_dict = {}
            for key_str in json_data.get(f"{json_prefix}_keys", []):
                # Convert string key back to float
                float_key = float(key_str)
                reconstructed_dict[float_key] = numpy_data.get(f"{numpy_prefix}_{key_str}")
            return reconstructed_dict
        return None

    # Handle leave_k_datasets
    kwargs["leave_k_datasets"] = _reconstruct_float_key_dict("leave_k_datasets", "leave_k_datasets")

    # Handle fit_params
    kwargs["fit_params"] = _reconstruct_float_key_dict("fit_params", "fit_params")

    # Handle log (list of datetime objects)
    log_iso_strings = json_data.get("log", [])
    kwargs["log"] = [datetime.fromisoformat(s) for s in log_iso_strings]


    data = leave_k_resampled_data(**kwargs)

    print(f"Data loaded from {json_path} and {npz_path}")
    return data

