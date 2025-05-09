#! /usr/bin/env python3

import argparse
import subprocess
import tempfile
import os
from typing import Literal
from pathlib import Path


def get_Tcont_trajs(
    in_traj: str = "./*.nc.001",
    out_traj: str | None = None,
    top: str = "./*.prmtop",
    start: int = 1,
    stop: int | Literal["last"] = "last",
    step: int = 1,
    verbosity: bool = False,
) -> None:
    """
    Generate thermodynamically continuous trajectories from the structurally
    continuous trajectories generated by AMBER temperature replica exchhange
    simulations.

    Parameters
    ----------

    in_traj : str
        Input trajectory for the REMD simulation. This is the trajectory file
        corresponding to the first replica, which should end with .001.

    out_traj : str | None
        Output trajectory name. Generates trajectories named
        template.[0-9]* in the same directory as the input trajectories.
        Default is out.nc.[0-9]+ in the same directory as the input.

    top : str
        Topology file for the system.

    start : int
        Starting frame

    stop : int
        Stopping frame

    step : int
        Step size when parsing frames

    verbosity : bool
        To determine whether to print the CppTraj output or not.
    """

    # If the parent directory does not exist, then create it.
    if out_traj:
        parent_dir = Path(out_traj).parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)

    else:
        out_traj = Path(in_traj).parent.joinpath("out.nc").as_posix()

    output_styles = [subprocess.DEVNULL, subprocess.PIPE]

    input_commands = f"""\
                    ensemble {in_traj} {start} {stop} {step}
                    trajout {out_traj}
                    run
                    exit
                    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_cpptraj_in:
        tmp_cpptraj_in.write(input_commands)
        tmp_filename = tmp_cpptraj_in.name
        tmp_cpptraj_in.close()
    subprocess.run(
        ["cpptraj", "-p", top, "-i", tmp_filename],
        check=True,
        stdout=output_styles[verbosity],
    )
    os.remove(tmp_filename)


def main():
    parser = argparse.ArgumentParser(
        prog="gen_T_cont_trajs",
        description="Generate thermodynamically continuous trajectories from \
        structurally continuous trajectories generated by AMBER temperature\
         replica exchange simulations.",
    )
    parser.add_argument(
        "-i",
        help="The path to the trajectory file corresponding to the first \
            replica.",
        default="./*.nc.001",
    )
    parser.add_argument(
        "-o",
        help="The path and naming template for cpptraj to generate \
            thermodynamically continuous output trajectories. The trajectories\
           will be named template.0, template.1 ...",
        default="./traj.nc",
    )
    parser.add_argument(
        "-p", required=True, help="Path to the topology file for the system."
    )
    parser.add_argument(
        "-s3",
        help="Start, stop and step for extract frames from the trajectory.",
        nargs="*",
    )
    parser.add_argument(
        "-v",
        help="Verbosity. Print output or not.",
    )

    ips = parser.parse_args()

    get_Tcont_trajs(ips.i, ips.o, ips.p)
