#!/usr/bin/env python3

import argparse
import tempfile
import subprocess
import os
from pathlib import Path
from typing import List


def gen_stripped_trajs(
    in_traj: str | None = None,
    out_traj: str | None = None,
    top: str | None = None,
    top_out: bool = False,
    outtop: str | None = None,
    strip_res: List[str] = ["Na+", "Cl-", "WAT"],
    verbosity: bool = False,
) -> None:
    """
    Strip residues from trajectory. Optionally, strip residues from topology
    and generate new topology.

    Parameters
    ----------

    in_traj : str
        Path to input trajectory.

    out_traj : str
        Output trajectory name (full path). By default, the function generates
        a the trajectory to the same directory as the input with the name
        stripped.nc.

    top : str
        Path to the topology file corresponding to the trajectory.

    top_out : str
        Set flag to generate a stripped topology file as well. (False => no new
        topology).

    outtop : str
        Name and path of output topology, only read if top_out is True. By
        default a topology in the same path as the input topology with the
        prefix "stripped" is generated.

    strip_res : List[str]
        Residues to strip from the trajectory.

    verbosity : bool
        Set flag to suppress output (False => no output)
    """

    joined_strip_res = ",".join(strip_res)

    output_styles = [subprocess.DEVNULL, subprocess.PIPE]

    input_commands = f"""\
                    trajin {in_traj}
                    autoimage
                    strip :{joined_strip_res}\n
                    """
    if not out_traj:
        out_traj = Path(in_traj).parent.joinpath("stripped.nc")

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_cpptraj_in:
        tmp_cpptraj_in.write(input_commands + f"\n trajout {out_traj}\n run\n exit")
        tmp_name = tmp_cpptraj_in.name
    subprocess.run(
        ["cpptraj", "-p", top, "-i", tmp_name],
        check=True,
        stdout=output_styles[verbosity],
    )
    os.remove(tmp_name)

    if top_out:
        if not outtop:
            outtop = Path(top).parent.as_posix() + "/stripped." + Path(top).name
        input_commands = f"""\
                        parmstrip :{joined_strip_res}\n
                        parmwrite out {outtop}\n
                        run\n
                        exit
                        """

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_cpptraj_in:
            tmp_cpptraj_in.write(input_commands)
            tmp_name = tmp_cpptraj_in.name
        subprocess.run(
            ["cpptraj", "-p", top, "-i", tmp_name],
            check=True,
            stdout=output_styles[verbosity],
        )
        os.remove(tmp_name)


def gen_stripped_top(
    top: str | None = None,
    outtop: str | None = None,
    strip_res: List[str] = ["Na+", "Cl-", "WAT"],
    verbosity: bool = False,
) -> None:
    """
    Generate stripped topology file using cpptraj.

    Parameters
    ----------

    top : str
        Path to topology file

    outtop : str
        Path of the output topology. By default, a topology file is generated
        in the same location as the input topology with the same name, with the
        prefix "stripped".

    strip_res : List[str]
        Names of residues to be stripped.

    verbosity : bool
        Verbosity

    """

    joined_strip_res = ",".join(strip_res)

    output_styles = [subprocess.DEVNULL, subprocess.PIPE]

    if not outtop:
        outtop = Path(top).parent.as_posix() + "/stripped." + Path(top).name

    input_commands = f"""\
                    parmstrip :{joined_strip_res}\n
                    parmwrite out {outtop}\n
                    run\n
                    exit
                    """

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_cpptraj_in:
        tmp_cpptraj_in.write(input_commands)
        tmp_name = tmp_cpptraj_in.name
    subprocess.run(
        ["cpptraj", "-p", top, "-i", tmp_name],
        check=True,
        stdout=output_styles[verbosity],
    )
    os.remove(tmp_name)


def main():
    parser = argparse.ArgumentParser(
        prog="Strip specified residues from the input trajectory. Optionally \
        strip the same residues from the passed topology and generate a new \
        topology file."
    )
    parser.add_argument(
        "-p",
        help="Topology for the trajectory to be stripped.",
        required=True,
        type=os.path.abspath,
    )
    parser.add_argument(
        "-i",
        help="Trajectory from which the residues are to be stripped.",
        required=True,
        type=os.path.abspath,
    )
    parser.add_argument(
        "-o",
        help="Output trajectory name (full path).",
        default="./out_traj.nc",
        type=os.path.abspath,
    )
    parser.add_argument(
        "-po",
        action="store_true",
        help="Include this flag to generate the stripped topology file.",
    )
    parser.add_argument(
        "-strip_res",
        help="The residues to strip from the trajectory. Add in a comma \
        separated list.",
        default="Na+,Cl-,WAT",
    )

    ips = parser.parse_args()
    cwd = Path("./")

    gen_stripped_trajs(ips.i, ips.o, ips.p, ips.po, ips.strip_res.split(","))
