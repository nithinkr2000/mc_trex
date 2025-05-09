import numpy as np
from typing import Optional, List, Any
from numpy.typing import NDArray
from scipy.constants import Boltzmann as kb

import sys
from mc_trex.post_processing.read_amber import ReadAMBER


class TRExEval(ReadAMBER):
    """
    Evaluate the temperature replica exchange simulation using
    1. Dwell/residence times
    2. Round trip times
    3. Exchange success rates (with immediate neighbour)
    4. Fluctuations in heat capacity
    5. Fraction of time spent in reference configurations

    Attributes
    ----------

    logfiles:List[str]
        List of paths to log files generated by the AMBER REMD simuation.

    outfiles:List[str]
        List of paths to output files generated by the AMBER REMD simulation.
        One can pass all output files in a concatenated list (without
        segregating the files corresponding to different replicas) since the
        generated dataframe will also contain thermostat information.

    therm_rep_restimes:List[List[NDArray[np.int64]]]
        Element i, j, k contains the time (in number of exchange attempts) that
        replica j spent with thermostat i on its kth visit.

    therm_rep_routimes:List[List[NDArray[np.int64]]]
        Element i, j, k contains the time (in number of exchange attempts) that
        replica j took to start from thermostat i and return to thermostat i for
        the kth time.

    Cvs:List[List[float]]:
        Estimate of heat capacity for each thermostat for as many blocks as
        specified.

    Methods
    -------

    __init__
        Constructor

    get_residence_times
        Calculate residence times

    """

    def __init__(
        self, logfiles: Optional[List[str]] = None, outfiles: Optional[List[str]] = None
    ):
        """Extract raw data and set variables."""

        self.logfiles = logfiles
        self.outfiles = outfiles
        self.therm_rep_restimes = None
        self.therm_rep_routimes = None
        self.Cvs = None

        super().__init__(logfiles=logfiles, outfiles=outfiles)

        self.thermostats, self.num_reps, self.num_exs, self.df_exchng = (
            super().generate_log_info(pass_all=True)
        )
        self.df_out = super().generate_out_info()

    def get_residence_times(self) -> List[List[NDArray[np.int64]]]:
        """
        Calculates the residence times for the replica exchange simulation from
        the log file(s).

        Returns
        -------

        therm_rep_restimes:List[List[NDArray[np.int64]]]
            Residence times for each thermostat with each replica expressed in
            units of
            number of exchanges.
        """

        therm_rep_restimes = []

        for thermostat in self.thermostats:
            rep_restimes = []
            const_thermostat_rows = self.df_exchng[self.df_exchng.Temp0 == thermostat]

            for group_idx, const_therm_steps in const_thermostat_rows.groupby(
                "Rep number"
            ):
                changes = (
                    np.where(
                        np.not_equal(
                            np.diff(const_therm_steps["Exchange number"].to_numpy()), 1
                        )
                    )[0]
                    + 1
                )

                rep_restimes.append(np.diff(changes))

            therm_rep_restimes.append(rep_restimes)

        self.therm_rep_restimes = therm_rep_restimes

        return therm_rep_restimes

    def get_round_trip_times(self) -> List[List[NDArray[np.int64]]]:
        """
        Calculate the round-trip times for the replica exchange simulation from
        the log file(s).

        Returns
        -------
        therm_rep_routimes:List[List[NDArray[np.int64]]]
            Round-trip times for each thermostat with each replica expressed in
            units of number exchanges.
        """

        therm_rep_routimes = []

        for thermostat in self.thermostats:
            rep_routimes: List[List[Any]] = [[] for _ in range(self.num_reps)]
            const_thermostat_rows = self.df_exchng[self.df_exchng.Temp0 == thermostat]

            track_visits = [[] for _ in range(self.num_reps)]

            for row_idx, row in const_thermostat_rows.iterrows():
                for i in range(self.num_reps):
                    if track_visits[i]:
                        track_visits[i].append(row["Rep number"])

                    else:
                        if i == row["Rep number"] - 1:
                            track_visits[i].append(row["Rep number"])
                        else:
                            continue

                    if (len(set(track_visits[i])) == self.num_reps) and (
                        track_visits[i][0] == track_visits[i][-1]
                    ):
                        rep_routimes[i].append(len(track_visits[i]) - 1)
                        track_visits[i] = [row["Rep number"]]

            therm_rep_routimes.append(rep_routimes)

        self.therm_rep_routimes = therm_rep_routimes

        return therm_rep_routimes

    def heat_capacity(self) -> List[float]:
        """
        Determine the heat capacities of the replicas using
        fluctuations in the total energies. Ordered in
        increasing order of thermostat temperature.

        Returns
        -------

        List[float]
            Contains Cvs for each thermostat

        """

        Cv_s = []
        for temperature in self.thermostats:
            temp = self.df_exchng[self.df_exchng.Temp0 == temperature].Eptot
            Cv_s.append((temp.std() ** 2) / (kb * (temperature**2)))

        self.Cvs = Cv_s

        return Cv_s
