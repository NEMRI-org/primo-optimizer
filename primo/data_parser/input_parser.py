#################################################################################
# PRIMO - The P&A Project Optimizer was produced under the Methane Emissions
# Reduction Program (MERP) and National Energy Technology Laboratory's (NETL)
# National Emissions Reduction Initiative (NEMRI).
#
# NOTICE. This Software was developed under funding from the U.S. Government
# and the U.S. Government consequently retains certain rights. As such, the
# U.S. Government has been granted for itself and others acting on its behalf
# a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
# reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit others to do so.
#################################################################################

# Standard libs
import argparse
import logging
import os
from typing import Dict

# Installed libs
import pandas as pd

# User-defined libs
from primo.data_parser.data_model import CampaignCandidates, OptInputs, Well
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


class InputParser:
    """
    InputParser class
    Implements methods for parsing input data from csv
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the class and read in a CSV file.

        Parameters
        ----------
        args : argparse.Namespace
            The arguments parsed from the command line.

        Raises
        ------
        ValueError
            If the specified file does not exist or is not a CSV file.
        """
        file_path = args.input_file
        # Check 1: Check file exists
        if not os.path.exists(file_path):
            raise_exception(f"File: {file_path} does not exist.", ValueError)

        # Check 2: Check file is a csv file
        if file_path.suffix != ".csv":
            raise_exception(
                f"File: {file_path} is expected to be an csv file", ValueError
            )

        self.well_data = pd.read_csv(file_path)
        self.args = args
        LOGGER.info(f"Initialized class and read file: {file_path}")

    def parse_data(
        self,
        mobilization_cost: Dict[int, float],
    ) -> OptInputs:
        """
        Parse data to create an OptInputs object that models all input parameters
        to set up the optimization model.

        Parameters
        ----------
        mobilization_cost : Dict[int, float]
            The cost associated with mobilization for a campaign based on the number
            of wells in a campaign.

        Returns
        -------
        opt_inputs : OptInputs
            A fully instantiated object for inputs to the optimization model.

        Raises
        ------
        ValueError
            If a well ID is repeated in the CSV file.
        """
        budget = self.args.total_budget
        max_wells_per_owner = self.args.max_wells_per_owner
        well_dict = {}
        candidates = {}
        # dac_budget_fraction is the percentage of budget that needs
        # to be spent on plugging wells in the disadvantaged community
        dac_budget_fraction = self.args.dac_weight_percent
        distance_threshold = self.args.maximum_well_distance

        # new data
        owner_well_count = (
            self.well_data.groupby("Operator Name")["Well ID"].apply(list).to_dict()
        )

        dac_weight = self.args.dac_weight

        if self.args.dac_weight_percent is not None:
            if dac_weight is not None:
                threshold = dac_weight / 100 * self.args.dac_weight_percent
            else:
                raise_exception(
                    "Disadvantaged Community Impact must be selected as a "
                    "priority factor if you wish to implement the DAC constraint.",
                    ValueError,
                )
        else:
            # When the user does not provide a value for disadvantaged community,
            # all wells are set as not disadvantaged.
            # The maximum score possible for disadvantaged_community_score is 100.
            threshold = 100
        for _, row in self.well_data.iterrows():
            well_id = row["Well ID"]
            lat = row["Latitude"]
            long = row["Longitude"]
            score = row["Priority Score [0-100]"]
            if dac_weight is not None:
                disadvantaged_community_score = row[f"DAC Score [0-{int(dac_weight)}]"]
                is_disadvantaged = float(disadvantaged_community_score > threshold)
            else:
                # When the user does not select DAC as a priority factor,
                # all wells are assumed to not be located in a disadvantaged community.
                is_disadvantaged = float(False)

            if well_id in well_dict:
                msg = f"Well id must be unique. Well id: {well_id} is repeated in csv file"
                raise_exception(msg, ValueError)

            well_dict[well_id] = Well(well_id, lat, long, score, is_disadvantaged)

        LOGGER.info("Parsing for campaign candidates")

        for _, row in self.well_data.iterrows():
            well_id = row["Well ID"]
            cluster = row["cluster"]
            if cluster not in candidates:
                candidates.setdefault(cluster, {})
            candidates[cluster][well_id] = well_dict[well_id]

        campaign_candidates = {}
        for campaign_id, campaign_dict in candidates.items():
            LOGGER.info(f"Preparing cluster: {campaign_id}")
            campaign_candidates[campaign_id] = CampaignCandidates(
                campaign_id, campaign_dict, len(campaign_dict)
            )

        return OptInputs(
            well_dict,
            campaign_candidates,
            budget,
            mobilization_cost,
            owner_well_count,
            max_wells_per_owner,
            dac_budget_fraction,
            distance_threshold,
        )
