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

# Standard lib
from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Dict, Union


# User defined lib
from primo.utils.geo_utils import get_distance
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)

# TODO: Write tests for entire module


@dataclass(frozen=True)
class Well:
    """
    A Well dataclass
    """

    __slots__ = (
        "well_id",
        "lat",
        "long",
        "score",
        "is_disadvantaged",
    )
    well_id: str
    lat: Union[float, None]
    long: Union[float, None]
    score: float
    is_disadvantaged: bool

    def haversine_distance(self, other_well: Well) -> float:
        """
        Returns the distance in miles between two wells calculated with Haversine formula.

        Parameters
        ----------
        other_well : Well
            The other well under consideration

        Returns
        -------
        distance : float
            The distance between self and other_well
        """
        origin = (self.lat, self.long)
        dest = (other_well.lat, other_well.long)
        return get_distance(origin, dest)


@dataclass(frozen=True)
class CampaignCandidates:
    """
    A dataclass that combines wells that are potential candidates to be included
    in a plugging campaign
    """

    __slots__ = ("campaign_id", "well_dict", "n_wells")
    campaign_id: str
    well_dict: Dict[str, Well]
    n_wells: int

    def __post_init__(self):
        """
        Run some validation checks
        """
        for well_id, well in self.well_dict.items():
            if well_id != well.well_id:
                msg = (
                    f"Dictionary for {self.campaign_id} is corrupted. Expect key "
                    "to be the same as well_id"
                )
                raise_exception(msg, ValueError)

        if self.n_wells != len(self.well_dict):
            LOGGER.warning(
                f"{self.n_wells} incorrectly initialized for {self.campaign_id}"
            )
            LOGGER.warning("Automatically correcting!")
            self.n_wells = len(self.well_dict)


@dataclass(frozen=True)
class OptInputs:
    """
    Succinctly captures all inputs relevant to the optimization model
    """

    __slots__ = (
        "well_dict",
        "campaign_candidates",
        "budget",
        "mobilization_cost",
        "owner_well_count",
        "max_wells_per_owner",
        "dac_budget_fraction",
        "distance_threshold",
    )
    well_dict: Dict[str, Well]
    campaign_candidates: Dict[str, CampaignCandidates]
    budget: float
    mobilization_cost: Dict[int, float]
    owner_well_count: Dict[str, Well]
    max_wells_per_owner: float
    dac_budget_fraction: float
    distance_threshold: float

    def __post_init__(self):
        """
        Run some validation checks
        """
        # Create a set to retrieve unique well_id (identifiers that do not repeat).
        well_id_set = set(self.well_dict.keys())

        if self.budget <= 0:
            msg = f"Budget cannot be negative. Received " f"{self.budget} for budget."
            raise_exception(msg, ValueError)

        for well_id, well in self.well_dict.items():
            if well_id != well.well_id:
                msg = (
                    f"Dictionary for optimization inputs corrupted. "
                    f"Expect key: {well_id} to be the same as"
                    f" well_id: {well.well_id}"
                )
                raise_exception(msg, ValueError)

        max_wells = 0
        # Iterate through each campaign using its campaign_id.
        for campaign_id, campaign in self.campaign_candidates.items():
            # Lines 167-173 aim to verify the accuracy of the
            # campaign_candidates construction.
            if campaign_id != campaign.campaign_id:
                msg = (
                    f"Dictionary of candidates corrupted. "
                    f"Expect key: {campaign_id} to be the same as "
                    f"campaign_id: {campaign.campaign_id}"
                )
                raise_exception(msg, ValueError)

            for well_id in campaign.well_dict:
                # Lines 178-183 aim to verify whether the wells in CampaignCandidates
                # are all sourced from the input well data.
                if well_id not in self.well_dict:
                    msg = (
                        f"Campaign candidate: {campaign_id} includes well: {well_id}"
                        f". This well is not provided in well_dict"
                    )
                    raise_exception(msg, ValueError)

                # Lines 187-190 aim to check if a well appears more than once within
                # the same campaign or across different campaigns.
                if well_id not in well_id_set:
                    msg = f"Well: {well_id} is included in multiple clusters (campaign candidates)."
                    raise_exception(msg, ValueError)
                well_id_set.remove(well_id)
                # Remove a well_id from the well_id_set the first time it is checked. If the well_id
                # appears a second time and has already been removed from the unique well_id_set,
                # an error message on line 188 will be raised.
                # Each cluster represents a campaign candidate.

            max_wells = max(max_wells, len(campaign.well_dict))

        # Upon iterating through all campaigns and well_ids within those campaigns, all
        # well_ids in well_id_set should be dropped. If not, it indicates that not all wells
        # are assigned to a cluster during the clustering step. An error message will be raised at
        #  Line 202.
        if well_id_set:
            msg = "The clustering is not exhaustive."
            raise_exception(msg, ValueError)
        for n_wells in range(1, max_wells + 1):
            cost = self.mobilization_cost.get(n_wells)
            if cost is None:
                msg = f"Mobilization cost for n_wells: {n_wells} is missing"
                raise_exception(msg, ValueError)

            if cost < 0:
                msg = (
                    f"Mobilization cost for campaign with n_wells: {n_wells}"
                    f" is negative: {cost}"
                )
                raise_exception(msg, ValueError)
