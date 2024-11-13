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

# Installed libs
from pyomo.common.config import (
    Bool,
    ConfigDict,
    ConfigValue,
    In,
    IsInstance,
    NonNegativeFloat,
)

# User-defined libs
from primo.data_parser.metric_data import EfficiencyMetrics, ImpactMetrics
from primo.utils.domain_validators import InRange


def data_config() -> ConfigDict:
    """
    Returns a Pyomo ConfigDict object that includes all user options
    associated with data processing
    """
    config = ConfigDict()
    config.declare(
        "census_year",
        ConfigValue(
            default=2020,
            domain=In(list(range(2020, 2101, 10))),
            doc="Year for collecting census data",
        ),
    )
    config.declare(
        "preliminary_data_check",
        ConfigValue(
            default=True, domain=Bool, doc="If True, performs preliminary data checks"
        ),
    )
    config.declare(
        "verify_operator_name",
        ConfigValue(
            default=True,
            domain=Bool,
            doc="Remove well if operator name is not provided",
        ),
    )
    config.declare(
        "missing_age",
        ConfigValue(
            default="fill",
            domain=In(["fill", "estimate", "remove"]),
            doc="Method for processing missing age information",
        ),
    )
    config.declare(
        "missing_depth",
        ConfigValue(
            default="fill",
            domain=In(["fill", "estimate", "remove"]),
            doc="Method for processing missing depth information",
        ),
    )
    config.declare(
        "fill_age",
        ConfigValue(
            default=100,
            # Assuming that no well is older than 350 years
            domain=InRange(0, 350),
            doc="Value to fill with, if the age is missing",
        ),
    )
    config.declare(
        "fill_depth",
        ConfigValue(
            default=1000,
            # Assuming that no well is deeper than 40,000 ft
            domain=InRange(0, 40000),
            doc="Value to fill with, if the depth is missing",
        ),
    )
    config.declare(
        "fill_well_type",
        ConfigValue(
            default="Oil",
            domain=In(["Oil", "Gas"]),
            doc="Well-type assumption if it is not specified",
        ),
    )
    config.declare(
        "fill_well_type_depth",
        ConfigValue(
            default="Deep",
            domain=In(["Deep", "Shallow"]),
            doc="Well-type (by depth) assumption if it is not specified",
        ),
    )
    config.declare(
        "fill_ann_gas_production",
        ConfigValue(
            default=0.0,
            domain=NonNegativeFloat,
            doc=(
                "Value to fill with, if the annual gas production "
                "[in Mcf/Year] is not specified"
            ),
        ),
    )
    config.declare(
        "fill_ann_oil_production",
        ConfigValue(
            default=0.0,
            domain=NonNegativeFloat,
            doc=(
                "Value to fill with, if the annual oil production "
                "[in bbl/Year] is not specified"
            ),
        ),
    )
    config.declare(
        "fill_life_gas_production",
        ConfigValue(
            default=0.0,
            domain=NonNegativeFloat,
            doc=(
                "Value to fill with, if the lifelong gas production [in Mcf] "
                "is not specified"
            ),
        ),
    )
    config.declare(
        "fill_life_oil_production",
        ConfigValue(
            default=0.0,
            domain=NonNegativeFloat,
            doc=(
                "Value to fill with, if the lifelong oil production [in bbl] "
                "is not specified"
            ),
        ),
    )
    config.declare(
        "threshold_gas_production",
        ConfigValue(
            domain=NonNegativeFloat,
            doc=(
                "If specified, wells whose lifelong gas production volume [in Mcf] is "
                "above the threshold production volume will be removed from the dataset"
            ),
        ),
    )
    config.declare(
        "threshold_oil_production",
        ConfigValue(
            domain=NonNegativeFloat,
            doc=(
                "If specified, wells whose lifelong oil production volume [in bbl] is "
                "above the threshold production volume will be removed from the dataset"
            ),
        ),
    )
    config.declare(
        "threshold_depth",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Threshold depth [in ft] for classifying a well as shallow or deep",
        ),
    )
    config.declare(
        "impact_metrics",
        ConfigValue(
            default=None,
            domain=IsInstance(ImpactMetrics),
            doc="Impact metrics for well priority ranking",
        ),
    )
    config.declare(
        "efficiency_metrics",
        ConfigValue(
            default=None,
            domain=IsInstance(EfficiencyMetrics),
            doc="Efficiency metrics for computing project efficiencies",
        ),
    )

    return config
