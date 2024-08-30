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
import os

# Installed libs
import pandas as pd
import pyomo.environ as pyo
import pytest

# User-defined libs
from primo.data_parser.input_parser import InputParser
from primo.opt_model.opt_model import OptModel
from primo.utils.setup_arg_parser import parse_args
from primo.utils.setup_logger import setup_logger

# expected selection
# 21, 6, 26,
expected_selection = [21, 6, 26]
# expected to not be selected
expected_not_be_selected = [23]


# Settings for sample
OIL_BUDGET = 20000000
VERBOSE_OUTPUT = 0
MAX_WELLS_PER_OWNER = 2
DAC_PERCENT = 50
DAC_WEIGHT = 80
PROJECT_BUDGET = 1000000

# Reading in sample data
screening_file = os.path.join("primo", "opt_model", "tests", "opt_toy_model.csv")
well_gdf = pd.read_csv(screening_file)

# Introducing sample mobilization cost scheme
mobilization_costs = {0: 0, 1: 12000, 2: 21000, 3: 28000, 4: 35000}
for n_wells in range(5, len(well_gdf)):
    mobilization_costs[n_wells] = n_wells * 8400

# Set up logger and sample arguments for opt model testing
args = parse_args(
    [
        "-f",
        screening_file,
        "-b",
        str(OIL_BUDGET),
        "-v",
        str(VERBOSE_OUTPUT),
        "-d",
        str(DAC_WEIGHT),
        "-owc",
        str(MAX_WELLS_PER_OWNER),
        "-df",
        str(DAC_PERCENT),
    ]
)
parser = InputParser(args)
opt_input = parser.parse_data(mobilization_costs)


@pytest.mark.parametrize(
    "opt_inputs,dac_percent_input,project_budget_input ",
    [
        (opt_input, DAC_PERCENT, PROJECT_BUDGET)
        # Add more test cases as needed
    ],
)
# Integration testing for building the optimization model
def test_build_opt_model(opt_inputs, dac_percent_input, project_budget_input):

    model = OptModel("PRIMO Model for Wells", opt_inputs)
    model.build_model(
        dac_budget_fraction=dac_percent_input, project_max_spend=project_budget_input
    )

    # Checking that each of the model components got built.
    assert isinstance(model.model.s_w, pyo.Set)
    assert isinstance(model.model.s_dw, pyo.Set)
    assert isinstance(model.model.s_cl, pyo.Set)
    assert isinstance(model.model.s_owc, pyo.Set)
    assert isinstance(model.model.p_B, pyo.Param)
    assert isinstance(model.model.p_B, pyo.Param)
    assert isinstance(model.model.p_v, pyo.Param)
    assert isinstance(model.model.p_owc, pyo.Param)
    assert isinstance(model.model.v_y, pyo.Var)
    assert isinstance(model.model.v_q, pyo.Var)
    assert isinstance(model.model.con_budget, pyo.Constraint)
    assert isinstance(model.model.owner_well_count, pyo.Constraint)
    assert isinstance(model.model.con_balanced_budgets, pyo.Constraint)
    assert isinstance(model.model.con_project_max_spend, pyo.Constraint)
    assert isinstance(model.model.obj, pyo.Objective)


@pytest.mark.parametrize(
    "opt_inputs,dac_percent_input, well_data_set,selected, not_selected",
    [
        (opt_input, DAC_PERCENT, well_gdf, expected_selection, expected_not_be_selected)
        # Add more test cases as needed
    ],
)

# Integration testing for solving the optimization model
def test_solve_opt_model(
    opt_inputs, dac_percent_input, well_data_set, selected, not_selected
):

    model = OptModel("PRIMO Model for Wells", opt_inputs)
    model.build_model(dac_budget_fraction=dac_percent_input)

    status = model.solve_model({"solver": "highs"})
    selected_wells = model.get_results()
    well_data_set["selected"] = well_data_set.apply(
        lambda row: "1" if row["Well ID"] in selected_wells else "0", axis=1
    )

    df_selected_wells = well_data_set[well_data_set["selected"] == "1"]

    # Make sure model reaches optimality
    assert status is True

    # Integration testing to make sure a proper solution was produced
    assert set(selected).issubset(set(df_selected_wells["Well ID"].tolist()))

    # Integration test to make sure max_owner_well count was successfully applied
    with pytest.raises(AssertionError):
        assert set(not_selected).issubset(set(df_selected_wells["Well ID"].tolist()))
