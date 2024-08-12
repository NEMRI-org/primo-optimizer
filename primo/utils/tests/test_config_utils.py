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

# Installed lib
import pytest

# User defined lib
from primo.utils.config_utils import copy_dict, is_valid


@pytest.mark.parametrize(
    "input_dict,output_dict,result, status_ok",
    [
        (
            {
                "input_str": "orange",
                "input_float": 42.0,
                "input_dict": {"nested_str": "banana", "nested_int": 42},
            },  # input_dict
            {
                "input_str": "apple",
                "input_float": 99.0,
                "input_dict": {"nested_str": "banana", "nested_int": 83},
                "input_default": "default",
            },  # output_dict
            {
                "input_str": "orange",
                "input_float": 42.0,
                "input_dict": {"nested_str": "banana", "nested_int": 42},
                "input_default": "default",
            },  # result
            True,
        ),
        ({}, {}, {}, True),
        (
            {"unknown_input": 33, "known_input": "OK"},  # input_dict
            {"known_input": "AOK"},  # output_dict
            {},  # Irrelevant result
            False,
        ),
    ],
)
def test_copy_dict(input_dict, output_dict, result, status_ok):
    if status_ok:
        assert copy_dict(input_dict, output_dict) == result
    else:
        with pytest.raises(ValueError):
            copy_dict(input_dict, output_dict)


@pytest.mark.parametrize(
    "input_dict,reference_dict,result",
    [
        (
            {
                "input_str": "orange",
                "input_float": 42.0,
                "input_dict": {"nested_str": "banana", "nested_int": 42},
            },  # input_dict
            {
                "input_str": "apple",
                "input_float": 99.0,
                "input_dict": {"nested_str": "banana", "nested_int": 83},
                "input_default": "default",
            },  # reference_dict
            True,
        ),
        ({}, {}, True),
        (
            {"unknown_input": 33, "known_input": "OK"},  # input_dict
            {"known_input": "AOK"},  # reference_dict
            False,
        ),
        (
            {
                "input_str": "orange",
                "input_float": 42.0,
            },  # input_dict
            {
                "input_str": "orange",
                "input_float": 42.0,
                "input_dict": {"nested_str": "banana", "nested_int": 42},
            },  # reference_dict
            True,
        ),
        (
            {
                "input_str": "orange",
                "input_float": 42.0,
            },  # input_dict
            {
                "input_string": "orange",
                "input_float": 42.0,
                "input_dict": {"nested_str": "banana", "nested_int": 42},
            },  # reference_dict
            False,
        ),
    ],
)
def test_is_valid(input_dict, reference_dict, result):
    assert is_valid(input_dict, reference_dict) == result
