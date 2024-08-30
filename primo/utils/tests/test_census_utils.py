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
import pytest

# User-defined libs
from primo.utils.census_utils import (
    get_block,
    get_block_group,
    get_county,
    get_fips_code,
    get_state,
    get_tract,
)


def test_get_state():
    assert get_state("53065") == "53"
    assert get_state("53065950101") == "53"

    with pytest.raises(ValueError):
        assert get_state("")


def test_get_county():
    assert get_county("53065") == "065"
    assert get_county("53065950101") == "065"

    with pytest.raises(ValueError):
        assert get_county("")
        assert get_county("53")


def test_get_tract():
    assert get_tract("53065950101") == "950101"
    assert get_tract("530659501012") == "950101"
    assert get_tract("530659501012022") == "950101"

    with pytest.raises(ValueError):
        assert get_tract("")
        assert get_tract("53")
        assert get_tract("53065")


def test_get_block_group():
    assert get_block_group("530659501012") == "2"
    assert get_block_group("530659501012022") == "2"

    with pytest.raises(ValueError):
        assert get_block_group("")
        assert get_block_group("53")
        assert get_block_group("53065")
        assert get_block_group("53065950101")


def test_get_block():
    assert get_block("530659501012022") == "022"

    with pytest.raises(ValueError):
        assert get_block("")
        assert get_block("53")
        assert get_block("53065")
        assert get_block("53065950101")
        assert get_block("530659501012")


def test_get_fips_code():
    assert get_fips_code(41, -76) == "420792166011027"
