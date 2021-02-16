"""
Tests for the utility library
"""
import pytest

from modypy.utils.uiuc_db import load_static_propeller


@pytest.mark.parametrize(
    "path",
    [
        "volume-1/data/ance_8.5x6_static_2848cm.txt",
        "volume-1/data/apce_11x5.5_static_kt0467.txt",
        "volume-1/data/apce_9x4.5_static_rd0995.txt",
        "volume-1/data/apcsf_8x6_static_2783rd.txt",
        "volume-1/data/apcsp_10x7_static_2654rd.txt",
        "volume-1/data/apcsp_14x13_static_jb1048.txt",
        "volume-1/data/apcsp_9x10_static_2513rd.txt",
        "volume-1/data/grcp_9x4_static_rd0923.txt",
        "volume-1/data/grsn_11x7_static_2988os.txt",
        "volume-1/data/gwssf_10x8_static_jb0880.txt",
        "volume-1/data/kavfk_9x6_static_2592rd.txt",
        "volume-1/data/magf_10x8_static_pg0719.txt",
        "volume-1/data/mas_10x7_static_kt0737.txt",
        "volume-1/data/ma_11x7_static_rd0586.txt",
        "volume-2/data/apcff_9x4_static_1016rd.txt",
        "volume-2/data/da4002_5x4.92_static_1190ga.txt",
        "volume-2/data/da4022_9x6.75_static_0767rd.txt",
        "volume-2/data/da4052_9x6.75_static_1041rd.txt",
        "volume-2/data/gwsdd_4.5x4_static_0332rd.txt",
        "volume-2/data/mit_5x4_static_0362rd.txt",
        "volume-2/data/nr640_9_15deg_static_0777rd.txt",
        "volume-3/data/ancf_10x6_static_0729od.txt",
        "volume-3/data/ancf_125x75_static_0914od.txt",
        "volume-3/data/ancf_13x11_static_0814od.txt",
        "volume-3/data/ancf_15x8_static_0975od.txt",
    ]
)
def test_propellers(path):
    prop_data = load_static_propeller(path)
    assert prop_data is not None
