"""Network-free test for the Fama-French CSV parser."""
from __future__ import annotations

import pandas as pd

from ovi.data.factors import parse_ff_factors_csv

FIXTURE = """This file was created by CMPT_ME_BEME ... (preamble)

,Mkt-RF,SMB,HML,RF
196307,  -0.39,  -0.41,  -0.97,   0.27
196308,   5.07,  -0.80,   1.80,   0.25
196309,  -1.57,  -0.48,   0.13,   0.27

 Annual Factors: January-December

,Mkt-RF,SMB,HML,RF
1963,   x,   x,   x,   x
"""


def test_parse_monthly_block_percent_to_decimal():
    df = parse_ff_factors_csv(FIXTURE)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 3  # only the three monthly rows; annual row excluded
    assert isinstance(df.index, pd.PeriodIndex)
    assert str(df.index[0]) == "1963-07"
    # percent -> decimal
    assert abs(df.loc[pd.Period("1963-08", "M"), "Mkt-RF"] - 0.0507) < 1e-9
    assert abs(df.loc[pd.Period("1963-07", "M"), "RF"] - 0.0027) < 1e-9
