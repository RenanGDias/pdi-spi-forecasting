import numpy as np

from spi_audit.overlap import overlap_table


def test_overlap_matches_professor_example():
    table = overlap_table(scales=[1, 12, 48])
    january_48 = table[(table.target.dt.month == 1) & (table.scale == 48)].iloc[0]
    december_48 = table[(table.target.dt.month == 12) & (table.scale == 48)].iloc[0]
    january_1 = table[(table.target.dt.month == 1) & (table.scale == 1)].iloc[0]

    assert january_48.known_months == 47
    assert np.isclose(january_48.known_fraction, 47 / 48)
    assert december_48.known_months == 36
    assert january_1.known_months == 0

