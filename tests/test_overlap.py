import numpy as np
import pytest

from spi_audit.overlap import annual_lag_overlap, annual_lag_overlap_table, overlap_table


def test_overlap_matches_professor_example():
    table = overlap_table(scales=[1, 12, 48])
    january_48 = table[(table.target.dt.month == 1) & (table.scale == 48)].iloc[0]
    december_48 = table[(table.target.dt.month == 12) & (table.scale == 48)].iloc[0]
    january_1 = table[(table.target.dt.month == 1) & (table.scale == 1)].iloc[0]

    assert january_48.known_months == 47
    assert np.isclose(january_48.known_fraction, 47 / 48)
    assert december_48.known_months == 36
    assert january_1.known_months == 0


def test_annual_lag_overlap_is_zero_up_to_twelve_months():
    """No desenho do artigo, escalas de até 12 meses não compartilham meses."""
    for scale in (1, 3, 6, 9, 12):
        assert annual_lag_overlap(scale) == 0.0
    assert np.isclose(annual_lag_overlap(18), 6 / 18)
    assert np.isclose(annual_lag_overlap(24), 0.5)
    assert np.isclose(annual_lag_overlap(48), 0.75)


def test_annual_lag_overlap_differs_from_known_fraction():
    """As duas medidas respondem a perguntas diferentes e não devem coincidir.

    Em 31/12/2014 o SPI-48 de janeiro/2015 já tem 47 de 48 meses conhecidos,
    mas a entrada da rede é janeiro/2014, cuja janela compartilha 36 meses.
    """
    known = overlap_table(scales=[48]).iloc[0]
    assert known.known_months == 47
    assert annual_lag_overlap(48) == 36 / 48


def test_annual_lag_overlap_table_and_validation():
    table = annual_lag_overlap_table(scales=[12, 24])
    assert table.shared_months.tolist() == [0, 12]
    with pytest.raises(ValueError):
        annual_lag_overlap(0)
    with pytest.raises(ValueError):
        annual_lag_overlap(24, lag_years=0)

