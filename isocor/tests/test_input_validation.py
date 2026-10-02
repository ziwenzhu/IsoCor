"""Test input validation and robustness of the correctors and of the databases."""

import math
from decimal import Decimal as D
import numpy as np
import pandas as pd
import pytest
import isocor as hrcor
from isocor.base import LabelledChemical
from isocor.ui.isocordb import EnvComputing


@pytest.mark.parametrize("purity", [[0.5, 0.9], [-0.1, 1.1], [0.1, 0.8]])
def test_invalid_tracer_purity(purity):
    """Tracer purity must contain probabilities that sum to 1."""
    with pytest.raises(ValueError):
        hrcor.MetaboliteCorrectorFactory("C3H7O6P", "13C", tracer_purity=purity)


def test_tracer_purity_float_rounding():
    """A purity that sums to 1 up to floating point errors is accepted."""
    purity = [0.6, 0.3, 0.1]
    assert sum(purity) != 1.
    hrcor.MetaboliteCorrectorFactory("C3H7O6P", "18O", tracer_purity=purity)


@pytest.mark.parametrize("formula", ["C3H7(OH)2", "C3H7O6.5", "c3h7", "C3H7O6P+", "C3H7-"])
def test_invalid_formula(formula):
    """Formulas that cannot be parsed entirely are rejected."""
    with pytest.raises(ValueError):
        hrcor.MetaboliteCorrectorFactory(formula, "13C")


def test_invalid_derivative_formula():
    with pytest.raises(ValueError):
        hrcor.MetaboliteCorrectorFactory("C3H7O6P", "13C", derivative_formula="Si(CH3)3")


def test_formula_whitespace():
    """Whitespace in formulas is ignored."""
    x = hrcor.MetaboliteCorrectorFactory(" C3 H7O6P ", "13C")
    assert dict(x.formula) == {"C": 3, "H": 7, "O": 6, "P": 1}


@pytest.mark.parametrize("formula,derivative", [("C3H7O6Xx", None), ("C3H7O6P", "C2Xx")])
def test_unknown_element(formula, derivative):
    """Elements without isotopic data are rejected at construction."""
    with pytest.raises(ValueError, match="Xx"):
        hrcor.MetaboliteCorrectorFactory(formula, "13C", derivative_formula=derivative)


def test_abundance_float_rounding():
    """Abundances that sum to 1 up to floating point errors are accepted."""
    abundance = [0.6779, 0.2239, 0.0982]
    assert math.fsum(abundance) != 1.
    data_iso = dict(LabelledChemical.DEFAULT_ISODATA)
    data_iso["O"] = {"abundance": abundance,
                     "mass": LabelledChemical.DEFAULT_ISODATA["O"]["mass"]}
    hrcor.MetaboliteCorrectorFactory("C3H7O6P", "13C", data_isotopes=data_iso)


def test_resolution_formula():
    """A custom resolution formula can be provided."""
    x = hrcor.MetaboliteCorrectorFactory("C3H7O6P", "13C", resolution=1e4, mz_of_resolution=400,
                                         charge=1, resolution_formula=lambda mw, res, at_mz: 0.01)
    assert x.correction_limit == pytest.approx(0.01)
    x = hrcor.HighResMetaboliteCorrector("C3H7O6P", "13C", 1e4, 400, "orbitrap", 1,
                                         resolution_formula=lambda mw, res, at_mz: 0.02)
    assert x.correction_limit == pytest.approx(0.02)


def test_direct_instantiation_defaults():
    """Correctors can be instantiated directly with default parameters."""
    x = hrcor.LowResMetaboliteCorrector("C3H7O6P", "13C")
    y = hrcor.MetaboliteCorrectorFactory("C3H7O6P", "13C")
    assert np.array_equal(x.correction_matrix, y.correction_matrix)


@pytest.mark.parametrize("resolution", [None, 1e5])
def test_float_masses(resolution):
    """Masses provided as floats give the same results as Decimal masses."""
    data_iso = {el: {"abundance": v["abundance"], "mass": [float(m) for m in v["mass"]]}
                for el, v in LabelledChemical.DEFAULT_ISODATA.items()}
    kwargs = {} if resolution is None else {"resolution": resolution,
                                            "mz_of_resolution": 400, "charge": 1}
    x = hrcor.MetaboliteCorrectorFactory("C6H12N2O4S2", "13C", data_isotopes=data_iso, **kwargs)
    y = hrcor.MetaboliteCorrectorFactory("C6H12N2O4S2", "13C", **kwargs)
    assert all(isinstance(m, D) for v in x.data_isotopes.values() for m in v["mass"])
    assert np.allclose(x.correction_matrix, y.correction_matrix, rtol=0, atol=1e-12)
    # user data must not be modified
    assert all(isinstance(m, float) for v in data_iso.values() for m in v["mass"])


@pytest.mark.parametrize("measurement", [[1, 2, np.nan, 4], [1, 2, np.inf, 4]])
def test_nonfinite_measurement(measurement):
    x = hrcor.MetaboliteCorrectorFactory("C3H7O6P", "13C")
    with pytest.raises(ValueError):
        x.correct(measurement)


def test_correction_vector_not_modified():
    """Building the correction matrix must not modify the correction vector in place."""
    x = hrcor.MetaboliteCorrectorFactory("C6", "13C", tracer_purity=[0.1, 0.9])
    assert x.get_mass_distribution_vector() == [1.]
    expected = np.zeros((7, 7))
    for i in range(7):
        for k in range(i + 1):
            expected[k, i] = math.factorial(i) // (math.factorial(k) * math.factorial(i - k)) * 0.9**k * 0.1**(i - k)
    assert np.allclose(x.correction_matrix, expected)


def test_isotopic_blocks_pruned():
    """Isotopic blocks below the probability threshold are removed, without changing results."""
    kwargs = {"resolution": 1e5, "mz_of_resolution": 400, "charge": 1}
    x = hrcor.MetaboliteCorrectorFactory("C21H27N7O14P2", "15N", **kwargs)
    assert x.threshold_p is not None
    blocks = x._get_isotopic_blocks()
    assert all(p > x.threshold_p for peaks in blocks.values() for _, p in peaks)
    y = hrcor.MetaboliteCorrectorFactory("C21H27N7O14P2", "15N", **kwargs)
    y.threshold_p = None
    assert sum(len(v) for v in y._get_isotopic_blocks().values()) > \
        sum(len(v) for v in blocks.values())
    assert np.allclose(x.correction_matrix, y.correction_matrix, rtol=0, atol=1e-9)


def test_strip_column_names(tmp_path):
    """Column names of the measurements file are stripped."""
    datafile = tmp_path / "data.tsv"
    datafile.write_text(" sample \tmetabolite\t derivative\tisotopologue\tarea \n"
                        "s1\tPEP\t\t0\t100\n", encoding="utf-8")
    env = EnvComputing(home=str(tmp_path))
    env.registerDatafile(datafile)
    assert list(env.dfDatafile.columns) == ["sample", "metabolite", "derivative",
                                            "isotopologue", "area"]
    df = pd.DataFrame({" a ": [1]})
    env._stripColNames(df)
    assert list(df.columns) == ["a"]
