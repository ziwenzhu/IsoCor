"""Test the non-negative least squares solver and the grouping of the measurements by label."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import isocor as hrcor
from isocor import mscorrectors
from isocor.ui.isocordb import EnvComputing

DATA_FILE = Path(hrcor.__file__).parent / "data" / "Data_example.tsv"
CORRECTORS = [("C6H11O9P", "13C", {}),
              ("C5H9NO4", "13C", {"derivative_formula": "C9H24Si3", "correct_NA_tracer": True,
                                  "tracer_purity": [0.01, 0.99]}),
              ("C10H16N5O13P3", "13C", {"resolution": 7e4, "mz_of_resolution": 400, "charge": 1}),
              ("C3H7O6P", "18O", {"tracer_purity": [0.01, 0.01, 0.98]})]


def optimality_error(matrix, measurement, solution):
    """Largest violation of the optimality (KKT) conditions of non-negative least squares,
    relative to the scale of the problem."""
    gradient = np.dot(matrix.T, np.dot(matrix, solution) - measurement)
    scale = np.linalg.norm(np.dot(matrix.T, measurement))
    return max(np.abs(gradient[solution > 0]).max(initial=0.),
               (-gradient[solution == 0]).max(initial=0.),
               (-solution).max(initial=0.)) / scale


@pytest.mark.parametrize("formula,tracer,kwargs", CORRECTORS)
def test_nnls_exact(formula, tracer, kwargs):
    """Corrected areas are the exact non-negative least squares solution, also with noisy data."""
    corrector = hrcor.MetaboliteCorrectorFactory(formula, tracer, **kwargs)
    matrix = corrector.correction_matrix
    rng = np.random.RandomState(0)
    for scale in (1., 1e6, 1e10):
        for _ in range(50):
            measurement = np.dot(matrix, rng.dirichlet(np.ones(matrix.shape[1]) * 0.3)) * scale
            measurement += rng.normal(0, 0.02 * measurement.max(), measurement.size)
            corrected_area = np.asarray(corrector.correct(list(measurement))[0])
            assert optimality_error(matrix, measurement, corrected_area) < 1e-12


def test_nnls_fallback(monkeypatch):
    """L-BFGS-B is used when nnls does not converge, with the same result."""
    corrector = hrcor.MetaboliteCorrectorFactory("C6H11O9P", "13C")
    measurement = list(np.dot(corrector.correction_matrix, [.5, .1, .1, .1, .1, .05, .05]) * 1e6)
    expected = corrector.correct(measurement)

    def not_converging(*args, **kwargs):
        raise RuntimeError("too many iterations")
    monkeypatch.setattr(mscorrectors, "nnls", not_converging)
    result = corrector.correct(measurement)
    assert np.allclose(result[1], expected[1], rtol=0, atol=1e-6)
    assert result[3] == pytest.approx(expected[3], abs=1e-6)


def data_series_by_scanning(env, label, useformula):
    """Measurements of a label as returned before getDataSerie used a lookup table."""
    length = 2 if useformula else 3
    series = []
    for i, j in env._groupbyDatafile:
        if i[:length] == label:
            series.append([i[length], list(j.sort_values(by=['isotopologue']).area.values)])
    return series


@pytest.mark.parametrize("useformula", [True, False])
def test_data_series(useformula):
    env = EnvComputing()
    env.registerDatafile(DATA_FILE, useformula)
    labels = env.getLabelsList(useformula)
    assert labels
    for label in labels:
        series, errors = env.getDataSerie(label, useformula)
        assert series == data_series_by_scanning(env, label, useformula) and not errors
    assert env.getDataSerie(("unknown", ""), useformula) == ([], [])


def test_data_series_new_file(tmp_path):
    """The measurements are those of the last registered file (the GUI reuses the environment)."""
    env = EnvComputing()
    env.registerDatafile(DATA_FILE)
    label = env.getLabelsList(True)[0]
    first = env.getDataSerie(label, True)[0]
    data = pd.read_csv(DATA_FILE, sep="\t")
    data["area"] = data["area"] * 2
    datafile = tmp_path / "data.tsv"
    data.to_csv(datafile, sep="\t", index=False)
    env.registerDatafile(datafile)
    second = env.getDataSerie(label, True)[0]
    assert [s[0] for s in second] == [s[0] for s in first]
    assert [s[1] for s in second] == [[2 * a for a in s[1]] for s in first]
