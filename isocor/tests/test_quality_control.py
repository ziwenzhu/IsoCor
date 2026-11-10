"""Test the quality controls: unlabeled pattern, uncertainty and unconstrained correction."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import isocor as hrcor
from isocor.ui import isocorcli, isocordb

DATA_DIR = Path(hrcor.__file__).parent / "data"
HIGH_RES = {"resolution": 1e5, "mz_of_resolution": 400, "charge": 1}


@pytest.mark.parametrize("formula,tracer,kwargs", [
    ("C6H11O9P", "13C", {}),
    ("C6H11O9P", "13C", {"derivative_formula": "C6H16Si2", "tracer_purity": [0.01, 0.99]}),
    ("C3H7O6P", "18O", {}),
    ("C6H11O9P", "13C", HIGH_RES),
    ("C5H11NO2S2", "34S", HIGH_RES),
])
@pytest.mark.parametrize("correct_NA_tracer", [False, True])
def test_unlabeled_pattern(formula, tracer, kwargs, correct_NA_tracer):
    """The unlabeled pattern is the measurement of the unlabeled isotopologue with the
    tracer element at natural abundance, whatever the correction options."""
    corrector = hrcor.MetaboliteCorrectorFactory(formula, tracer, correct_NA_tracer=correct_NA_tracer, **kwargs)
    reference = hrcor.MetaboliteCorrectorFactory(formula, tracer, correct_NA_tracer=True, **kwargs)
    expected = reference.correction_matrix[:, 0] / reference.correction_matrix[:, 0].sum()
    pattern = corrector.get_unlabeled_pattern()
    assert np.allclose(pattern, expected, rtol=0, atol=1e-12)
    assert np.allclose(corrector.compare_to_unlabeled(list(pattern * 1e6)), 0, atol=1e-12)


def test_unlabeled_reveals_wrong_formula():
    """A missing derivative gives a perfect fit (null residuum) but is revealed by an unlabeled sample."""
    right = hrcor.MetaboliteCorrectorFactory("C6H11O9P", "13C", derivative_formula="C6H16Si2")
    wrong = hrcor.MetaboliteCorrectorFactory("C6H11O9P", "13C")
    measurement = list(right.get_unlabeled_pattern() * 1e6)
    _, _, residuum, _ = wrong.correct(measurement)
    assert np.abs(residuum).max() < 1e-6
    assert np.abs(right.compare_to_unlabeled(measurement)).max() < 1e-12
    assert np.abs(wrong.compare_to_unlabeled(measurement)).max() > 0.1


def test_compare_to_unlabeled_null_measurement():
    corrector = hrcor.MetaboliteCorrectorFactory("C3H7O6P", "13C")
    assert np.all(np.isnan(corrector.compare_to_unlabeled([0, 0, 0, 0])))


def test_unconstrained():
    """Unconstrained correction equals the constrained one when it is positive, and is unbiased otherwise."""
    corrector = hrcor.MetaboliteCorrectorFactory("C5H9NO4", "13C", derivative_formula="C9H24Si3")
    matrix = corrector.correction_matrix
    fractions = np.array([.5, .1, .1, .1, .1, .1])
    measurement = list(np.dot(matrix, fractions) * 1e6)
    area, fraction, enrichment = corrector.correct_unconstrained(measurement)
    constrained = corrector.correct(measurement)
    assert np.allclose(fraction, fractions, atol=1e-10)
    assert np.allclose(area, constrained[0], rtol=1e-6)
    assert enrichment == pytest.approx(constrained[3], abs=1e-8)
    # noise on absent isotopologues
    rng = np.random.RandomState(0)
    fractions = np.array([.9, 0., 0., 0., 0., .1])
    noiseless = np.dot(matrix, fractions) * 1e6
    enrichments = np.array([[corrector.correct(list(m))[3], corrector.correct_unconstrained(list(m))[2]]
                            for m in noiseless + rng.normal(0, 0.02 * noiseless.max(), (500, 6))])
    assert abs(enrichments[:, 1].mean() - 0.1) < 0.002
    assert enrichments[:, 0].mean() - 0.1 > 0.004


def test_uncertainty():
    """Propagated standard deviations agree with simulations."""
    corrector = hrcor.MetaboliteCorrectorFactory("C5H9NO4", "13C", derivative_formula="C9H24Si3",
                                                 correct_NA_tracer=True)
    noiseless = np.dot(corrector.correction_matrix, [.5, .1, .1, .1, .1, .1]) * 1e6
    rng = np.random.RandomState(0)
    simulated = []
    for measurement in noiseless + rng.normal(size=(4000, 6)) * np.sqrt(500**2 + (0.03 * noiseless)**2):
        _, fraction, enrichment = corrector.correct_unconstrained(list(measurement))
        simulated.append(list(fraction) + [enrichment])
    fraction_sd, enrichment_sd = corrector.get_uncertainty(list(noiseless), relative_sd=0.03, absolute_sd=500)
    assert np.allclose(np.std(simulated, axis=0), list(fraction_sd) + [enrichment_sd], rtol=0.1)
    # no measurement error, no uncertainty
    fraction_sd, enrichment_sd = corrector.get_uncertainty(list(noiseless))
    assert np.all(fraction_sd == 0) and enrichment_sd == 0
    with pytest.raises(ValueError):
        corrector.get_uncertainty(list(noiseless), relative_sd=-1)


def run_cli(capsys, options):
    """Run the command line interface on the example data and return its results."""
    args = [str(DATA_DIR / "Data_example.tsv"), "-t", "13C"]
    for name in ("Metabolites", "Derivatives", "Isotopes"):
        args += ["-" + name[0], str(DATA_DIR / (name + ".dat"))]
    root_logger = logging.getLogger()
    handlers = list(root_logger.handlers)
    try:
        isocorcli.process(isocorcli.parseArgs().parse_args(args + options))
    finally:
        for handler in root_logger.handlers[:]:
            if handler not in handlers:
                root_logger.removeHandler(handler)
    captured = capsys.readouterr()
    lines = [line.split("\t") for line in captured.out.strip().split("\n")]
    return pd.DataFrame(lines[1:], columns=lines[0]), captured.err


def test_cli_quality_options(capsys):
    """Optional columns are added to the results only when requested."""
    default, _ = run_cli(capsys, [])
    assert list(default.columns[-2:]) == ["residuum", "mean_enrichment"]
    results, log = run_cli(capsys, ["--relative_sd", "0.02", "--unconstrained", "-u", "Sample_1"])
    assert list(results.columns[-4:]) == isocordb.extra_columns(relative_sd=0.02, unconstrained=True)
    assert results[default.columns].equals(default)
    assert "unlabeled control" in log
    # standard deviations are provided, except for metabolites without signal
    measured = results["corrected_area"].astype(float) > 0
    assert len(results) == 87 and (results.loc[measured, "isotopologue_fraction_sd"].astype(float) > 0).all()


def test_cli_unknown_unlabeled_sample(capsys):
    with pytest.raises(ValueError, match="not found"):
        run_cli(capsys, ["-u", "unknown_sample"])
