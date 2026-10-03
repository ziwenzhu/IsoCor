"""Test the graphical interface (skipped when tkinter or a display is not available)."""

import logging
import time
from pathlib import Path
import pytest

tk = pytest.importorskip("tkinter")
from tkinter import messagebox
import isocor
from isocor.ui import isocorgui

DATA_FILE = Path(isocor.__file__).parent / "data" / "Data_example.tsv"


@pytest.fixture
def gui(tmp_path, monkeypatch):
    """GUI using a temporary home folder, with dialogs replaced by recorders."""
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    dialogs = []
    answers = {"askyesno": True}
    for name in ("showerror", "showwarning", "showinfo"):
        monkeypatch.setattr(messagebox, name,
                            lambda *args, name=name, **kwargs: dialogs.append((name,) + args))

    def askyesno(*args, **kwargs):
        dialogs.append(("askyesno",) + args)
        return answers["askyesno"]
    monkeypatch.setattr(messagebox, "askyesno", askyesno)
    app = isocorgui.GUIinterface(master=root, home=str(tmp_path / "home"))
    app.dialogs, app.answers = dialogs, answers
    app.openDataFile(str(DATA_FILE))
    app.varOutputPath.set(str(tmp_path))
    root.update()
    yield app
    logging.getLogger().removeHandler(app.scroll_handler)
    root.destroy()


def wait_until_done(app, timeout=60):
    """Process Tk events until the worker thread has finished and its outcome is handled."""
    start = time.time()
    while app._thread is not None:
        assert time.time() - start < timeout, "the correction process did not finish"
        app.update()
        time.sleep(0.01)
    app.update()


def log_text(app):
    return app.logstream.get("1.0", tk.END)


def test_process(gui, tmp_path):
    """Results are saved, their location is logged, and the button is reset."""
    gui.start_process()
    assert gui.processButon.cget("text") == "Stop"
    wait_until_done(gui)
    out_file = tmp_path / "Data_example_res.tsv"
    assert out_file.is_file() and (tmp_path / "Data_example.log").is_file()
    assert str(out_file) in log_text(gui)
    assert gui.processButon.cget("text") == "Process"
    assert gui.dialogs == []


def test_missing_output_folder(gui, tmp_path):
    gui.varOutputPath.set(str(tmp_path / "missing"))
    gui.start_process()
    assert gui._thread is None
    assert gui.dialogs[0][0] == "showerror" and "does not exist" in gui.dialogs[0][2]


def test_overwrite_declined(gui, tmp_path):
    out_file = tmp_path / "Data_example_res.tsv"
    out_file.write_text("previous results")
    gui.answers["askyesno"] = False
    gui.start_process()
    assert gui._thread is None
    assert out_file.read_text() == "previous results"


def test_error_during_process(gui, tmp_path):
    """An error in the worker thread is shown, the button is reset and the log file is closed."""
    (tmp_path / "Data_example_res.tsv").mkdir()
    gui.start_process()
    wait_until_done(gui)
    assert [d[0] for d in gui.dialogs] == ["askyesno", "showerror"]
    assert gui.processButon.cget("text") == "Process"
    log_file = str(tmp_path / "Data_example.log")
    assert not any(getattr(h, "baseFilename", None) == log_file for h in logging.getLogger().handlers)


def test_stop(gui, tmp_path, monkeypatch):
    """Stop interrupts the process, and a new process cannot start before it has stopped."""
    factory = isocorgui.hr.MetaboliteCorrectorFactory

    def slow_factory(*args, **kwargs):
        time.sleep(0.05)
        return factory(*args, **kwargs)
    monkeypatch.setattr(isocorgui.hr, "MetaboliteCorrectorFactory", slow_factory)
    gui.start_process()
    worker = gui._thread
    gui.stop_process()
    assert str(gui.processButon.cget("state")) == "disabled"
    gui.start_process()
    assert gui._thread is worker
    wait_until_done(gui)
    assert gui._result[0] == "cancelled"
    assert not (tmp_path / "Data_example_res.tsv").exists()
    assert gui.processButon.cget("text") == "Process"


def test_invalid_purity(gui):
    gui.purityManager.tracer_purity[0].set("abc")
    gui.start_process()
    assert gui._thread is None
    assert "¹²C" in gui.dialogs[0][2] and "abc" in gui.dialogs[0][2]


def test_purity_entries_visible(gui):
    """For a tracer with many isotopes, the entry of the tracer isotope is scrolled into view."""
    tracer = "³⁶S"
    gui.isotopictracerCBB.set(tracer)
    gui.updatePurity(None)
    gui.update()
    manager = gui.purityManager
    assert manager.isotope_names[-1] == tracer
    assert [v.get() for v in manager.tracer_purity] == ["0", "0", "0", "0", "1"]
    assert len(manager.find_all()) == 1
    assert manager.cget("scrollregion")
    assert manager.yview()[1] == 1.0


def test_datafile_formula_ignored_at_low_resolution(gui):
    gui.R2.invoke()
    gui.formulaEntered.set("datafile")
    gui.R1.invoke()
    params = gui.getParameters()
    assert params["useformula"] and not params["HR"]


def test_data_preview(gui):
    assert list(gui.datatable["columns"]) == ["sample", "metabolite", "derivative", "isotopologue",
                                              "area", "resolution"]
    assert len(gui.datatable.get_children()) == 87
    assert gui.inputDataEntry.xview()[1] == 1.0
