import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from tkinter import scrolledtext
from tkinter import filedialog
from tkinter import messagebox
from isocor.ui.isocordb import EnvComputing, results_dataframe
import logging
import pandas as pd
import isocor as hr
from pathlib import Path
import numpy as np
import re
import math
import queue
import webbrowser
import threading
import urllib.request

UTF8_TABLE_SUBCRIPS_INT = {'0': '\u2070', '1': '\u00B9', '2': '\u00B2', '3': '\u00B3',
                           '4': '\u2074', '5': '\u2075', '6': '\u2076', '7': '\u2077', '8': '\u2078', '9': '\u2079'}

# m/z passed to the correctors when the resolution formula does not use it
# ('constant' and 'datafile' formulas); the factory requires a positive value.
UNUSED_MZ_OF_RESOLUTION = 400.0


class Tooltip:
    """
    It creates a tooltip for a given widget as the mouse goes on it.

    see:

    http://stackoverflow.com/questions/3221956/
           what-is-the-simplest-way-to-make-tooltips-
           in-tkinter/36221216#36221216

    http://www.daniweb.com/programming/software-development/
           code/484591/a-tooltip-class-for-tkinter

    - Originally written by vegaseat on 2014.09.09.

    - Modified to include a delay time by Victor Zaccardo on 2016.03.25.

    - Modified
        - to correct extreme right and extreme bottom behavior,
        - to stay inside the screen whenever the tooltip might go out on
          the top but still the screen is higher than the tooltip,
        - to use the more flexible mouse positioning,
        - to add customizable background color, padding, waittime and
          wraplength on creation
      by Alberto Vassena on 2016.11.05.
    """

    def __init__(self, widget,
                 *,
                 bg='#FFFFEA',
                 pad=(5, 3, 5, 3),
                 text='widget info',
                 waittime=400,
                 wraplength=250):

        self.waittime = waittime  # in miliseconds, originally 500
        self.wraplength = wraplength  # in pixels, originally 180
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<ButtonPress>", self.on_leave)
        self.bg = bg
        self.pad = pad
        self.id = None
        self.tw = None

    def on_enter(self, event=None):
        self.schedule()

    def on_leave(self, event=None):
        self.unschedule()
        self.hide()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.show)

    def unschedule(self):
        id_ = self.id
        self.id = None
        if id_:
            self.widget.after_cancel(id_)

    def show(self):
        def tip_pos_calculator(widget, label,
                               *,
                               tip_delta=(10, 5), pad=(5, 3, 5, 3)):

            w = widget

            s_width, s_height = w.winfo_screenwidth(), w.winfo_screenheight()

            width, height = (pad[0] + label.winfo_reqwidth() + pad[2],
                             pad[1] + label.winfo_reqheight() + pad[3])

            mouse_x, mouse_y = w.winfo_pointerxy()

            x1, y1 = mouse_x + tip_delta[0], mouse_y + tip_delta[1]
            x2, y2 = x1 + width, y1 + height

            x_delta = x2 - s_width
            if x_delta < 0:
                x_delta = 0
            y_delta = y2 - s_height
            if y_delta < 0:
                y_delta = 0

            offscreen = (x_delta, y_delta) != (0, 0)

            if offscreen:

                if x_delta:
                    x1 = mouse_x - tip_delta[0] - width

                if y_delta:
                    y1 = mouse_y - tip_delta[1] - height

            offscreen_again = y1 < 0  # out on the top

            if offscreen_again:
                # No further checks will be done.

                # TIP:
                # A further mod might automagically augment the
                # wraplength when the tooltip is too high to be
                # kept inside the screen.
                y1 = 0

            return x1, y1

        bg = self.bg
        pad = self.pad
        widget = self.widget

        # creates a toplevel window
        self.tw = tk.Toplevel(widget)

        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)

        win = tk.Frame(self.tw,
                       background=bg,
                       borderwidth=0)
        label = tk.Label(win,
                         text=self.text,
                         justify=tk.LEFT,
                         background=bg,
                         relief=tk.SOLID,
                         borderwidth=0,
                         wraplength=self.wraplength)

        label.grid(padx=(pad[0], pad[2]),
                   pady=(pad[1], pad[3]),
                   sticky=tk.NSEW)
        win.grid()

        x, y = tip_pos_calculator(widget, label)

        self.tw.wm_geometry("+%d+%d" % (x, y))

    def hide(self):
        tw = self.tw
        if tw:
            tw.destroy()
        self.tw = None


class ProcessCancelled(Exception):
    """Raised in the worker thread when the user stops the correction process."""
    pass


class TextHandler(logging.Handler):
    """This class allows you to log to a Tkinter Text or ScrolledText widget

    Records may be emitted from any thread: they are queued, and written to the widget
    by :py:meth:`~flush_to_widget`, which must be called from the Tk main loop
    (Tk widgets must only be used from the main thread).
    """

    def __init__(self, text):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        # Store a reference to the Text it will log to
        self.text = text
        self.queue = queue.Queue()

    def emit(self, record):
        self.queue.put(self.format(record))

    def flush_to_widget(self):
        """Write the queued messages to the widget."""
        lines = []
        while True:
            try:
                lines.append(self.queue.get_nowait())
            except queue.Empty:
                break
        if lines:
            self.text.configure(state='normal')
            self.text.insert(tk.END, '\n'.join(lines) + '\n')
            self.text.configure(state='disabled')
            # Autoscroll to the bottom
            self.text.yview(tk.END)


class PurityTracerManager(tk.Canvas):
    def __init__(self, master=None, **kwargs):
        tk.Canvas.__init__(self, master, **kwargs)
        self.tracer_purity = []
        self.isotope_names = []
        self._entries = []
        self._bind_wheel(self)

    def _initFrameInWindows(self):
        self.frame = ttk.Frame(self)
        self.create_window((0, 0), anchor="nw", window=self.frame)
        self._bind_wheel(self.frame)

    def _bind_wheel(self, widget):
        """Scroll the entries with the mouse wheel (Windows/macOS and X11 events)."""
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(event, self._on_wheel)

    def _on_wheel(self, event):
        if event.num == 4 or getattr(event, 'delta', 0) > 0:
            self.yview_scroll(-1, "units")
        else:
            self.yview_scroll(1, "units")

    def show_entry(self, index):
        """Scroll the canvas so that the entry at the given index is visible."""
        self.update_idletasks()
        total_height = self.frame.winfo_reqheight()
        if not self._entries or total_height <= 0:
            return
        entry = self._entries[index]
        top, bottom = entry.winfo_y(), entry.winfo_y() + entry.winfo_reqheight()
        # before the window is shown, use the requested height of the canvas
        view_height = self.winfo_height() if self.winfo_ismapped() else int(self.cget('height'))
        view_top = self.canvasy(0)
        if top < view_top:
            self.yview_moveto(top / total_height)
        elif bottom > view_top + view_height:
            self.yview_moveto((bottom - view_height) / total_height)

    def changeEntries(self, df, tracer):
        self.tracer_purity = []
        self.isotope_names = []
        self._entries = []
        # canvas clear all method
        self.delete("all")
        self._initFrameInWindows()
        purity = (df['subscriptName'] ==
                  tracer.iloc[0]['subscriptName']).astype(int)
        tracer_row = 0
        for row, entry in enumerate(df.itertuples()):
            purityentry = tk.StringVar()
            purityentry.set(str(purity[entry.Index]))
            label_entry = ttk.Label(self.frame, text=entry.subscriptName)
            label_entry.grid(row=row, column=0, sticky="news")
            purity_entry = ttk.Entry(self.frame, textvariable=purityentry)
            purity_entry.grid(row=row, column=1, sticky="news")
            # keep the entry visible when it is reached with the keyboard
            purity_entry.bind("<FocusIn>", lambda event, i=row: self.show_entry(i))
            self._bind_wheel(label_entry)
            self._bind_wheel(purity_entry)
            self.tracer_purity.append(purityentry)
            self.isotope_names.append(entry.subscriptName)
            self._entries.append(purity_entry)
            if purity[entry.Index]:
                tracer_row = row
        self.frame.update_idletasks()
        self.config(scrollregion=self.bbox("all"))
        self.yview_moveto(0)
        # the entry of the tracer isotope must be visible
        self.show_entry(tracer_row)


class GUIinterface(ttk.Frame):
    """GUI interface for isocor in tk widget"""

    def __init__(self, master=None, home=None):
        super().__init__(master)
        self.pack(fill='both', expand=True)
        self.baseenv = EnvComputing() if home is None else EnvComputing(home)
        # check the database files exists in the default path,
        # otherwise copy them from the example folder
        self.baseenv.initializeDB()
        # isotope database should be loaded at init and not changed afterwards
        isotopesfile = Path(self.baseenv.db_path, "Isotopes.dat")
        try:
            self.baseenv.registerIsopotes(isotopesfile)
        except Exception as err:
            messagebox.showerror("Error", err)
            raise SystemExit(1)

        self.addSubcriptingName()
        self.cleanListTracer()
        self.log_level = 'INFO'
        self.createWidgets()
        # worker thread running the correction, event used to stop it, and its outcome
        self._thread, self._stop_event, self._result = None, threading.Event(), None
        self._poll()

    def start_process(self):
        """Check the parameters, then run the correction in a worker thread."""
        if self._thread is not None:
            return
        params = self.getParameters()
        if params is None:
            return
        self.cleanLog()
        self._stop_event = threading.Event()
        self._result = None
        self._thread = threading.Thread(target=self._run_process, args=(params,), daemon=True)
        self._thread.start()
        self.processButon.configure(text="Stop", command=self.stop_process)

    def stop_process(self):
        """Ask the worker thread to stop as soon as possible."""
        if self._thread is not None:
            self._stop_event.set()
            self.processButon.configure(text="Stopping...", state='disabled')

    def _check_stop(self):
        if self._stop_event.is_set():
            raise ProcessCancelled()

    def _run_process(self, params):
        """Worker thread: run the correction and store its outcome (no Tk calls here)."""
        try:
            self._result = ('done', self.process(params))
        except ProcessCancelled:
            self.logger.warning("Process stopped by the user. No results were saved.")
            self._result = ('cancelled', None)
        except Exception as err:
            self.logger.error("Process failed: {}".format(err))
            self._result = ('error', str(err))

    def _poll(self):
        """Periodically display the logs and handle the end of the worker thread (main thread)."""
        self.scroll_handler.flush_to_widget()
        if self._thread is not None and not self._thread.is_alive():
            self._thread = None
            self.processButon.configure(text="Process", command=self.start_process, state='normal')
            status, message = self._result if self._result else ('error', 'Unexpected error.')
            if status == 'error':
                messagebox.showerror("Error", "The correction process failed:\n\n{}\n\nSee the logs for details.".format(
                    message))
        self.after(100, self._poll)

    def addSubcriptingName(self):
        self.baseenv.dfIsotopes['subscriptName'] = self.baseenv.dfIsotopes['isotope'].apply(
            self.subscriptingInt) + self.baseenv.dfIsotopes['element']

    def subscriptingInt(self, myint):
        return ''.join([UTF8_TABLE_SUBCRIPS_INT[i] for i in str(myint)])

    @staticmethod
    def _positive_float(value):
        """Return value as a positive float, or None if it is not a positive number."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def getParameters(self):
        """Read and check the correction parameters (main thread).

        Returns:
            dict: correction parameters, or None if a parameter is invalid (an error is shown)
        """
        params = {}
        params['tracer'] = self.baseenv.dfIsotopes[self.baseenv.dfIsotopes['subscriptName']
                                                   == self.isotopictracerCBB.get()]['name'].values[0]
        params['correct_NA_tracer'] = bool(self.chVarNatAbTracer.get())
        params['data_isotopes'] = self.baseenv.dictIsotopes
        # check critical parameters and cancel processing if errors
        tracer_purity = []
        for name, var in zip(self.purityManager.isotope_names, self.purityManager.tracer_purity):
            try:
                value = float(var.get())
            except ValueError:
                value = None
            if value is None or not (0 <= value <= 1):
                messagebox.showerror("Error", "Invalid purity value for {}: '{}'.\n\nPurity values should be"
                                              " numbers within the range [0, 1].".format(name, var.get()))
                return None
            tracer_purity.append(value)
        if not math.isclose(math.fsum(tracer_purity), 1, abs_tol=hr.LowResMetaboliteCorrector.SUM_TOLERANCE):
            messagebox.showerror("Error", "Purity values sum to {:g}, but their sum should be 1.".format(
                math.fsum(tracer_purity)))
            return None
        params['tracer_purity'] = tracer_purity

        params['HR'] = bool(self.chVarHR.get())
        params['resolution_formula_code'] = self.formulaEntered.get() if params['HR'] else None
        params['useformula'] = params['resolution_formula_code'] != 'datafile'
        params['resolution'], params['mz_of_resolution'] = None, None
        if params['HR']:
            # only check the fields used by the selected formula (the others are disabled)
            if params['useformula']:
                params['resolution'] = self._positive_float(self.varMass.get())
                if params['resolution'] is None:
                    messagebox.showerror("Error", "Resolution should be a positive number.")
                    return None
            if params['resolution_formula_code'] in ('constant', 'datafile'):
                params['mz_of_resolution'] = UNUSED_MZ_OF_RESOLUTION
            else:
                params['mz_of_resolution'] = self._positive_float(self.varMZ.get())
                if params['mz_of_resolution'] is None:
                    messagebox.showerror("Error", "mz at which resolution is measured should be a positive number.")
                    return None

        params['database_path'] = self.varDatabasePath.get()
        params['input_file'] = self.varInputPath.get()
        if not params['input_file'] or not Path(params['input_file']).is_file():
            messagebox.showerror("Error", "Please load a measurements file first.")
            return None
        output_dir = Path(self.varOutputPath.get())
        if not output_dir.is_dir():
            messagebox.showerror("Error", "The output folder does not exist:\n'{}'.\n\nPlease select another"
                                          " folder with 'Output Data Path'.".format(output_dir))
            return None
        fin_base = Path(params['input_file']).stem
        params['out_file'] = output_dir.joinpath(fin_base + '_res.tsv')
        params['log_file'] = output_dir.joinpath(fin_base + '.log')
        if params['out_file'].exists():
            overwrite = messagebox.askyesno(
                "Overwrite results?",
                "The results file already exists:\n'{}'.\n\nOverwrite it (and its log file)?".format(
                    params['out_file']))
            if not overwrite:
                return None
        params['log_level'] = self.log_level
        return params

    def process(self, params):
        """Run the correction (worker thread): must not use any Tk widget.

        Returns:
            Path: the results file
        """
        derivativesfile = Path(params['database_path'], "Derivatives.dat")
        self.baseenv.registerDerivativesDB(derivativesfile)
        metabolitesfile = Path(params['database_path'], "Metabolites.dat")
        self.baseenv.registerMetabolitesDB(metabolitesfile)
        useformula = params['useformula']
        input_file = params['input_file']
        self.baseenv.registerDatafile(input_file, useformula)
        tracer = params['tracer']
        correct_NA_tracer = params['correct_NA_tracer']
        tracer_purity = params['tracer_purity']
        data_isotopes = params['data_isotopes']
        resolution = params['resolution']
        mz_of_resolution = params['mz_of_resolution']
        resolution_formula_code = params['resolution_formula_code']

        # add a filehandler to the logger (to redirect logs to a file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
        log_file = params['log_file']
        file_handler = logging.FileHandler(str(log_file), mode='w+')
        file_handler.setLevel(params['log_level'])
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        try:
            # log general information on the process
            self.logger.info('------------------------------------------------')
            self.logger.info("Correction process")
            self.logger.info('------------------------------------------------')
            self.logger.info("   data files")
            self.logger.info("      data file: {}".format(input_file))
            self.logger.info("      derivatives database: {}".format(derivativesfile))
            self.logger.info("      metabolites database: {}".format(metabolitesfile))
            self.logger.info("   correction parameters")
            self.logger.info("      isotopic tracer: {}".format(tracer))
            self.logger.info("      correct natural abundance of the tracer element: {}".format(correct_NA_tracer))
            self.logger.info("      isotopic purity of the tracer: {}".format(tracer_purity))
            if params['HR']:
                self.logger.info("      mode: high-resolution")
                self.logger.info("         formula code: {}".format(resolution_formula_code))
                if useformula:
                    self.logger.info("         instrument resolution: {}".format(resolution))
                if resolution_formula_code not in ['datafile', 'constant']:
                    self.logger.info("         at mz: {}".format(mz_of_resolution))
            else:
                self.logger.info("      mode: low-resolution")
            self.logger.info("   natural abundance of isotopes")
            self.logger.info("   {}".format(data_isotopes))
            self.logger.info("   IsoCor version: {}".format(hr.__version__))

            # initialize error dict
            errors = {'labels': [], 'measurements': []}

            # construct correctors for all (metabolite, derivative)
            labels = self.baseenv.getLabelsList(useformula)
            self.logger.info('------------------------------------------------')
            self.logger.info('Constructing correctors for all (metabolite, derivative)...')
            self.logger.info('------------------------------------------------')
            dictMetabolites = {}
            for label in labels:
                self._check_stop()
                try:
                    self.logger.debug("constructing {}...".format(label))
                    if params['HR']:
                        if not useformula:
                            resolution = label[2]
                            resolution_formula_code = 'constant'
                        dictMetabolites[label] = hr.MetaboliteCorrectorFactory(
                            formula=self.baseenv.getMetaboliteFormula(label[0]), tracer=tracer, resolution=resolution,
                            label=label[0],
                            data_isotopes=data_isotopes, mz_of_resolution=mz_of_resolution,
                            derivative_formula=self.baseenv.getDerivativeFormula(label[1]), tracer_purity=tracer_purity,
                            correct_NA_tracer=correct_NA_tracer, resolution_formula_code=resolution_formula_code,
                            charge=self.baseenv.getMetaboliteCharge(label[0]),
                            inchi=self.baseenv.getMetaboliteInChI(label[0]))
                    else:
                        dictMetabolites[label] = hr.MetaboliteCorrectorFactory(
                            formula=self.baseenv.getMetaboliteFormula(label[0]), tracer=tracer, label=label[0],
                            data_isotopes=data_isotopes,
                            derivative_formula=self.baseenv.getDerivativeFormula(label[1]), tracer_purity=tracer_purity,
                            correct_NA_tracer=correct_NA_tracer, inchi=self.baseenv.getMetaboliteInChI(label[0]))
                    self.logger.info("{} successfully constructed.".format(label))
                except Exception as err:
                    dictMetabolites[label] = None
                    errors['labels'] = errors['labels'] + [label]
                    self.logger.error("cannot construct {}: {}".format(label, err))

            # correct measurements for naturally occuring isotopes
            # note: the correction matrix is constructed only once (at first correction of a (metabolite, derivative))
            self.logger.info('------------------------------------------------')
            self.logger.info('Correcting raw MS data...')
            self.logger.info('------------------------------------------------')
            rows, index = [], []
            for label in labels:
                metabo = dictMetabolites[label]
                series, series_err = self.baseenv.getDataSerie(label, useformula)
                for s_err in series_err:
                    errors['measurements'] = errors['measurements'] + ["{} - {}".format(s_err, label)]
                    self.logger.error(
                        "{} - {}: Measurement vector is incomplete, some isotopologues are not provided.".format(s_err,
                                                                                                                 label))
                for serie in series:
                    self._check_stop()
                    if metabo:
                        try:
                            isotopic_inchi = metabo.isotopic_inchi
                            valuesCorrected = metabo.correct(serie[1])
                            self.logger.info("{} - {}: processed".format(serie[0], label))
                        except Exception as err:
                            isotopic_inchi = [''] * len(serie[1])
                            valuesCorrected = (
                            [np.nan] * len(serie[1]), [np.nan] * len(serie[1]), [np.nan] * len(serie[1]), np.nan)
                            self.logger.error("{} - {}: {}".format(serie[0], label, err))
                            errors['measurements'] = errors['measurements'] + ["{} - {}".format(serie[0], label)]
                    else:
                        isotopic_inchi = [''] * len(serie[1])
                        valuesCorrected = (
                        [np.nan] * len(serie[1]), [np.nan] * len(serie[1]), [np.nan] * len(serie[1]), np.nan)
                        errors['measurements'] = errors['measurements'] + ["{} - {}".format(serie[0], label)]
                        self.logger.error(
                            "{} - {}: (metabolite, derivative) corrector could not be constructed.".format(serie[0], label))

                    for i, line in enumerate(zip(*(serie[1], valuesCorrected[0], valuesCorrected[1], valuesCorrected[2],
                                                   [valuesCorrected[3]] * len(valuesCorrected[0])))):
                        rows.append(line)
                        index.append((serie[0], label[0], label[1], i, isotopic_inchi[i]))

            # save results
            df = results_dataframe(rows, index)
            out_file = params['out_file']
            df.to_csv(str(out_file), sep='\t')

            # summary results for logs
            self.logger.info('------------------------------------------------')
            self.logger.info("Correction process summary")
            self.logger.info('------------------------------------------------')
            self.logger.info("   number of samples: {}".format(len(self.baseenv.getSamplesList())))
            if useformula:
                self.logger.info("   number of (metabolite, derivative): {}".format(len(labels)))
            else:
                self.logger.info("   number of (metabolite, derivative, resolution): {}".format(len(labels)))
            nb_errors = len(errors['labels']) + len(errors['measurements'])
            self.logger.info("   errors: {}".format(nb_errors))
            if nb_errors:
                self.logger.info("      {} errors during construction of (metabolite, derivative) correctors".format(
                    len(errors['labels'])))
                self.logger.info("      {} errors during correction of measurements".format(len(errors['measurements'])))
                self.logger.info("      detailed information on errors are provided above.")
            self.logger.info("   results saved in: {}".format(out_file))
            self.logger.info("   log saved in: {}".format(log_file))
        finally:
            # remove filehandler from the logger, and close the file
            self.logger.removeHandler(file_handler)
            file_handler.close()
        return out_file

    def cleanLog(self):
        self.logstream.config(state="normal")
        self.logstream.delete(1.0, tk.END)
        self.logstream.config(state="disabled")

    def cleanData(self):
        self.datatable.delete(*self.datatable.get_children())
        self.datatable.configure(columns=())

    def cleanListTracer(self):
        self.cleanDfIsotopes = self.baseenv.dfIsotopes[(self.baseenv.dfIsotopes.abundance > 0) & (
                self.baseenv.dfIsotopes.abundance < 1.)]
        dfMaxAbundance = self.cleanDfIsotopes.groupby(
            "element", as_index=False)["abundance"].max()
        self.cleanDfIsotopes = self.cleanDfIsotopes[~(self.cleanDfIsotopes['abundance'].isin(
            dfMaxAbundance['abundance']) & self.cleanDfIsotopes['element'].isin(dfMaxAbundance['element']))]

    def loadData(self):
        "load data Callback"
        current = self.varInputPath.get()
        name = filedialog.askopenfilename(initialdir=str(Path(current).parent) if current else str(self.baseenv.home),
                                          filetypes=(
                                              ("Data File", "*.tsv"), ("All Files", "*.*")),
                                          title="Choose a file."
                                          )
        if name:
            self.openDataFile(name)

    def openDataFile(self, name):
        """Select the measurements file and show its content."""
        try:
            with open(name, 'r', encoding='utf-8') as fp:
                data = pd.read_csv(fp, delimiter='\t', dtype=str, keep_default_na=False)
        except Exception as err:
            messagebox.showerror("Error", "Cannot read the measurements file:\n'{}'.\n\n{}".format(name, err))
            return
        self.cleanLog()
        self.cleanData()
        self.varInputPath.set(name)
        self.varOutputPath.set(Path(name).parent)
        self.showData(data)

    def showData(self, data):
        """Show a table of measurements in the data preview."""
        columns = [str(c) for c in data.columns]
        self.datatable.configure(columns=columns)
        text_font = tkfont.nametofont('TkDefaultFont')
        heading_font = tkfont.nametofont('TkHeadingFont')
        for i, col in enumerate(columns):
            # column width fitted to the heading and to the first values
            width = max([heading_font.measure(col)] + [text_font.measure(v) for v in data.iloc[:, i].head(200)])
            self.datatable.heading(col, text=col, anchor='w')
            self.datatable.column(col, width=min(width + 16, 300), stretch=False, anchor='w')
        for values in data.itertuples(index=False):
            self.datatable.insert('', tk.END, values=list(values))

    def outputDir(self):
        "gui output path"
        path = filedialog.askdirectory()
        if path:
            self.varOutputPath.set(path)

    def databaseDir(self):
        "gui database path"
        path = filedialog.askdirectory()
        if path:
            self.varDatabasePath.set(path)
            self.update_DBpath()

    def enableHR(self):
        if self.chVarHR.get():
            for child in self.highResFrame.winfo_children():
                child.configure(state='normal')
            self.formulaEntered.configure(state='readonly')
        else:
            for child in self.highResFrame.winfo_children():
                child.configure(state='disabled')
        self.enableAtmz(None)

    def enableAtmz(self, event):
        if not self.chVarHR.get():
            self.mzEntry.configure(state='disabled')
            self.mzlbl.configure(state='disabled')
            self.masslbl.configure(state='disabled')
            self.massEntry.configure(state='disabled')
            self.formulalbl.configure(state='disabled')
            self.formulaEntered.configure(state="disabled")
        elif self.formulaEntered.get() == 'constant':
            self.mzEntry.configure(state='disabled')
            self.mzlbl.configure(state='disabled')
            self.masslbl.configure(state='normal')
            self.massEntry.configure(state='normal')
            self.formulalbl.configure(state='normal')
            self.formulaEntered.configure(state="readonly")
        elif self.formulaEntered.get() == 'datafile':
            self.mzEntry.configure(state='disabled')
            self.mzlbl.configure(state='disabled')
            self.masslbl.configure(state='disabled')
            self.massEntry.configure(state='disabled')
            self.formulalbl.configure(state='normal')
            self.formulaEntered.configure(state="readonly")
        else:
            self.mzEntry.configure(state='normal')
            self.mzlbl.configure(state='normal')
            self.masslbl.configure(state='normal')
            self.massEntry.configure(state='normal')
            self.formulalbl.configure(state='normal')
            self.formulaEntered.configure(state="readonly")

    def updatePurity(self, event):
        tracer = self.baseenv.dfIsotopes[self.baseenv.dfIsotopes['subscriptName']
                                         == self.isotopictracerCBB.get()]
        dfSymbol = self.baseenv.dfIsotopes[self.baseenv.dfIsotopes['element']
                                           == tracer.iloc[0]['element']]
        self.purityManager.changeEntries(df=dfSymbol, tracer=tracer)

    def updateLogLevel(self):
        if self.chVarVerboseLog.get():
            self.log_level = "DEBUG"
        else:
            self.log_level = "INFO"
        self.logger.setLevel(self.log_level)

    def update_DBpath(self):
        self.baseenv.db_path = Path(self.varDatabasePath.get())

    @staticmethod
    def _show_end_of_path(var, entry):
        """Keep the end of the path (i.e. the file name) visible in the entry."""
        var.trace_add('write', lambda *args: entry.after_idle(entry.xview_moveto, 1.0))
        entry.bind('<Configure>', lambda event: entry.xview_moveto(1.0))

    def createWidgets(self):
        content = ttk.Frame(self, padding=(3, 3, 12, 12))
        dataFrame = ttk.Frame(content, padding=(3, 3, 12, 12))
        optionFrame = ttk.Frame(content, padding=(3, 3, 12, 12))

        self.TracerOptFrame = ttk.LabelFrame(
            optionFrame, text='Tracer correction options')

        tr_lab = ttk.Label(text="Isotopic purity of the tracer (?)")
        purityLblFrame = ttk.LabelFrame(
            self.TracerOptFrame, labelwidget=tr_lab)
        self.scrollPurity = ttk.Scrollbar(purityLblFrame, orient='vertical')
        self.purityManager = PurityTracerManager(
            purityLblFrame, width=200, height=90, highlightthickness=0, yscrollcommand=self.scrollPurity.set)
        self.scrollPurity.config(command=self.purityManager.yview)

        tracer_list = list(self.cleanDfIsotopes['subscriptName'])
        self.isotopictracerEntered = tk.StringVar()
        self.isotopictracerlbl = ttk.Label(
            optionFrame, text="Isotopic tracer")
        self.isotopictracerCBB = ttk.Combobox(
            optionFrame, textvariable=self.isotopictracerEntered, values=tracer_list, state="readonly")
        self.isotopictracerCBB.bind("<<ComboboxSelected>>", self.updatePurity)
        # default value: 13C
        try:
            default_tracer = tracer_list.index(u"\u00B9\u00B3\u0043")
        except ValueError:
            default_tracer = 0
        self.isotopictracerCBB.current(default_tracer)
        self.updatePurity(None)

        self.chVarHR = tk.IntVar(value=0)
        self.R1 = ttk.Radiobutton(optionFrame, text="Low resolution (?)", variable=self.chVarHR, value=0,
                                  command=self.enableHR)
        self.R2 = ttk.Radiobutton(optionFrame, text="High resolution (?)", variable=self.chVarHR, value=1,
                                  command=self.enableHR)

        self.highResFrame = ttk.LabelFrame(
            optionFrame, text='High resolution parameters')
        self.formulalbl = ttk.Label(self.highResFrame, text="Resolution formula", )
        self.formulaEntered = ttk.Combobox(
            self.highResFrame, values=self.baseenv.formulas_code, state="readonly")
        self.formulaEntered.bind("<<ComboboxSelected>>", self.enableAtmz)
        self.formulaEntered.current(0)
        self.varMass = tk.StringVar()
        self.varMZ = tk.StringVar()
        self.varMass.set("60000")
        self.varMZ.set('400.0')
        self.masslbl = ttk.Label(self.highResFrame, text="instrument resolution")
        self.massEntry = ttk.Entry(
            self.highResFrame, textvariable=self.varMass)
        self.mzlbl = ttk.Label(self.highResFrame, text="at m/z")
        self.mzEntry = ttk.Entry(self.highResFrame, textvariable=self.varMZ)

        self.chVarVerboseLog = tk.IntVar()
        self.chVarNatAbTracer = tk.IntVar()
        self.chVerboseLog = ttk.Checkbutton(
            content, text="Verbose logs (?)", variable=self.chVarVerboseLog, command=self.updateLogLevel)
        self.processButon = ttk.Button(
            content, text=" Process ", command=self.start_process)
        self.chNatAbTracer = ttk.Checkbutton(
            self.TracerOptFrame, text="Correct natural abundance of the tracer element (?)",
            variable=self.chVarNatAbTracer)
        self.varInputPath = tk.StringVar()
        self.varOutputPath = tk.StringVar()
        self.varDatabasePath = tk.StringVar()
        self.inputDataEntry = ttk.Entry(
            dataFrame, textvariable=self.varInputPath, state='readonly')
        self.loadbutton = ttk.Button(
            dataFrame, text=" Load Data ", command=self.loadData)
        self.outputDataEntry = ttk.Entry(
            dataFrame, textvariable=self.varOutputPath, state='readonly')
        self.outputPathSubmit = ttk.Button(
            dataFrame, text=" Output Data Path ", command=self.outputDir)
        self.databaseEntry = ttk.Entry(
            dataFrame, textvariable=self.varDatabasePath, state='readonly')
        self.databasePathSubmit = ttk.Button(
            dataFrame, text=" Databases Path (?)", command=self.databaseDir)
        for var, entry in ((self.varInputPath, self.inputDataEntry), (self.varOutputPath, self.outputDataEntry),
                           (self.varDatabasePath, self.databaseEntry)):
            self._show_end_of_path(var, entry)
        self.varOutputPath.set(self.baseenv.home)
        self.varDatabasePath.set(self.baseenv.default_db)
        scrolH = 10
        # preview of the measurements file, as a table
        # fixed requested size: the window must not grow with the number of columns of the table
        previewFrame = ttk.Frame(dataFrame, width=330, height=190)
        previewFrame.grid_propagate(False)
        self.datatable = ttk.Treeview(previewFrame, show='headings', height=8, selectmode='none')
        dataYScroll = ttk.Scrollbar(previewFrame, orient='vertical', command=self.datatable.yview)
        dataXScroll = ttk.Scrollbar(previewFrame, orient='horizontal', command=self.datatable.xview)
        self.datatable.configure(yscrollcommand=dataYScroll.set, xscrollcommand=dataXScroll.set)
        self.logstream = scrolledtext.ScrolledText(
            content, height=scrolH, wrap=tk.WORD, state="disabled")

        for child in self.highResFrame.winfo_children():
            child.configure(state='disabled')

        # layout: the data preview and the logs grow with the window
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        content.rowconfigure(2, weight=1)
        dataFrame.columnconfigure(0, weight=1)
        dataFrame.rowconfigure(2, weight=1)
        previewFrame.columnconfigure(0, weight=1)
        previewFrame.rowconfigure(0, weight=1)
        content.grid(column=0, row=0, sticky='NSWE')
        dataFrame.grid(column=0, row=0, sticky='NSWE')
        optionFrame.grid(column=1, row=0, sticky='NSW')
        self.isotopictracerlbl.grid(column=0, row=0, sticky='NW')
        self.isotopictracerCBB.grid(column=0, row=1, sticky='NW')
        self.R1.grid(column=0, row=2, sticky='NW')
        self.R2.grid(column=0, row=2, sticky='NE')
        self.TracerOptFrame.grid(column=0, row=4, sticky='NWE', )
        self.highResFrame.grid(column=0, row=3, sticky='NWE', )
        self.formulalbl.grid(column=0, row=0, sticky='NW')
        self.formulaEntered.grid(column=0, row=1, sticky='NW')
        self.masslbl.grid(column=0, row=2, sticky='NW')
        self.massEntry.grid(column=0, row=3, sticky='NW')
        self.mzlbl.grid(column=1, row=2, sticky='NW')
        self.mzEntry.grid(column=1, row=3, sticky='NW')
        purityLblFrame.grid(column=0, row=5, sticky='NW')
        self.purityManager.grid(column=0, row=1, sticky='NW')
        self.scrollPurity.grid(column=1, row=1, sticky='NS')
        self.chNatAbTracer.grid(column=0, row=0, sticky='NW')
        self.inputDataEntry.grid(column=0, row=0, sticky='NWE')
        self.loadbutton.grid(column=0, row=1, sticky='NWE')
        self.outputDataEntry.grid(column=0, row=3, sticky='NWE')
        self.outputPathSubmit.grid(column=0, row=4, sticky='NWE')
        self.databaseEntry.grid(column=0, row=5, sticky='NWE')
        self.databasePathSubmit.grid(column=0, row=6, sticky='NWE')
        previewFrame.grid(column=0, row=2, sticky='NSWE')
        self.datatable.grid(column=0, row=0, sticky='NSWE')
        dataYScroll.grid(column=1, row=0, sticky='NS')
        dataXScroll.grid(column=0, row=1, sticky='WE')
        self.processButon.grid(column=0, row=1, columnspan=2, sticky='NWE')
        self.logstream.grid(column=0, row=2, columnspan=2, sticky='NSWE')
        self.chVerboseLog.grid(column=0, row=3, sticky='NW')
        ttk.Label(content, text="Hover over items marked (?) for help.").grid(column=1, row=3, sticky='NE')

        # create tooltip helpers
        Tooltip(self.chNatAbTracer,
                text="Correct for the contribution of naturally occurring isotopes of the tracer element at unlabeled positions. This only concerns the tracer element: natural abundance of other elements is always corrected.")
        Tooltip(self.R1, text="For measurements collected at unitary resolution (e.g. on quadrupole instruments).")
        Tooltip(self.R2,
                text="For measurements collected at high or ultrahigh resolution (e.g. on Orbitrap or FT-ICR instruments).")
        Tooltip(tr_lab,
                text="Correct for the contribution of isotopic impurities of the tracer at labeled positions. The isotopic purity is typically obtained from the manufacturer.\ne.g. for \u00B9\u00B3C-substrates with purity of 99%, use 0.01 for \u00B9\u00B2C and 0.99 for \u00B9\u00B3C.")
        Tooltip(self.chVerboseLog, text="Log more details. Useful in case of trouble: attach the log file when reporting an issue on GitHub.")
        Tooltip(self.databasePathSubmit, text="Folder containing all database files.")

        # create texthandler and formatter to display logs
        self.scroll_handler = TextHandler(self.logstream)
        formatter = logging.Formatter(
            '%(levelname)s - %(message)s')
        self.scroll_handler.setFormatter(formatter)

        # create logger (should be root to catch 'mscorrectors' logger)
        self.logger = logging.getLogger()
        self.updateLogLevel()

        # add the text handler to logger
        self.logger.addHandler(self.scroll_handler)


def openDoc():
    webbrowser.open_new(r"https://isocor.readthedocs.io/en/latest/")


def openGit():
    webbrowser.open_new(r"https://github.com/MetaSys-LISBP/IsoCor/")


def checkupdateto():
    """Return the latest IsoCor version available online, or None if it cannot be retrieved."""
    try:
        # Get the distant __init__.py and read its version as it done in setup.py
        response = urllib.request.urlopen("https://github.com/MetaSys-LISBP/IsoCor/raw/master/isocor/__init__.py",
                                          timeout=10)
        data = response.read()
        txt = data.decode('utf-8').rstrip()
        return re.findall(r"^__version__ = ['\"]([^'\"]*)['\"]", txt, re.M)[0]
    except Exception:
        return None  # silently ignore everything that just happened


def start_gui():
    root = tk.Tk()
    # create menu
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    filemenu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=filemenu)
    filemenu.add_command(label="Exit", command=root.quit)
    helpmenu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=helpmenu)
    helpmenu.add_command(label="IsoCor project", command=openGit)
    helpmenu.add_command(label="Documentation", command=openDoc)
    # start GUI
    app = GUIinterface(master=root)
    app.master.title("IsoCor {}".format(hr.__version__))
    # the window can be enlarged, but not made smaller than its content
    root.update_idletasks()
    root.minsize(root.winfo_reqwidth(), root.winfo_reqheight())
    # check version in a specific thread, and show the result from the main thread
    lastversion = []
    threading.Thread(target=lambda: lastversion.append(checkupdateto()), daemon=True).start()

    def showUpdate():
        if not lastversion:
            root.after(500, showUpdate)
        elif lastversion[0] is not None and lastversion[0] != hr.__version__:
            messagebox.showwarning('Version {} available'.format(lastversion[0]),
                                   'You can update IsoCor with:\n"pip install --upgrade isocor"\nCheck the documentation for more information.')
    root.after(500, showUpdate)
    app.mainloop()
