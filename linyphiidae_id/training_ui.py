import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .model import GenusClassifier
from .training import train


class TrainingPanel(ttk.Frame):
    def __init__(self, parent, app, root_directory):
        super().__init__(parent, padding=12)
        self.app = app
        self.messages = queue.Queue()
        self.cancel = threading.Event()
        self.running = False
        self.manifest = None
        self.columnconfigure(1, weight=1)
        self.rowconfigure(9, weight=1)
        self.dataset = tk.StringVar(value=str(root_directory / 'datasets' / 'linyphiidae'))
        self.inventory = tk.StringVar()
        self.output = tk.StringVar(value=str(root_directory / 'models' / 'linyphiidae'))
        for row, (title, value, is_file) in enumerate([
                ('Dataset folder', self.dataset, False), ('Reviewed inventory CSV', self.inventory, True),
                ('Model output folder', self.output, False)]):
            ttk.Label(self, text=title).grid(row=row, column=0, sticky='w', padx=(0, 12), pady=4)
            ttk.Entry(self, textvariable=value).grid(row=row, column=1, sticky='ew', pady=4)
            ttk.Button(self, text='Browse', command=lambda v=value, f=is_file: self.browse(v, f)).grid(row=row, column=2, padx=(8, 0))
        options = ttk.Frame(self)
        options.grid(row=3, column=0, columnspan=3, sticky='w', pady=10)
        self.epochs = tk.IntVar(value=10)
        self.batch = tk.IntVar(value=16)
        self.rate = tk.DoubleVar(value=0.001)
        self.target = tk.DoubleVar(value=0.9)
        self.seed = tk.IntVar(value=42)
        for i, (title, variable, minimum, maximum, increment) in enumerate([
                ('Epochs', self.epochs, 1, 1000, 1), ('Batch size', self.batch, 1, 256, 1),
                ('Learning rate', self.rate, 0.00001, 1, 0.0001),
                ('Target accuracy', self.target, 0.01, 1, 0.01),
                ('Split seed', self.seed, 0, 4294967295, 1)]):
            ttk.Label(options, text=title).grid(row=0, column=i, sticky='w', padx=(0, 16))
            ttk.Spinbox(options, textvariable=variable, from_=minimum, to=maximum,
                        increment=increment, width=12).grid(row=1, column=i, padx=(0, 16))
        self.pretrained = tk.BooleanVar(value=True)
        self.reviewed = tk.BooleanVar(value=False)
        for variable in (self.dataset, self.inventory):
            variable.trace_add('write', lambda *_: self.reviewed.set(False))
        ttk.Checkbutton(self, text='Use ImageNet weights (downloads when not cached)', variable=self.pretrained).grid(
            row=4, column=0, columnspan=3, sticky='w')
        ttk.Checkbutton(self, text='Genus labels and specimen IDs have been reviewed', variable=self.reviewed).grid(
            row=5, column=0, columnspan=3, sticky='w', pady=6)
        toolbar = ttk.Frame(self)
        toolbar.grid(row=6, column=0, columnspan=3, sticky='ew', pady=6)
        self.start_button = ttk.Button(toolbar, text='Train and Evaluate', command=self.start)
        self.start_button.pack(side='left')
        self.stop_button = ttk.Button(toolbar, text='Cancel', command=self.cancel.set, state='disabled')
        self.stop_button.pack(side='left', padx=8)
        self.load_button = ttk.Button(toolbar, text='Use Trained Model', command=self.use_model, state='disabled')
        self.load_button.pack(side='right')
        self.summary = ttk.Label(self, text='ResNet-18 | CPU | Specimen-based train / validation / test split', wraplength=680)
        self.summary.grid(row=7, column=0, columnspan=3, sticky='w', pady=6)
        self.progress = ttk.Progressbar(self, mode='indeterminate')
        self.progress.grid(row=8, column=0, columnspan=3, sticky='ew')
        self.log = tk.Text(self, height=10, wrap='word', state='disabled')
        self.log.grid(row=9, column=0, columnspan=2, sticky='nsew', pady=8)
        scroll = ttk.Scrollbar(self, command=self.log.yview)
        scroll.grid(row=9, column=2, sticky='ns')
        self.log.configure(yscrollcommand=scroll.set)
        app.root.after(150, self.poll)

    def browse(self, variable, is_file):
        if self.app.busy:
            return
        value = filedialog.askopenfilename(filetypes=[('CSV inventory', '*.csv')]) if is_file else filedialog.askdirectory()
        if value:
            variable.set(value)

    def append(self, message):
        self.log.configure(state='normal')
        self.log.insert('end', message + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def poll(self):
        while not self.messages.empty():
            self.append(self.messages.get_nowait())
        if self.running and not self.app.busy:
            self.running = False
            self.progress.stop()
            self.stop_button.configure(state='disabled')
            self.start_button.configure(state='normal')
        self.app.root.after(150, self.poll)

    def start(self):
        if self.app.busy:
            return
        if not self.reviewed.get():
            messagebox.showerror('Review required', 'Review genus labels and assign specimen IDs in the inventory first.')
            return
        if not self.pretrained.get() and not messagebox.askyesno(
                'Random features', 'Without ImageNet weights this is a pipeline test, not a useful identification baseline. Continue?'):
            return
        try:
            options = dict(root=self.dataset.get(), inventory=self.inventory.get(), output=self.output.get(),
                           epochs=self.epochs.get(), batch_size=self.batch.get(), learning_rate=self.rate.get(),
                           target_accuracy=self.target.get(), pretrained=self.pretrained.get(), seed=self.seed.get())
            if not Path(options['root']).is_dir() or not Path(options['inventory']).is_file() or not options['output'].strip():
                raise ValueError('Select a dataset folder, inventory CSV, and output folder.')
        except (ValueError, tk.TclError) as error:
            messagebox.showerror('Training settings', str(error))
            return
        self.cancel.clear()
        self.manifest = None
        self.load_button.configure(state='disabled')
        self.running = True
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.progress.start()
        self.summary.configure(text='Training in progress')
        self.app.status.configure(text='Training genus classifier...')
        self.app.background(lambda: train(**options, progress=self.messages.put, cancel=self.cancel), self.completed)

    def completed(self, result):
        self.manifest = result['manifest']
        report = result['report']
        test = report['test']
        summary = f'Test accuracy: {test["accuracy"]:.1%} | Macro F1: {test["macro_f1"]:.3f} | Accepted: {test["coverage"]:.1%}'
        self.summary.configure(text=summary)
        self.append(summary)
        for genus, values in test['per_genus'].items():
            self.append(f'{genus}: precision {values["precision"]:.1%}, recall {values["recall"]:.1%}, test images {values["support"]}')
        self.append('Evaluation report: ' + str(Path(self.manifest).with_name('evaluation.json')))
        if report['threshold_policy']['reject_all']:
            self.append('Validation target was not met. All predictions will remain Uncertain.')
        self.load_button.configure(state='normal')
        self.app.status.configure(text='Training complete. Test results are estimates, not taxonomic validation.')

    def failed(self, error):
        self.summary.configure(text='Training did not complete')
        self.append(error)

    def use_model(self):
        if self.app.busy or self.manifest is None:
            return
        def loaded(classifier):
            self.app.classifier = classifier
            self.app.clear_result()
            self.app.model_label.configure(text=Path(self.manifest).parent.name + ' | ResNet-18')
            self.app.outcome.configure(text='Awaiting identification')
            self.app.status.configure(text='Trained model loaded; family remains assumed.')
        self.app.background(lambda: GenusClassifier(self.manifest), loaded)
