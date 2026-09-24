import csv
import queue
import threading
from collections import Counter
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from .dataset import inspect_dataset, export_inventory
from .model import GenusClassifier
from .training_ui import TrainingPanel

ROOT = Path(__file__).resolve().parent.parent


class Application:
    def __init__(self, root):
        self.root = root
        self.image = None
        self.image_path = None
        self.classifier = None
        self.inventory = []
        self.result = None
        self.busy = False
        self.events = queue.Queue()
        root.title('Linyphiidae ID')
        root.geometry('1040x760')
        root.minsize(760, 600)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        ttk.Label(root, text='Linyphiidae ID', font=('Arial', 22, 'bold')).grid(
            row=0, column=0, sticky='w', padx=20, pady=(16, 4))
        ttk.Label(root, text='Family supplied by the user; not verified by this classifier.').grid(
            row=1, column=0, sticky='w', padx=20, pady=(0, 12))
        tabs = ttk.Notebook(root)
        tabs.grid(row=2, column=0, sticky='nsew', padx=16)
        identify = ttk.Frame(tabs, padding=12)
        dataset = ttk.Frame(tabs, padding=12)
        tabs.add(identify, text='Identify Specimen')
        tabs.add(dataset, text='Genus Dataset')
        self.training_panel = TrainingPanel(tabs, self, ROOT)
        tabs.add(self.training_panel, text='Train Model')
        root.protocol('WM_DELETE_WINDOW', self.close)
        identify.columnconfigure(0, weight=1)
        identify.rowconfigure(2, weight=1)
        toolbar = ttk.Frame(identify)
        toolbar.grid(row=0, column=0, sticky='ew')
        ttk.Button(toolbar, text='Open Image', command=self.open_image).pack(side='left')
        ttk.Button(toolbar, text='Load Genus Model', command=self.load_model).pack(side='left', padx=8)
        self.identify_button = ttk.Button(toolbar, text='Identify Genus', command=self.identify, state='disabled')
        self.identify_button.pack(side='left')
        self.export_button = ttk.Button(toolbar, text='Export Result', command=self.export_result, state='disabled')
        self.export_button.pack(side='right')
        self.model_label = ttk.Label(identify, text='No Linyphiidae model loaded', wraplength=680)
        self.model_label.grid(row=1, column=0, sticky='w', pady=10)
        self.preview = tk.Canvas(identify, bg='#f3f4f6', highlightthickness=0)
        self.preview.grid(row=2, column=0, sticky='nsew')
        self.preview.bind('<Configure>', self.draw_image)
        self.outcome = ttk.Label(identify, text='Awaiting specimen image', font=('Arial', 13, 'bold'), wraplength=680)
        self.outcome.grid(row=3, column=0, sticky='w', pady=10)
        self.predictions = ttk.Treeview(identify, columns=('genus', 'score'), show='headings', height=3)
        for column, heading in [('genus', 'Candidate Genus'), ('score', 'Model Score')]:
            self.predictions.heading(column, text=heading)
        self.predictions.grid(row=4, column=0, sticky='ew')
        dataset.columnconfigure(0, weight=1)
        dataset.rowconfigure(2, weight=1)
        bar = ttk.Frame(dataset)
        bar.grid(row=0, column=0, sticky='ew')
        ttk.Button(bar, text='Open Dataset Folder', command=self.open_dataset).pack(side='left')
        self.inventory_button = ttk.Button(bar, text='Export Inventory', command=self.save_inventory, state='disabled')
        self.inventory_button.pack(side='right')
        self.dataset_label = ttk.Label(dataset, text='No dataset selected', wraplength=680)
        self.dataset_label.grid(row=1, column=0, sticky='w', pady=10)
        self.dataset_table = ttk.Treeview(dataset, columns=('genus', 'count', 'issues'), show='headings')
        for col, title in [('genus', 'Folder Label'), ('count', 'Images'), ('issues', 'Images Needing Review')]:
            self.dataset_table.heading(col, text=title)
        self.dataset_table.grid(row=2, column=0, sticky='nsew')
        scroll = ttk.Scrollbar(dataset, orient='vertical', command=self.dataset_table.yview)
        scroll.grid(row=2, column=1, sticky='ns')
        self.dataset_table.configure(yscrollcommand=scroll.set)
        self.status = ttk.Label(root, text='Model training required before genus identification.', wraplength=720)
        self.status.grid(row=3, column=0, sticky='w', padx=20, pady=12)
        root.after(100, self.poll)

    def clear_result(self):
        self.result = None
        self.predictions.delete(*self.predictions.get_children())
        self.export_button.configure(state='disabled')

    def close(self):
        if self.training_panel.running:
            self.training_panel.cancel.set()
            self.status.configure(text='Cancelling training; close again after it stops.')
            return
        self.root.destroy()

    def refresh(self):
        self.identify_button.configure(state='normal' if self.image and self.classifier and not self.busy else 'disabled')

    def background(self, action, success):
        if self.busy:
            return
        self.busy = True
        self.refresh()
        def worker():
            try:
                self.events.put((success, action(), None))
            except Exception as error:
                self.events.put((success, None, str(error)))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            success, value, error = self.events.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            if error:
                if self.training_panel.running:
                    self.training_panel.failed(error)
                self.status.configure(text='Operation failed: ' + error)
                messagebox.showerror('Unable to complete operation', error, parent=self.root)
            else:
                success(value)
            self.refresh()
        self.root.after(100, self.poll)

    def open_image(self):
        if self.busy:
            return
        name = filedialog.askopenfilename(filetypes=[('Images', '*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp')])
        if not name:
            return
        try:
            with Image.open(name) as image:
                self.image = ImageOps.exif_transpose(image).convert('RGB')
        except (OSError, ValueError) as error:
            messagebox.showerror('Image error', str(error))
            return
        self.image_path = name
        self.clear_result()
        self.outcome.configure(text=Path(name).name)
        self.draw_image()
        self.refresh()

    def draw_image(self, event=None):
        self.preview.delete('all')
        if self.image is None:
            return
        width, height = max(self.preview.winfo_width()-16, 1), max(self.preview.winfo_height()-16, 1)
        self.photo = ImageTk.PhotoImage(ImageOps.contain(self.image, (width, height)))
        self.preview.create_image(self.preview.winfo_width()/2, self.preview.winfo_height()/2, image=self.photo)

    def load_model(self):
        if self.busy:
            return
        name = filedialog.askopenfilename(initialdir=ROOT / 'models' / 'linyphiidae', filetypes=[('Model manifest', '*.json')])
        if not name:
            return
        self.status.configure(text='Loading model...')
        def loaded(classifier):
            self.classifier = classifier
            self.clear_result()
            self.outcome.configure(text='Awaiting identification')
            self.model_label.configure(text=Path(name).name + ' | ' + str(len(classifier.metadata['genera'])) + ' genera')
            self.status.configure(text='Model loaded. Genus labels are supplied by its manifest.')
        self.background(lambda: GenusClassifier(name), loaded)

    def identify(self):
        if self.busy or self.image is None or self.classifier is None:
            return
        self.clear_result()
        self.status.configure(text='Identifying genus...')
        def completed(value):
            ranked, accepted = value
            decision = ranked[0][0] if accepted else 'Uncertain'
            self.result = (decision, ranked)
            self.outcome.configure(text='Predicted genus: ' + decision)
            for genus, score in ranked:
                self.predictions.insert('', 'end', values=(genus, f'{score:.2%}'))
            self.export_button.configure(state='normal')
            self.status.configure(text='Prediction is not a confirmed identification; family remains assumed.')
        self.background(lambda: self.classifier.predict(self.image), completed)

    def export_result(self):
        if self.result is None:
            return
        name = filedialog.asksaveasfilename(defaultextension='.csv', initialfile=Path(self.image_path).stem + '_result.csv')
        if name:
            try:
                with open(name, 'w', newline='', encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['image', 'assumed_family', 'decision', 'candidate_genus', 'model_score'])
                    for genus, score in self.result[1]:
                        writer.writerow([self.image_path, 'Linyphiidae', self.result[0], genus, score])
            except OSError as error:
                messagebox.showerror('Export error', str(error))

    def open_dataset(self):
        if self.busy:
            return
        name = filedialog.askdirectory(initialdir=ROOT / 'datasets' / 'linyphiidae')
        if not name:
            return
        self.status.configure(text='Checking dataset images...')
        def completed(rows):
            self.inventory = rows
            self.training_panel.dataset.set(name)
            self.dataset_table.delete(*self.dataset_table.get_children())
            counts = Counter(row['genus'] for row in rows)
            issues = Counter(row['genus'] for row in rows if row['issue'])
            for genus, count in sorted(counts.items()):
                self.dataset_table.insert('', 'end', values=(genus or '(unlabeled)', count, issues[genus]))
            self.dataset_label.configure(text=f'{name}\n{len(rows)} images; {len([g for g in counts if g])} genus folders')
            self.inventory_button.configure(state='normal' if rows else 'disabled')
            self.status.configure(text='Folder labels are unverified. Specimen IDs and expert label review are required before training.')
        self.background(lambda: inspect_dataset(name), completed)

    def save_inventory(self):
        name = filedialog.asksaveasfilename(defaultextension='.csv', initialfile='specimen_inventory.csv')
        if name:
            try:
                export_inventory(self.inventory, name)
                self.training_panel.inventory.set(name)
            except OSError as error:
                messagebox.showerror('Export error', str(error))


def main():
    root = tk.Tk()
    Application(root)
    root.mainloop()
