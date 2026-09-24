import os
import unittest


@unittest.skipUnless(os.environ.get('LINYPHIIDAE_GUI_TESTS') == '1',
                     'Set LINYPHIIDAE_GUI_TESTS=1 with a display available')
class GuiSmokeTests(unittest.TestCase):
    def test_application_starts_and_tabs_render(self):
        import tkinter as tk
        from tkinter import ttk

        from linyphiidae_id.app import Application

        root = tk.Tk()
        self.addCleanup(root.destroy)
        callback_errors = []
        root.report_callback_exception = lambda *error: callback_errors.append(error)
        self.assertGreaterEqual(
            tuple(map(int, root.tk.call('package', 'provide', 'Tk').split('.')[:2])),
            (8, 6))
        app = Application(root)
        root.update()
        self.assertEqual(root.title(), 'Linyphiidae ID')
        notebook = next(child for child in root.winfo_children()
                        if isinstance(child, ttk.Notebook))
        self.assertEqual([notebook.tab(tab, 'text') for tab in notebook.tabs()],
                         ['Identify Specimen', 'Genus Dataset', 'Train Model'])
        for tab in notebook.tabs():
            notebook.select(tab)
            root.update()
            self.assertTrue(root.nametowidget(tab).winfo_ismapped())
        self.assertTrue(app.identify_button.instate(['disabled']))
        self.assertFalse(app.training_panel.running)
        self.assertEqual(callback_errors, [])
