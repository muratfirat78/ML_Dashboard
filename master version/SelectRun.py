from pathlib import Path

import ipywidgets as widgets


class SelectRun:
    # Lets the instructor download the latest student data and pick which
    # cohort/run to build the overview for.

    def __init__(self, controller):
        self.controller = controller

        self.run_dropdown = widgets.Dropdown(description='Run:')
        self.load_button = widgets.Button(description='Load runs')
        self.show_button = widgets.Button(description='Show overview')
        self.status_label = widgets.Label(value='')

        self.load_button.on_click(self.load_runs)
        self.show_button.on_click(self.select_run)

        self.vbox = widgets.VBox([
            widgets.HBox([self.load_button, self.run_dropdown, self.show_button]),
            self.status_label,
        ])

    def get_ui(self):
        return self.vbox

    def load_runs(self, _=None):
        self.status_label.value = 'Downloading student data...'
        self.controller.drive.get_performances_master()

        runs = self.find_runs()
        self.run_dropdown.options = runs

        if runs:
            self.run_dropdown.value = runs[0]
            self.status_label.value = f'Found {len(runs)} run(s).'
        else:
            self.status_label.value = 'No saved student data found yet.'

    def find_runs(self):
        # run labels are the part of the filename before "__", e.g. "2026-11__03-11-2026 13-00-24.txt"
        drive_path = Path('../drive')
        runs = set()

        if drive_path.exists():
            for file in drive_path.glob('*/*__*.txt'):
                run, _, _ = file.stem.partition('__')
                runs.add(run)

        return sorted(runs, reverse=True)

    def select_run(self, _=None):
        self.controller.run = self.run_dropdown.value
        self.controller.show_overview()
