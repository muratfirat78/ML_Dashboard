import ipywidgets as widgets
from IPython.display import display

from SelectRun import SelectRun
from Overview import Overview


class Controller:
    def __init__(self, drive, online_version):
        self.run = None
        self.drive = drive
        self.online_version = online_version

        self.selectRun = SelectRun(self)
        self.overview = Overview(self)

        self.ui = widgets.VBox([
            self.selectRun.get_ui(),
            self.overview.get_ui(),
        ])

    def start(self):
        display(self.ui)

    def show_overview(self):
        self.overview.refresh()
