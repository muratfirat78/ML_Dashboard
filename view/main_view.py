from ipywidgets import *

class MainView:
    def __init__(self):
            self.dt_features = None
            self.feat_page = None
            self.process_page = None
            self.right_page = None
            self.process_types = None
            self.f_box = None
            self.trg_lbl = None
            self.dt_ftslay = None
            self.ftlaycl = None
            self.prdtsk_lbl = None
            self.datasets = None
            self.tab_set = None
            self.tabs = None

            
    def set_tabs(self,tab_1,tab_2,tab_3,tab_4, tab_5):
        self.tabs = [tab_1,tab_2,tab_3, tab_4, tab_5]
        tab_set = widgets.Tab(self.tabs)
        tab_set.set_title(0, 'Data Selection')
        tab_set.set_title(1, 'Data Cleaning')
        tab_set.set_title(2, 'Data Processing')
        tab_set.set_title(3, 'Predictive Modeling')
        tab_set.set_title(4, 'Logging')
        tab_set.layout.display = 'none'
        self.tab_set = tab_set

    def show_tabs(self):
         self.tab_set.layout = widgets.Layout(visibility = 'visible')

    def get_tabs(self):
         return self.tab_set

    def get_ui(self, login, tabs):
         return widgets.VBox([login, tabs])
    
    def close_tab(self, index):
         self.tabs[index].close()