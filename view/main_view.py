from ipywidgets import *
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

class MainView:
    # The view class for variables and elements shared across views or affecting multiple views
    def __init__(self):
            self.dt_features = None
            self.feat_page = None
            self.process_page = None
            self.vis_page = None
            self.right_page = None
            self.process_types = None
            self.f_box = None
            self.trg_lbl = None
            self.dt_ftslay = None
            self.prdtsk_lbl = None
            self.datasets = None
            self.tab_set = None
            self.tabs = None
            self.cleaningoutput = None

    
    def set_tabs(self,tab_1,tab_2,tab_3,tab_4, tab_5, tab_6):
        #initialize the tabs
        self.tabs = [tab_1,tab_2,tab_3, tab_4, tab_5, tab_6]
        tab_set = widgets.Tab(self.tabs)
        tab_set.set_title(0, 'Data Information')
        tab_set.set_title(1, 'Data Cleaning')
        tab_set.set_title(2, 'Data Processing')
        tab_set.set_title(3, 'Predictive Modeling')
        tab_set.set_title(4, 'Logging')
        tab_set.set_title(5, 'Topic information')
        tab_set.layout.width='100%'
        tab_set.layout.display = 'none' 
        self.tab_set = tab_set

    def show_tabs(self):
         self.tab_set.layout = widgets.Layout(visibility = 'visible')

    def get_tabs(self):
         return self.tab_set

    def get_ui(self, login, task_selection, tabs):
        #create the main view
        main_vbox = VBox([login, task_selection, tabs],
                         layout=Layout(flex='1', display='flex', width='100%'))

        ui = HBox([main_vbox],
                  layout=Layout(width='100%', display='flex'))
        return ui
    
    def close_tab(self, index):
         self.tabs[index].close()
     
    def set_title(self, index, titlename):
         self.tab_set.set_title(index, titlename)

    def switch_tab_to_topic_info(self):
         self.tab_set.selected_index = 5