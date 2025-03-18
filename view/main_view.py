from ipywidgets import widgets

class MainView:
    def start(self,tab_1, tab_2, tab_3, tab_4):
        tab_set = widgets.Tab([tab_1,tab_2,tab_3,tab_4])
        tab_set.set_title(
            0, 'Data Selection')

        tab_set.set_title(1, 'Data Cleaning')
        tab_set.set_title(2, 'Data Processing')
        tab_set.set_title(3, 'Predictive Modeling')

        tab_set
