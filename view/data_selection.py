from IPython.display import clear_output
from IPython import display
from ipywidgets import *


class DataSelectionView:
    def __init__(self, controller, main_view):
        self.controller = controller
        self.datafolder = None
        self.main_view = main_view
        self.DFPage = None
        self.InfoPage = None

    def fileClick(self, event):
        global wslay,wsheets,butlay

        self.controller.file_Click(self.controller.get_datafolder(),self.main_view.datasets.value,wsheets)
        wslay.display = 'block'
        wsheets.value = wsheets.options[0]    
        wsheets.layout = wslay
            
        butlay.display = 'block'

        return

    def read_dataset(self,b):  
        global InfoPage,wsheets
        rowheight = 20
        self.controller.read_data_set(self.controller.get_datafolder(),self.main_view.datasets.value,wsheets.value)

        df = self.controller.get_curr_df()
        info = self.controller.get_curr_info()
        self.main_view.dt_ftslay.height = str(rowheight*len(df.columns))+'px'
        self.main_view.dt_features.layout = self.main_view.dt_ftslay
        self.main_view.dt_features.options = [col for col in df.columns]


        self.main_view.ftlaycl.display = 'block'
        self.main_view.ftlaycl.height = str(rowheight*len(df.columns))+'px'
        self.main_view.featurescl.layout = self.main_view.ftlaycl
        self.main_view.featurescl.options = [col+'('+str(df[col].isnull().sum())+')' for col in df.columns]
        
        self.main_view.process_types.value = self.main_view.process_types.options[0]

        with self.main_view.feat_page:
            clear_output()
        with self.main_view.process_page:
            clear_output()
        
        with self.DFPage:
            clear_output()
            #####################################
            display.display(df.info()) 
            display.display(df.describe()) 
            display.display(df) 
            #####################################

       
        nrlines = 0
        with self.InfoPage:
            clear_output()
            
            infotxt = str(info)
            if len(infotxt) > 0:
                display.display("Dataset Description:")
            while infotxt.find("\n") > -1:
                line = infotxt[:infotxt.find("\n")]
                infotxt = infotxt[infotxt.find("\n")+1:]
                if (infotxt == '') or (infotxt == None):
                    continue
                display.display(line)
                nrlines+=1
            #####################################
            #####################################
        self.InfoPage.layout.height = str(18*nrlines)+'px'

        with self.main_view.right_page:
            clear_output()

        return

    def on_submit_func(self, event):
        self.controller.on_submitfunc(self.datafolder.value,self.main_view.datasets)
        return

    def get_data_selection_tab(self):
        global wsheets, wslay, butlay, DFPage
        wslay =  widgets.Layout()
        wslay.display = 'none'
        wsheets = widgets.Dropdown( options=[], description='..', disabled=False, layout = wslay)

        self.datafolder=widgets.Text(description ='Folder name:',value = 'DataSets')
        self.main_view.datasets = widgets.Dropdown(options=[], description='DataSets:',layout = Layout(width='50%'))

        butlay = Layout(width='75px')
        butlay.display = 'none'
        readfile = widgets.Button(description="Read",layout = butlay)

        self.datafolder.on_submit(self.on_submit_func)
        self.main_view.datasets.observe(self.fileClick,'value')
        readfile.on_click(self.read_dataset)


        filelay =  widgets.Layout(height = '60px',width='99%')
        tablayout = widgets.Layout(height='500px')
        fthboxlay = widgets.Layout(height='500px')

        self.DFPage = widgets.Output(layout=Layout(width='50%',height='250px',align_items='center',overflow="visible"))
        self.InfoPage = widgets.Output(layout=Layout(height='150px',align_items='center',overflow="visible"))

        tab_1 = VBox(children=[
            HBox(children = [self.datafolder,self.main_view.datasets,wsheets,readfile],layout=filelay),
            HBox(children = [self.DFPage,self.InfoPage],layout = fthboxlay)],layout=tablayout)
        return tab_1
        