from IPython.display import clear_output
from IPython import display
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import *
from IPython.display import  HTML

display.display(HTML("<style>.red_label { color:red }</style>"))
display.display(HTML("<style>.blue_label { color:blue }</style>"))


class DataSelectionView:
    def __init__(self, controller, main_view, task_menu):
        self.controller = controller
        self.datafolder = None
        self.main_view = main_view
        self.DFPage = None
        self.HeadPage = None
        self.InfoPage = None
        self.task_menu = task_menu
        self.infotext = None
        self.infolbl = widgets.Label(value="Dataset Information",layout = widgets.Layout(width='50%'))
        self.infolbl.add_class("red_label")
        self.statlbl = widgets.Label(value="Statistical Summary")
        self.statlbl.add_class("red_label")


    
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
        featurelist_width = '100%'
        self.controller.read_data_set(self.controller.get_datafolder(),self.main_view.datasets.value,wsheets.value)

        df = self.controller.get_curr_df()
        info = self.controller.get_curr_info()
        self.main_view.dt_ftslay.height = str(rowheight*len(df.columns))+'px'
        self.main_view.dt_features.layout = self.main_view.dt_ftslay

        self.main_view.ftlaycl.display = 'block'
        self.main_view.ftlaycl.height = str(rowheight*len(df.columns))+'px'
        self.main_view.ftlaycl.width = featurelist_width
        self.main_view.featurescl.layout = self.main_view.ftlaycl


        missings = []
        for col in self.controller.get_curr_df().columns:
            missings.append((self.controller.get_curr_df()[col].isnull().sum(),col))

        new_list = sorted(missings, key=lambda x: x[0], reverse=True)
        
        self.main_view.dt_features.options = [col for (miss,col) in new_list]
        self.main_view.featurescl.options = [col for (miss,col) in new_list]

        df = df[[col for (miss,col) in new_list]]
        

        self.main_view.right_page.layout.display = 'block'
        self.main_view.right_page.layout.visibility = 'visible'
        
        with self.main_view.right_page:
            clear_output()
            totalmisses = 0
            missing_df = pd.DataFrame(columns=['feature','missing values'])
            for col in df.columns:
                row = {'feature': col, 'missing values':df[col].isnull().sum()}
                new_df = pd.DataFrame([row])
                missing_df = pd.concat([missing_df, new_df], axis=0, ignore_index=True)
                totalmisses+=df[col].isnull().sum()

            g = sns.barplot(x='feature', y='missing values', data=missing_df)
            g.set_xticklabels(g.get_xticklabels(),rotation= 45)
            plt.title('Total Missing Values: '+str(totalmisses))
            plt.show()

                
        
        self.main_view.process_types.value = self.main_view.process_types.options[0]

        with self.main_view.feat_page:
            clear_output()
        with self.main_view.process_page:
            clear_output()
        
        with self.DFPage:
            clear_output()
            #####################################
            display.display(df.info())            
            #####################################
        with self.HeadPage:
            clear_output()
            #####################################
            display.display(df.describe()) 
            display.display(df) 
            #####################################


       
        nrlines = 0

        self.InfoPage.value = ''
        

        infotxt = str(info)
        self.InfoPage.value=infotxt
        self.InfoPage.layout.visibility = 'visible'

        self.readfile.disabled = True
        return

    def on_submit_func(self, event):
        self.controller.on_submitfunc(self.datafolder.value,self.main_view.datasets)
        return

    def settaskmenu(self,monitormode):
  
        if not monitormode:
            self.task_menu.layout.display = 'block'
            self.task_menu.layout.visibility  = 'visible'
    
        else:
            self.task_menu.layout.visibility  = 'hidden'
            self.task_menu.layout.display = 'none'
        
        return

    def get_data_selection_tab(self):
        global wsheets, wslay, butlay, DFPage
        wslay =  widgets.Layout()
        wslay.display = 'none'
        wsheets = widgets.Dropdown( options=[], description='..', disabled=False, layout = wslay)

        self.datafolder=widgets.Text(description ='Folder name:',value = 'DataSets')
        self.main_view.datasets = widgets.Dropdown(options=[], description='DataSets:',layout = Layout(width='50%'))
        # if not self.controller.developer_mode:
        #      self.datafolder.disabled = True
        #      self.main_view.datasets.disabled = True

        butlay = Layout(width='75px')
        butlay.display = 'none'
        self.readfile = widgets.Button(description="Read",layout = butlay)

        self.datafolder.on_submit(self.on_submit_func)
        self.main_view.datasets.observe(self.fileClick,'value')
        self.readfile.on_click(self.read_dataset)


        filelay =  widgets.Layout(height = '60px',width='99%')
      

        self.DFPage = widgets.Output(layout=Layout(width='50%',height='150px',align_items='center',overflow="visible"))
        self.HeadPage = widgets.Output(layout=Layout(width='99%',height='200px',align_items='center',overflow="visible"))
        self.InfoPage = widgets.Textarea(layout=Layout(width='500px',height='150px',align_items='center',overflow="visible", visibility="hidden"))

        self.infotext = widgets.Label(value="Dataset Context")
        self.infotext .add_class("red_label")

        tab_1 = VBox(children=[
            self.task_menu,
            HBox(children = [self.datafolder,self.main_view.datasets,wsheets,self.readfile]),
            HBox(children = [self.infolbl,self.infotext]),
            widgets.Box(layout=widgets.Layout(border='solid 1px lightblue', width='99%', height='1px', margin='5px 0px',style={'background': "#C7EFFF"})),
            HBox(children = [self.DFPage,VBox(children=[widgets.Label(value="  "),self.InfoPage])]),
            widgets.Box(layout=widgets.Layout(border='solid 1px lightblue', width='99%', height='1px', margin='5px 0px',style={'background': "#C7EFFF"})),
            self.statlbl,
            HBox(children = [self.HeadPage]) ])
        tab_1.layout.height = '720px'
        return tab_1
        