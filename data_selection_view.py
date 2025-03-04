from IPython.display import clear_output
from IPython import display
from ipywidgets import *
from data_selection_manager import File_Click, on_submitfunc, read_data_set
import settings

def FileClick(change):
    global wslay,wsheets,butlay

    File_Click(settings.online_version,settings.DataFolder.value,settings.datasets.value,wsheets,wslay,butlay)

    return

def read_dataset(b):  
    global DFPage,wsheets
    Pages = (settings.FeatPage,settings.ProcssPage,DFPage,settings.RightPage)
    
    read_data_set(settings.online_version,settings.DataFolder.value,settings.datasets.value,wsheets.value,settings.processtypes,Pages,settings.dt_features,settings.dt_ftslay,settings.featurescl,settings.ftlaycl)
    
    return

def on_submit_func(sender):
    on_submitfunc(settings.online_version,settings.DataFolder.value,settings.datasets)

    return

def get_data_selection_tab():
    global wsheets, wslay, butlay, DFPage
    wslay =  widgets.Layout()
    wslay.display = 'none'
    wsheets = widgets.Dropdown( options=[], description='..', disabled=False, layout = wslay)

    settings.DataFolder=widgets.Text(description ='Folder name:',value = 'DataSets')
    settings.datasets = widgets.Dropdown(options=[], description='DataSets:',layout = Layout(width='50%'))

    butlay = Layout(width='75px')
    butlay.display = 'none'
    readfile = widgets.Button(description="Read",layout = butlay)

    settings.DataFolder.on_submit(on_submit_func)
    settings.datasets.observe(FileClick,'value')
    readfile.on_click(read_dataset)


    filelay =  widgets.Layout(height = '60px',width='99%')
    tablayout = widgets.Layout(height='500px')
    fthboxlay = widgets.Layout(height='500px')

    DFPage = widgets.Output(layout=Layout(height='250px',align_items='center',overflow="visible"))

    tab_1 = VBox(children=[
        HBox(children = [settings.DataFolder,settings.datasets,wsheets,readfile],layout=filelay),
        HBox(children = [DFPage],layout = fthboxlay)
                        ],layout=tablayout)
    return tab_1
    