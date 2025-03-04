from IPython.display import clear_output
from IPython import display
from ipywidgets import *
from data_processing_manager import SelectProcess_Type, assign_target, featureprclick, make_balanced, make_encoding, make_scaling, make_split, remove_outliers, savedata
import settings

predictiontask = None

def savecurrdata(change):
    savedata(settings.DataFolder, settings.datasets.value)

def makebalanced(event):  
    global balncacts
      
    make_balanced(settings.dt_features,balncacts,settings.ProcssPage)
    
    return

def makescaling(event):    
    global scalingacts,result2exp

    make_scaling(settings.dt_features,settings.ProcssPage,scalingacts,result2exp)
    
    return

def makeencoding(event):   
    global encodingacts,result2exp
    make_encoding(settings.dt_features,encodingacts,result2exp)
    
    return

def makesplit(event):  
    global splt_txt,splt_btn,result2exp
    
    make_split(splt_txt,splt_btn,result2exp)
    
    return

def featurepr_click(event):  
    global scalingacts
    
    featureprclick(settings.dt_features,settings.FeatPage,settings.processtypes,settings.ProcssPage,scalingacts)

    return

def SelectProcessType(change):   
    global sclblly,scalelbl,prctlay,scalingacts,imblncdlay,balncacts,imbllbllly,imbllbl,encdlbl,encodingacts 
    global outrmvlay,outrmvbtn,encdblly,ecndlay,fxctlbl,fxctingacts,fxctblly,fxctlay

    
    SelectProcess_Type([settings.processtypes,sclblly,scalelbl,prctlay,scalingacts,imblncdlay,balncacts,imbllbllly,imbllbl,outrmvlay,outrmvbtn,encdlbl,encodingacts,encdblly,ecndlay,fxctlbl,fxctingacts,fxctblly,fxctlay])
    
    return

def removeoutliers(change): 
    remove_outliers()

    return

def AssignTarget(change): 
    global result2exp,trg_btn,predictiontask

    assign_target(settings.trg_lbl,settings.dt_features,settings.prdtsk_lbl,result2exp,trg_btn,predictiontask)

    return

def get_data_processing_tab():
    global scalingacts, result2exp, trg_btn, balncacts, imblncdlay, imbllbllly, imbllbl, encdlbl, encodingacts, prctlay, scalelbl, sclblly, splt_txt,splt_btn
    global outrmvlay,outrmvbtn,encdblly,ecndlay,fxctlbl,fxctingacts,fxctblly,fxctlay
    fpgelay = Layout(width="100%")
    settings.FeatPage = widgets.Output(layout = fpgelay)
    settings.ProcssPage = widgets.Output(layout=fpgelay)

    svprtlay = Layout(width='150px')
    sveprbtn = widgets.Button(description="Save Processed Data",layout = svprtlay)
    sveprbtn.on_click(savecurrdata)

    sclblly = widgets.Layout(width="25%")
    scalelbl = widgets.Label(value ='Methods',layout = sclblly)
    sclblly.visibility = 'hidden'
    scalelbl.layout = sclblly


    imbllbllly = widgets.Layout()
    imbllbl = widgets.Label(value ='Methods',layout = imbllbllly)
    imbllbllly.visibility = 'hidden'
    imbllbl.layout = imbllbllly

    imblncdlay = widgets.Layout()
    balncacts = widgets.Dropdown( options=['Select','Upsample','DownSample'], description='', disabled=False,layout = imblncdlay)
    balncacts.observe(makebalanced,'value')

    imblncdlay.visibility = 'hidden'
    balncacts.layout = imblncdlay


    outrmvlay = Layout(width='150px',visibility = 'hidden')
    outrmvbtn = widgets.Button(description="Remove Outliers",layout = outrmvlay)
    outrmvbtn.on_click(removeoutliers)
    outrmvbtn.layout = outrmvlay

    settings.processtypes = widgets.Dropdown( options=['Select Processing','Scaling','Encoding','Feature Extraction','Outlier','Imbalancedness'], description='', disabled=False)
    settings.processtypes.observe(SelectProcessType,'value')



    settings.dt_ftslay =  widgets.Layout( width="99%",display = 'block')
    settings.dt_features = widgets.Select(options=[],description = '',layout = settings.dt_ftslay)
    settings.dt_features.observe(featurepr_click, 'value')

    splt_txt =widgets.Dropdown(description ='Split (Test%):',options=[20,25,30,35])
    spltlay = Layout(width='150px')
    splt_btn = widgets.Button(description="Apply Split",layout = spltlay)
    splt_btn.on_click(makesplit)

    settings.trg_lbl =widgets.Text(description ='Target:',value = '',disabled = True)
    settings.prdtsk_lbl =widgets.Text(description ='Pred. Task:',value = '',disabled = True)
    trglay = Layout(width='150px')
    trg_btn = widgets.Button(description="Assign Target",layout = trglay)
    trg_btn.on_click(AssignTarget)


    prctlay = widgets.Layout(width="25%",display = 'none')
    scalingacts = widgets.Dropdown( options=['Select','Standardize','Normalize'], description='', disabled=False,layout = prctlay)
    scalingacts.observe(makescaling,'value')


    encdblly = widgets.Layout(width="25%",visibility = 'hidden')
    encdlbl = widgets.Label(value ='Methods',layout = encdblly)

    ecndlay = widgets.Layout(width="25%",display = 'none')
    encodingacts = widgets.Dropdown( options=['Select','Label Encoding','One Hot Encoding'], description='', disabled=False,layout = ecndlay)
    encodingacts.observe(makeencoding,'value')

    fxctblly = widgets.Layout(width="25%",visibility = 'hidden')
    fxctlbl = widgets.Label(value ='Methods',layout = encdblly)

    fxctlay = widgets.Layout(width="25%",display = 'none')
    fxctingacts = widgets.Dropdown( options=['Select','PCA','Correlation'], description='', disabled=False,layout = ecndlay)


    sboxxlay = widgets.Layout()
    sel_box = VBox(children=[trg_btn,settings.trg_lbl,settings.prdtsk_lbl,splt_txt,splt_btn,HBox(children=[widgets.Label(value ='Process Types'),settings.processtypes])
                            ,HBox(children=[scalelbl,scalingacts]),
                            HBox(children=[imbllbl,balncacts]),HBox(children=[encdlbl,encodingacts]),
                            HBox(children=[fxctlbl,fxctingacts]),outrmvbtn],layout = sboxxlay)


    fbox2alay = widgets.Layout(width = '30%')
    settings.f_box = VBox(children=[widgets.Label(value ='Features'),HBox(children=[settings.dt_features])],layout = fbox2alay)


    res2lay = widgets.Layout(height='150px',width='99%')
    result2exp = widgets.Textarea(value='', placeholder='',description='',disabled=True,layout = res2lay)

    vb1lay =  widgets.Layout()
    prboxlay = widgets.Layout()
    vbox1 = VBox(children = [HBox(children=[settings.f_box,sel_box],layout = prboxlay),result2exp,sveprbtn],layout = vb1lay)

    vb2lay =  widgets.Layout()
    vbox2 = VBox(children = [settings.FeatPage,settings.ProcssPage],layout = vb2lay)
    tab_3 = HBox(children=[vbox1,vbox2])
    return tab_3