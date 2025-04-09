from IPython.display import clear_output
from IPython import display
from ipywidgets import *
import matplotlib.pyplot as plt
import seaborn as sns

predictiontask = None

class DataProcessingView:
    def __init__(self, controller, main_view):
        self.controller = controller
        self.main_view = main_view
        self.coltype = None
        self.selcl = None
        self.ApplyButton = None
        self.trg_btn = None
        self.splt_btn = None
        

    def featureprclick(self,features2,FeatPage,processtypes,ProcssPage,scalingacts):  
        colname = features2.value

        display_df = self.controller.get_curr_df()


        
        if self.controller.main_model.datasplit:
            if colname in self.controller.get_XTrain().columns: 
                display_df = self.controller.get_XTrain()
            else: 
                ytrain_df = self.controller.main_model.getYtrain().to_frame()
                if colname in ytrain_df.columns: 
                    display_df = ytrain_df
                else: 
                    return


        if not colname in display_df.columns:
            return

        self.selcl.value = "Column: "+str(colname)
        self.coltype.value = "Column Type: "+str(display_df[colname].dtype)
        
        with FeatPage:
            clear_output()

            if (display_df[colname].dtype == 'float64') or (display_df[colname].dtype == 'int64') or (display_df[colname].dtype == 'int32'):

                fig, (axbox, axhist) = plt.subplots(1,2)
        
                sns.boxplot(x=colname,data=display_df, ax=axbox)
                bxtitle = 'Box plot'
                if self.controller.main_model.datasplit:
                    bxtitle+=' (train)'
                axbox.set_title(bxtitle) 
                sns.distplot(display_df[colname],ax=axhist)
                title = 'Histogram' 
                if self.controller.main_model.datasplit:
                    title+=' (train)'
                axhist.set_title(title) 
                plt.legend(['Mean '+str(round(display_df[colname].mean(),2)),'Stdev '+str(round(display_df[colname].std(),2))], bbox_to_anchor=(0.6, 0.6))
                plt.show()
        
            if (display_df[colname].dtype == 'object') or (display_df[colname].dtype== 'string'):
                
                nrclasses = len(display_df[colname].unique())
                if nrclasses < 250:
                    g = sns.countplot(display_df, x=colname)
                    g.set_xticklabels(g.get_xticklabels(),rotation= 45)
                    plt.show()
                else:
                    display.display('Number of classes: ',nrclasses)
                    
        with ProcssPage:
            clear_output()            
        

                
        scalingacts.value = scalingacts.options[0]
        return
    
    def savecurrdata(self,event):
        self.controller.savedata(self.controller.get_datafolder(), self.main_view.datasets.value)

    def makebalanced(self,event):  
        global balncacts
        
        self.controller.make_balanced(self.main_view.dt_features,balncacts,self.main_view.process_page)
        
        return

    def makescaling(self,event):    
        global scalingacts,result2exp

        self.controller.make_scaling(self.main_view.dt_features,self.main_view.feat_page,scalingacts,result2exp)
        
        return

   

    def makesplit(self,event):  
        global result2exp
        
        self.testratiolbl.value = 'Test Ratio(%): '+str(self.splt_txt.value)
        self.controller.make_split(self.splt_txt,self.splt_btn,result2exp)
        
        self.testratiolbl.layout.display = 'block'
        self.testratiolbl.layout.visibility = 'visible'
        
        
        return

    def featurepr_click(self,event):  
        global scalingacts
        
        self.featureprclick(self.main_view.dt_features,self.main_view.feat_page,self.main_view.process_types,self.main_view.process_page,scalingacts)

        return

    def selectProcessType(self,event):   
        global sclblly,scalelbl,prctlay,scalingacts,imblncdlay,balncacts,imbllbllly,imbllbl,encdlbl,encodingacts 
        global outrmvlay,outrmvbtn,encdblly,ecndlay,fxctlbl,fxctingacts,fxctblly,fxctlay

        self.selectProcess_Type([self.main_view.process_types,sclblly,scalelbl,prctlay,scalingacts,imblncdlay,balncacts,imbllbllly,imbllbl,outrmvlay,outrmvbtn,encdlbl,encodingacts,encdblly,ecndlay,fxctlbl,fxctingacts,fxctblly,fxctlay])
        
        return

    def removeoutliers(self, event): 
        self.controller.remove_outliers()

        return

    def ApplyMethod(self,event):  
        global scalingacts,result2exp
        #'Select Processing','Scaling','Encoding','Feature Extraction','Outlier','Imbalancedness'

        if self.main_view.process_types.value == "Scaling":
            self.controller.make_scaling(self.main_view.dt_features,self.main_view.process_page,scalingacts,result2exp)
        if self.main_view.process_types.value == "Encoding":
            self.controller.make_encoding(self.main_view.dt_features,encodingacts,result2exp)

        
        return

    def assignTarget(self, event): 
        global result2exp,predictiontask

        
        self.controller.assign_target(self.main_view.trg_lbl,self.main_view.dt_features,self.main_view.prdtsk_lbl,result2exp,self.trg_btn,predictiontask)
        self.trg_btn.layout.visibility = 'hidden'
        self.trg_btn.layout.display = 'none'
        return

    def get_data_processing_tab(self):
        global scalingacts, result2exp, balncacts, imblncdlay, imbllbllly, imbllbl, encdlbl, encodingacts, prctlay, scalelbl, sclblly
        global outrmvlay,outrmvbtn,encdblly,ecndlay,fxctlbl,fxctingacts,fxctblly,fxctlay
        fpgelay = Layout(width="100%")
        self.main_view.feat_page = widgets.Output(layout = fpgelay)
        self.main_view.process_page = widgets.Output(layout=fpgelay)

        svprtlay = Layout(width='150px')
        sveprbtn = widgets.Button(description="Save Processed Data",layout = svprtlay)
        sveprbtn.on_click(self.savecurrdata)

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
        balncacts.observe(self.makebalanced,'value')

        imblncdlay.visibility = 'hidden'
        balncacts.layout = imblncdlay


        outrmvlay = Layout(width='150px',visibility = 'hidden')
        outrmvbtn = widgets.Button(description="Remove Outliers",layout = outrmvlay)
        outrmvbtn.on_click(self.removeoutliers)
        outrmvbtn.layout = outrmvlay

        self.main_view.process_types = widgets.Dropdown( options=['Select Processing','Scaling','Encoding','Feature Extraction','Outlier','Imbalancedness'], description='', disabled=False)
        self.main_view.process_types.observe(self.selectProcessType,'value')
        self.main_view.process_types.layout.width = '200px'



        self.main_view.dt_ftslay =  widgets.Layout( width="99%",display = 'block')
        self.main_view.dt_features = widgets.Select(options=[],description = '',layout = self.main_view.dt_ftslay)
        self.main_view.dt_features.observe(self.featurepr_click, 'value')
       
        self.testratiolbl  =widgets.Label(value = 'Test Ratio(%):',disabled = True)
        self.testratiolbl.layout.visibility = 'hidden'
        self.testratiolbl.layout.display = 'none'
        self.splt_txt =widgets.Dropdown(description ='Test Ratio(%): ',options=[20,25,30,35])
        self.splt_txt.layout.width = '160px'
        spltlay = Layout(width='150px')
        self.splt_btn = widgets.Button(description="Apply Split",layout = spltlay)
        self.splt_btn.on_click(self.makesplit)

        self.main_view.trg_lbl = widgets.Label(value ='Target: -',disabled = True)
        
        #self.main_view.trg_lbl.layout.display = 'none'
        self.main_view.prdtsk_lbl =widgets.Label(value = 'Prediction Type: - ',disabled = True)
        trglay = Layout(width='150px')
        self.trg_btn = widgets.Button(description="Assign Target",layout = trglay)
        self.trg_btn.on_click(self.assignTarget)


        prctlay = widgets.Layout(width="25%",display = 'none')
        scalingacts = widgets.Dropdown( options=['Select','Standardize','Normalize'], description='', disabled=False,layout = prctlay)
        #scalingacts.observe(self.makescaling,'value')


        encdblly = widgets.Layout(width="25%",visibility = 'hidden')
        encdlbl = widgets.Label(value ='Methods',layout = encdblly)

        ecndlay = widgets.Layout(width="25%",display = 'none')
        encodingacts = widgets.Dropdown( options=['Select','Label Encoding','One Hot Encoding'], description='', disabled=False,layout = ecndlay)
     
        fxctblly = widgets.Layout(width="25%",visibility = 'hidden')
        fxctlbl = widgets.Label(value ='Methods',layout = encdblly)

        fxctlay = widgets.Layout(width="25%",display = 'none')
        fxctingacts = widgets.Dropdown( options=['Select','PCA','Correlation'], description='', disabled=False,layout = ecndlay)

        
        self.selcl = widgets.Label(value ='Column: -',disabled = True)
        self.coltype =widgets.Label(value ='Column Type: -',disabled = True)
        self.ApplyButton = widgets.Button(description="Apply")
        self.ApplyButton.on_click(self.ApplyMethod)

        sboxxlay = widgets.Layout()
        sel_box = VBox(children=[self.selcl,self.coltype,
                                 HBox(children=[self.trg_btn,self.main_view.trg_lbl]),
                                 self.main_view.prdtsk_lbl,
                                 HBox(children=[self.testratiolbl,self.splt_txt,self.splt_btn]),
                                 
                                 HBox(children=[widgets.Label(value ='Process Types'),self.main_view.process_types,self.ApplyButton])
                                ,HBox(children=[scalelbl,scalingacts]),
                                HBox(children=[imbllbl,balncacts]),HBox(children=[encdlbl,encodingacts]),
                                HBox(children=[fxctlbl,fxctingacts]),outrmvbtn],layout = sboxxlay)


        fbox2alay = widgets.Layout(width = '30%')
        self.main_view.f_box = VBox(children=[widgets.Label(value ='Features'),HBox(children=[self.main_view.dt_features])],layout = fbox2alay)


        res2lay = widgets.Layout(height='150px',width='99%')
        result2exp = widgets.Textarea(value='', placeholder='',description='',disabled=True,layout = res2lay)

        vb1lay =  widgets.Layout()
        prboxlay = widgets.Layout()
        vbox1 = VBox(children = [HBox(children=[self.main_view.f_box,sel_box],layout = prboxlay),result2exp,sveprbtn],layout = vb1lay)

        vb2lay =  widgets.Layout()
        vbox2 = VBox(children = [self.main_view.feat_page,self.main_view.process_page],layout = vb2lay)
        tab_3 = HBox(children=[vbox1,vbox2])
        return tab_3
    
    def selectProcess_Type(self,vis_list):
        processtypes = vis_list[0]
        sclblly = vis_list[1]
        scalelbl = vis_list[2]
        prctlay = vis_list[3]
        scalingacts = vis_list[4]
        imblncdlay = vis_list[5]
        balncacts = vis_list[6]
        imbllbllly = vis_list[7]
        imbllbl = vis_list[8]
        outrmvlay = vis_list[9]
        outrmvbtn = vis_list[10]
        encdlbl = vis_list[11]
        encodingacts = vis_list[12]
        encdblly = vis_list[13]
        ecndlay = vis_list[14]
        fxctlbl = vis_list[15]
        fxctingacts = vis_list[16]
        fxctblly = vis_list[17]
        fxctlay = vis_list[18]
    
        self.ResetProcessMenu(vis_list)

        
        if processtypes.value == 'Scaling':
            sclblly.display = 'block'
            sclblly.visibility = 'visible'
            scalelbl.layout = sclblly
            prctlay.display = 'block'
            prctlay.visibility = 'visible'
            scalingacts.layout = prctlay
            
        if processtypes.value == 'Imbalancedness':
            
            imbllbllly.display = 'block'
            imbllbllly.visibility = 'visible'
            imbllbl.layout = imbllbllly
            imblncdlay.display = 'block'
            imblncdlay.visibility = 'visible'
            balncacts.layout = imblncdlay
        
        if processtypes.value == 'Outlier':     
            outrmvlay.display = 'block'
            outrmvlay.visibility = 'visible'
            outrmvbtn.layout = outrmvlay

        if processtypes.value == 'Encoding':     
            encdblly.display = 'block'
            encdblly.visibility = 'visible'
            encdblly.layout = sclblly
            ecndlay.display = 'block'
            ecndlay.visibility = 'visible'
            encodingacts.layout = ecndlay

        if processtypes.value == 'Feature Extraction':
            fxctblly.display = 'block'
            fxctblly.visibility = 'visible'
            fxctlbl.layout = fxctblly
            fxctlay.display = 'block'
            fxctlay.visibility = 'visible'
            fxctingacts.layout = fxctlay
            

        return

    def ResetProcessMenu(self,vis_list):

        processtypes = vis_list[0]
        sclblly = vis_list[1]
        scalelbl = vis_list[2]
        prctlay = vis_list[3]
        scalingacts = vis_list[4]
        imblncdlay = vis_list[5]
        balncacts = vis_list[6]
        imbllbllly = vis_list[7]
        imbllbl = vis_list[8]
        outrmvlay = vis_list[9]
        outrmvbtn = vis_list[10]
        encdlbl = vis_list[11]
        encodingacts = vis_list[12]
        encdblly = vis_list[13]
        ecndlay = vis_list[14]
        fxctlbl = vis_list[15]
        fxctingacts = vis_list[16]
        fxctblly = vis_list[17]
        fxctlay = vis_list[18]
    

        fxctblly.display = 'none'
        fxctlbl.layout = fxctblly

        fxctlay.display = 'none'
        fxctingacts.layout = fxctlay


        sclblly.display = 'none'
        scalelbl.layout = sclblly

        ecndlay.display = 'none'
        encodingacts.layout = ecndlay

        encdblly.display = 'none'
        encdlbl.layout = encdblly

    
        outrmvlay.display = 'none'
        outrmvbtn.layout = outrmvlay
        
        imbllbllly.display = 'none'
        imbllbl.layout = imbllbllly

        prctlay.display = 'none'
        scalingacts.layout = prctlay

        imblncdlay.display = 'none'
        balncacts.layout = imblncdlay

        return