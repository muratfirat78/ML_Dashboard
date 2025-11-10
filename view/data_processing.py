from IPython.display import clear_output,HTML
from IPython import display
from ipywidgets import *
import matplotlib.pyplot as plt
import seaborn as sns



predictiontask = None

class DataProcessingView:
    def __init__(self, controller, main_view, task_menu):
        self.controller = controller
        self.main_view = main_view
        self.coltype = None
        self.selcl = None
        self.ApplyButton = None
        self.trg_btn = None
        self.splt_btn = None
        self.pca_btn = None
        self.pcaselect = None
        self.task_menu = task_menu
        self.ordinalenconding = None
        self.ord_btn = None
        self.ord_btn2 = None
        self.feattitle = None
        self.proctitle = None

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



        color = "gray"
        mytext =str(colname)
        mytext2 = " -> "+str(display_df[colname].dtype)
        self.selcl.value = f'<span style="color:{color};"><b>{mytext}</b>{mytext2}</span>'
      
        
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
        
            if (display_df[colname].dtype == 'object') or (display_df[colname].dtype== 'string') or (display_df[colname].dtype== 'bool')  :
                
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
        global balncacts,result2exp
        
        
        
        return


    def makesplit(self,event):  
        global result2exp
        
        self.testratiolbl.value = 'Test Ratio(%): '+str(self.splt_txt.value)
        self.controller.make_split(self.splt_txt,self.splt_btn,result2exp)
        
        self.testratiolbl.layout.display = 'block'
        self.testratiolbl.layout.visibility = 'visible'

        opts = []
        if not self.controller.main_model.datasplit:
            opts =  [col for col in self.controller.main_model.get_curr_df().columns]
        else:
            opts = [col for col in self.controller.main_model.get_XTrain().columns]
            for col in self.controller.main_model.getYtrain().to_frame().columns:
                opts.append(col)
          
        self.main_view.dt_features.options =[x for x in opts]
        self.main_view.featurescl.options  = [x for x in opts] 
        
        
        return

    def featurepr_click(self,event):  
        global scalingacts
        
        self.featureprclick(self.main_view.dt_features,self.main_view.feat_page,self.main_view.process_types,self.main_view.process_page,scalingacts)

        return

    def selectProcessType(self,event):   
        global sclblly,scalelbl,prctlay,scalingacts,imblncdlay,balncacts,imbllbllly,imbllbl,encdlbl,encodingacts 
        global outrmvlay,outrmvbtn,encdblly,ecndlay,fxctlbl,fxctingacts,fxctblly,fxctlay

        if  self.main_view.process_types.value == "Assign Target":
            
            self.main_view.trg_lbl.layout.display = 'block'
            self.main_view.trg_lbl.layout.visibility = 'visible'
            self.trg_btn.layout.display = 'block'
            self.trg_btn.layout.visibility = 'visible'
            self.main_view.prdtsk_lbl.layout.display = 'block'
            self.main_view.prdtsk_lbl.layout.visibility = 'visible'
            self.splt_txt.layout.visibility = 'hidden'
            self.splt_txt.layout.display = 'none'
            self.splt_btn.layout.visibility = 'hidden'
            self.splt_btn.layout.display = 'none'
 
        else: 
            self.main_view.trg_lbl.layout.visibility = 'hidden'
            self.main_view.trg_lbl.layout.display = 'none'
            self.trg_btn.layout.visibility = 'hidden'
            self.trg_btn.layout.display = 'none'
            self.main_view.prdtsk_lbl.layout.visibility = 'hidden'
            self.main_view.prdtsk_lbl.layout.display = 'none'

            if self.main_view.process_types.value == "Data Split":
                self.splt_txt.layout.display = 'block'
                self.splt_txt.layout.visibility = 'visible'
                self.splt_btn.layout.display = 'block'
                self.splt_btn.layout.visibility = 'visible'
     
            else:
                self.splt_txt.layout.visibility = 'hidden'
                self.splt_txt.layout.display = 'none'
                self.splt_btn.layout.visibility = 'hidden'
                self.splt_btn.layout.display = 'none'
                self.selectProcess_Type([self.main_view.process_types,sclblly,scalelbl,prctlay
                                     ,scalingacts,imblncdlay,balncacts,imbllbllly,imbllbl,outrmvlay,
                                     outrmvbtn,encdlbl,encodingacts,encdblly,
                                     ecndlay,fxctlbl,fxctingacts,fxctblly,fxctlay])
        
        return



    def ApplyMethod(self,event):  
        global scalingacts,result2exp,balncacts,fxctingacts
        #'Select Processing','Scaling','Encoding','Feature Extraction','Outlier','Imbalancedness'
        refreshFeatures = True



        
        with self.main_view.vis_page:
            clear_output()
    
        if self.main_view.process_types.value == "Scaling":
            self.controller.make_scaling(self.main_view.dt_features,self.main_view.process_page,scalingacts,result2exp)
        if self.main_view.process_types.value == "Encoding":
            self.controller.make_encoding(self.main_view.dt_features,encodingacts,self.ordinalenconding,result2exp)
        if self.main_view.process_types.value == "Outlier":
            self.controller.remove_outliers(self.main_view.dt_features,result2exp)
        if self.main_view.process_types.value == "Imbalancedness":
            self.controller.make_balanced(self.main_view.dt_features,balncacts,self.main_view.process_page,result2exp)
        if self.main_view.process_types.value == "Feature Extraction":
            if fxctingacts.value == "Correlation":
                self.controller.showCorrHeatMap(self.main_view.feat_page,fxctingacts,result2exp)
                refreshFeatures = False
            if fxctingacts.value == "PCA":
                self.controller.ApplyPCA(self.main_view.dt_features,self.pcaselect,result2exp)
                self.pcaselect.options = []
                self.pca_btn.layout.visibility = 'hidden'
                self.pca_btn.layout.display = 'none'
                self.pcaselect.layout.visibility = 'hidden'
                self.pcaselect.layout.display = 'none'
            fxctingacts.value = fxctingacts.options[0]
        if self.main_view.process_types.value == "Convert Feature 0/1->Bool":
            self.controller.make_featconvert(self.main_view.dt_features,result2exp)
            
        if refreshFeatures:
            opts = []
            if not self.controller.main_model.datasplit:
                opts =  [col for col in self.controller.main_model.get_curr_df().columns]
            else:
                opts = [col for col in self.controller.main_model.get_XTrain().columns]
                for col in self.controller.main_model.getYtrain().to_frame().columns:
                    opts.append(col)

            self.main_view.dt_features.options = [x for x in opts]
            self.main_view.featurescl.options  = [x for x in opts] 

        
        return

    def assignTarget(self, event): 
        global result2exp,predictiontask

        
        self.controller.assign_target(self.main_view.trg_lbl,self.main_view.dt_features,self.main_view.prdtsk_lbl,result2exp,self.trg_btn,predictiontask)
        self.trg_btn.layout.visibility = 'hidden'
        self.trg_btn.layout.display = 'none'
        self.splt_btn.disabled = False


        return

    def get_data_processing_tab(self):
        global scalingacts, result2exp, balncacts, imblncdlay, imbllbllly, imbllbl, encdlbl, encodingacts, prctlay, scalelbl, sclblly
        global outrmvlay,outrmvbtn,encdblly,ecndlay,fxctlbl,fxctingacts,fxctblly,fxctlay
        fpgelay = Layout(width="100%")
        self.main_view.feat_page = widgets.Output(layout = fpgelay)
        fpgelay = Layout(width="100%")
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
    
        imblncdlay.visibility = 'hidden'
        balncacts.layout = imblncdlay


        outrmvlay = Layout(width='150px',visibility = 'hidden')
        outrmvbtn = widgets.Button(description="Remove Outliers",layout = outrmvlay)
       
        outrmvbtn.layout = outrmvlay

        self.proctitle = widgets.HTML("")
        color = "gray"
        mytext ="Process types"
        self.proctitle.value = f'<span style="color:{color};"><b>{mytext}</b></span>'

        self.main_view.process_types = widgets.Select(description = '',options=['Scaling','Encoding','Assign Target','Data Split','Feature Extraction','Outlier','Imbalancedness','Convert Feature 0/1->Bool'],disabled=False)
        self.main_view.process_types.observe(self.selectProcessType,'value')
        self.main_view.process_types.layout.width = '250px'



        self.main_view.dt_ftslay =  widgets.Layout( width="99%",display = 'block')
        self.main_view.dt_features = widgets.Select(options=[],description = '',layout = self.main_view.dt_ftslay)
        self.main_view.dt_features.observe(self.featurepr_click, 'value')
       
     
        self.splt_txt =widgets.Dropdown(description ='Test Ratio(%): ',options=[20,25,30,35])
        self.splt_txt.layout.width = '160px'
        self.splt_btn = widgets.Button(description="Split",disabled = True)
        self.splt_btn.layout.width = '100px'
        self.splt_btn.on_click(self.makesplit)

        
        self.main_view.trg_lbl = widgets.Label(value ='Target: -',disabled = True)
        
        #self.main_view.trg_lbl.layout.display = 'none'
        self.main_view.prdtsk_lbl =widgets.Label(value = ' | Prediction Task: - ',disabled = True)
        trglay = Layout(width='150px')
        self.trg_btn = widgets.Button(description="Assign Target",layout = trglay)
        self.trg_btn.layout.width = '125px'
        self.trg_btn.on_click(self.assignTarget)


        prctlay = widgets.Layout(width="25%",display = 'none')
        scalingacts = widgets.Dropdown( options=['Select','Standardize','Normalize'], description='', disabled=False,layout = prctlay)
        #scalingacts.observe(self.makescaling,'value')


        encdblly = widgets.Layout(width="25%",visibility = 'hidden')
        encdlbl = widgets.Label(value ='Methods',layout = encdblly)

        ecndlay = widgets.Layout(width="25%",display = 'none')
        encodingacts = widgets.Dropdown( options=['Select','Label Encoding','One Hot Encoding','Ordinal Encoding'], description='', disabled=False,layout = ecndlay)
        encodingacts.observe(self.OrdinalView)
     
        fxctblly = widgets.Layout(width="25%",visibility = 'hidden')
        fxctlbl = widgets.Label(value ='Methods',layout = fxctblly)

        fxctlay = widgets.Layout(width="25%",display = 'none')
        fxctingacts = widgets.Dropdown( options=['Select','PCA','Correlation'], description='', disabled=False,layout = fxctlay)
        fxctingacts.observe(self.PCAView)

        
        self.selcl = widgets.HTML("", layout=widgets.Layout(height="20px", width="55%", text_align="center"))
        
        self.ApplyButton = widgets.Button(description="Apply",layout=widgets.Layout(width='60px'))
        self.ApplyButton.on_click(self.ApplyMethod)

        self.main_view.vis_page = widgets.Output()

        self.pca_btn = widgets.Button(description=">> Add PCA >> ")
        self.pca_btn.layout.visibility = 'hidden'
        self.pca_btn.layout.display = 'none'
        self.pca_btn.on_click(self.AddftPCA)
        
        self.pcaselect = widgets.Select(options=[],description = '')
        self.pcaselect.layout.visibility = 'hidden'
        self.pcaselect.layout.display = 'none'

        self.ordinalenconding = widgets.Select(options=[],description = '')
        self.ordinalenconding.layout.visibility = 'hidden'
        self.ordinalenconding.layout.display = 'none'


        self.ord_btn = widgets.Button(description="Select ->",layout=widgets.Layout(width='80px'))
        self.ord_btn.layout.visibility = 'hidden'
        self.ord_btn.layout.display = 'none'
        self.ord_btn.on_click(self.SelectOrdFeature)
        
        self.ord_btn2 = widgets.Button(description=" Move up ",layout=widgets.Layout(width='80px'))
        self.ord_btn2.layout.visibility = 'hidden'
        self.ord_btn2.layout.display = 'none'
        #self.ord_btn.on_click(self.AddftPCA)
        self.ord_btn2.on_click(self.OrdinalMoveup)

        self.splt_txt.layout.visibility = 'hidden'
        self.splt_txt.layout.display = 'none'
        self.splt_btn.layout.visibility = 'hidden'
        self.splt_btn.layout.display = 'none'

        self.main_view.trg_lbl.layout.visibility = 'hidden'
        self.main_view.trg_lbl.layout.display = 'none'
        self.trg_btn.layout.visibility = 'hidden'
        self.trg_btn.layout.display = 'none'
        self.main_view.prdtsk_lbl.layout.visibility = 'hidden'
        self.main_view.prdtsk_lbl.layout.display = 'none'


    

        sboxxlay = widgets.Layout()
        sel_box = VBox(children=[self.selcl,
                                  widgets.Box(layout=widgets.Layout(border='solid 1px lightblue', width='99%', height='1px', margin='5px 0px',style={'background': "#C7EFFF"})),
                                 self.proctitle,
                                 
                                 HBox(children=[self.main_view.process_types,self.ApplyButton]),
                                 HBox(children=[self.main_view.trg_lbl,self.trg_btn,self.main_view.prdtsk_lbl]),
                                 HBox(children=[self.splt_txt,self.splt_btn]),
                                 HBox(children=[scalelbl,scalingacts]),
                                 HBox(children=[imbllbl,balncacts]),HBox(children=[encdlbl,encodingacts]),
                                 HBox(children=[fxctlbl,fxctingacts]),
                                 HBox(children=[self.pca_btn,self.pcaselect]),
                                 HBox(children=[VBox(children=[self.ord_btn,self.ord_btn2]),self.ordinalenconding]),
                                 self.main_view.vis_page
                                ],layout = sboxxlay)


        fbox2alay = widgets.Layout(width = '35%')
        self.feattitle = widgets.HTML("Features", layout=widgets.Layout(height="30px", width="55%", text_align="center"))
        color = "gray"
        mytext ="Features"
        
        self.feattitle.value = f'<span style="color:{color};"><b>{mytext}</b></span>'
        
        self.main_view.f_box = VBox(children=[self.feattitle,HBox(children=[self.main_view.dt_features])],layout = fbox2alay)
       

        res2lay = widgets.Layout(height='150px',width='99%')
        result2exp = widgets.Textarea(value='', placeholder='',description='',disabled=True,layout = res2lay)

        vb1lay =  widgets.Layout()
        prboxlay = widgets.Layout()
        vbox1 = VBox(children = [HBox(children=[self.main_view.f_box,sel_box],layout = prboxlay),result2exp],layout = vb1lay)

        vb2lay =  widgets.Layout()
        vbox2 = VBox(children = [self.main_view.feat_page,self.main_view.process_page],layout = vb2lay)
        tab_3 = VBox([self.task_menu,HBox([vbox1,vbox2])])
        #tab_3.layout.height = '700px'
        return tab_3

    def PCAView(self,event):
        global fxctingacts

        

        if fxctingacts.value == "PCA":
            with self.main_view.vis_page:
                clear_output()
            self.pcaselect.options = []
            self.pca_btn.layout.display = 'block'
            self.pca_btn.layout.visibility = 'visible'
            self.pcaselect.layout.display = 'block'
            self.pcaselect.layout.visibility = 'visible'
        else: 
            self.pcaselect.options = []
            self.pca_btn.layout.visibility = 'hidden'
            self.pca_btn.layout.display = 'none'
            self.pcaselect.layout.visibility = 'hidden'
            self.pcaselect.layout.display = 'none'
  
        return
        
    def OrdinalView(self,event):
         
        global encodingacts

        

        if encodingacts.value == 'Ordinal Encoding':
            with self.main_view.vis_page:
                clear_output()
            self.ordinalenconding.options = []
            self.ord_btn.layout.display = 'block'
            self.ord_btn.layout.visibility = 'visible'
            self.ord_btn2.layout.display = 'block'
            self.ord_btn2.layout.visibility = 'visible'
            self.ordinalenconding.layout.display = 'block'
            self.ordinalenconding.layout.visibility = 'visible'
        else: 
            self.ordinalenconding.options = []
            self.ord_btn.layout.visibility = 'hidden'
            self.ord_btn.layout.display = 'none'
            self.ord_btn2.layout.visibility = 'hidden'
            self.ord_btn2.layout.display = 'none'
            self.ordinalenconding.layout.visibility = 'hidden'
            self.ordinalenconding.layout.display = 'none'
  
        return
    def AddftPCA(self,event):

        ftname = self.main_view.dt_features.value

        if not ftname in self.pcaselect.options:
            newosp = [op for op in self.pcaselect.options]
            newosp.append(ftname)
            self.pcaselect.options = newosp
            

        return

    def SelectOrdFeature(self,event):

        ftname = self.main_view.dt_features.value

        data_df = self.controller.get_curr_df()

        # add checking if this is a categorical feature type
        self.ordinalenconding.options= [val for val in data_df[ftname].unique()]

        return

    def OrdinalMoveup(self,event):

        classname = self.ordinalenconding.value
        allclasses = [cls for cls in self.ordinalenconding.options]

        classind = allclasses.index(classname)

        if classind > 0: 
            allclasses.remove(classname)
            allclasses.insert(classind-1,classname)
            self.ordinalenconding.options= [val for val in allclasses]
     

        return


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

        self.pca_btn.layout.visibility = 'hidden'
        self.pca_btn.layout.display = 'none'
        self.pcaselect.layout.visibility = 'hidden'
        self.pcaselect.layout.display = 'none'

       
    
    

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