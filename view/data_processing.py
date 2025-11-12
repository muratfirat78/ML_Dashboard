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
       
        self.pca_btn = None
        self.pcaselect = None
        self.task_menu = task_menu
        self.ordinalenconding = None
        self.ord_btn = None
        self.ord_btn2 = None
        self.feattitle = None
        self.proctitle = None
        self.nooutliers = None
        self.progress = None

        
        self.methodslbl = None
        self.processmethods = dict()
        self.processvisuals = dict()
        
        self.methodsmenu = None

    def featureprclick(self,features2,FeatPage,processtypes,ProcssPage):  
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
        

        return
    
    def savecurrdata(self,event):
        self.controller.savedata(self.controller.get_datafolder(), self.main_view.datasets.value)

    def makebalanced(self,event):  
      
        
        
        return


   
    def featurepr_click(self,event):  
        
        self.featureprclick(self.main_view.dt_features,self.main_view.feat_page,self.main_view.process_types,self.main_view.process_page)

        
        return

    def selectProcessType(self,event):   

        self.methodslbl.layout.visibility = 'hidden'
        self.methodslbl.layout.display = 'none'

        self.methodsmenu.layout.visibility =  'hidden'
        self.methodsmenu.layout.display = 'none'

        self.ApplyButton.layout.display = 'block'
        self.ApplyButton.layout.visibility = 'visible'

       

        self.progress.value+="In select process type.."+"\n"

        for selectedprocess,myitem in self.processvisuals.items():
            if isinstance(myitem, list):
                for visitem in myitem:
                    visitem.layout.visibility = 'hidden'
                    visitem.layout.display = 'none'
                    
            if isinstance(myitem, dict):
                for methodname,methoditems in myitem.items():
                    for methoditem in methoditems:
                        methoditem.layout.visibility = 'hidden'
                        methoditem.layout.display = 'none'  
            

        selectedprocess = self.main_view.process_types.value

        self.progress.value+="Select process type.."+selectedprocess+"\n"


        if selectedprocess in self.processmethods:
            
            self.methodsmenu.options = [x for x in self.processmethods[selectedprocess]]
            self.methodsmenu.value = self.methodsmenu.options[0]

            self.methodslbl.layout.display = 'block'
            self.methodslbl.layout.visibility = 'visible'
            self.methodsmenu.layout.display = 'block'
            self.methodsmenu.layout.visibility = 'visible'

        if selectedprocess in self.processvisuals:
            if isinstance(self.processvisuals[selectedprocess], list):
                for visitem in self.processvisuals[selectedprocess]:
                    visitem.layout.display = 'block'
                    visitem.layout.visibility = 'visible'
          
        return



    def ApplyMethod(self,event):  

        global predictiontask
        refreshFeatures = True


        with self.main_view.vis_page:
            clear_output()
        
        processtype = self.methodsmenu.value

         

        if self.main_view.process_types.value == "Data Split":
            if not self.controller.main_model.datasplit: 
                self.controller.make_split(self.splt_txt,self.progress)
                self.controller.main_model.datasplit = True
            else:
                self.progress.value+="> Data is already split... "+"\n"
                
        if self.main_view.process_types.value == "Assign Target":
            self.progress.value+="> Target ... "+"\n"  
            if self.controller.main_model.targetcolumn == None:
                self.controller.assign_target(self.main_view.trg_lbl,self.main_view.dt_features,self.main_view.prdtsk_lbl,self.progress,predictiontask)
            else:
                self.progress.value+="> Target is already assigned... "+"\n"   
                
           
        if self.main_view.process_types.value == "Scaling":
            self.controller.make_scaling(self.main_view.dt_features,self.main_view.process_page,processtype,self.progress)
        if self.main_view.process_types.value == "Encoding":
            self.controller.make_encoding(self.main_view.dt_features,processtype,self.ordinalenconding,self.progress)
        if self.main_view.process_types.value == "Outlier":
            self.controller.remove_outliers(self.main_view.dt_features,self.methodsmenu.value,self.progress)
        if self.main_view.process_types.value == "Imbalancedness":
            self.controller.make_balanced(self.main_view.dt_features,processtype,self.main_view.process_page,self.progress)
        if self.main_view.process_types.value == "Feature Extraction":
            if processtype == "Correlation":
                self.controller.showCorrHeatMap(self.main_view.feat_page,processtype,self.progress)
                refreshFeatures = False
            if processtype == "PCA":
                self.controller.ApplyPCA(self.main_view.dt_features,self.pcaselect,self.progress)
                self.pcaselect.options = []
                self.pca_btn.layout.visibility = 'hidden'
                self.pca_btn.layout.display = 'none'
                self.pcaselect.layout.visibility = 'hidden'
                self.pcaselect.layout.display = 'none'
                
        if self.main_view.process_types.value == "Convert Feature 0/1->Bool":
            self.controller.make_featconvert(self.main_view.dt_features,self.progress)
            
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
            self.featureprclick(self.main_view.dt_features,self.main_view.feat_page,self.main_view.process_types,self.main_view.process_page)


        
        return

   

    def get_data_processing_tab(self):
     
        fpgelay = Layout(width="100%")
        self.main_view.feat_page = widgets.Output(layout = fpgelay)
        fpgelay = Layout(width="100%")
        self.main_view.process_page = widgets.Output(layout=fpgelay)

    
        self.processmethods = dict()
        self.processmethods['Scaling'] = ['Standardize','Normalize']
        self.processmethods['Encoding'] = ['Label Encoding','One Hot Encoding','Ordinal Encoding']
        self.processmethods['Feature Extraction'] = ['PCA','Correlation']
        self.processmethods['Outlier'] = ['IQR','Z-scores']
        self.processmethods['Imbalancedness'] = ['Upsample','DownSample']

        self.processvisuals = dict()
       

        processmethods = [ x for x in self.processmethods.keys()]
        processmethods.insert(2,'Convert Feature 0/1->Bool')
        processmethods.insert(3,'Assign Target')
        processmethods.insert(4,'Data Split')

        self.main_view.process_types = widgets.Select(description = '',options= processmethods ,disabled=False)
        self.main_view.process_types.observe(self.selectProcessType,'value')
        self.main_view.process_types.layout.width = '200px'
        self.main_view.process_types.layout.height = '150px'

    
        

       

        
        self.ApplyButton = widgets.Button(description="Apply",layout=widgets.Layout(width='60px'))
        self.ApplyButton.on_click(self.ApplyMethod)

        self.ApplyButton.layout.width =  self.main_view.process_types.layout.width 

     
        self.main_view.trg_lbl = widgets.Label(value ='Target: -',disabled = True)
        self.main_view.prdtsk_lbl =widgets.Label(value = ' | Prediction Task: - ',disabled = True)
        trglay = Layout(width='150px')
        
      

        self.processvisuals['Assign Target'] = [self.main_view.trg_lbl,self.main_view.prdtsk_lbl]

        self.splt_txt =widgets.Dropdown(description ='Test Ratio(%): ',options=[20,25,30,35])
        self.splt_txt.layout.width = '160px'
       
        self.processvisuals['Data Split'] = [self.splt_txt]


        self.main_view.vis_page = widgets.Output()

        self.pca_btn = widgets.Button(description=">> Add PCA >> ")
        self.pca_btn.layout.visibility = 'hidden'
        self.pca_btn.layout.display = 'none'
        self.pca_btn.layout.width = '120px'
        self.pca_btn.on_click(self.AddftPCA)

        self.pcaselect = widgets.Select(options=[],description = '')
        self.pcaselect.layout.visibility = 'hidden'
        self.pcaselect.layout .display = 'none'

        self.processvisuals['Feature Extraction'] = dict()
        self.processvisuals['Feature Extraction']['PCA'] =[self.pca_btn,self.pcaselect]
        

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
        self.ord_btn2.on_click(self.OrdinalMoveup)

        self.processvisuals['Encoding'] = dict()
        self.processvisuals['Encoding']['Ordinal Encoding'] =[self.ordinalenconding,self.ord_btn,self.ord_btn2]


        self.main_view.dt_ftslay =  widgets.Layout( width="99%",display = 'block')
        self.main_view.dt_features = widgets.Select(options=[],description = '',layout = self.main_view.dt_ftslay)
        self.main_view.dt_features.observe(self.featurepr_click, 'value')
       
        self.proctitle = widgets.HTML("")
        color = "gray"
        mytext ="Process types"
        self.proctitle.value = f'<span style="color:{color};"><b>{mytext}</b></span>'

    
        self.methodslbl = widgets.Label(value ='Methods',layout = widgets.Layout(width="99%",visibility = 'hidden'))

        self.methodsmenu = widgets.Dropdown( options=[], description='', disabled=False,layout = widgets.Layout(width="99%",display = 'none'))
        
        self.methodsmenu.observe(self.MethodView)
        
        self.selcl = widgets.HTML("", layout=widgets.Layout(height="20px", width="99%", text_align="center"))
        
    
        self.nooutliers = widgets.Label(value ='Outliers: ',layout = widgets.Layout(width="25%",visibility = 'hidden'))
        self.thrshlds = widgets.Label(value ='Thresholds: ',layout = widgets.Layout(width="25%",visibility = 'hidden'))

       

       

        self.splt_txt.layout.visibility = 'hidden'
        self.splt_txt.layout.display = 'none'
    

        self.main_view.trg_lbl.layout.visibility = 'hidden'
        self.main_view.trg_lbl.layout.display = 'none'
        
        self.main_view.prdtsk_lbl.layout.visibility = 'hidden'
        self.main_view.prdtsk_lbl.layout.display = 'none'
        self.ApplyButton.layout.visibility = 'hidden'
        self.ApplyButton.layout.display = 'none'
        self.nooutliers.layout.visibility = 'hidden'
        self.nooutliers.layout.display = 'none'
       
        self.methodslbl.layout.visibility = 'hidden'
        self.methodslbl.layout.display = 'none'


      

        sboxxlay = widgets.Layout()
        sel_box = VBox(children=[self.selcl,
                                  widgets.Box(layout=widgets.Layout(border='solid 1px lightblue', width='99%', height='1px', margin='5px 0px',style={'background': "#C7EFFF"})),
                                 self.proctitle, 
                                 HBox(children=[self.main_view.process_types]),
                                 self.ApplyButton,
                                 HBox(children=[self.main_view.trg_lbl,self.main_view.prdtsk_lbl]),
                                 HBox(children=[self.splt_txt]),
                                 HBox(children=[self.methodslbl,self.methodsmenu]),    
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
        self.progress = widgets.Textarea(value='', placeholder='',description='',disabled=True,layout = res2lay)

        vb1lay =  widgets.Layout()
        prboxlay = widgets.Layout()
        vbox1 = VBox(children = [HBox(children=[self.main_view.f_box,sel_box],layout = prboxlay),self.progress],layout = vb1lay)

        vb2lay =  widgets.Layout()
        vbox2 = VBox(children = [self.main_view.feat_page,self.main_view.process_page],layout = vb2lay)
        tab_3 = VBox([self.task_menu,HBox([vbox1,vbox2])])
        return tab_3

    def MethodView(self,event):

        selectedprocess = self.main_view.process_types.value
        selectedmethod = self.methodsmenu.value

        self.progress.value+="> Method observe... "+", "+str(selectedprocess)+","+str(selectedmethod)+"\n"   
        
        for myprocess,myitem in self.processvisuals.items():
            if isinstance(myitem, dict):
                for methodname,methoditems in myitem.items():
                    for methoditem in methoditems:
                        methoditem.layout.visibility = 'hidden'
                        methoditem.layout.display = 'none'  

       
        if selectedprocess in self.processvisuals:
            self.progress.value+="> in visuals... "+"\n" 
            if isinstance(self.processvisuals[selectedprocess], dict):
                self.progress.value+="> dict... "+selectedprocess+"\n" 

                self.progress.value+=str(self.processvisuals[selectedprocess].keys())+"\n" 
                if selectedmethod in self.processvisuals[selectedprocess]:
                    self.progress.value+="> in second list... "+"\n" 
                    for methoditem in self.processvisuals[selectedprocess][selectedmethod]:
                        self.progress.value+="> item "+str(type(methoditem))+"\n" 
                        methoditem.layout.display = 'block'  
                        methoditem.layout.visibility = 'visible'
    
    
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


    