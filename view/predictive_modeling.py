from IPython.display import clear_output
from IPython import display
from ipywidgets import *
from sklearn import tree,metrics
import matplotlib.pyplot as plt

class PredictiveModelingView:
    def __init__(self, controller, main_view):
        self.controller = controller
        self.main_view = main_view
        self.dtdepth = None
        self.dtcrit = None
        self.t4_models = None
        self.dtminseg = None
        self.mdplbl = None
        self.mgplbl = None
        self.knnkval = None
        self.knnmetric = None
        self.rfnrest = None
        self.rfcrit  = None
        self.svcc = None
        self.svckrnl= None
        self.performpage = None
        
    def models_click(self,change):     
        global  trmodels,model_sumry

        model_sumry.value = ''

        for mdl in self.controller.get_trained_models():
            if trmodels.value == mdl.getName():
                if len(mdl.GetPerformanceDict()) > 0:
                    model_sumry.value += 'Performance scores: '+'\n'

                for prf,val in mdl.GetPerformanceDict().items():
                    if (prf == 'ROCFPR') or (prf == 'ROCTPR'):
                        continue
                    model_sumry.value += '  > '+prf+': '+str(val)+'\n'
                    
                if ('ROCFPR' in mdl.GetPerformanceDict()) and ('ROCTPR' in mdl.GetPerformanceDict()):
              
                    try: 
                        with self.performpage:
                            
                            clear_output()
                            if mdl.getTask() == "Classification":
                                display.display('Confusion Matrix: ')
                                display.display(mdl.getConfMatrix())
                             
                            roc_auc = metrics.auc(mdl.GetPerformanceDict()['ROCFPR'],mdl.GetPerformanceDict()['ROCTPR'])
                            display0 = metrics.RocCurveDisplay(fpr=mdl.GetPerformanceDict()['ROCFPR'], tpr=mdl.GetPerformanceDict()['ROCTPR'], roc_auc=roc_auc, estimator_name=mdl.getName())
                            display0.plot()
                            plt.show()
                    except Exception as e: 
                        with self.performpage: 
                            clear_output()
                            display.display('Error occured: '+str(e))
                                
              
                        
                        
                
        return
        
    def ModelChange(self,event):

        self.dtdepth.layout.visibility = 'hidden'
        self.dtdepth.layout.display = 'none'
        self.dtcrit.layout.visibility = 'hidden'
        self.dtcrit.layout.display = 'none'
        self.dtminseg.layout.visibility = 'hidden'
        self.dtminseg.layout.display = 'none'
        self.mdplbl.layout.visibility = 'hidden'
        self.mdplbl.layout.display = 'none'
        self.mgplbl.layout.visibility = 'hidden'
        self.mgplbl.layout.display = 'none'
        self.knnkval.layout.visibility = 'hidden'
        self.knnkval.layout.display = 'none'
        self.knnmetric.layout.visibility = 'hidden'
        self.knnmetric.layout.display = 'none'
        self.rfnrest.layout.visibility = 'hidden'
        self.rfnrest.layout.display = 'none'
        self.rfcrit.layout.visibility = 'hidden'
        self.rfcrit.layout.display = 'none'
        self.svcc.layout.visibility = 'hidden'
        self.svcc.layout.display = 'none'
        self.svckrnl.layout.visibility = 'hidden'
        self.svckrnl.layout.display = 'none'


        
    
        if self.t4_models.value == "Decision Tree":
            self.dtdepth.layout.display = 'block'
            self.dtdepth.layout.visibility = 'visible'
            self.dtcrit.layout.display = 'block'
            self.dtcrit.layout.visibility = 'visible'
            self.dtminseg.layout.display = 'block'
            self.dtminseg.layout.visibility = 'visible'
            self.mdplbl.layout.display = 'block'
            self.mdplbl.layout.visibility = 'visible'
            self.mgplbl.layout.display = 'block'
            self.mgplbl.layout.visibility = 'visible'
        if self.t4_models.value == "KNN":
            self.knnkval.layout.display = 'block'
            self.knnkval.layout.visibility = 'visible'
            self.knnmetric.layout.display = 'block'
            self.knnmetric.layout.visibility = 'visible'
        if self.t4_models.value == "Random Forest":
            self.rfnrest.layout.display = 'block'
            self.rfnrest.layout.visibility = 'visible'
            if self.main_view.prdtsk_lbl.value[self.main_view.prdtsk_lbl.value.find(":")+2:] == 'Classification': 
                self.rfcrit.layout.display = 'block'
                self.rfcrit.layout.visibility = 'visible'
        if self.t4_models.value == "SVM":
            self.svcc.layout.display = 'block'
            self.svcc.layout.visibility = 'visible'
            self.svckrnl.layout.display = 'block'
            self.svckrnl.layout.visibility = 'visible'
            
                
        return

    def TrainModel(self,event): 
        global t4_results,trmodels

        t4_results.value += 'Train Model...'+'\n'
        params= []

        if self.t4_models.value == "Decision Tree":
            params= [self.dtdepth.value,self.dtminseg.value,self.dtcrit.value]
        if self.t4_models.value == "KNN":
            params= [self.knnkval.value,self.knnmetric.value]
        if self.t4_models.value == "Random Forest":
            params= [self.rfnrest.value,self.rfcrit.value]
        if self.t4_models.value == "SVM":
            params= [self.svcc.value,self.svckrnl.value]


    
        self.controller.train_Model(self.main_view.prdtsk_lbl.value,self.t4_models.value,t4_results,trmodels,params)
        return

    def get_predictive_modeling_tab(self):
        
        global t4_results,trmodels, model_sumry
 
        trnml_btn = widgets.Button(description="Train")
        trnml_btn.layout.width = '150px'
        trnml_btn.on_click(self.TrainModel)

        t4_rslay = widgets.Layout(height='150x', width="99%")
        t4_results = widgets.Textarea(value='', placeholder='',description='',disabled=True,layout = t4_rslay)

        t4_results.layout.height = '100px'

        
        self.t4_models = widgets.Dropdown(options=['Select','Decision Tree','KNN','Random Forest','SVM','Logistic Regression','Linear Model'],description = '')
        self.t4_models.layout.width = '200px'
        self.t4_models.observe(self.ModelChange)
       


        trmodels = widgets.Select(options=[],description = '')
        trmodels.observe(self.models_click, 'value')
     
        model_sumry = widgets.Textarea(options=[],description = '',disabled = True)
        model_sumry.layout.height = '150px'
       

     
        mdsmrylbl = widgets.Label( value="Model summary: ")
        self.dtdepth = widgets.Dropdown(options=[i for i in range(1,16)],description = 'MaxDepth')
        self.dtdepth.layout.width = '125px'
        self.dtdepth.layout.visibility = 'hidden'
        self.dtdepth.layout.display = 'none'
        self.dtcrit = widgets.Dropdown(options=['entropy','gini'],description = 'Criterion')
        self.dtcrit.layout.width = '175px'
        self.dtcrit.layout.display = 'none'
        self.dtminseg = widgets.Dropdown(options=[i for i in range(1,6)],description = 'MinSeg')
        self.dtminseg.layout.width = '125px'
        self.dtminseg.layout.display = 'none'

        self.knnkval = widgets.Dropdown(options=[i for i in range(1,15)],description = 'k')
        self.knnkval.layout.width = '125px'
        self.knnkval.layout.display = 'none'
        self.knnmetric = widgets.Dropdown(options=['minkowski','euclidean','manhattan'],description = 'distance')
        self.knnmetric.layout.width = '125px'
        self.knnmetric.layout.display = 'none'

        self.mdplbl = widgets.Label( value="MaxDepth: ")
        self.mdplbl.layout.visibility = 'hidden'
        self.mdplbl.layout.display = 'none'
        self.mgplbl = widgets.Label( value="MinSeg: ")
        self.mgplbl.layout.display = 'none'

        self.lmlib = widgets.Dropdown(options=['SklearnLR','OLS',],description = 'Library')
        self.knnmetric.layout.width = '125px'
        self.knnmetric.layout.display = 'none'


        self.svcc =widgets.Dropdown(options=[0.1*i for i in range(0,11)],description = 'C')
        self.svcc.layout.display = 'none'
        self.svckrnl = widgets.Dropdown(options=['linear','poly','rbf','sigmoid'],description = 'kernel')
        self.svckrnl.layout.display = 'none'

        self.rfnrest =widgets.Dropdown(options=[i for i in range(15,150,15)],description = 'estimators')
        self.rfnrest.layout.display = 'none'
        self.rfcrit = widgets.Dropdown(options=['entropy','gini'],description = 'criterion')
        self.rfcrit.layout.display = 'none'


        
      
        sel_box = VBox(children=[HBox(children=[self.main_view.trg_lbl,self.main_view.prdtsk_lbl]),
                                 HBox(children=[self.dtdepth,self.dtminseg,self.dtcrit]),
                                 HBox(children=[self.knnkval,self.knnmetric]),
                                 HBox(children=[self.rfnrest,self.rfcrit]),
                                 HBox(children=[self.svcc,self.svckrnl]),
                                 HBox(children=[self.t4_models,trnml_btn]),
                                 trmodels,model_sumry
                                ])

        vbox1 = VBox(children = [HBox(children=[self.main_view.f_box,sel_box]),t4_results],layout= Layout(width='65%'))

        self.performpage = widgets.Output()


        vbox2 = VBox(children = [self.performpage])
        tab_4 = HBox(children=[vbox1,vbox2])
        
        tab_4.layout.height = '450px'
        return tab_4