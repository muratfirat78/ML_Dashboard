from IPython.display import clear_output
from IPython import display
from ipywidgets import *
from sklearn import tree,metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from array import array

class PredictiveModelingView:
    def __init__(self, controller, main_view,task_menu):
        self.controller = controller
        self.main_view = main_view
        self.dtcrit = None
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
        self.task_menu = task_menu
        self.modelmenu = None
        self.mdltitle = None
        self.paramtitle = None
        self.parammenu = None
        self.paramvalues = None
        self.selectedmodel = None
        self.selectedparam = None
        self.selectedparamval = None
        self.paramvals = None
        self.paramedit = None
        self.modelslbl = None
        self.trmodels = None
        self.perlbl = None
        self.tasklbl = None
        self.trnml_btn = None

        self.paramchangective = False

        self.prm1lbl = None
        self.prm2lbl = None
        self.prm3lbl = None
        self.paramlbls = None
        self.performlbl = None
        
        self.progress = None
        self.selectedmodel = None
        
    def models_click(self,change):     
   

        for mdl in self.controller.get_trained_models():
            if self.trmodels.value == mdl.getName():

                self.selectedmodel  = mdl
                self.progress.value += 'model found...'+str(mdl.getName())+'\n'

                self.progress.value += 'labels...'+str(len(self.paramlbls))+'\n'

                for mylbl in self.paramlbls:
                    self.progress.value += 'lbl type...'+str(type(mylbl))+'\n'
                    mylbl.value = ''

                self.progress.value += 'start...'+str(mdl.getName())+'\n'


                self.progress.value += 'params...'+str(len(mdl.getModelParams()))+'\n'
                
                prind = 0
                for param,val in mdl.getModelParams().items():
                    self.progress.value += str(param)+": "+str(val)+'\n'
                    self.paramlbls[prind].value = str(param)+": "+str(val)
                    prind+=1

                if 'Accuracy' in mdl.GetPerformanceDict():
                    self.performlbl.value = " Accuracy: "+str(mdl.GetPerformanceDict()['Accuracy'])

                if 'MSE' in mdl.GetPerformanceDict():
                    self.performlbl.value = " MSE: "+str(mdl.GetPerformanceDict()['MSE'])

                self.progress.value += 'labels set...'+'\n'
                    


                with self.performpage:          
                    clear_output()
                    
                    if mdl.getTask() == "Classification":
                      
                        classes = [cls for cls in self.controller.main_model.getYtrain().to_frame()[self.controller.main_model.targetcolumn].unique()]

                        # Normalized confusion matrix
                        cm = mdl.getConfMatrix() / mdl.getConfMatrix().sum(axis = 1, keepdims = True)
                        
                        # Color map
                        cmap = sns.diverging_palette(220, 20, as_cmap = True)
                        
                        # Seaborn Heatmap
                        sns.heatmap(cm, cmap = cmap, center = 0, annot = mdl.getConfMatrix(), fmt = 'd',
                                    xticklabels = classes, yticklabels = classes, annot_kws = {'size': 12})
                        
                        
                        plt.xlabel('Predicted Class', fontsize=12)
                        plt.ylabel('Actual Class', fontsize=12)
                        plt.title('Confusion Matrix', fontsize=14)
                        plt.show()

                        display.display('__________________________________')                    
            
                        if len(self.controller.main_model.getYtrain().to_frame()[self.controller.main_model.targetcolumn].unique()) == 2:
                            perf = mdl.GetPerformanceDict()
                            if 'ROCFPR' in perf and 'ROCTPR' in perf:
                                roc_auc = metrics.auc(perf['ROCFPR'],perf['ROCTPR'])
                                display0 = metrics.RocCurveDisplay(fpr=perf['ROCFPR'], tpr=perf['ROCTPR'], roc_auc=roc_auc, estimator_name=mdl.getName())
                                display0.plot()
                                plt.title('ROC Curve', fontsize=14)
                                plt.show()
                        # display.display('Confusion Matrix: ')
                        # display.display(mdl.getConfMatrix())
                        
                    else:
                        
                        if not mdl.istraindatachanged():
                            
                            self.plt_btn.visible = 'visible'
                            self.plt_btn.disabled = False
        
                        else: 
                            self.plt_btn.visible = 'hidden'

                        
              
                        
                        
                
        return

    def ShowParamOpts(self,event):


        self.paramchangective = True
        self.progress.value+="********** ShowOptions***********"+"\n"

        self.progress.value+="Model "+str(self.selectedmodel)+"\n"

        self.progress.value+="Parameter: "+str(self.selectedparam)+"\n"
        
        self.progress.value+="Parameter menu value: "+str(self.parammenu.value)+"\n"
        

        self.selectedparam = self.parammenu.value

        self.progress.value+="Set Parameter: "+str(self.selectedparam)+"\n"
        
        

        if self.selectedparam == '' or self.selectedparam == None:
            self.paramchangective = False
            return


        if self.selectedparam in self.mlmodels[self.selectedmodel]:
            self.progress.value+="parameter options..: "+str(self.selectedparam)+"\n"

            self.paramoptions.options = [x for x in self.mlmodels[self.selectedmodel][self.selectedparam]] 
            self.paramoptions.value = self.mlmodelparams[self.selectedmodel][self.selectedparam]

            self.progress.value+="set done options: "+str(self.selectedparam)+"\n"
            
        else:
            self.paramoptions.options = []

     
        self.paramchangective = False
        return

    def ChangeParamVal(self,event):

        self.progress.value+="**********ChangeParam***********"+"\n"
        self.progress.value+="in change paramter..."+"\n"

        if self.paramchangective:
            return

        self.progress.value+="Model "+str(self.selectedmodel)+"\n"

        self.progress.value+="Parameter: "+str(self.selectedparam)+"\n"
    

        if self.selectedparam == '' or self.selectedparam == None:
            return

       
        self.progress.value+="Parameter value: "+ str(self.selectedparamval)+"\n"

        if self.paramoptions.value == ''  or self.paramoptions.value==  None:
            return

        self.selectedparamval = self.paramoptions.value
     
        self.mlmodelparams[self.selectedmodel][self.selectedparam] = self.selectedparamval
        
        self.paramvalues.options = [self.mlmodelparams[self.selectedmodel][x] for x in self.mlmodels[self.selectedmodel].keys()] 
        
       

        return
        
    

    def ShowModel(self,event):

      
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
        self.lmlib.layout.visibility = 'hidden'
        self.lmlib.layout.display = 'none'

        self.progress.value+="**********ShowModel***********"+"\n"
  
        self.selectedmodel = self.modelmenu.value

        self.progress.value+=" Set Model "+str(self.selectedmodel)+"\n"
   


        self.paramvalues.options = [self.mlmodelparams[self.selectedmodel][x] for x in self.mlmodels[self.selectedmodel].keys()] 
        self.parammenu.options = [x for x in self.mlmodels[self.selectedmodel].keys()] 
        
    
        if self.modelmenu.value == "Decision Tree":
          
            self.dtcrit.layout.display = 'block'
            self.dtcrit.layout.visibility = 'visible'
            self.dtminseg.layout.display = 'block'
            self.dtminseg.layout.visibility = 'visible'
            self.mdplbl.layout.display = 'block'
            self.mdplbl.layout.visibility = 'visible'
            self.mgplbl.layout.display = 'block'
            self.mgplbl.layout.visibility = 'visible'
        if self.modelmenu.value == "KNN":
            self.knnkval.layout.display = 'block'
            self.knnkval.layout.visibility = 'visible'
            self.knnmetric.layout.display = 'block'
            self.knnmetric.layout.visibility = 'visible'
        if self.modelmenu.value == "Random Forest":
            self.rfnrest.layout.display = 'block'
            self.rfnrest.layout.visibility = 'visible'
            if self.main_view.prdtsk_lbl.value[self.main_view.prdtsk_lbl.value.find(":")+2:] == 'Classification': 
                self.rfcrit.layout.display = 'block'
                self.rfcrit.layout.visibility = 'visible'
        if self.modelmenu.value == "SVM":
            self.svcc.layout.display = 'block'
            self.svcc.layout.visibility = 'visible'
            self.svckrnl.layout.display = 'block'
            self.svckrnl.layout.visibility = 'visible'
        if self.modelmenu.value == "Linear Model":
            self.lmlib.layout.display = 'block'
            self.lmlib.layout.visibility = 'visible'
            
                
        return

    def PlotGraph(self,event):

        with self.performpage:          
            clear_output()
            Xtrain_df = self.controller.main_model.get_XTrain()
            ytrain_df = self.controller.main_model.getYtrain()
            ytest_df = self.controller.main_model.get_YTest()
            Xtest_df = self.controller.main_model.get_XTest()
    
            pred_df = pd.DataFrame(columns = ['y_true','y_pred','tag'])
        
            preds = self.selectedmodel.GetPredictions(Xtest_df)
            pred_df['y_true'] = [x for x in ytest_df]
            pred_df['y_pred'] = [x for x in preds]
            pred_df['tag'] = ['test' for i in preds]
        
            trpred_df = pd.DataFrame(columns = ['y_true','y_pred','tag'])
        
            ytr_pred = self.selectedmodel.GetPredictions(Xtrain_df)
            ytr_pred = ytr_pred.tolist()
            ytrain_df = ytrain_df.to_list()
        
                                
            trpred_df['y_true'] = ytrain_df
            trpred_df['y_pred'] = ytr_pred
            trpred_df['tag'] = ['train' for i in ytr_pred]
        
            pred_df = pd.concat([pred_df, trpred_df], ignore_index=True)
        
        
            g = sns.lmplot(x='y_true', y ='y_pred', data=pred_df, hue='tag')
            g.fig.suptitle('True Vs Pred', y= 1.02)
            g.set_axis_labels('y_true', 'y_pred');
            plt.show()  

        
        
        
        self.plt_btn.disabled = True


        return 


    def TrainModel(self,event): 


        self.trnml_btn.disabled = True

        self.progress.value += 'Train Model...'+str(self.selectedmodel)+'\n'

        
        params= dict()

        for param,val in self.mlmodelparams[self.selectedmodel].items():
            params[param] = val

        self.progress.value += 'Params...'+str(params)+'\n'

        
        self.controller.train_Model(self.selectedmodel,self.progress,self.trmodels ,params)
 

        self.trnml_btn.disabled = False
        return

    def get_predictive_modeling_tab(self):
        
   
 
        self.trnml_btn= widgets.Button(description="Train")
        self.trnml_btn.layout.width = '98px'
        self.trnml_btn.on_click(self.TrainModel)

        self.plt_btn= widgets.Button(description="Plot true vs predicted values")
        self.plt_btn.layout.width = '200px'
        self.plt_btn.on_click(self.PlotGraph)

        self.plt_btn.visible = 'hidden'

        t4_rslay = widgets.Layout(height='150x', width="99%")
        
        self.progress = widgets.Textarea(value='', placeholder='',description='',disabled=True,layout = t4_rslay)
        self.progress.layout.height = '100px'

        self.mlmodels = dict()
        self.mlmodelparams = dict()
   

        self.mlmodels['Decision Tree'] = dict()
        self.mlmodels['KNN'] = dict()
        self.mlmodels['Random Forest'] = dict()
        self.mlmodels['SVM'] = dict()
        self.mlmodels['Logistic Regression'] = dict()
        self.mlmodels['Linear Model'] = dict()
        self.mlmodels['Gaussian NB'] = dict()

        self.mlmodelparams['Decision Tree'] = dict()
        self.mlmodelparams['KNN'] = dict()
        self.mlmodelparams['Random Forest'] = dict()
        self.mlmodelparams['SVM'] = dict()
        self.mlmodelparams['Logistic Regression'] = dict()
        self.mlmodelparams['Linear Model'] = dict()
        self.mlmodelparams['Gaussian NB'] = dict()

        

        self.mdltitle = widgets.HTML("")
        color = "gray"
        mytext ="ML Models"
        
        self.mdltitle.value = f'<span style="color:{color};"><b>{mytext}</b></span>'
       


        self.trmodels = widgets.Select(options=[],description = '')
        self.trmodels.layout.width = '220px'
        self.trmodels.layout.height = '140px'
        self.trmodels.observe(self.models_click, 'value')
     


     
        self.mlmodels['Decision Tree']['max_depth'] = [x for x in range(1,16)]
        self.mlmodels['Decision Tree']['criterion'] = ['entropy','gini']
        self.mlmodels['Decision Tree']['min_samples_split'] = [x for x in range(1,5)]

        self.mlmodelparams['Decision Tree']['max_depth'] = 5
        self.mlmodelparams['Decision Tree']['criterion'] = 'gini'
        self.mlmodelparams['Decision Tree']['min_samples_split'] = 3


        

        self.mlmodels['KNN']['n_neighbors'] = [x for x in range(1,16)]
        self.mlmodels['KNN']['weights'] = ['uniform','distance']
        self.mlmodels['KNN']['metric'] = ['minkowski','euclidean']

        self.mlmodelparams['KNN']['n_neighbors'] = 8
        self.mlmodelparams['KNN']['weights'] = 'uniform'
        self.mlmodelparams['KNN']['metric'] = 'euclidean'

        self.mlmodels['Random Forest']['estimators'] = [i for i in range(15,150,15)]
        self.mlmodels['Random Forest']['criterion'] = ['entropy','gini']
        self.mlmodels['Random Forest']['max_depth'] = [x for x in range(1,7)]

        self.mlmodelparams['Random Forest']['estimators'] = 90
        self.mlmodelparams['Random Forest']['criterion'] = 'entropy'
        self.mlmodelparams['Random Forest']['max_depth'] = 4

        self.mlmodels['SVM']['kernel'] = ['linear','poly','rbf','sigmoid']
        self.mlmodels['SVM']['C'] = [0.1*i for i in range(10,1,-1)]
        self.mlmodels['SVM']['degree'] = ['minkowski','euclidean']

        self.mlmodelparams['SVM']['kernel'] = 'linear'
        self.mlmodelparams['SVM']['C'] = 1.0
        self.mlmodelparams['SVM']['degree'] = 'minkowski'


        
        self.mlmodels['Logistic Regression']['C'] = [round(0.1*i,2) for i in range(10,1,-1)]
        self.mlmodels['Logistic Regression']['penalty'] = ['l1', 'l2', 'elasticnet']


        self.mlmodelparams['Logistic Regression']['C'] = 1.0
        self.mlmodelparams['Logistic Regression']['penalty'] = 'l1'


        self.mlmodels['Linear Model']['validation_fraction'] = [round(0.1*i,2) for i in range(10,0,-1)]
        self.mlmodels['Linear Model']['penalty'] = ['l1', 'l2', 'elasticnet']

        self.mlmodelparams['Linear Model']['validation_fraction'] = 0.1
        self.mlmodelparams['Linear Model']['penalty'] = 'l2'

        


        self.modelmenu = widgets.Select(options=[x for x in self.mlmodels.keys()])
        self.modelmenu.layout.height = '175px'
        self.modelmenu.layout.width = '120px'
        self.modelmenu.observe(self.ShowModel)

        self.tasklbl = widgets.Label(value ='Perdiction task: -')
        
     
        mdsmrylbl = widgets.Label( value="Model summary: ")

        self.parammenu = widgets.Select(options=[])
        self.parammenu.layout.height = '80px'
        self.parammenu.layout.width = '120px'
        self.parammenu.observe(self.ShowParamOpts)

        self.paramvalues = widgets.Select(options=[],disabled = True)
        self.paramvalues.layout.height = '80px'
        self.paramvalues.layout.width = '120px'
       

        self.paramoptions = widgets.Dropdown(options=[],description = '')
        self.paramoptions.layout.width = '100px'
        self.paramoptions.observe(self.ChangeParamVal)


        self.prm1lbl = widgets.Label( value="")
        self.prm1lbl.layout.width = '120px'
        self.prm2lbl = widgets.Label( value="")
        self.prm2lbl.layout.width = '120px'
        self.prm3lbl = widgets.Label( value="")
        self.prm3lbl.layout.width = '120px'
        self.performlbl = widgets.Label( value="")

        self.paramlbls = [self.prm1lbl,self.prm2lbl,self.prm3lbl]

        
        self.dtdepth = widgets.Dropdown(options=[i for i in range(1,16)],description = 'MaxDepth')
        self.dtdepth.layout.width = '125px'
        self.dtdepth.layout.visibility = 'hidden'
        self.dtdepth.layout.display = 'none'
        self.dtcrit = widgets.Dropdown(options=['entropy','gini'],description = 'Criterion')
        self.dtcrit.layout.width = '140px'
        self.dtcrit.layout.display = 'none'
        self.dtminseg = widgets.Dropdown(options=[i for i in range(1,6)],description = 'MinSeg')
        self.dtminseg.layout.width = '125px'
        self.dtminseg.layout.display = 'none'

        self.knnkval = widgets.Dropdown(options=[i for i in range(1,15)],description = 'N.Size')
        self.knnkval.layout.width = '125px'
        self.knnkval.layout.display = 'none'
        self.knnmetric = widgets.Dropdown(options=['minkowski','euclidean','manhattan'],description = 'Dist.')
        self.knnmetric.layout.width = '125px'
        self.knnmetric.layout.display = 'none'

        self.mdplbl = widgets.Label( value="MaxDep")
        self.mdplbl.layout.visibility = 'hidden'
        self.mdplbl.layout.display = 'none'
        self.mgplbl = widgets.Label( value="MinSeg: ")
        self.mgplbl.layout.display = 'none'

        self.lmlib = widgets.Dropdown(options=['SklearnLR','OLS',],description = 'Library')
        self.lmlib.layout.width = '125px'
        self.lmlib.layout.display = 'none'


        self.svcc =widgets.Dropdown(options=[0.1*i for i in range(10,1,-1)],description = 'C')
        self.svcc.layout.display = 'none'
        self.svckrnl = widgets.Dropdown(options=['linear','poly','rbf','sigmoid'],description = 'kernel')
        self.svckrnl.layout.display = 'none'

        self.rfnrest =widgets.Dropdown(options=[i for i in range(15,150,15)],description = 'estimators')
        self.rfnrest.layout.display = 'none'
        self.rfcrit = widgets.Dropdown(options=['entropy','gini'],description = 'criterion')
        self.rfcrit.layout.display = 'none'


        self.paramtitle = widgets.HTML("")
        color = "gray"
        mytext ="Parameters"
        self.paramtitle.value = f'<span style="color:{color};"><b>{mytext}</b></span>'
        self.paramtitle.layout.width = '120px'

        self.paramedit = widgets.HTML("")
        color = "gray"
        mytext ="Modify"
        self.paramedit.value = f'<span style="color:{color};"><b>{mytext}</b></span>'
        self.paramedit.layout.width = '100px'

        self.paramvals = widgets.HTML("")
        color = "gray"
        mytext ="Values"
        self.paramvals.value = f'<span style="color:{color};"><b>{mytext}</b></span>'
        self.paramvals.layout.width = '120px'

        self.modelslbl = widgets.HTML("")
        color = "gray"
        mytext ="Trained Models"
        self.modelslbl.value = f'<span style="color:{color};"><b>{mytext}</b></span>'
        self.modelslbl.layout.width = '120px'


        self.perlbl = widgets.HTML("")
        color = "gray"
        mytext ="Performance"
        self.perlbl.value = f'<span style="color:{color};"><b>{mytext}</b></span>'
        self.perlbl.layout.width = '120px'

        horzbar = widgets.Box(layout=widgets.Layout(border='solid 1px lightblue', width='99%', height='1px', margin='5px 0px',style={'background': "#C7EFFF"}))
      
        sel_box = VBox(children=[ self.tasklbl,
            HBox(children=[self.paramtitle,self.paramedit,self.paramvals]),
                               horzbar,
                                 HBox(children=[self.parammenu,VBox(children=[self.paramoptions, self.trnml_btn]),self.paramvalues]),
                                 self.modelslbl,horzbar,
                                 HBox(children=[
                                     VBox(children = [self.trmodels,self.plt_btn]),
                                     VBox(children = [self.prm1lbl,self.prm2lbl,self.prm3lbl,self.perlbl,horzbar,self.performlbl])])
                                ])

        fbox2alay = widgets.Layout(width = '35%')

        self.f_box = VBox(children=[self.mdltitle,
                                    HBox(children=[self.modelmenu])],layout = fbox2alay)

        vb1lay =  widgets.Layout(width='55%')
        prboxlay = widgets.Layout(width= '99%')

        vbox1 = VBox(children = [HBox(children=[self.f_box,sel_box],layout = prboxlay),self.progress],layout = vb1lay)


        self.performpage = widgets.Output()
        separator = widgets.Box(layout=widgets.Layout(border='solid 1px lightblue', width='1px', height='90%', margin='5px 0px',style={'background': "#C7EFFF"}))


        vbox2 = VBox(children = [self.performpage])

        tab_4 = VBox([self.task_menu,HBox([vbox1,vbox2])])
        
        #tab_4 = VBox([self.task_menu, HBox([VBox(children=[self.mdltitle,self.modelmenu]), separator,vbox1,vbox2])])
        
        tab_4.layout.height = '700px'
        return tab_4