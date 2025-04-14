from IPython.display import clear_output
from IPython import display
from ipywidgets import *

class PredictiveModelingView:
    def __init__(self, controller, main_view):
        self.controller = controller
        self.main_view = main_view

    def models_click(self,change):     
        global  trmodels,model_sumry

        model_sumry.value = ''

        for mdl in self.controller.get_trained_models():
            if trmodels.value == mdl.getType():
                model_sumry.value += 'Model-> '+mdl.getType()+'\n'
                for prf,val in mdl.GetPerformanceDict().items():
                    model_sumry.value += 'Performance-> '+prf+': '+str(round(val,3))+'\n'
                break
        return

    def TrainModel(self,event): 
        global t4_models,t4_results,trmodels

        t4_results.value += 'Train Model...'+'\n'
    
        self.controller.train_Model(self.main_view.prdtsk_lbl.value,t4_models.value,t4_results,trmodels)
        return

    def get_predictive_modeling_tab(self):
        global t4_models,t4_results,trmodels, model_sumry
        trnmllay = Layout(width='150px')
        trnml_btn = widgets.Button(description="Train Model",layout = trnmllay)
        trnml_btn.on_click(self.TrainModel)

        t4_rslay = widgets.Layout(height='150x', width="99%")
        t4_results = widgets.Textarea(value='', placeholder='',description='',disabled=True,layout = t4_rslay)

        t4_ftlay2 =  widgets.Layout( width="90%")
        t4_models = widgets.Dropdown(options=['Decision Tree','KNN','Random Forest','SVM','Logistic Regression','Linear Model'],description = 'Models',layout = t4_ftlay2)
        t4_vb1lay =  widgets.Layout( width="50%")
        t4_vb2lay =  widgets.Layout( width="50%")

        trmodels = widgets.Select(options=[],description = '')
        trmodels.observe(self.models_click, 'value')
        mdlsum_lay = widgets.Layout( height ="150x")
        model_sumry = widgets.Textarea(options=[],description = '',layout = mdlsum_lay)


        trmlbl = widgets.Label( value="Trained models: ")
        mdsmrylbl = widgets.Label( value="Model summary: ")


        mlsboxxlay = widgets.Layout()
        mlsel_box = VBox(children=[self.main_view.trg_lbl,self.main_view.prdtsk_lbl,HBox(children=[t4_models,trnml_btn]),trmlbl,trmodels,mdsmrylbl,model_sumry],layout = mlsboxxlay)

        t4_vb1lay = widgets.Layout()
        tb4_vbox1 = VBox(children = [HBox(children=[self.main_view.f_box,mlsel_box]),t4_results],layout = t4_vb1lay)

        tab_4 = HBox(children=[tb4_vbox1])
        return tab_4