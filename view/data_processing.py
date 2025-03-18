from IPython.display import clear_output
from IPython import display
from ipywidgets import *
import settings
import matplotlib.pyplot as plt
import seaborn as sns

predictiontask = None

class DataProcessingView:
    def __init__(self, controller):
        self.controller = controller

    def featureprclick(self,features2,FeatPage,processtypes,ProcssPage,scalingacts):  
        colname = features2.value

        if not colname in settings.curr_df.columns:
            return
        
        with FeatPage:
            clear_output()
                
            if (settings.curr_df[colname].dtype == 'float64') or (settings.curr_df[colname].dtype == 'int64'):

                fig, (axbox, axhist) = plt.subplots(1,2)
        
                sns.boxplot(x=colname,data=settings.curr_df, ax=axbox)
                axbox.set_title('Box plot') 
                sns.distplot(settings.curr_df[colname],ax=axhist)
                axhist.set_title('Histogram') 
                plt.legend(['Mean '+str(round(settings.curr_df[colname].mean(),2)),'Stdev '+str(round(settings.curr_df[colname].std(),2))], bbox_to_anchor=(0.6, 0.6))
                plt.show()
                
            
                    
                    ############################################################################################################
            '''
                if processtypes.value == 'Imbalancedness':
                    if len(settings.curr_df[colname].unique()) == 2: # binary detection
            
                        plt.figure(figsize=(6, 2))
                        ax = sns.countplot(x=colname,data=settings.curr_df, palette="cool_r")
                        for p in ax.patches:
                            ax.annotate("{:.1f}".format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
                        plt.show()
            '''           
            
            if (settings.curr_df[colname].dtype == 'object') or (settings.curr_df[colname].dtype== 'string'):
            
            
                nrclasses = len(settings.curr_df[colname].unique())
                if nrclasses < 250:
                    g = sns.countplot(settings.curr_df, x=colname)
                    g.set_xticklabels(g.get_xticklabels(),rotation= 45)
                    
                        #sns.distplot(settings.curr_df[settings.curr_df.columns[optind]]).set_title('Histogram of feature '+settings.curr_df.columns[optind])
                    plt.show()
                else:
                    display.display('Number of classes: ',nrclasses)
                    
        with ProcssPage:
            clear_output()            
        

                
        scalingacts.value = scalingacts.options[0]
        return
    
    def savecurrdata(self,event):
        self.controller.savedata(settings.DataFolder, settings.datasets.value)

    def makebalanced(self,event):  
        global balncacts
        
        self.controller.make_balanced(settings.dt_features,balncacts,settings.ProcssPage)
        
        return

    def makescaling(self,event):    
        global scalingacts,result2exp

        self.controller.make_scaling(settings.dt_features,settings.ProcssPage,scalingacts,result2exp)
        
        return

    def makeencoding(self,event):   
        global encodingacts,result2exp
        self.controller.make_encoding(settings.dt_features,encodingacts,result2exp)
        
        return

    def makesplit(self,event):  
        global splt_txt,splt_btn,result2exp
        
        self.controller.make_split(splt_txt,splt_btn,result2exp)
        
        return

    def featurepr_click(self,event):  
        global scalingacts
        
        self.featureprclick(settings.dt_features,settings.FeatPage,settings.processtypes,settings.ProcssPage,scalingacts)

        return

    def selectProcessType(self,event):   
        global sclblly,scalelbl,prctlay,scalingacts,imblncdlay,balncacts,imbllbllly,imbllbl,encdlbl,encodingacts 
        global outrmvlay,outrmvbtn,encdblly,ecndlay,fxctlbl,fxctingacts,fxctblly,fxctlay

        
        self.controller.selectProcess_Type([settings.processtypes,sclblly,scalelbl,prctlay,scalingacts,imblncdlay,balncacts,imbllbllly,imbllbl,outrmvlay,outrmvbtn,encdlbl,encodingacts,encdblly,ecndlay,fxctlbl,fxctingacts,fxctblly,fxctlay])
        
        return

    def removeoutliers(self, event): 
        self.controller.remove_outliers()

        return

    def assignTarget(self, event): 
        global result2exp,trg_btn,predictiontask

        self.controller.assign_target(settings.trg_lbl,settings.dt_features,settings.prdtsk_lbl,result2exp,trg_btn,predictiontask)

        return

    def get_data_processing_tab(self):
        global scalingacts, result2exp, trg_btn, balncacts, imblncdlay, imbllbllly, imbllbl, encdlbl, encodingacts, prctlay, scalelbl, sclblly, splt_txt,splt_btn
        global outrmvlay,outrmvbtn,encdblly,ecndlay,fxctlbl,fxctingacts,fxctblly,fxctlay
        fpgelay = Layout(width="100%")
        settings.FeatPage = widgets.Output(layout = fpgelay)
        settings.ProcssPage = widgets.Output(layout=fpgelay)

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

        settings.processtypes = widgets.Dropdown( options=['Select Processing','Scaling','Encoding','Feature Extraction','Outlier','Imbalancedness'], description='', disabled=False)
        settings.processtypes.observe(self.selectProcessType,'value')



        settings.dt_ftslay =  widgets.Layout( width="99%",display = 'block')
        settings.dt_features = widgets.Select(options=[],description = '',layout = settings.dt_ftslay)
        settings.dt_features.observe(self.featurepr_click, 'value')

        splt_txt =widgets.Dropdown(description ='Split (Test%):',options=[20,25,30,35])
        spltlay = Layout(width='150px')
        splt_btn = widgets.Button(description="Apply Split",layout = spltlay)
        splt_btn.on_click(self.makesplit)

        settings.trg_lbl =widgets.Text(description ='Target:',value = '',disabled = True)
        settings.prdtsk_lbl =widgets.Text(description ='Pred. Task:',value = '',disabled = True)
        trglay = Layout(width='150px')
        trg_btn = widgets.Button(description="Assign Target",layout = trglay)
        trg_btn.on_click(self.assignTarget)


        prctlay = widgets.Layout(width="25%",display = 'none')
        scalingacts = widgets.Dropdown( options=['Select','Standardize','Normalize'], description='', disabled=False,layout = prctlay)
        scalingacts.observe(self.makescaling,'value')


        encdblly = widgets.Layout(width="25%",visibility = 'hidden')
        encdlbl = widgets.Label(value ='Methods',layout = encdblly)

        ecndlay = widgets.Layout(width="25%",display = 'none')
        encodingacts = widgets.Dropdown( options=['Select','Label Encoding','One Hot Encoding'], description='', disabled=False,layout = ecndlay)
        encodingacts.observe(self.makeencoding,'value')

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