from IPython.display import clear_output
from IPython import display
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import *

class DataCleaningView:
    def __init__(self, controller, main_view):
        self.controller = controller
        self.main_view = main_view
        self.min_lbl = None
        self.max_lbl = None
        self.min_text = None
        self.max_text = None
        self.missacts = None
        self.applybutton = None

    def featureclclick(self,trgcl_lbl,featurescl,miss_lbl):  
        #settings.curr_df,trgcl_lbl,featurescl,miss_lbl
        colname = featurescl.value

        missng_vals = 0
        totalmisses  = 0

        display_df = self.controller.get_curr_df()

        for col in display_df.columns:
            curr_miss = display_df[col].isnull().sum()  
            totalmisses+=curr_miss
        missng_vals = display_df[colname].isnull().sum()


        if self.controller.main_model.datasplit:
            totalmisses  = 0
            if colname in self.controller.get_XTrain().columns: 
                display_df = self.controller.main_model.get_XTrain()
                missng_vals = display_df[colname].isnull().sum()
                missng_vals+= self.controller.main_model.get_XTest()[colname].isnull().sum()

                for col in display_df.columns:
                    curr_miss = display_df[col].isnull().sum()  
                    totalmisses+=curr_miss

                for col in self.controller.main_model.get_XTest().columns:
                    curr_miss = self.controller.main_model.get_XTest()[col].isnull().sum()  
                    totalmisses+=curr_miss

            else: 
                ytrain_df = self.controller.main_model.getYtrain().to_frame()
                if colname in ytrain_df.columns: 
                    display_df = ytrain_df
                    for col in display_df.columns:
                        curr_miss = display_df[col].isnull().sum()  
                        totalmisses+=curr_miss
                else: 
                    return
        
        
 
        

        trgcl_lbl.value = " Column: "+colname
        trgcl_lbl.value = "Column: ["+str(colname)+"] | Type -> "+str(display_df[colname].dtype)
        
        miss_lbl.value =" Missing values: " + str(missng_vals)+" ( Total: "+str(totalmisses)+")"


        
        if (display_df[colname].dtype == 'int64') or (display_df[colname].dtype == 'float64') or (display_df[colname].dtype == 'int32'):
            self.min_lbl.value = "Min: "+str(display_df[colname].min())
            self.max_lbl.value = "Max: "+str(display_df[colname].max())
        else:
            self.min_lbl.value = "Min: -"
            self.max_lbl.value = "Max: -"
     

        
        return

    def featurecl_click(self,event):  
        global trgcl_lbl,miss_lbl
        
        self.featureclclick(trgcl_lbl,self.main_view.featurescl,miss_lbl)

        return

    def makecleaning(self,event):
        
        global result2aexp

        params = []
        if  self.missacts.value == "Edit Range":
            params = [self.min_text,self.max_text]

        
        
        self.controller.make_cleaning(self.main_view.featurescl,result2aexp,self.missacts,self.main_view.dt_features,params) 

        self.missacts.value = "Select"

        missings = []
        if not self.controller.main_model.datasplit:
            for col in self.controller.get_curr_df().columns:
                missings.append((self.controller.get_curr_df()[col].isnull().sum(),col))

            curr_df = self.controller.get_curr_df()
            new_list = sorted(missings, key=lambda x: x[0], reverse=True)
            curr_df = curr_df[[col for (miss,col) in new_list]]

            self.controller.set_curr_df(curr_df)
            
        else:
            for col in self.controller.main_model.get_XTrain().columns:
                missings.append((self.controller.main_model.get_XTrain()[col].isnull().sum()+self.controller.main_model.get_XTest()[col].isnull().sum(),col))
        
            new_list = sorted(missings, key=lambda x: x[0], reverse=True)
            Xtrain = self.controller.main_model.get_XTrain()
            Xtrain = Xtrain[[col for (miss,col) in new_list]]
            Xtest = self.controller.main_model.get_XTest()
            Xtest = Xtest[[col for (miss,col) in new_list]]

            for col in self.controller.main_model.getYtrain().to_frame().columns: 
                missings.append((self.controller.main_model.getYtrain().to_frame()[col].isnull().sum(),col))
                
            new_list = sorted(missings, key=lambda x: x[0], reverse=True)
            
            self.controller.main_model.set_XTest(Xtest)
            self.controller.main_model.set_XTrain(Xtrain)
        

        with self.main_view.right_page:
            clear_output()
            
            missing_df = pd.DataFrame(columns=['feature','missing values'])
            totalmisses  = 0
            if not self.controller.main_model.datasplit:
                for col in self.controller.get_curr_df().columns:
                    curr_miss = self.controller.get_curr_df()[col].isnull().sum()
                    row = {'feature': col, 'missing values':curr_miss}
                    new_df = pd.DataFrame([row])
                    missing_df = pd.concat([missing_df, new_df], axis=0, ignore_index=True)
                    totalmisses+=curr_miss

            else:
               
                for col in self.controller.main_model.get_XTrain().columns:
                    curr_miss = self.controller.main_model.get_XTrain()[col].isnull().sum()
                    curr_miss +=  self.controller.main_model.get_XTest()[col].isnull().sum()
                    row = {'feature': col, 'missing values':curr_miss}
                    new_df = pd.DataFrame([row])
                    missing_df = pd.concat([missing_df, new_df], axis=0, ignore_index=True)
                    totalmisses+=curr_miss
                
                for col in self.controller.main_model.getYtrain().to_frame().columns:   
                    trg_miss = self.controller.main_model.getYtrain().to_frame()[col].isnull().sum()
                    trg_miss +=  self.controller.main_model.get_YTest().to_frame()[col].isnull().sum()
                    totalmisses+=trg_miss
                    row = {'feature': self.controller.main_model.targetcolumn , 'missing values':trg_miss}
                    new_df = pd.DataFrame([row])
                    missing_df = pd.concat([missing_df, new_df], axis=0, ignore_index=True)
                   

                
            #display.display(missing_df.head(20))  
            g = sns.barplot(x='feature', y='missing values', data=missing_df)
            g.set_xticklabels(g.get_xticklabels(),rotation= 45)
            plt.title('Total Missing Values: '+str(totalmisses))
            plt.show()
                    

        self.main_view.dt_features.options = [col for (miss,col) in new_list]
        self.main_view.featurescl.options = [col for (miss,col) in new_list]

      
        return

    def makerangedit(self,event):

        if  self.missacts.value == "Edit Range":
            self.min_text.layout.width = '50px'
            self.min_text.layout.visibility = 'visible'
            self.min_lbl.layout.visibility = 'hidden'

            self.min_text.value = self.min_lbl.value[self.min_lbl.value.find(":")+1:]
           
            self.max_text.layout.width = '50px'
            self.max_text.layout.visibility = 'visible'
            self.max_lbl.layout.visibility = 'hidden'

            self.max_text.value = self.max_lbl.value[self.max_lbl.value.find(":")+1:]
           
            
        else:
            self.min_lbl.layout.visibility = 'visible'
            
            self.min_text.layout.width = '1px'
            self.min_text.layout.visibility = 'hidden'

            self.max_lbl.layout.visibility = 'visible'
           
            self.max_text.layout.width = '1px'
            self.max_text.layout.visibility = 'hidden'

        return

    def get_data_cleaning_tab(self):
        global trgcl_lbl,miss_lbl, result2aexp
        RP_lay=Layout(height='250px',width ='70%',align_items='center',overflow="visible")

        self.main_view.right_page = widgets.Output(layout = RP_lay)

        self.main_view.ftlaycl =  widgets.Layout(display = 'none')
        
        self.main_view.featurescl = widgets.Select(options=[],description = '',layout = self.main_view.ftlaycl)
        self.main_view.featurescl.observe(self.featurecl_click, 'value')

        misslycl =  widgets.Layout(display = 'none')
        misscl = widgets.Select(options=[],description = '',layout = misslycl)


        mssactlay = widgets.Layout(display = 'block')
        self.missacts = widgets.Dropdown(description='Actions',options=['Select','Drop Column','Remove-Missing','Replace-Mean','Replace-Median','Replace-Mode','Edit Range'], disabled=False,layout = mssactlay)

        self.missacts.observe(self.makerangedit)

        msshndlay = Layout(width='125px')
        self.applybutton = widgets.Button(description="Apply",layout = msshndlay)
        self.applybutton.on_click(self.makecleaning)


        trgcl_lbl = widgets.Label(value ='Column: -',disabled = True)
     
        miss_lbl =widgets.Label(value ='Missing values: -',disabled = True)


        self.min_lbl =widgets.Label(value ='Min: -',disabled = True)
        self.max_lbl =widgets.Label(value ='Max: -',disabled = True)

        self.min_text =widgets.Text(value ='',disabled = False)
        self.min_text.layout.width = '1px'
        self.min_text.layout.visibility = 'hidden'
        self.max_text =widgets.Text(value ='',disabled = False)
        self.max_text.layout.width = '1px'
        self.max_text.layout.visibility = 'hidden'

        

        fbox2alay = widgets.Layout(width = '30%')
        f2a_box = VBox(children=[widgets.Label(value ='Features'),HBox(children=[self.main_view.featurescl,misscl])],layout = fbox2alay)




        sboxcllay = widgets.Layout()
        selcl_box = VBox(children=[trgcl_lbl,miss_lbl,
                                   HBox(children=[self.min_lbl,self.min_text,self.max_lbl,self.max_text]),
                                   HBox(children=[self.missacts,self.applybutton],layout = Layout(align_items='flex-start'))],layout = sboxcllay)


        res2alay = widgets.Layout(height='150px')
        result2aexp = widgets.Textarea(value='', placeholder='',description='',disabled=True,layout = res2alay)


        tab2a_lay = widgets.Layout(witdh='99%')
        tab2_lftbox = VBox(children = [HBox(children=[f2a_box,selcl_box]),result2aexp],layout = tab2a_lay)

        tab_2 = HBox(children=[tab2_lftbox,self.main_view.right_page])
        return tab_2