from IPython.display import clear_output
from IPython import display
from ipywidgets import *

class DataCleaningView:
    def __init__(self, controller, main_view):
        self.controller = controller
        self.main_view = main_view

    def featureclclick(self,trgcl_lbl,featurescl,trgtyp_lbl,miss_lbl):  
        #settings.curr_df,trgcl_lbl,featurescl,trgtyp_lbl,miss_lbl
        bk_ind = 0
        for c in reversed(featurescl.value):
            if c == '(':
                break
            bk_ind-=1

        colname = featurescl.value[:bk_ind-1]

        trgcl_lbl.value = " Column: "+colname
        trgtyp_lbl.value= " Type: " +str(self.controller.get_curr_df()[colname].dtype)
        miss_lbl.value =" Missing values: " + str(self.controller.get_curr_df()[colname].isnull().sum())
        
        return

    def featurecl_click(self,event):  
        global trgcl_lbl,trgtyp_lbl,miss_lbl
        
        self.featureclclick(trgcl_lbl,self.main_view.featurescl,trgtyp_lbl,miss_lbl)

        return

    def makecleaning(self,event):
        global result2aexp,missacts
        self.controller.make_cleaning(self.main_view.featurescl,result2aexp,missacts,self.main_view.dt_features) 

        return

    def get_data_cleaning_tab(self):
        global trgcl_lbl, trgtyp_lbl, miss_lbl, result2aexp, missacts
        RP_lay=Layout(height='250px',width ='70%',align_items='center',overflow="visible")

        self.main_view.right_page = widgets.Output(layout = RP_lay)

        self.main_view.ftlaycl =  widgets.Layout(display = 'none')
        self.main_view.featurescl = widgets.Select(options=[],description = '',layout = self.main_view.ftlaycl)
        self.main_view.featurescl.observe(self.featurecl_click, 'value')

        misslycl =  widgets.Layout(display = 'none')
        misscl = widgets.Select(options=[],description = '',layout = misslycl)


        mssactlay = widgets.Layout(display = 'block')
        missacts = widgets.Dropdown(description='Handling',options=['Select','Drop Column','Remove Missing','Replace-Mean','Replace-Median','Replace-Mode'], disabled=False,layout = mssactlay)


        msshndlay = Layout(width='125px')
        mssapplybtn = widgets.Button(description="Apply",layout = msshndlay)
        mssapplybtn.on_click(self.makecleaning)


        trgcl_lbl =widgets.Label(value ='Column: -',disabled = True)
        trgtyp_lbl =widgets.Label(value ='Type: -',disabled = True)
        miss_lbl =widgets.Label(value ='Missing values: -',disabled = True)


        min_lbl =widgets.Label(value ='Min: ',disabled = True)
        max_lbl =widgets.Label(value ='Max: ',disabled = True)


        fbox2alay = widgets.Layout(width = '30%')
        f2a_box = VBox(children=[widgets.Label(value ='Features'),HBox(children=[self.main_view.featurescl,misscl])],layout = fbox2alay)




        sboxcllay = widgets.Layout()
        selcl_box = VBox(children=[trgcl_lbl,trgtyp_lbl,miss_lbl,HBox(children=[missacts,mssapplybtn],layout = Layout(align_items='flex-start'))],layout = sboxcllay)


        res2alay = widgets.Layout(height='150px')
        result2aexp = widgets.Textarea(value='', placeholder='',description='',disabled=True,layout = res2alay)


        tab2a_lay = widgets.Layout(witdh='99%')
        tab2_lftbox = VBox(children = [HBox(children=[f2a_box,selcl_box]),result2aexp],layout = tab2a_lay)

        tab_2 = HBox(children=[tab2_lftbox,self.main_view.right_page])
        return tab_2