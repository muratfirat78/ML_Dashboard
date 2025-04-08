from IPython.display import clear_output
from IPython import display
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

        
        if (self.controller.get_curr_df()[colname].dtype == 'int64') or (self.controller.get_curr_df()[colname].dtype == 'float64'):
            self.min_lbl.value = "Min: "+str(self.controller.get_curr_df()[colname].min())
            self.max_lbl.value = "Max: "+str(self.controller.get_curr_df()[colname].max())
        else:
            self.min_lbl.value = "Min: -"
            self.max_lbl.value = "Max: -"
     
         

        
        return

    def featurecl_click(self,event):  
        global trgcl_lbl,trgtyp_lbl,miss_lbl
        
        self.featureclclick(trgcl_lbl,self.main_view.featurescl,trgtyp_lbl,miss_lbl)

        return

    def makecleaning(self,event):
        global result2aexp

        params = []
        if  self.missacts.value == "Edit Range":
            params = [self.min_text,self.max_text]
        
        self.controller.make_cleaning(self.main_view.featurescl,result2aexp,self.missacts,self.main_view.dt_features,params) 

        bk_ind = 0
        for c in reversed(self.main_view.featurescl.value):
            if c == '(':
                break
            bk_ind-=1

        colname = self.main_view.featurescl.value[:bk_ind-1]
        if (self.controller.get_curr_df()[colname].dtype == 'int64') or (self.controller.get_curr_df()[colname].dtype == 'float64'):
            self.min_lbl.value = "Min: "+str(self.controller.get_curr_df()[colname].min())
            self.max_lbl.value = "Max: "+str(self.controller.get_curr_df()[colname].max())

        self.missacts.value = "Select"
        
         
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
        global trgcl_lbl, trgtyp_lbl, miss_lbl, result2aexp
        RP_lay=Layout(height='250px',width ='70%',align_items='center',overflow="visible")

        self.main_view.right_page = widgets.Output(layout = RP_lay)

        self.main_view.ftlaycl =  widgets.Layout(display = 'none')
        self.main_view.featurescl = widgets.Select(options=[],description = '',layout = self.main_view.ftlaycl)
        self.main_view.featurescl.observe(self.featurecl_click, 'value')

        misslycl =  widgets.Layout(display = 'none')
        misscl = widgets.Select(options=[],description = '',layout = misslycl)


        mssactlay = widgets.Layout(display = 'block')
        self.missacts = widgets.Dropdown(description='Actions',options=['Select','Drop Column','Remove Missing','Replace-Mean','Replace-Median','Replace-Mode','Edit Range'], disabled=False,layout = mssactlay)

        self.missacts.observe(self.makerangedit)

        msshndlay = Layout(width='125px')
        self.applybutton = widgets.Button(description="Apply",layout = msshndlay)
        self.applybutton.on_click(self.makecleaning)


        trgcl_lbl = widgets.Label(value ='Column: -',disabled = True)
        trgtyp_lbl =widgets.Label(value ='Type: -',disabled = True)
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
        selcl_box = VBox(children=[trgcl_lbl,trgtyp_lbl,miss_lbl,
                                   HBox(children=[self.min_lbl,self.min_text,self.max_lbl,self.max_text]),
                                   HBox(children=[self.missacts,self.applybutton],layout = Layout(align_items='flex-start'))],layout = sboxcllay)


        res2alay = widgets.Layout(height='150px')
        result2aexp = widgets.Textarea(value='', placeholder='',description='',disabled=True,layout = res2alay)


        tab2a_lay = widgets.Layout(witdh='99%')
        tab2_lftbox = VBox(children = [HBox(children=[f2a_box,selcl_box]),result2aexp],layout = tab2a_lay)

        tab_2 = HBox(children=[tab2_lftbox,self.main_view.right_page])
        return tab_2