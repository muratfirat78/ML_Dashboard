from log import *
import pandas as pd
from IPython.display import clear_output
from IPython import display
from pathlib import Path
import os

rowheight = 20
colabpath = '/content/CPP_Datasets'

class DataSelectionModel:
    def __init__(self, main_model):
        self.main_model = main_model
        self.datafolder = None
        
    def on_submitfunc(self,online_version,foldername,datasets):
        self.datafolder = foldername
        dtsetnames = [] 
        
        if online_version: 

            directory_files = os.listdir(colabpath)
        
            for file in directory_files:
                if (file.find('.csv')>-1) or (file.find('.xlsx')>-1) or (file.find('.xls')>-1) or(file.find('.tsv')>-1):
                    dtsetnames.append(file)
        else:
            
            rel_path = foldername
            abs_file_path = os.path.join(Path.cwd(), rel_path)

            #print('Path:',abs_file_path)
            
            for root, dirs, files in os.walk(abs_file_path):
                for file in files:

                    if (file.find('.csv')>-1) or (file.find('.xlsx')>-1)or (file.find('.xls')>-1)  or(file.find('.tsv')>-1):
                        dtsetnames.append(file)


        datasets.options = dtsetnames
        if len(dtsetnames) > 0:
            datasets.value = dtsetnames[0]
        return

    def read_data_set(self,online_version,foldername,filename,sheetname,processtypes,Pages,dt_features,dt_ftslay,featurescl,ftlaycl):

        FeatPage,ProcssPage,DFPage,RightPage = Pages
        
        rel_path = foldername+'\\'+filename
        
        if online_version:
            abs_file_path = colabpath+'/'+filename
        else:
            abs_file_path = os.path.join(Path.cwd(), rel_path)
            

        if abs_file_path.find('.csv') > -1:
            self.main_model.curr_df = pd.read_csv(abs_file_path, sep=sheetname) 
        if (abs_file_path.find('.xlsx') > -1) or (filename.find('.xls') > -1):
            xls = pd.ExcelFile(abs_file_path)
            self.main_model.curr_df = pd.read_excel(xls,sheetname)
        if abs_file_path.find('.tsv') > -1:    
        
            self.main_model.curr_df = pd.read_csv(abs_file_path, sep="\t")
            
        self.main_model.curr_df.convert_dtypes()
        
        datasetname = filename[:filename.find('.')]  
        
        dt_ftslay.height = str(rowheight*len(self.main_model.curr_df.columns))+'px'
        dt_features.layout = dt_ftslay
        dt_features.options = [col for col in self.main_model.curr_df.columns]


        ftlaycl.display = 'block'
        ftlaycl.height = str(rowheight*len(self.main_model.curr_df.columns))+'px'
        featurescl.layout = ftlaycl
        featurescl.options = [col+'('+str(self.main_model.curr_df[col].isnull().sum())+')' for col in self.main_model.curr_df.columns]
        
        processtypes.value = processtypes.options[0]
        
        with FeatPage:
            clear_output()
        with ProcssPage:
            clear_output()
        
        with DFPage:
            clear_output()
            #####################################
            display.display(self.main_model.curr_df.info()) 
            display.display(self.main_model.curr_df.describe()) 
            display.display(self.main_model.curr_df) 
            #####################################

        with RightPage:
            clear_output()

        logging.info('Data Selection: Read data set' + filename)
        return 


    ################################################################################################################

    def file_Click(self,online_version,foldername,filename,wsheets,wslay,butlay): 
        abs_file_path = ''
        
        if online_version:
            abs_file_path = colabpath+'/'+filename
        else:
            rel_path = foldername+'\\'+filename
            script_dir = Path.cwd()
            abs_file_path = os.path.join(script_dir, rel_path)

        
        if filename.find('.csv') > -1:
            wsheets.description = 'Separator'
            wsheets.options = [',',';']
        
        if filename.find('.tsv') > -1:
            wsheets.description = 'Separator'
            wsheets.options = ['\\t']
            
        if (filename.find('.xlsx') > -1) or (filename.find('.xls') > -1) :
    
            wsheets.description = 'Worksheets'
            xls = pd.ExcelFile(abs_file_path)
            wsheets.options = xls.sheet_names
            
        wslay.display = 'block'
        wsheets.value = wsheets.options[0]    
        wsheets.layout = wslay
            
        butlay.display = 'block'
            
        return
    
    def get_datafolder(self):
        return self.datafolder