from log import *
import pandas as pd
from IPython.display import clear_output
from IPython import display
from pathlib import Path
from datetime import timedelta
import os

colabpath = '/content/ML_Dashboard/DataSets'

class DataSelectionModel:
    def __init__(self, main_model, logger):
        self.main_model = main_model
        self.datafolder = None
        self.logger = logger
        
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
            
            for root, dirs, files in os.walk(abs_file_path):
                for file in files:

                    if (file.find('.csv')>-1) or (file.find('.xlsx')>-1)or (file.find('.xls')>-1)  or(file.find('.tsv')>-1):
                        dtsetnames.append(file)


        datasets.options = dtsetnames
        if len(dtsetnames) > 0:
            datasets.value = dtsetnames[0]
        return

    def read_data_set(self,online_version,foldername,filename,sheetname):

        stemfile = filename[:filename.find(".")]
        rel_path = foldername+'\\'+filename
        infopath = foldername+'\\'+"Info_"+stemfile+".txt"
        
        if online_version:
            abs_file_path = colabpath+'/'+filename
            abs_info_path = "/content/ML_Dashboard/DataSets/" +"Info_"+stemfile+".txt"
        else:
            abs_file_path = os.path.join(Path.cwd(), rel_path)
            abs_info_path = os.path.join(Path.cwd(), infopath)

        if abs_file_path.find('.csv') > -1:
            self.main_model.curr_df = pd.read_csv(abs_file_path, sep=sheetname) 

            for col in self.main_model.curr_df.columns:
                if str(col).find("Date") > -1:
                    try: 
                        self.main_model.curr_df[col] = pd.to_datetime(self.main_model.curr_df[col], format='%Y-%m-%d')

                        self.main_model.curr_df['year']=[x.year for x  in self.main_model.curr_df[col]]
    
                        minyear = min(self.main_model.curr_df['year'])
                        self.main_model.curr_df['month']=[x.month for x  in self.main_model.curr_df[col]]
                        self.main_model.curr_df['week']=[x.isocalendar()[1]+52*((x+timedelta(days= 7-x.weekday())).year-minyear) for x in self.main_model.curr_df[col]]
                       
                        self.main_model.curr_df['dayofweek']=[x.weekday() for x  in self.main_model.curr_df[col]]
                        self.main_model.curr_df.sort_values(by=[col], inplace=True)
                    except:
                        pass
      
            
        if (abs_file_path.find('.xlsx') > -1) or (filename.find('.xls') > -1):
            xls = pd.ExcelFile(abs_file_path)
            self.main_model.curr_df = pd.read_excel(xls,sheetname)
        if abs_file_path.find('.tsv') > -1:    
            self.main_model.curr_df = pd.read_csv(abs_file_path, sep="\t")

       
        try: 
            f = open(abs_info_path)
            self.main_model.currinfo = f.read()
            f.close()
        except: 
            self.main_model.currinfo = "No info found for this file"
            
            
        self.main_model.curr_df.convert_dtypes()
        
        filename[:filename.find('.')]  

        self.logger.add_action(['SelectData', 'DataSet'], filename)
        logging.info('Data Selection: Read data set' + filename)
        return 
    
    def file_Click(self,online_version,foldername,filename,wsheets): 
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
            
        return
    
    def get_datafolder(self):
        return self.datafolder
    
    def set_datafolder(self, foldername):
        self.datafolder = foldername