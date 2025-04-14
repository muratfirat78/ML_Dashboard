from log import *

class DataCleaningModel:
    def __init__(self, main_model, logger):
        self.main_model = main_model
        self.logger = logger

    def make_cleaning(self,featurescl,result2aexp,missacts,dt_features,params): 
        curr_df = self.main_model.get_curr_df()
      

        colname = featurescl.value

        handling = missacts.value
        write_log('col '+colname+', action '+handling+', coltype '+str(curr_df[colname].dtype), result2aexp, 'Data cleaning')
        write_log('Initial data size'+str(len(curr_df)),  result2aexp, 'Data cleaning')  

        self.logger.add_action(['DataCleaning', handling], [colname])


        if handling == 'Edit Range':

            prev_size = len(curr_df)
            
            minval = params[0].value; maxval = params[1].value
            
            if (curr_df[colname].dtype == 'int64'):
                minval =int(minval); maxval =int(maxval); 

            if (curr_df[colname].dtype == 'float64'):
                minval =float(minval); maxval =float(maxval); 
            
            curr_df = curr_df[(curr_df[colname] >= minval) & (curr_df[colname] <= maxval)]

            write_log('Edit Range is selected.. ',  result2aexp, 'Data cleaning')
     
            
        elif handling == 'Drop Column':
            del curr_df[colname]
        else:    
            if (curr_df[colname].dtype == 'float64') or (curr_df[colname].dtype == 'int64'):
                if handling in ['Replace-Mean','Replace-Median','Remove']:
                    if handling == 'Replace-Mean': 
                        curr_df[colname].fillna(curr_df[colname].mean(), inplace=True)
                    if handling == 'Replace-Median': 
                        curr_df[colname].fillna(curr_df[colname].median(), inplace=True)
    
                    if handling == 'Remove': 
                        curr_df = curr_df.dropna(subset = [colname])
                else:
                    write_log('Improper action is selected.. ',  result2aexp, 'Data cleaning')
                    return
            else: 
                write_log('mode.. '+str(curr_df[colname].mode()[0]), result2aexp, 'Data cleaning')
                if handling == 'Replace-Mode': 
                    curr_df[colname].fillna(curr_df[colname].mode()[0], inplace=True)
                    
        self.main_model.set_curr_df(curr_df)
        write_log('Cleaning action done..'+str(self.main_model.get_curr_df().columns),  result2aexp, 'Data cleaning') 
        write_log('Final data size'+str(len(self.main_model.get_curr_df())),  result2aexp, 'Data cleaning')  

        return