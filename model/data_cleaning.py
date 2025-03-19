from log import *

class DataCleaningModel:
    def __init__(self, main_model):
        self.main_model = main_model

    def make_cleaning(self,featurescl,result2aexp,missacts,dt_features): 

        bk_ind = 0
        for c in reversed(featurescl.value):
            if c == '(':
                break
            bk_ind-=1

        colname = featurescl.value[:bk_ind-1]

        handling = missacts.value
        write_log('col '+colname+', action '+handling+', coltype '+str(self.main_model.curr_df[colname].dtype), result2aexp, 'Data cleaning')

        if handling == 'Drop Column':
            del self.main_model.curr_df[colname]
        else:    
            if (self.main_model.curr_df[colname].dtype == 'float64') or (self.main_model.curr_df[colname].dtype == 'int64'):
                if handling in ['Replace-Mean','Replace-Median','Remove']:
                    if handling == 'Replace-Mean': 
                        self.main_model.curr_df[colname].fillna(self.main_model.curr_df[colname].mean(), inplace=True)
                    if handling == 'Replace-Median': 
                        self.main_model.curr_df[colname].fillna(self.main_model.curr_df[colname].median(), inplace=True)

                    if handling == 'Remove': 
                        self.main_model.curr_df = self.main_model.curr_df.dropna(subset = [colname])
                else:
                    write_log('Improper action is selected.. ',  result2aexp, 'Data cleaning')
                    return
            else: 
                write_log('mode.. '+str(self.main_model.curr_df[colname].mode()[0]), result2aexp, 'Data cleaning')
                if handling == 'Replace-Mode': 
                    self.main_model.curr_df[colname].fillna(self.main_model.curr_df[colname].mode()[0], inplace=True)
                
        featurescl.options = [col+'('+str(self.main_model.curr_df[col].isnull().sum())+')' for col in self.main_model.curr_df.columns]
        dt_features.options = [col for col in self.main_model.curr_df.columns]

        return