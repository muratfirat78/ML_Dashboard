class DataCleaningModel:
    def __init__(self, main_model, logger):
        self.main_model = main_model
        self.logger = logger

    def make_cleaning(self,featurescl,result2aexp,missacts,dt_features): 
        curr_df = self.main_model.curr_df
        bk_ind = 0
        for c in reversed(featurescl.value):
            if c == '(':
                break
            bk_ind-=1

        colname = featurescl.value[:bk_ind-1]

        handling = missacts.value
        self.logger.write_log('col '+colname+', action '+handling+', coltype '+str(curr_df[colname].dtype), result2aexp, 'Data cleaning')

        if handling == 'Drop Column':
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
                    self.logger.write_log('Improper action is selected.. ',  result2aexp, 'Data cleaning')
                    return
            else: 
                self.logger.write_log('mode.. '+str(curr_df[colname].mode()[0]), result2aexp, 'Data cleaning')
                if handling == 'Replace-Mode': 
                    curr_df[colname].fillna(curr_df[colname].mode()[0], inplace=True)
                
        featurescl.options = [col+'('+str(curr_df[col].isnull().sum())+')' for col in curr_df.columns]
        dt_features.options = [col for col in curr_df.columns]

        return