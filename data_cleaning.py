import settings
from log import *

def make_cleaning(featurescl,result2aexp,missacts,dt_features): 

    bk_ind = 0
    for c in reversed(featurescl.value):
        if c == '(':
            break
        bk_ind-=1

    colname = featurescl.value[:bk_ind-1]

    handling = missacts.value
    write_log('col '+colname+', action '+handling+', coltype '+str(settings.curr_df[colname].dtype), result2aexp, 'Data cleaning')

    if handling == 'Drop Column':
        del settings.curr_df[colname]
    else:    
        if (settings.curr_df[colname].dtype == 'float64') or (settings.curr_df[colname].dtype == 'int64'):
            if handling in ['Replace-Mean','Replace-Median','Remove']:
                if handling == 'Replace-Mean': 
                    settings.curr_df[colname].fillna(settings.curr_df[colname].mean(), inplace=True)
                if handling == 'Replace-Median': 
                    settings.curr_df[colname].fillna(settings.curr_df[colname].median(), inplace=True)

                if handling == 'Remove': 
                    settings.curr_df = settings.curr_df.dropna(subset = [colname])
            else:
                write_log('Improper action is selected.. ',  result2aexp, 'Data cleaning')
                return
        else: 
            write_log('mode.. '+str(settings.curr_df[colname].mode()[0]), result2aexp, 'Data cleaning')
            if handling == 'Replace-Mode': 
                settings.curr_df[colname].fillna(settings.curr_df[colname].mode()[0], inplace=True)
            
    featurescl.options = [col+'('+str(settings.curr_df[col].isnull().sum())+')' for col in settings.curr_df.columns]
    dt_features.options = [col for col in settings.curr_df.columns]

    return