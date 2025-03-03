from sklearn import preprocessing 
import settings
from log import *
from sklearn.model_selection import train_test_split 



def remove_outliers():
    settings.curr_df = settings.curr_df[settings.curr_df["outlier"] == False]
    settings.curr_df = settings.curr_df.drop(["outlier"], axis=1)
    logging.info('Data preprocessing, outlier detection and removal')
    
    return
##################################################################################

def assign_target(trg_lbl,dt_features,prdtsk_lbl,result2exp,trg_btn,predictiontask):

    global targetcolumn

    targetcolumn = dt_features.value

    trg_lbl.value = targetcolumn
    trg_btn.disabled = True

    
    if (settings.curr_df[targetcolumn].dtype == 'float64') or (settings.curr_df[targetcolumn].dtype == 'int64'):
        predictiontask = "Regression"
    else:
        predictiontask = "Classification" 

    prdtsk_lbl.value = predictiontask 
    write_log('Target assigned: '+targetcolumn, result2exp, 'Data processing')
    

    return 
############################################################################################################    
def make_split(splt_txt,splt_btn,result2exp):
    global targetcolumn

    if targetcolumn is None:
        return
  
    y = settings.curr_df[targetcolumn] # Target variable 
    column_list = [col for col in settings.curr_df.columns]
    column_list.remove(targetcolumn)
    X = settings.curr_df[column_list]
    
    ratio_percnt = int(splt_txt.value) 
    write_log('Split ratio, '+str(ratio_percnt/100), result2exp, 'Data processing')
    settings.Xtrain_df,settings.Xtest_df, settings.ytrain_df, settings.ytest_df = train_test_split(X, y, test_size=ratio_percnt/100, random_state=16)
    splt_btn.disabled = True

    write_log('Split, Train size: '+str(len(settings.Xtrain_df)), result2exp, 'Data processing')


    return
############################################################################################################    
def make_encoding(features2,encodingacts,result2exp):

    colname = features2.value

    
    write_log('Encoding.. col '+colname+' is list:'+str(int(isinstance(settings.curr_df, list))), result2exp, 'Data processing')

    if colname is None:
        return

    
  
    # Encode column  
    if len(settings.Xtrain_df) > 0:
        
        if encodingacts.value == "Label Encoding":
            label_encoder = preprocessing.LabelEncoder() 
            write_log('Encoding-> '+features2.value+' (train) current classes: '+str(data_df[0][colname].unique()), result2exp, 'Data processing')
            data_df[0][colname] = label_encoder.fit_transform(data_df[0][colname]) # train 
            write_log('Encoding-> '+features2.value+' (train) after labeling classes: '+str(data_df[0][colname].unique()), result2exp, 'Data processing')
            write_log('Encoding-> '+features2.value+' (test) current classes: '+str(data_df[1][colname].unique()), result2exp, 'Data processing')
            data_df[1][colname] = label_encoder.fit_transform(data_df[1][colname]) # test
            write_log('Encoding-> '+features2.value+' (test) after labeling classes: '+str(data_df[1][colname].unique()), result2exp, 'Data processing')

            data_df = data_df[0],data_df[1]
            
        if encodingacts.value == "One Hot Encoding":
            categorical_columns = [colname]
            
            encoder = preprocessing.OneHotEncoder(sparse_output=False)  # Initialize OneHotEncoder
            one_hot_encoded = encoder.fit_transform(data_df[0][categorical_columns])  # Fit and transform the categorical columns          
            one_hot_df = pd.DataFrame(one_hot_encoded,columns=encoder.get_feature_names_out(categorical_columns)) # Create a DataFrame
            data1_df = pd.concat([data_df[0].drop(categorical_columns, axis=1), one_hot_df], axis=1)
            write_log('One Hot Encoding-> (train) after one-hot features: '+str(data1_df.columns), result2exp, 'Data processing')

            encoder = preprocessing.OneHotEncoder(sparse_output=False)  # Initialize OneHotEncoder
            one_hot_encoded = encoder.fit_transform(data_df[1][categorical_columns])  # Fit and transform the categorical columns          
            one_hot_df = pd.DataFrame(one_hot_encoded,columns=encoder.get_feature_names_out(categorical_columns)) # Create a DataFrame 
            data2_df = pd.concat([data_df[1].drop(categorical_columns, axis=1), one_hot_df], axis=1)
            write_log('One Hot Encoding-> (test) after one-hot features: '+str(data2_df.columns), result2exp, 'Data processing')

            data_df = data1_df,data2_df

            features2.options = [col+'('+str(data1_df[col].isnull().sum())+')' for col in data1_df.columns]
    else:
        write_log('Encoding-> '+encodingacts.value+', '+str(encodingacts.value == "One Hot Encoding"), result2exp, 'Data processing')
        write_log('Encoding-> '+colname+' current classes: '+str(len(settings.curr_df)), result2exp, 'Data processing')
        if encodingacts.value == "Label Encoding":
            label_encoder = preprocessing.LabelEncoder() 
            write_log('Encoding-> '+features2.value+' current classes: '+str(settings.curr_df[colname].unique()), result2exp, 'Data processing')
            settings.curr_df[colname] = label_encoder.fit_transform(settings.curr_df[colname]) 
            write_log('Encoding-> '+features2.value+' after labeling classes: '+str([cls for cls in settings.curr_df[colname].unique()]), result2exp, 'Data processing')
            
        if encodingacts.value == "One Hot Encoding":
            write_log('Encoding-> '+colname+' current classes: '+str(settings.curr_df[colname].unique()), result2exp, 'Data processing')
            categorical_columns = [colname]
            encoder = preprocessing.OneHotEncoder(sparse_output=False)  # Initialize OneHotEncoder
            one_hot_encoded = encoder.fit_transform(settings.curr_df[categorical_columns])  # Fit and transform the categorical columns          
            one_hot_df = pd.DataFrame(one_hot_encoded,columns=encoder.get_feature_names_out(categorical_columns)) # Create a DataFrame
            settings.curr_df = pd.concat([settings.curr_df.drop(categorical_columns, axis=1), one_hot_df], axis=1)
            write_log('One Hot Encoding-> after one-hot features: '+str(settings.curr_df.columns), result2exp, 'Data processing')

        features2.options = [col for col in settings.curr_df.columns]
    
    return
