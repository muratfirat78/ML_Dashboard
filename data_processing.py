from sklearn import preprocessing 
import settings
from log import *
from sklearn.model_selection import train_test_split 
from IPython.display import clear_output
from IPython import display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
import pandas as pd
import os
import shutil
from datetime import timedelta,date, datetime

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
##################################################################################

def NormalizeColumn(df,colname):
    logging.info('Data preprocessing, feature scaling: normalization of column '+ colname)
    col_min = min(df[colname])
    col_max = max(df[colname])
    
    if col_max == col_min: 
        return
    
    df[colname] = (df[colname]- col_min)/(col_max-col_min)
    
    return df

############################################################################################################    

def make_scaling(dt_features,ProcssPage,scalingacts,result2exp):  
  
    colname = dt_features.value

    if colname is None:
        return

    write_log('Scaling-> '+scalingacts.value+': '+colname, result2exp, 'Data processing')
    
    if (settings.curr_df[colname].dtype == 'object') or (settings.curr_df[colname].dtype== 'string'):
        with ProcssPage:
            clear_output()
            display.display('Selected column is not a numerical type..')
        return

    if scalingacts.value == 'Standardize':
        if len(settings.Xtrain_df)>0:
            if colname in settings.Xtrain_df.columns:
                colmean = settings.Xtrain_df[colname].mean(); colstd = settings.Xtrain_df[colname].std()
                settings.Xtrain_df[colname] = (settings.Xtrain_df[colname]- colmean)/colstd
                settings.Xtest_df[colname] = (settings.Xtest_df[colname]- colmean)/colstd
            if colname in settings.ytrain_df.columns:
                colmean = settings.ytrain_df[colname].mean(); colstd = settings.ytrain_df[colname].std()
                settings.ytrain_df[colname] = (settings.ytrain_df[colname]- colmean)/colstd
                settings.ytest_df[colname] = (settings.ytest_df[colname]- colmean)/colstd
                 
        colmean = settings.curr_df[colname].mean()
        settings.curr_df[colname] = (settings.curr_df[colname]- colmean)/settings.curr_df[colname].std()
        logging.info('Data preprocessing, feature scaling: standardization of column '+ colname)


    if scalingacts.value == 'Normalize':

        if len(settings.Xtrain_df)>0:
            if colname in settings.Xtrain_df.columns:
                col_min = min(settings.Xtrain_df[colname]); col_max = max(settings.Xtrain_df[colname])
                denominator = (col_max-col_min)
                if denominator== 0:
                    settings.Xtrain_df[colname] = (settings.Xtrain_df[colname]/col_min)
                    settings.Xtest_df[colname] = (settings.Xtest_df[colname]/col_min)
                else:
                    settings.Xtrain_df[colname] = (settings.Xtrain_df[colname]-col_min)/denominator
                    settings.Xtest_df[colname] = (settings.Xtest_df[colname]-col_min)/denominator
               
            if colname in settings.ytrain_df.columns:
                col_min = min(settings.ytrain_df[colname]); col_max = max(settings.ytrain_df[colname])
                denominator = (col_max-col_min)
                if denominator== 0:
                    settings.ytrain_df[colname] = (settings.ytrain_df[colname]/col_min)
                    settings.ytest_df[colname] = (settings.ytest_df[colname]/col_min)
                else:
                    settings.ytrain_df[colname] = (settings.ytrain_df[colname]-col_min)/denominator
                    settings.ytest_df[colname] = (settings.ytest_df[colname]-col_min)/denominator

        
        col_min = min(settings.curr_df[colname]); col_max = max(settings.curr_df[colname])
        denominator = (col_max-col_min)

        if denominator== 0:
            settings.curr_df[colname] = (settings.curr_df[colname]/col_min)
        else:
            settings.curr_df[colname] = (settings.curr_df[colname]-col_min)/denominator

        logging.info('Data preprocessing, feature scaling: normalization of column '+ colname)

    with ProcssPage:
        clear_output()
        fig, (axbox, axhist) = plt.subplots(1,2)
     
        sns.boxplot(x=colname,data=settings.curr_df, ax=axbox)
        axbox.set_title('Box plot') 
        sns.distplot(settings.curr_df[colname],ax=axhist)
        axhist.set_title('Histogram') 
        plt.legend(['Mean '+str(round(settings.curr_df[colname].mean(),2)),'Stdev '+str(round(settings.curr_df[colname].std(),2))], bbox_to_anchor=(0.6, 0.6))
        plt.show()
    
 
    return
#################################################################################################################
def make_balanced(features2,balncacts,ProcssPage):  


    colname = features2.value

    if balncacts.value == 'Upsample':
         
        if len(settings.curr_df[colname].unique()) == 2: # binary detection
            
            colvals = settings.curr_df[colname].unique()
            ColmFirst = settings.curr_df[ settings.curr_df[colname] == colvals[0]]
            ColmOther = settings.curr_df[ settings.curr_df[colname] == colvals[1]]
          
            if len(ColmFirst) < len(ColmOther):
                upsampled_First = resample(ColmFirst, replace=True, n_samples=len(ColmOther), random_state=27) 
                settings.curr_df = pd.concat([ColmOther, upsampled_First])
            else:
                upsampled_Other= resample(ColmOther, replace=True, n_samples=len(ColmFirst), random_state=27) 
                settings.curr_df = pd.concat([ColmFirst, upsampled_Other])
                
            with ProcssPage:
                
                clear_output()
                plt.figure(figsize=(6, 2))
                ax = sns.countplot(x=colname,data=settings.curr_df, palette="cool_r")
                for p in ax.patches:
                    ax.annotate("{:.1f}".format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
                plt.show()
    logging.info('Data preprocessing, checking and handling unbalancedness')
    return

#####################################################################################################################


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

def savedata( dataFolder, datasetname):
    datasetname = os.path.splitext(os.path.basename(datasetname))[0]
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = dataFolder.value + '/' + datasetname + '_' + current_datetime
    shutil.copy('output.log', filename + '.txt')
    settings.curr_df.to_csv(filename + '.csv')

    
    version = 0

def SelectProcess_Type(vis_list):
    
 
    processtypes = vis_list[0]
    sclblly = vis_list[1]
    scalelbl = vis_list[2]
    prctlay = vis_list[3]
    scalingacts = vis_list[4]
    imblncdlay = vis_list[5]
    balncacts = vis_list[6]
    imbllbllly = vis_list[7]
    imbllbl = vis_list[8]
    outrmvlay = vis_list[9]
    outrmvbtn = vis_list[10]
    encdlbl = vis_list[11]
    encodingacts = vis_list[12]
    encdblly = vis_list[13]
    ecndlay = vis_list[14]
    fxctlbl = vis_list[15]
    fxctingacts = vis_list[16]
    fxctblly = vis_list[17]
    fxctlay = vis_list[18]
   
    ResetProcessMenu(vis_list)

    
    if processtypes.value == 'Scaling':
        sclblly.display = 'block'
        sclblly.visibility = 'visible'
        scalelbl.layout = sclblly
        prctlay.display = 'block'
        prctlay.visibility = 'visible'
        scalingacts.layout = prctlay
        
    if processtypes.value == 'Imbalancedness':
        
        imbllbllly.display = 'block'
        imbllbllly.visibility = 'visible'
        imbllbl.layout = imbllbllly
        imblncdlay.display = 'block'
        imblncdlay.visibility = 'visible'
        balncacts.layout = imblncdlay
     
    if processtypes.value == 'Outlier':     
        outrmvlay.display = 'block'
        outrmvlay.visibility = 'visible'
        outrmvbtn.layout = outrmvlay

    if processtypes.value == 'Encoding':     
        encdblly.display = 'block'
        encdblly.visibility = 'visible'
        encdblly.layout = sclblly
        ecndlay.display = 'block'
        ecndlay.visibility = 'visible'
        encodingacts.layout = ecndlay

    if processtypes.value == 'Feature Extraction':
        fxctblly.display = 'block'
        fxctblly.visibility = 'visible'
        fxctlbl.layout = fxctblly
        fxctlay.display = 'block'
        fxctlay.visibility = 'visible'
        fxctingacts.layout = fxctlay
        

    return

def ResetProcessMenu(vis_list):

    processtypes = vis_list[0]
    sclblly = vis_list[1]
    scalelbl = vis_list[2]
    prctlay = vis_list[3]
    scalingacts = vis_list[4]
    imblncdlay = vis_list[5]
    balncacts = vis_list[6]
    imbllbllly = vis_list[7]
    imbllbl = vis_list[8]
    outrmvlay = vis_list[9]
    outrmvbtn = vis_list[10]
    encdlbl = vis_list[11]
    encodingacts = vis_list[12]
    encdblly = vis_list[13]
    ecndlay = vis_list[14]
    fxctlbl = vis_list[15]
    fxctingacts = vis_list[16]
    fxctblly = vis_list[17]
    fxctlay = vis_list[18]
   

    fxctblly.display = 'none'
    fxctlbl.layout = fxctblly

    fxctlay.display = 'none'
    fxctingacts.layout = fxctlay


    sclblly.display = 'none'
    scalelbl.layout = sclblly

    ecndlay.display = 'none'
    encodingacts.layout = ecndlay

    encdblly.display = 'none'
    encdlbl.layout = encdblly

 
    outrmvlay.display = 'none'
    outrmvbtn.layout = outrmvlay
    
    imbllbllly.display = 'none'
    imbllbl.layout = imbllbllly

    prctlay.display = 'none'
    scalingacts.layout = prctlay

    imblncdlay.display = 'none'
    balncacts.layout = imblncdlay

    return


def featureclclick(trgcl_lbl,featurescl,trgtyp_lbl,miss_lbl):  

    
    #settings.curr_df,trgcl_lbl,featurescl,trgtyp_lbl,miss_lbl
    bk_ind = 0
    for c in reversed(featurescl.value):
        if c == '(':
            break
        bk_ind-=1

    colname = featurescl.value[:bk_ind-1]

    trgcl_lbl.value = " Column: "+colname
    trgtyp_lbl.value= " Type: " +str(settings.curr_df[colname].dtype)
    miss_lbl.value =" Missing values: " + str(settings.curr_df[colname].isnull().sum())
    
    return

def featureprclick(features2,FeatPage,processtypes,ProcssPage,scalingacts):  
    colname = features2.value

    if not colname in settings.curr_df.columns:
        return
    
    with FeatPage:
        clear_output()
            
        if (settings.curr_df[colname].dtype == 'float64') or (settings.curr_df[colname].dtype == 'int64'):

            fig, (axbox, axhist) = plt.subplots(1,2)
     
            sns.boxplot(x=colname,data=settings.curr_df, ax=axbox)
            axbox.set_title('Box plot') 
            sns.distplot(settings.curr_df[colname],ax=axhist)
            axhist.set_title('Histogram') 
            plt.legend(['Mean '+str(round(settings.curr_df[colname].mean(),2)),'Stdev '+str(round(settings.curr_df[colname].std(),2))], bbox_to_anchor=(0.6, 0.6))
            plt.show()
             
         
                
                ############################################################################################################
        '''
            if processtypes.value == 'Imbalancedness':
                if len(settings.curr_df[colname].unique()) == 2: # binary detection
          
                    plt.figure(figsize=(6, 2))
                    ax = sns.countplot(x=colname,data=settings.curr_df, palette="cool_r")
                    for p in ax.patches:
                        ax.annotate("{:.1f}".format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
                    plt.show()
         '''           
        
        if (settings.curr_df[colname].dtype == 'object') or (settings.curr_df[colname].dtype== 'string'):
        
        
            nrclasses = len(settings.curr_df[colname].unique())
            if nrclasses < 250:
                g = sns.countplot(settings.curr_df, x=colname)
                g.set_xticklabels(g.get_xticklabels(),rotation= 45)
                  
                    #sns.distplot(settings.curr_df[settings.curr_df.columns[optind]]).set_title('Histogram of feature '+settings.curr_df.columns[optind])
                plt.show()
            else:
                display.display('Number of classes: ',nrclasses)
                
    with ProcssPage:
        clear_output()            
    

            
    scalingacts.value = scalingacts.options[0]
    return