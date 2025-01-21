# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:46:59 2024

@author: mfirat
"""

##### import ipywidgets as widgets
from IPython.display import clear_output
from IPython import display
from ipywidgets import *
from datetime import timedelta,date
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import os
from pathlib import Path
import pandas as pd
import warnings
import sys
from sklearn.model_selection import train_test_split 
from sklearn import tree,neighbors,linear_model,ensemble,svm
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn import preprocessing 
import numpy as np

dtsetnames = [] 
rowheight = 20


colabpath = '/content/CPP_Datasets'
warnings.filterwarnings("ignore")

curr_df = pd.DataFrame()
Xtrain_df = pd.DataFrame()
Xtest_df = pd.DataFrame()
ytrain_df = pd.DataFrame()
ytest_df = pd.DataFrame()
clean_df = pd.DataFrame()
original_df = pd.DataFrame()
datasetname = ''
ShowMode = True

targetcolumn = None
predictiontask = None

TrainedModels = []

#######################################################################################################################

class MLModel: 
    def __init__(self,data,target,tasktype,mytype,report):

        #data  = [trdf,tr_tgtdf,tstdf,tst_tgtdf] 
        self.train_df = data[0]
        self.traintrg_df = data[1]
        self.test_df = data[2]
        self.testtrg_df = data[3]

        self.modelsetting = dict()
        self.performance = dict()
        
        self.Type = mytype
        self.myTask = tasktype
        self.PythonObject = None
        self.PreprocessingSteps = [] 

        if self.Type == 'Decision Tree':
            if self.myTask == 'Classification': 
                self.PythonObject = tree.DecisionTreeClassifier()
            if self.myTask == 'Regression': 
                self.PythonObject = tree.DecisionTreeRegressor(random_state = 0) 
        if self.Type == 'KNN':
            if self.myTask == 'Classification': 
                self.PythonObject = neighbors.KNeighborsClassifier(n_neighbors=5)
            if self.myTask == 'Regression': 
                self.PythonObject = neighbors.KNeighborsRegressor(n_neighbors=5)
        if self.Type == 'Linear Model':
            if self.myTask == 'Classification': 
                self.PythonObject = linear_model.SGDClassifier()       
            if self.myTask == 'Regression': 
                self.PythonObject = linear_model.LinearRegression() # try before standardization..
        if self.Type == 'Random Forest':
            if self.myTask == 'Classification': 
                self.PythonObject = ensemble.RandomForestClassifier()       
            if self.myTask == 'Regression': 
                self.PythonObject = ensemble.RandomForestRegressor(n_estimators=15, random_state=0,)   # try before standardization..
        if self.Type == 'SVM':
            if self.myTask == 'Classification': 
                self.PythonObject = svm.SVC(kernel='linear', gamma='auto',probability = True)
            if self.myTask == 'Regression': 
                self.PythonObject = svm.SVR(kernel = 'rbf')
        if self.Type == 'Logistic Regression':
            self.PythonObject = linear_model.LogisticRegression(random_state=16)   # Initialize the model object 
           

        report.value += 'Model.. Type '+str(type(self.PythonObject))+'.. \n' 
        return


    def GetPredictions(self):
        if self.myTask == 'Classification':
            return self.PythonObject.predict(self.test_df) 
        if self.myTask == 'Regression':
            if self.Type == 'Logistic Regression': 
                return self.PythonObject.predict_proba(self.test_df)
            else:
                return self.PythonObject.predict(self.test_df)
      
    def getSkLearnModel(self):
        return self.PythonObject
        
    def getData(self):
        return self.train_df,self.traintrg_df,self.test_df,self.testtrg_df
        
    def getType(self):
        return self.Type

    def GetPerformanceDict(self):
        return self.performance

def Train_Model(tasktype,mytype,results,trmodels):

    global  Xtrain_df,Xtest_df, ytrain_df, ytest_df,curr_df,targetcolumn

    data = [Xtrain_df,ytrain_df,Xtest_df,ytest_df]

    mymodel = MLModel(data,targetcolumn,tasktype,mytype,results)

    model = mymodel.getSkLearnModel().fit(data[0], data[1]) 

    y_pred = mymodel.GetPredictions()

    if tasktype == 'Classification': 
        mymodel.GetPerformanceDict()['Accuracy'] = accuracy_score(data[3], y_pred)
    
    if tasktype == 'Regression': 
        mymodel.GetPerformanceDict()['MSE'] = mean_squared_error(data[3], y_pred)

    TrainedModels.append(mymodel)

    results.value += 'Train Model-> '+mytype+'\n'
    for prf,val in mymodel.GetPerformanceDict().items():
        results.value += 'Model Performance-> '+prf+': '+str(val)+'\n'

    trmodels.options = [mdl.getType() for mdl in TrainedModels]
   

    return 
############################################################################################################    
def make_encoding(features2,encodingacts,result2exp):

    global  Xtrain_df,Xtest_df, ytrain_df, ytest_df,curr_df 

    colname = features2.value

    
    result2exp.value += 'Encoding.. col '+colname+' is list:'+str(int(isinstance(curr_df, list)))+'\n'

    if colname is None:
        return

    
  
    # Encode column  
    if len(Xtrain_df) > 0:
        
        if encodingacts.value == "Label Encoding":
            label_encoder = preprocessing.LabelEncoder() 
            result2exp.value += 'Encoding-> '+features2.value+' (train) current classes: '+str(data_df[0][colname].unique())+'\n'
            data_df[0][colname] = label_encoder.fit_transform(data_df[0][colname]) # train 
            result2exp.value += 'Encoding-> '+features2.value+' (train) after labeling classes: '+str(data_df[0][colname].unique())+'\n'
            result2exp.value += 'Encoding-> '+features2.value+' (test) current classes: '+str(data_df[1][colname].unique())+'\n'
            data_df[1][colname] = label_encoder.fit_transform(data_df[1][colname]) # test
            result2exp.value += 'Encoding-> '+features2.value+' (test) after labeling classes: '+str(data_df[1][colname].unique())+'\n'

            data_df = data_df[0],data_df[1]
            
        if encodingacts.value == "One Hot Encoding":
            categorical_columns = [colname]
            
            encoder = preprocessing.OneHotEncoder(sparse_output=False)  # Initialize OneHotEncoder
            one_hot_encoded = encoder.fit_transform(data_df[0][categorical_columns])  # Fit and transform the categorical columns          
            one_hot_df = pd.DataFrame(one_hot_encoded,columns=encoder.get_feature_names_out(categorical_columns)) # Create a DataFrame
            data1_df = pd.concat([data_df[0].drop(categorical_columns, axis=1), one_hot_df], axis=1)
            result2exp.value += 'One Hot Encoding-> (train) after one-hot features: '+str(data1_df.columns)+'\n'

            encoder = preprocessing.OneHotEncoder(sparse_output=False)  # Initialize OneHotEncoder
            one_hot_encoded = encoder.fit_transform(data_df[1][categorical_columns])  # Fit and transform the categorical columns          
            one_hot_df = pd.DataFrame(one_hot_encoded,columns=encoder.get_feature_names_out(categorical_columns)) # Create a DataFrame 
            data2_df = pd.concat([data_df[1].drop(categorical_columns, axis=1), one_hot_df], axis=1)
            result2exp.value += 'One Hot Encoding-> (test) after one-hot features: '+str(data2_df.columns)+'\n'

            data_df = data1_df,data2_df

            features2.options = [col+'('+str(data1_df[col].isnull().sum())+')' for col in data1_df.columns]
    else:
        result2exp.value += 'Encoding-> '+encodingacts.value+', '+str(encodingacts.value == "One Hot Encoding")+'\n'
        result2exp.value += 'Encoding-> '+colname+' current classes: '+str(len(curr_df))+'\n'
        if encodingacts.value == "Label Encoding":
            label_encoder = preprocessing.LabelEncoder() 
            result2exp.value += 'Encoding-> '+features2.value+' current classes: '+str(curr_df[colname].unique())+'\n'
            curr_df[colname] = label_encoder.fit_transform(curr_df[colname]) 
            result2exp.value += 'Encoding-> '+features2.value+' after labeling classes: '+str([cls for cls in curr_df[colname].unique()])+'\n'
            
        if encodingacts.value == "One Hot Encoding":
            result2exp.value += 'Encoding-> '+colname+' current classes: '+str(curr_df[colname].unique())+'\n'
            categorical_columns = [colname]
            encoder = preprocessing.OneHotEncoder(sparse_output=False)  # Initialize OneHotEncoder
            one_hot_encoded = encoder.fit_transform(curr_df[categorical_columns])  # Fit and transform the categorical columns          
            one_hot_df = pd.DataFrame(one_hot_encoded,columns=encoder.get_feature_names_out(categorical_columns)) # Create a DataFrame
            curr_df = pd.concat([curr_df.drop(categorical_columns, axis=1), one_hot_df], axis=1)
            result2exp.value += 'One Hot Encoding-> after one-hot features: '+str(curr_df.columns)+'\n'

        features2.options = [col for col in curr_df.columns]
    
    return
#############################################################################################################
def assign_target(trg_lbl,dt_features,prdtsk_lbl,result2exp,trg_btn,predictiontask):

    global curr_df,targetcolumn

    targetcolumn = dt_features.value

    trg_lbl.value = targetcolumn
    trg_btn.disabled = True

    
    if (curr_df[targetcolumn].dtype == 'float64') or (curr_df[targetcolumn].dtype == 'int64'):
        predictiontask = "Regression"
    else:
        predictiontask = "Classification" 

    prdtsk_lbl.value = predictiontask 
    result2exp.value += 'Target assigned: '+targetcolumn+'\n'
    

    return 
####################################################################################################################
def make_split(splt_txt,splt_btn,result2exp):
    global curr_df,Xtrain_df,Xtest_df, ytrain_df, ytest_df,targetcolumn

    if targetcolumn is None:
        return
  
    y = curr_df[targetcolumn] # Target variable 
    column_list = [col for col in curr_df.columns]
    column_list.remove(targetcolumn)
    X = curr_df[column_list]
    
    ratio_percnt = int(splt_txt.value) 
    result2exp.value += 'Split ratio, '+str(ratio_percnt/100)+'\n'
    Xtrain_df,Xtest_df, ytrain_df, ytest_df = train_test_split(X, y, test_size=ratio_percnt/100, random_state=16)
    splt_btn.disabled = True

    result2exp.value += 'Split, Train size: '+str(len(Xtrain_df))+'\n' 


    return

def read_data_set(online_version,foldername,filename,sheetname,processtypes,Pages,dt_features,dt_ftslay,featurescl,ftlaycl):

    FeatPage,ProcssPage,DFPage,RightPage = Pages
    global curr_df
    
    rel_path = foldername+'\\'+filename
    
    if online_version:
        abs_file_path = colabpath+'/'+filename
    else:
        abs_file_path = os.path.join(Path.cwd(), rel_path)
        

    if abs_file_path.find('.csv') > -1:
        curr_df = pd.read_csv(abs_file_path, sep=sheetname) 
    if (abs_file_path.find('.xlsx') > -1) or (filename.find('.xls') > -1):
        xls = pd.ExcelFile(abs_file_path)
        curr_df = pd.read_excel(xls,sheetname)
    if abs_file_path.find('.tsv') > -1:    
       
        curr_df = pd.read_csv(abs_file_path, sep="\t")
        
    curr_df.convert_dtypes()
     
    datasetname = filename[:filename.find('.')]  
    
    dt_ftslay.height = str(rowheight*len(curr_df.columns))+'px'
    dt_features.layout = dt_ftslay
    dt_features.options = [col for col in curr_df.columns]


    ftlaycl.display = 'block'
    ftlaycl.height = str(rowheight*len(curr_df.columns))+'px'
    featurescl.layout = ftlaycl
    featurescl.options = [col+'('+str(curr_df[col].isnull().sum())+')' for col in curr_df.columns]
    
    processtypes.value = processtypes.options[0]
    
    with FeatPage:
        clear_output()
    with ProcssPage:
        clear_output()
    
    with DFPage:
        clear_output()
        #####################################
        display.display(curr_df.info()) 
        display.display(curr_df.describe()) 
        display.display(curr_df) 
        #####################################

    with RightPage:
        clear_output()

    return 


################################################################################################################
def File_Click(online_version,foldername,filename,wsheets,wslay,butlay):
    
    # filename = datasets.value
    # foldername = DataFolder.value

    
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

##################################################################################################################
#########################################################################################################

def on_submitfunc(online_version,foldername,datasets):
    
    #  foldername = DataFolder.value
    
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
#######################################################################################################
###########################################  TAB: Data Cleaning ##################################################



##########################################################################################################
def make_cleaning(featurescl,result2aexp,missacts,dt_features): 

    global curr_df
    bk_ind = 0
    for c in reversed(featurescl.value):
        if c == '(':
            break
        bk_ind-=1

    colname = featurescl.value[:bk_ind-1]

    handling = missacts.value

    result2aexp.value+= 'Data cleaning: col '+colname+', action '+handling+', coltype '+str(curr_df[colname].dtype)+'\n' 

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
                result2aexp.value+= 'Data cleaning: Improper action is selected.. '+'\n'
                return
        else: 
            result2aexp.value+= 'Data cleaning: mode.. '+str(curr_df[colname].mode()[0])+'\n'
            if handling == 'Replace-Mode': 
                curr_df[colname].fillna(curr_df[colname].mode()[0], inplace=True)
            
    featurescl.options = [col+'('+str(curr_df[col].isnull().sum())+')' for col in curr_df.columns]
    dt_features.options = [col for col in curr_df.columns]
    
    result2aexp.value+= 'Done.. '+'\n'    

    return
##################################################################################################################
def savecurrdata(change):
    
    global curr_df,DataFolder
    
    version = 0
##################################################################################################################
def drawlmplot(curr_df,xdrop,ydrop,huedrop,VisualPage):
    
    x_feat = xdrop.value
    y_feat = ydrop.value
   
    hue_feat = huedrop.value
    
    with VisualPage:
        
        clear_output()
    
        if hue_feat!= '':
            sns.lmplot(x=x_feat, y=y_feat, data=curr_df, hue=hue_feat, palette="Set1",ci = 0)
        else:
            sns.lmplot(x=x_feat, y=y_feat, data=curr_df, palette="Set1",ci = 0)
            
        plt.show()
    
    version = 0
    
    return
##################################################################################################################
#########################################  TAB: Data Processing   ################################################
##################################################################################################################
from sklearn.utils import resample
##################################################################################

def NormalizeColumn(df,colname):
    
    col_min = min(df[colname])
    col_max = max(df[colname])
    
    if col_max == col_min: 
        return
    
    df[colname] = (df[colname]- col_min)/(col_max-col_min)
    
    return df

###############
def featureprclick(features2,FeatPage,processtypes,ProcssPage,scalingacts):  

    global curr_df
 
    colname = features2.value

    if not colname in curr_df.columns:
        return
    
    with FeatPage:
        clear_output()
            
        if (curr_df[colname].dtype == 'float64') or (curr_df[colname].dtype == 'int64'):

            fig, (axbox, axhist) = plt.subplots(1,2)
     
            sns.boxplot(x=colname,data=curr_df, ax=axbox)
            axbox.set_title('Box plot') 
            sns.distplot(curr_df[colname],ax=axhist)
            axhist.set_title('Histogram') 
            plt.legend(['Mean '+str(round(curr_df[colname].mean(),2)),'Stdev '+str(round(curr_df[colname].std(),2))], bbox_to_anchor=(0.6, 0.6))
            plt.show()
             
         
                
                ############################################################################################################
        '''
            if processtypes.value == 'Imbalancedness':
                if len(curr_df[colname].unique()) == 2: # binary detection
          
                    plt.figure(figsize=(6, 2))
                    ax = sns.countplot(x=colname,data=curr_df, palette="cool_r")
                    for p in ax.patches:
                        ax.annotate("{:.1f}".format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
                    plt.show()
         '''           
        
        if (curr_df[colname].dtype == 'object') or (curr_df[colname].dtype== 'string'):
        
        
            nrclasses = len(curr_df[colname].unique())
            if nrclasses < 250:
                g = sns.countplot(curr_df, x=colname)
                g.set_xticklabels(g.get_xticklabels(),rotation= 45)
                  
                    #sns.distplot(curr_df[curr_df.columns[optind]]).set_title('Histogram of feature '+curr_df.columns[optind])
                plt.show()
            else:
                display.display('Number of classes: ',nrclasses)
                
    with ProcssPage:
        clear_output()            
    

            
    scalingacts.value = scalingacts.options[0]
    return
##############
###############
def featureclclick(trgcl_lbl,featurescl,trgtyp_lbl,miss_lbl):  

    global curr_df
    #curr_df,trgcl_lbl,featurescl,trgtyp_lbl,miss_lbl
    bk_ind = 0
    for c in reversed(featurescl.value):
        if c == '(':
            break
        bk_ind-=1

    colname = featurescl.value[:bk_ind-1]

    trgcl_lbl.value = " Column: "+colname
    trgtyp_lbl.value= " Type: " +str(curr_df[colname].dtype)
    miss_lbl.value =" Missing values: " + str(curr_df[colname].isnull().sum())
    
    return
##############
def vistypeclick(curr_df,ShowMode,vboxvis1,vbvs1lay,visualtypes,vlmpltcomps,vboxlmplot,vblmpltlay,VisualPage,changenew):  
 
    if not ShowMode:
        return
    
    
    if visualtypes.value == 'Lmplot':
        vblmpltlay.display = 'block'
        
        vlmpltcomps[0].options = [str(curr_df.columns[colid]) for colid in range(len(curr_df.columns))]
        vlmpltcomps[1].options = [str(curr_df.columns[colid]) for colid in range(len(curr_df.columns))]
        vlmpltcomps[2].options = [str(curr_df.columns[colid]) for colid in range(len(curr_df.columns))]
        vblmpltlay.height ='200px'
        vboxlmplot.layout = vblmpltlay 
        
        vbvs1lay.height = str(int(vblmpltlay.height[:vblmpltlay.height.find('px')])+150)+'px'
        vboxvis1.layout = vbvs1lay
        
    
      
    else:
        vblmpltlay.display = 'none'
        vboxlmplot.layout = vblmpltlay 
        with VisualPage:
            clear_output()
            

        
    return
##############
def make_scaling(dt_features,ProcssPage,scalingacts,result2exp):  

    global  Xtrain_df,Xtest_df, ytrain_df, ytest_df,curr_df
  
    colname = dt_features.value

    if colname is None:
        return

    result2exp.value += 'Scaling-> '+scalingacts.value+': '+colname+'\n'
    
    if (curr_df[colname].dtype == 'object') or (curr_df[colname].dtype== 'string'):
        with ProcssPage:
            clear_output()
            display.display('Selected column is not a numerical type..')
        return

    if scalingacts.value == 'Standardize':

        if len(Xtrain_df)>0:
            if colname in Xtrain_df.columns:
                colmean = Xtrain_df[colname].mean(); colstd = Xtrain_df[colname].std()
                Xtrain_df[colname] = (Xtrain_df[colname]- colmean)/colstd
                Xtest_df[colname] = (Xtest_df[colname]- colmean)/colstd
            if colname in ytrain_df.columns:
                colmean = ytrain_df[colname].mean(); colstd = ytrain_df[colname].std()
                ytrain_df[colname] = (ytrain_df[colname]- colmean)/colstd
                ytest_df[colname] = (ytest_df[colname]- colmean)/colstd
                 
        colmean = curr_df[colname].mean()
        curr_df[colname] = (curr_df[colname]- colmean)/curr_df[colname].std()


    if scalingacts.value == 'Normalize':

        if len(Xtrain_df)>0:
            if colname in Xtrain_df.columns:
                col_min = min(Xtrain_df[colname]); col_max = max(Xtrain_df[colname])
                denominator = (col_max-col_min)
                if denominator== 0:
                    Xtrain_df[colname] = (Xtrain_df[colname]/col_min)
                    Xtest_df[colname] = (Xtest_df[colname]/col_min)
                else:
                    Xtrain_df[colname] = (Xtrain_df[colname]-col_min)/denominator
                    Xtest_df[colname] = (Xtest_df[colname]-col_min)/denominator
               
            if colname in ytrain_df.columns:
                col_min = min(ytrain_df[colname]); col_max = max(ytrain_df[colname])
                denominator = (col_max-col_min)
                if denominator== 0:
                    ytrain_df[colname] = (ytrain_df[colname]/col_min)
                    ytest_df[colname] = (ytest_df[colname]/col_min)
                else:
                    ytrain_df[colname] = (ytrain_df[colname]-col_min)/denominator
                    ytest_df[colname] = (ytest_df[colname]-col_min)/denominator

        
        col_min = min(curr_df[colname]); col_max = max(curr_df[colname])
        denominator = (col_max-col_min)

        if denominator== 0:
            curr_df[colname] = (curr_df[colname]/col_min)
        else:
            curr_df[colname] = (curr_df[colname]-col_min)/denominator

    with ProcssPage:
        clear_output()
        fig, (axbox, axhist) = plt.subplots(1,2)
     
        sns.boxplot(x=colname,data=curr_df, ax=axbox)
        axbox.set_title('Box plot') 
        sns.distplot(curr_df[colname],ax=axhist)
        axhist.set_title('Histogram') 
        plt.legend(['Mean '+str(round(curr_df[colname].mean(),2)),'Stdev '+str(round(curr_df[colname].std(),2))], bbox_to_anchor=(0.6, 0.6))
        plt.show()
    
 
    return

#################################################################################################################
def make_balanced(features2,balncacts,ProcssPage):  

    global curr_df
    colname = features2.value

    if balncacts.value == 'Upsample':
         
        if len(curr_df[colname].unique()) == 2: # binary detection
            
            colvals = curr_df[colname].unique()
            ColmFirst = curr_df[ curr_df[colname] == colvals[0]]
            ColmOther = curr_df[ curr_df[colname] == colvals[1]]
          
            if len(ColmFirst) < len(ColmOther):
                upsampled_First = resample(ColmFirst, replace=True, n_samples=len(ColmOther), random_state=27) 
                curr_df = pd.concat([ColmOther, upsampled_First])
            else:
                upsampled_Other= resample(ColmOther, replace=True, n_samples=len(ColmFirst), random_state=27) 
                curr_df = pd.concat([ColmFirst, upsampled_Other])
                
            with ProcssPage:
                
                clear_output()
                plt.figure(figsize=(6, 2))
                ax = sns.countplot(x=colname,data=curr_df, palette="cool_r")
                for p in ax.patches:
                    ax.annotate("{:.1f}".format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
                plt.show()

    return
####################################################################################################################

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

##################################################################################
def remove_outliers():

    global curr_df
    curr_df = curr_df[curr_df["outlier"] == False]
    curr_df = curr_df.drop(["outlier"], axis=1)
   
    return

##################################################################################
    
    if datasetname.find('_clean') > 0:
        
        checkstr = datasetname[datasetname.find('_clean')+1:]
        
     
        version = int(checkstr[checkstr.find('_v')+2:])
        
        dsname = datasetname[:datasetname.find('_clean')]

        filename = DataFolder.value+'/'+dsname+'_clean_v'+str(version+1)+'.csv'
        curr_df.to_csv(filename, index=False) 
            
    else:
        
        filename = DataFolder.value+'/'+datasetname+'_clean_v'+str(version)+'.csv'
        curr_df.to_csv(filename, index=False) 
    
    return
######################################################################################################################