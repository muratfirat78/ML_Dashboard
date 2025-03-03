# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:46:59 2024

@author: mfirat
"""
import shutil

##### import ipywidgets as widgets
from IPython.display import clear_output
from IPython import display
from ipywidgets import *
from datetime import timedelta,date, datetime
import settings
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import os
from log import *
from pathlib import Path
import pandas as pd
import warnings
import sys
import logging
from sklearn.model_selection import train_test_split 
from sklearn import tree,neighbors,linear_model,ensemble,svm
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn import preprocessing 
import numpy as np

dtsetnames = [] 
rowheight = 20

colabpath = '/content/CPP_Datasets'
warnings.filterwarnings("ignore")

datasetname = ''
ShowMode = True

targetcolumn = None
predictiontask = None

TrainedModels = []

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
##################################################################################################################
def savedata(curr_df, dataFolder, datasetname):
    datasetname = os.path.splitext(os.path.basename(datasetname))[0]
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = dataFolder.value + '/' + datasetname + '_' + current_datetime
    shutil.copy('output.log', filename + '.txt')
    settings.curr_df.to_csv(filename + '.csv')

    
    version = 0
##################################################################################################################
def drawlmplot(curr_df,xdrop,ydrop,huedrop,VisualPage):
    
    x_feat = xdrop.value
    y_feat = ydrop.value
   
    hue_feat = huedrop.value
    
    with VisualPage:
        
        clear_output()
    
        if hue_feat!= '':
            sns.lmplot(x=x_feat, y=y_feat, data=settings.curr_df, hue=hue_feat, palette="Set1",ci = 0)
        else:
            sns.lmplot(x=x_feat, y=y_feat, data=settings.curr_df, palette="Set1",ci = 0)
            
        plt.show()
    
    version = 0
    
    return
##################################################################################################################
#########################################  TAB: Data Processing   ################################################
##################################################################################################################
from sklearn.utils import resample

###############
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
##############
###############
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
##############
def vistypeclick(curr_df,ShowMode,vboxvis1,vbvs1lay,visualtypes,vlmpltcomps,vboxlmplot,vblmpltlay,VisualPage,changenew):  
 
    if not ShowMode:
        return
    
    
    if visualtypes.value == 'Lmplot':
        vblmpltlay.display = 'block'
        
        vlmpltcomps[0].options = [str(settings.curr_df.columns[colid]) for colid in range(len(settings.curr_df.columns))]
        vlmpltcomps[1].options = [str(settings.curr_df.columns[colid]) for colid in range(len(settings.curr_df.columns))]
        vlmpltcomps[2].options = [str(settings.curr_df.columns[colid]) for colid in range(len(settings.curr_df.columns))]
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
#############


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
    
    # if datasetname.find('_clean') > 0:
        
    #     checkstr = datasetname[datasetname.find('_clean')+1:]
        
     
    #     version = int(checkstr[checkstr.find('_v')+2:])
        
    #     dsname = datasetname[:datasetname.find('_clean')]

    #     filename = DataFolder.value+'/'+dsname+'_clean_v'+str(version+1)+'.csv'
    #     settings.curr_df.to_csv(filename, index=False) 
            
    # else:
        
    #     filename = DataFolder.value+'/'+datasetname+'_clean_v'+str(version)+'.csv'
    #     settings.curr_df.to_csv(filename, index=False) 
    
    # return
######################################################################################################################