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
import pandas as pd
import warnings
import seaborn as sns
import os
from pathlib import Path



colabpath = '/content/CPP_Datasets'
warnings.filterwarnings("ignore")


file_names = {'csv':['apple_quality','aug_train','diet','housing','loan_data','worldcities','flight_data','titanic','Telecom_Churn']}
file_names['xlsx'] = ['crop_yield']
file_names['tsv'] = ['gapminder']

#######################################################################################################################

def read_data_set(curr_df,online_version,foldername,filename,sheetname,reslay,resultexp,processtypes,FeatPage,ProcssPage,DFPage):
    
   
    
    #  filename = datasets.value 
    #  foldername = DataFolder.value
    #  sheetname = wsheets.value
    
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
    
     
    reslay.height = '100px'
    resultexp.value = '' 
    resultexp.layout = reslay
    
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
  
    return curr_df


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

def Activate_Tab1(online_version,rowheight,curr_df,ShowMode,mhndslay,ftlay,dtlay,featurevals,features,dtypes,missing,msslay,misshands,svebtn):
    

    
    ShowMode = False
    
    msslay.display = 'block'
    msslay.height ='30px'
    mhndslay.display = 'block'
    mhndslay.height ='30px'
    ftlay.display = 'block'
    ftlay.height ='30px'
    dtlay.display = 'block'
    dtlay.height ='30px'
    
    if online_version: 
        svebtn.disabled = True
    
    featurevals.options = []
     
    features.options = curr_df.columns
    dtypes.options =[str(colid)+'-'+str(curr_df[curr_df.columns[colid]].dtype) for colid in  range(len(curr_df.columns))]

    currhghttxt = int(ftlay.height[:ftlay.height.find('px')])
    ftlay.height = str(min(450,currhghttxt+len(curr_df.columns)*rowheight))+'px'
    features.layout = ftlay 

    currhghttxt = int(dtlay.height[:dtlay.height.find('px')])
    dtlay.height = str(min(450,currhghttxt+len(curr_df.columns)*rowheight))+'px'
    dtypes.layout = dtlay 

       
    missing.options = [str(colid)+'-'+str(curr_df[curr_df.columns[colid]].isnull().sum()) for colid in  range(len(curr_df.columns))]
    msslay.height = ftlay.height 
    missing.layout = msslay

    misshands.options = [str(colid)+'-'+'Keep'+','+'Remove' for colid in  range(len(curr_df.columns))]
    mhndslay.height = ftlay.height 
    misshands.layout = mhndslay
    
    ShowMode = True
     
    return curr_df

##########################################################################################################
def Handle_Missing_Values(curr_df,resultexp,misshands): 
    
   
    clean_df = curr_df.copy()
    optind = 0
    rowheight = 16
  
    inisize = len(curr_df)
    inicols = len(curr_df.columns)
    
    resultexp.value += 'Initial data: '+str(inicols)+', columns,  size '+str(inisize)+'\n'
    
    nrrows = 0
    
    for opt in misshands.options:
        curact = misshands.options[optind]  
        check = curact[curact.find('-')+1:]  
        check = check[:check.find(',')]
      
        if check != 'Drop':
            if curact[curact.find(',')+1:] == 'Remove':  
               
                clean_df = clean_df.dropna(subset = [clean_df.columns[optind]])   
            if curact[curact.find(',')+1:] == 'Mean':
                clean_df[clean_df.columns[optind]].fillna(clean_df[clean_df.columns[optind]].mean(), inplace=True)
            if curact[curact.find(',')+1:] == 'Median':
                clean_df[clean_df.columns[optind]].fillna((clean_df[clean_df.columns[optind]].median()), inplace=True)    
        optind+=1

    # check dropped columns
    optind = 0
    dropped = 0 
    for opt in misshands.options:
        curact = misshands.options[optind]
        curact = curact[curact.find('-')+1:]  
        curact = curact[:curact.find(',')]
       
        if curact == 'Drop':
            resultexp.value += 'Column drop: '+clean_df.columns[optind]+'\n'
            nrrows+=1
            clean_df = clean_df.drop([clean_df.columns[optind-dropped]], axis=1)
            dropped+=1

        optind+=1 
        
    curr_df = clean_df

   
    resultexp.value += 'Cleaned data: '+str(len(clean_df.columns))+', columns, size  '+str(len(clean_df))+'\n'
    
    
    
    
    return curr_df
######################################################################################################################
def featureclick(ShowMode,features,featurevals,featurename,curr_df,dtypes,missing,misshands,HCPage):  

    
    if not ShowMode:
        return
    
    
    featurename.value = 'Selected Feature: '+features.value
    ratiosum = 0
    optind = 0
    datasize = len(curr_df)
    
  
    
    for opt in features.options:
        if features.value == opt:
           
            dtypes.value = dtypes.options[optind]
            missing.value = missing.options[optind]
            misshands.value = misshands.options[optind]
            typecheck = dtypes.value
            checkissing = missing.value
            break
            
            
            '''
            if (checkissing[checkissing.find('-')+1:] == '0') & (len(curr_df[curr_df.columns[optind]].unique())) <= 250:
             
                featurevals.options = [x for x in curr_df[curr_df.columns[optind]].unique()]
              
                if (typecheck[typecheck.find('-')+1:] == 'float64') | (typecheck[typecheck.find('-')+1:] == 'int64'):
                    featurevals.options = sorted(featurevals.options, key=lambda x: x, reverse=False)
                    ratiosum =sum([len(curr_df[curr_df[curr_df.columns[optind]] == x])/datasize for x in featurevals.options])
              
                featurevals.options = [str(x)+' ('+str(round(100*len(curr_df[curr_df[curr_df.columns[optind]] == x])/datasize,3))+'%)' for x in featurevals.options]       
                featurevals.value = featurevals.options[0]
            '''

           
        optind+=1
        
   
   

    return

#####################################################################################################################
def actionclick(features,dtypes,newchg,misshands):  
    
 
    selectid = 0
    optind = 0
    firstpart=  ''
    
   
    for opt in features.options:
       
        if opt == features.value:
            checktype = dtypes.options[optind]
            if (newchg == 'Mean') or  (newchg == 'Median'):
                if checktype[checktype.find('-')+1:] != 'float64':
                    return
            selectid = optind
            firstpart = misshands.options[selectid][:misshands.options[selectid].find(',')]
            break
        optind+=1
  
            
    newopt = firstpart+','+newchg
 
    misshands.options = [misshands.options[optind] if optind!= selectid else newopt for optind in range(len(misshands.options))]
 
   
    
    return

##################################################################################################################
def columnclick(colmnacts,features,misshands,newchg):  

    if colmnacts.value == colmnacts.options[0]:
        return
 
    selectid = 0
    
    
    
    optind = 0   
    for opt in features.options:
        if opt == features.value:
            selectid = optind
            break
        optind+=1
        
    prevstr = misshands.options[selectid]
  

    newopt = str(selectid)+'-'+newchg+','+prevstr[prevstr.find(',')+1:]
    
    misshands.options = [misshands.options[optind] if optind!= selectid else newopt for optind in range(len(misshands.options))]
    colmnacts.value = colmnacts.options[0]
    
    return
#################################################################################################################
def featval_click(change):  
    global misshands,colmnacts,features,dtypes,missingacts,curr_df,featurevals,valchangeacts,resultexp,reslay
 
    selectid = 0      
     
    
    optind = 0   
    for opt in featurevals.options:
        if opt == featurevals.value:
            selectid = optind
            break
        optind+=1
    
    colname = features.value 
    
   
    
    featureval =  featurevals.value
 
    featureval =  featureval[:featureval.find('(')-1]
  
    
    prevsize = len(curr_df)
    
    if valchangeacts.value == 'Remove':
        curr_df = curr_df.drop(curr_df[curr_df[colname] == featureval].index)
     
    
    resultexp.value += 'Feature: '+colname+' value '+featureval+ ' removed. Size: '+str(prevsize)+'->'+str(len(curr_df))+'\n'
    valchangeacts.value == valchangeacts.options[0]
    
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
def StandardizeColumn(df,colname):
    
    colmean = df[colname].mean()
    
    df[colname] = (df[colname]- colmean)/df[colname].std()
    
    return df

def NormalizeColumn(df,colname):
    
    col_min = min(df[colname])
    col_max = max(df[colname])
    
    if col_max == col_min: 
        return
    
    df[colname] = (df[colname]- col_min)/(col_max-col_min)
    
    return df

####################################################################################
def Activate_Tab2(curr_df,ftlay2,features2,rowheight,sveprbtn,online_version):
    
    
    ftlay2.display = 'block'
    ftlay2.height ='30px'
    
    if online_version:
        sveprbtn.disabled = True
    
    features2.options = curr_df.columns

   
    currhghttxt = int(ftlay2.height[:ftlay2.height.find('px')])
    ftlay2.height = str(min(450,currhghttxt+len(curr_df.columns)*rowheight))+'px'
    features2.layout = ftlay2 
    
    return

###################################################################################
def Activate_Tab3(curr_df,t4_ftlay,t4_vb1lay,tb4_vbox1,t4_features,rowheight,online_version):
    #curr_df,t4_ftlay,t4_vb1lay,tb4_vbox1,t4_features,rowheight,online_version

    if online_version:
        sveprbtn.disabled = True
    
    #visualtypes.options = curr_df.columns
    t4_features.options = curr_df.columns
   
    
    
    return

###############
def featureprclick(curr_df,ShowMode,features2,FeatPage,processtypes,ProcssPage,scalingacts):  
 
    if not ShowMode:
        return
    
    colname = features2.value
    
    
    with FeatPage:
        clear_output()
            
        if (curr_df[colname].dtype == 'float64') or (curr_df[colname].dtype == 'int64'):
            
            if processtypes.value == 'Scaling':
                sns.distplot(curr_df[colname]).set_title('Histogram of feature '+colname)

                plt.legend(['Mean '+str(round(curr_df[colname].mean(),2)),'Stdev '+str(round(curr_df[colname].std(),2))], bbox_to_anchor=(0.6, 0.6))

                plt.show()
                       
            if processtypes.value == 'Outlier':
                #######################################################################################################
                
                quantiles = curr_df[colname].quantile([0.25,0.5,0.75])
                IQR = quantiles[0.75] - quantiles[0.25]
                boxplot_outlierLB =  quantiles[0.25]-1.5*IQR
                boxplot_outlierUB =  quantiles[0.75]+1.5*IQR
            
                sns.boxplot(curr_df[colname]).set_title('Box plot of '+colname+' (Boundaries '+
                                                        str(round(boxplot_outlierLB,2))
                                                        +','+str(round(boxplot_outlierUB,2))+')')
                plt.show()
             
             
         
                
                ############################################################################################################
            if processtypes.value == 'Imbalancedness':
                if len(curr_df[colname].unique()) == 2: # binary detection
          
                    plt.figure(figsize=(6, 2))
                    ax = sns.countplot(x=colname,data=curr_df, palette="cool_r")
                    for p in ax.patches:
                        ax.annotate("{:.1f}".format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
                    plt.show()
                    
        
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
def make_scaling(curr_df,features2,ProcssPage,scalingacts):  
    
  
    colname = features2.value
    
    if (curr_df[colname].dtype == 'object') or (curr_df[colname].dtype== 'string'):
        with ProcssPage:
            clear_output()
            display.display('Selected column is not a numerical type..')
        return
    
    if scalingacts.value == 'Standardize':
        curr_df =  StandardizeColumn(curr_df,colname)

        with ProcssPage:
            clear_output()
            sns.distplot(curr_df[colname]).set_title('Histogram of standardized feature '+colname)
            plt.legend(['Mean '+str(round(curr_df[colname].mean(),2)),'Stdev '+str(round(curr_df[colname].std(),2))], bbox_to_anchor=(0.6, 0.6))
          
            #plt.text(20,20, 'Mean= '+str(round(curr_df[colname].mean(),2)), dict(size=10))
            #plt.text(25,25, 'Stdev= '+str(round(curr_df[colname].std(),2)), dict(size=10))
            plt.show()
            
    if scalingacts.value == 'Normalize':
        curr_df =  NormalizeColumn(curr_df,colname)

        with ProcssPage:
            clear_output()

            sns.distplot(curr_df[colname]).set_title('Histogram of normalized feature '+colname)
            plt.legend(['Mean '+str(round(curr_df[colname].mean(),4)),'Stdev '+str(round(curr_df[colname].std(),4))], bbox_to_anchor=(0.6, 0.6))
          
            #plt.text(20,20, 'Mean= '+str(round(curr_df[colname].mean(),2)), dict(size=10))
            #plt.text(25,25, 'Stdev= '+str(round(curr_df[colname].std(),2)), dict(size=10))
            plt.show()
 
    return
#################################################################################################################
def make_balanced(curr_df,features2,balncacts,ProcssPage):  

    
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
    
    sclblly.display = 'none'
    sclblly.visibility = 'hidden'
    scalelbl.layout = sclblly
    
    outrmvlay.display = 'none'
    outrmvlay.visibility = 'hidden'
    outrmvbtn.layout = outrmvlay
    
    imbllbllly.display = 'none'
    imbllbllly.visibility = 'hidden'
    imbllbl.layout = imbllbllly
    
    prctlay.visibility = 'hidden'
    scalingacts.layout = prctlay
    
    imblncdlay.visibility = 'hidden'
    balncacts.layout = imblncdlay
    
    if processtypes.value == 'Scaling':
        sclblly.display = 'block'
        sclblly.visibility = 'visible'
        scalelbl.layout = sclblly
        prctlay.visibility = 'visible'
        scalingacts.layout = prctlay
        
    if processtypes.value == 'Imbalancedness':
        
        imbllbllly.display = 'block'
        imbllbllly.visibility = 'visible'
        imbllbl.layout = imbllbllly
        imblncdlay.visibility = 'visible'
        balncacts.layout = imblncdlay
     
    if processtypes.value == 'Outlier':     
        outrmvlay.display = 'block'
        outrmvlay.visibility = 'visible'
        outrmvbtn.layout = outrmvlay
    


    
    return

##################################################################################
def remove_outliers(curr_df):
    
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

####################################################################################