from sklearn import preprocessing 
from sklearn.decomposition import PCA
from log import *
from sklearn.model_selection import train_test_split 
from IPython.display import clear_output
from IPython import display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
import pandas as pd
import numpy as np
import os
import shutil
from datetime import timedelta,date, datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class DataProcessingModel:
    def __init__(self, main_model, logger):
        self.main_model = main_model
        self.logger = logger

    def showCorrHeatMap(self,ProcssPage,fxctingacts,result2exp):

        write_log('Correlation: '+fxctingacts.value,result2exp, 'Correlation')

        if fxctingacts.value == "Correlation":
            current_df = self.main_model.get_curr_df()

            write_log('Correlation: '+str(len(current_df)),result2exp, 'Correlation')
            
            if self.main_model.datasplit:
                
                Xtrain_df = self.main_model.get_XTrain()
                ytrain_df = self.main_model.getYtrain()
                current_df = pd.concat([Xtrain_df,ytrain_df],axis=1)

            corrcols = [col for col in current_df.columns if current_df[col].dtype in ['float64','int64','int32']]

            with ProcssPage:
                clear_output()
                axis_corr = sns.heatmap( current_df[corrcols].corr(), vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(50, 500, n=500),square=True)
                plt.show()


        return 

    def ApplyPCA(self,features2,pca_features,result2exp):


        pcafeats = [ftname for ftname in pca_features.options]
        self.logger.add_action(['DataProcessing', 'PCA'], pcafeats)

        if self.main_model.targetcolumn in pcafeats:
            write_log('PCA: Returned due to inclusion of target in PCA',result2exp, 'PCA')
            return

        if self.main_model.datasplit:

                  
            Xtest = self.main_model.get_XTest()
            XTrain = self.main_model.get_XTrain()

            for col in pcafeats:
                if (XTrain[col].dtype == 'object') or (XTrain[col].dtype== 'string'):
                    write_log('PCA: Returned due to categorical feature selection',result2exp, 'PCA')
                    return
                if XTrain[col].isnull().sum()  > 0:
                    write_log('PCA: Returned feature '+col+' has missing values',result2exp, 'PCA')
                    return
      
            tr_prev_indices = XTrain.index
            write_log('PCA (split): ',result2exp, 'PCA')

            ss  = StandardScaler()
            XTrain_0 = XTrain.loc[:,pcafeats].values
            XTrain_sc = ss.fit_transform(XTrain_0) 

            Xtest_0 = Xtest.loc[:,pcafeats].values
            Xtest_sc = ss.transform(Xtest_0)
            
            pca = PCA(n_components=1)
            #principalComponents = pca.fit_transform(x)

           

            pca.fit(XTrain_sc)

            Xtrain_pca = pca.transform(XTrain_sc)
            Xtest_pca = pca.transform(Xtest_sc)

            pcacolname = 'Princ_Comp'
        
            pcacols = [col for col in XTrain.columns if col.find(pcacolname) > -1]
            write_log('PCA: explained variance'+str(pca.explained_variance_ratio_),result2exp, 'PCA')
            
            pcacols = [int(col[col.find(pcacolname)+11:]) for col in pcacols]

            pcid = 0
            if len(pcacols)>0:
                pcid = max(pcacols)+1
                
            write_log('PCA: name of column '+pcacolname+"_"+str(pcid)+",type "+str(type(Xtrain_pca)),result2exp, 'PCA')

            Xtrain_pca_df = pd.DataFrame(Xtrain_pca, index=tr_prev_indices) 
            
            

            #write_log('PCA: column names ? '+Xtrain_pca_df.columns,result2exp, 'PCA')

            write_log('PCA (split): cols initial '+str(XTrain.columns),result2exp, 'PCA')

            for col in pcafeats:
                del XTrain[col]


            write_log('PCA (split): size of PCA '+str(len(Xtrain_pca_df)),result2exp, 'PCA')

            write_log('PCA (split): cols before '+str(len(XTrain.columns)),result2exp, 'PCA')

            XTrain = pd.concat([XTrain,Xtrain_pca_df],axis=1)

            write_log('PCA (split): cols after '+str(len(XTrain.columns)),result2exp, 'PCA')

            

            XTrain.rename(columns={XTrain.columns[-1]: pcacolname+"_"+str(pcid)},inplace=True) 

            self.main_model.set_XTrain(XTrain)
          
            write_log('PCA (split): size of final df'+str(len(self.main_model.get_XTrain())),result2exp, 'PCA')

    
        else:
            current_df = self.main_model.get_curr_df()

            for col in pcafeats:
                if (current_df[col].dtype == 'object') or (current_df[col].dtype== 'string'):
                    write_log('PCA: Returned due to categorical feature selection',result2exp, 'PCA')
                    return
                if current_df[col].isnull().sum()  > 0:
                    write_log('PCA: Returned feature '+col+' has missing values',result2exp, 'PCA')
                    return
            
            
            write_log('PCA: '+str(pcafeats),result2exp, 'PCA')
            x = current_df.loc[:,pcafeats].values
            x = StandardScaler().fit_transform(x) # normalizing the features

            pca = PCA(n_components=1)
            principalComponents = pca.fit_transform(x)

            
            pcacolname = 'Princ_Comp'
        
            pcacols = [col for col in current_df.columns if col.find(pcacolname) > -1]
            write_log('PCA: explained variance'+str(pca.explained_variance_ratio_),result2exp, 'PCA')
            
            pcacols = [int(col[col.find(pcacolname)+11:]) for col in pcacols]
    
    
            pcid = 0
            if len(pcacols)>0:
                pcid = max(pcacols)+1
                
            write_log('PCA: name of column '+pcacolname+"_"+str(pcid),result2exp, 'PCA')
            principal_Df = pd.DataFrame(data = principalComponents, columns = [pcacolname+"_"+str(pcid)])
    
            write_log('PCA: size of PCA'+str(len(principal_Df)),result2exp, 'PCA')

            for col in pcafeats:
                del current_df[col]

            current_df = pd.concat([current_df,principal_Df],axis=1)
            self.main_model.set_curr_df(current_df)
            
            write_log('PCA: size of final df'+str(len(self.main_model.get_curr_df())),result2exp, 'PCA')


        return
     
    def remove_outliers(self,dt_features,result2exp): 

        colname = dt_features.value
        write_log('Outlier removal: '+colname,result2exp, 'Outlier removal')
        self.logger.add_action(['DataProcessing', 'outlier'], colname)
      
        
        if self.main_model.datasplit:
            write_log('Outlier removal: (split)-> '+': '+colname, result2exp, 'Data processing')

            Xtest_df = self.main_model.get_XTest()
            Xtrain_df = self.main_model.get_XTrain()
            ytrain_df = self.main_model.getYtrain()
            ytest_df = self.main_model.get_YTest().to_frame()
            
            if colname in Xtrain_df.columns:                                                                     

                quantiles = Xtrain_df[colname].quantile([0.25,0.5,0.75])
                IQR = quantiles[0.75] - quantiles[0.25]
                boxplot_outlierLB =  quantiles[0.25]-1.5*IQR
                boxplot_outlierUB =  quantiles[0.75]+1.5*IQR

                prev_size = len(Xtrain_df)
                outliers = Xtrain_df[(Xtrain_df[colname]>boxplot_outlierUB) | (Xtrain_df[colname]<boxplot_outlierLB)]
                Xtrain_df = Xtrain_df.drop(outliers.index)

                write_log('Outlier removal: (split)-> X_train'+str(prev_size)+'->'+str(len(Xtrain_df))+': '+colname, result2exp, 'Data processing')

                ytrain_df = ytrain_df.drop(outliers.index)

                write_log('Outlier removal: (split)-> Y_train'+str(prev_size)+'->'+str(len(ytrain_df))+': '+colname, result2exp, 'Data processing')
                self.main_model.set_XTest(Xtest_df)
                self.main_model.set_XTrain(Xtrain_df)
                self.main_model.set_YTrain(ytrain_df)
                self.main_model.set_YTest(ytest_df.squeeze())

            else:
                ytrain_df = ytrain_df.to_frame()
                
                if colname in ytrain_df.columns:   
                    write_log('Outlier removal-target: (split): '+colname, result2exp, 'Data processing')
      
                    quantiles = ytrain_df[colname].quantile([0.25,0.5,0.75])
                    IQR = quantiles[0.75] - quantiles[0.25]
                    boxplot_outlierLB =  quantiles[0.25]-1.5*IQR
                    boxplot_outlierUB =  quantiles[0.75]+1.5*IQR
                    prev_size = len(ytrain_df)
                    outliers = ytrain_df[(ytrain_df[colname]>boxplot_outlierUB) | (ytrain_df[colname]<boxplot_outlierLB)]
                    
                    ytrain_df = ytrain_df.drop(outliers.index)
                    Xtrain_df = Xtrain_df.drop(outliers.index)
    
                    write_log('Outlier removal-target: (split)-> X_train'+str(prev_size)+'->'+str(len(Xtrain_df))+': '+colname, result2exp, 'Data processing')
    
    
                    write_log('Outlier removal-target: (split)-> Y_train'+str(prev_size)+'->'+str(len(ytrain_df))+': '+colname, result2exp, 'Data processing')
    
                    self.main_model.set_XTest(Xtest_df)
                    self.main_model.set_XTrain(Xtrain_df)
                    self.main_model.set_YTrain(ytrain_df.squeeze())
                    self.main_model.set_YTest(ytest_df.squeeze())
                    
          
        else:
            
            write_log('Outlier removal: (no split)-> '+': '+colname, result2exp, 'Data processing')
            curr_df = self.main_model.get_curr_df()
            quantiles = curr_df[colname].quantile([0.25,0.5,0.75])
            IQR = quantiles[0.75] - quantiles[0.25]
            boxplot_outlierLB =  quantiles[0.25]-1.5*IQR
            boxplot_outlierUB =  quantiles[0.75]+1.5*IQR

            write_log('Outlier removal: (no split)-> '+': boxplot_outlierLB'+str(round(boxplot_outlierLB,2))+", boxplot_outlierUB"+str(round(boxplot_outlierUB,2)), result2exp, 'Data processing')

            prev_size = len(curr_df)
            
            outliers = curr_df[(curr_df[colname]>boxplot_outlierUB) | (curr_df[colname]<boxplot_outlierLB)]
            curr_df = curr_df.drop(outliers.index)

            write_log('Outlier removal: (no split)-> '+str(prev_size)+'->'+str(len(curr_df))+': '+colname, result2exp, 'Data processing')
            self.main_model.set_curr_df(curr_df)
            

            
        #self.main_model.curr_df[self.main_model.curr_df["outlier"] == False]
        #self.main_model.curr_df.drop(["outlier"], axis=1)
        #logging.info('Data preprocessing, outlier detection and removal')
        #self.logger.add_action(['DataProcessing', 'outlier'], 'All columns')
        
        return
    ##################################################################################

    def assign_target(self,trg_lbl,dt_features,prdtsk_lbl,result2exp,trg_btn,predictiontask):

        self.main_model.targetcolumn = dt_features.value

        trg_lbl.value = "Target: ["+self.main_model.targetcolumn+"]"
        trg_btn.disabled = True

        curr_df = self.main_model.get_curr_df()
        target_column = self.main_model.targetcolumn

        if (curr_df[self.main_model.targetcolumn].dtype == 'float64') or (curr_df[target_column].dtype == 'int64'):
            predictiontask = "Regression"
        else:
            predictiontask = "Classification" 

        prdtsk_lbl.value = "| Prediction Task: "+predictiontask 
        write_log('Target assigned: '+target_column, result2exp, 'Data processing')
        self.logger.add_action(['DataProcessing', 'AssignTarget'], target_column)
        

        return 
    ############################################################################################################    
    def make_featconvert(self,dt_features,result2exp):
        colname = dt_features.value
        self.logger.add_action(['DataProcessing', 'ConvertToBoolean'], colname)
                    
        write_log('Convert Feature Type-> '+colname, result2exp, 'Data processing')

        if self.main_model.datasplit:
            Xtrain_df = self.main_model.get_XTrain()
            ytrain_df = self.main_model.getYtrain().to_frame()
            ytest_df = self.main_model.get_YTest().to_frame()
            
            write_log('Convert Feature Type-> (split)-> '+colname, result2exp, 'Data processing')
                         
            if colname in Xtrain_df.columns:
                write_log('Convert Feature Type (split)-> Returned, only target variable can be converted..', result2exp, 'Data processing')
                return
            if colname in ytrain_df.columns:
                if ytrain_df[colname].dtype in ['float64','int64','int32']:
                    if len(ytrain_df[colname].unique()) == 2:
                        ytrain_df.iloc[:,ytrain_df.columns.get_loc(colname)] = ytrain_df.iloc[:,ytrain_df.columns.get_loc(colname)].astype(bool)
                        ytest_df.iloc[:,ytest_df.columns.get_loc(colname)] = ytest_df.iloc[:,ytest_df.columns.get_loc(colname)].astype(bool)
                        
                        self.main_model.set_YTrain(ytrain_df.squeeze())
                        self.main_model.set_YTest(ytest_df.squeeze())
                    else:
                        write_log('Convert Feature Type (split)-> Returned, target variable has more than two levels..', result2exp, 'Data processing')
                        return
                else:
                    write_log('Convert Feature Type-> Returned, feature is not convenient for num->bool conversion..', result2exp, 'Data processing')
                    return
            else:
                write_log('Convert Feature Type (split)-> Returned, feature is unknown', result2exp, 'Data processing')
                return
                    

        else:
            curr_df = self.main_model.get_curr_df()
            write_log('Convert Feature Type (no split)-> '+': '+str(len(curr_df)), result2exp, 'Data processing')
            if curr_df[colname].dtype in ['float64','int64','int32']:
                if len(curr_df[colname].unique()) == 2:
                    curr_df.iloc[:,curr_df.columns.get_loc(colname)] = curr_df.iloc[:,curr_df.columns.get_loc(colname)].astype(bool)
                    write_log('Convert Feature Type -> done.. '+'  feat type '+str(curr_df[colname].dtype), result2exp, 'Data processing')
                    self.main_model.set_curr_df(curr_df)
                else:
                    write_log('Convert Feature Type-> Returned, feature has more than two levels..', result2exp, 'Data processing')
                    return 
            else:
                write_log('Convert Feature Type-> Returned, feature is not convenient for num->bool conversion..', result2exp, 'Data processing')
                return
                
      
        return

    def make_scaling(self,dt_features,FeatPage,scalingacts,result2exp):
                    
        write_log('Scaling-> '+scalingacts.value, result2exp, 'Data processing')
       
        scalingtype = scalingacts.value
        colname = dt_features.value
      
        if colname is None:
            return

        write_log('Scaling-> '+scalingtype+': '+colname, result2exp, 'Data processing')
    
        
        if scalingtype == 'Standardize':
            
            if self.main_model.datasplit:

                write_log('Scaling (split)-> '+scalingtype+': '+colname, result2exp, 'Data processing')

                Xtest_df = self.main_model.get_XTest()
                Xtrain_df = self.main_model.get_XTrain()
                ytrain_df = self.main_model.getYtrain().to_frame()
                ytest_df = self.main_model.get_YTest().to_frame()

                # use parameters of training data for scaling
                write_log('Scaling (split)-> '+scalingtype+': '+colname, result2exp, 'Data processing')
                
                
                if colname in Xtrain_df.columns:

                    if (Xtrain_df[colname].dtype == 'object') or (Xtrain_df[colname].dtype== 'string'):
                        write_log('Scaling (split)-> Returned due to non-numerical feature: '+colname, result2exp, 'Data processing')
                        return
                    
                    colmean = Xtrain_df[colname].mean(); colstd = Xtrain_df[colname].std()
                    Xtest_df[colname] = (Xtest_df[colname]- colmean)/colstd
                    Xtrain_df[colname] = (Xtrain_df[colname]- colmean)/colstd
                
                if colname in ytrain_df.columns:
                    if (ytrain_df[colname].dtype == 'object') or (ytrain_df[colname].dtype== 'string'):
                        write_log('Scaling (split)-> Returned due to non-numerical target feature: '+colname, result2exp, 'Data processing')
                        return
                        
                    colmean = ytrain_df[colname].mean(); colstd = ytrain_df[colname].std()
                    ytrain_df[colname] = (ytrain_df[colname]- colmean)/colstd
                    ytest_df[colname] = (ytest_df[colname]- colmean)/colstd
            else:
                curr_df = self.main_model.get_curr_df()
                write_log('Scaling (no split)-> '+': '+str(len(curr_df)), result2exp, 'Data processing')
                if (curr_df[colname].dtype == 'object') or (curr_df[colname].dtype== 'string'):
                    write_log('Scaling (split)-> Returned due to non-numerical feature: '+colname, result2exp, 'Data processing')
                    return
                # standardization before splitting data
                colmean = curr_df[colname].mean();colstd = curr_df[colname].std()
                curr_df[colname] = (curr_df[colname]- colmean)/colstd

           
            self.logger.add_action(['DataProcessing', 'Standardize'], [colname])
            logging.info('Data preprocessing, feature scaling: standardization of column '+ colname)


        if scalingtype == 'Normalize':

            if self.main_model.datasplit:
                Xtest_df = self.main_model.get_XTest()
                Xtrain_df = self.main_model.get_XTrain()
                ytrain_df = self.main_model.getYtrain().to_frame()
                ytest_df = self.main_model.get_YTest().to_frame()
                if colname in Xtrain_df.columns:

                    if (Xtrain_df[colname].dtype == 'object') or (Xtrain_df[colname].dtype== 'string'):
            
                        with FeatPage:
                            clear_output()
                            display.display('Selected column is not a numerical type..')
                        return
                    
                    col_min = min(Xtrain_df[colname]); col_max = max(Xtrain_df[colname])
                    denominator = (col_max-col_min)
                    if denominator== 0:
                        Xtest_df[colname] = (Xtest_df[colname]/col_min)
                        Xtrain_df[colname] = (Xtrain_df[colname]/col_min)
                    else:
                        Xtest_df[colname] = (Xtest_df[colname]-col_min)/denominator
                        Xtrain_df[colname] = (Xtrain_df[colname]-col_min)/denominator
                
                if colname in ytrain_df.columns:
                    col_min = min(ytrain_df[colname]); col_max = max(ytrain_df[colname])
                    denominator = (col_max-col_min)
                    if denominator== 0:
                        ytrain_df[colname] = (ytrain_df[colname]/col_min)
                        ytest_df[colname] = (ytest_df[colname]/col_min)
                    else:
                        ytrain_df[colname] = (ytrain_df[colname]-col_min)/denominator
                        ytest_df[colname] = (ytest_df[colname]-col_min)/denominator
            else:    
                curr_df = self.main_model.get_curr_df() 
                if (curr_df[colname].dtype == 'object') or (curr_df[colname].dtype== 'string'):
            
                    with FeatPage:
                        clear_output()
                        display.display('Selected column is not a numerical type..')
                    return
                col_min = min(curr_df[colname]); col_max = max(curr_df[colname])
                denominator = (col_max-col_min)
                
                if denominator== 0:
                    curr_df[colname] = (curr_df[colname]/col_min)
                else:
                    curr_df[colname] = (curr_df[colname]-col_min)/denominator

            self.logger.add_action(['DataProcessing', 'Normalize'], [colname])
            logging.info('Data preprocessing, feature scaling: normalization of column '+ colname)

        with FeatPage:
            clear_output()
            fig, (axbox, axhist) = plt.subplots(1,2)

            if self.main_model.datasplit:
                Xtrain_df=  self.main_model.get_XTrain()
                if colname in Xtrain_df.columns:

                    sns.boxplot(x=colname,data=Xtrain_df, ax=axbox)
                    axbox.set_title('Box plot (train)') 
                    sns.distplot(Xtrain_df[colname],ax=axhist)
                    axhist.set_title('Histogram (train)') 
                    plt.legend(['Mean '+str(round(Xtrain_df[colname].mean(),2)),'Stdev '+str(round(Xtrain_df[colname].std(),2))], bbox_to_anchor=(0.6, 0.6))
                    plt.show()
                else: 
                    ytrain_df = self.main_model.getYtrain().to_frame()
                    if colname in ytrain_df.columns:
                        sns.boxplot(x=colname,data=ytrain_df, ax=axbox)
                        axbox.set_title('Box plot (train)') 
                        sns.distplot(ytrain_df[colname],ax=axhist)
                        axhist.set_title('Histogram (train)') 
                        plt.legend(['Mean '+str(round(ytrain_df[colname].mean(),2)),'Stdev '+str(round(ytrain_df[colname].std(),2))], bbox_to_anchor=(0.6, 0.6))
                        plt.show()
                    
            else:
                sns.boxplot(x=colname,data=curr_df, ax=axbox)
                axbox.set_title('Box plot') 
                sns.distplot(curr_df[colname],ax=axhist)
                axhist.set_title('Histogram') 
                plt.legend(['Mean '+str(round(curr_df[colname].mean(),2)),'Stdev '+str(round(curr_df[colname].std(),2))], bbox_to_anchor=(0.6, 0.6))
                plt.show()

        if self.main_model.datasplit:
            self.main_model.set_XTest(Xtest_df)
            self.main_model.set_XTrain(Xtrain_df)
            self.main_model.set_YTrain(ytrain_df.squeeze())
            self.main_model.set_YTest(ytest_df.squeeze())
        else:
            self.main_model.set_curr_df(curr_df)
        
    
        return
    #################################################################################################################
    def make_balanced(self,features2,balncacts,ProcssPage,result2exp):  


        write_log('Balancing-> '+balncacts.value, result2exp, 'Data processing')
       
        balancetype = balncacts.value
        colname = features2.value

        if not self.main_model.datasplit:
            write_log('Balancing-> No split, improper balancing ', result2exp, 'Data processing')
            return

        if colname != self.main_model.targetcolumn:
            write_log('Balancing-> Non-target feature is attempted for balancing', result2exp, 'Data processing')
            return
      
      
        if colname is None:
            return

        write_log('Balancing-> '+balancetype+': '+colname, result2exp, 'Data processing')


        Xtrain_df = self.main_model.get_XTrain()
        ytrain_df = self.main_model.getYtrain()
        prev_size = len(ytrain_df)
        write_log('Balancing (split)-> '+colname, result2exp, 'Data processing')
    
      
        if len(ytrain_df.to_frame()[colname].unique()) == 2: # binary detection      

            if balancetype == 'Upsample':
                oversampler = RandomOverSampler()
                X_oversampled, y_oversampled = oversampler.fit_resample(Xtrain_df,ytrain_df)
                    
                self.main_model.set_XTrain(X_oversampled)
                self.main_model.set_YTrain(y_oversampled)

                write_log('Balancing: (split)-> '+str(prev_size)+'->'+str(len(y_oversampled))+': '+colname, result2exp, 'Data processing')
                
            if balancetype == 'DownSample':
                downsampler = RandomUnderSampler(random_state=42)
                X_res, y_res = downsampler.fit_resample(Xtrain_df, ytrain_df)

                self.main_model.set_XTrain(X_res)
                self.main_model.set_YTrain(y_res)
                write_log('Balancing: (split)-> '+str(prev_size)+'->'+str(len(y_res))+': '+colname, result2exp, 'Data processing')
    
        else:
            write_log('Balancing (split)-> Feature has more than 2 unique values', result2exp, 'Data processing')
                    
      
       
                    
        logging.info('Data preprocessing, checking and handling unbalancedness')
        self.logger.add_action(['DataProcessing', 'Unbalancedness ' + balancetype ], [colname])
        return

    #####################################################################################################################


    def make_split(self,splt_txt,splt_btn,result2exp):
        curr_df = self.main_model.curr_df
        targetcolumn = self.main_model.targetcolumn
        if targetcolumn is None:
            return
    
        y = curr_df[targetcolumn] # Target variable 
        column_list = [col for col in curr_df.columns]
        column_list.remove(targetcolumn)
        X = curr_df[column_list]
        
        ratio_percnt = int(splt_txt.value) 
        write_log('Split ratio, '+str(ratio_percnt/100), result2exp, 'Data processing')
        X_indices = X.index
        y_indices = y.index

        xtrain,xtest, ytrain, ytest = train_test_split(X, y, test_size=ratio_percnt/100, random_state=16)
        self.main_model.set_XTest(xtest)
        self.main_model.set_XTrain(xtrain)
        self.main_model.set_YTrain(ytrain)
        self.main_model.set_YTest(ytest)
        splt_txt.layout.visibility = 'hidden'
        splt_txt.layout.display = 'none'
        splt_btn.layout.visibility = 'hidden'
        splt_btn.disabled = True
        splt_btn.layout.display = 'none'


        write_log('Split, XTrain size: '+str(len(self.main_model.get_XTrain())), result2exp, 'Data processing')
        write_log('Split, XTest size: '+str(len(self.main_model.get_XTest())), result2exp, 'Data processing')
        write_log('Split, yTrain size: '+str(len(self.main_model.getYtrain())), result2exp, 'Data processing')
        write_log('Split, yTrain indices: '+str(len(self.main_model.getYtrain().index)), result2exp, 'Data processing')
        write_log('Split, yTest size: '+str(len(self.main_model.get_YTest())), result2exp, 'Data processing')
        self.logger.add_action(['DataProcessing', 'Split'], str(ratio_percnt) + '%')
        self.main_model.datasplit = True


    
       
        
        return
    ############################################################################################################    
    def make_encoding(self,features2,encodingacts,result2exp):

        colname = features2.value

        write_log('Encoding.. col '+colname, result2exp, 'Data processing')

        if (colname is None) or (colname == ''):
            return

        encodingtype = encodingacts.value

        write_log('Encoding.. col '+colname+', split '+str(self.main_model.datasplit)+', type '+encodingtype, result2exp, 'Data processing')
       
        # Encode column  
        if self.main_model.datasplit:

            Xtest_df = self.main_model.get_XTest()
            Xtrain_df = self.main_model.get_XTrain()
            ytrain_df = self.main_model.getYtrain().to_frame()
            ytest_df = self.main_model.get_YTest().to_frame()
    
            if encodingtype == "Label Encoding":
                
                if colname in Xtrain_df.columns:

                    
                    if Xtrain_df[colname].dtype in ['float64','int64','int32']:
                        write_log('Encoding-> Attempt to encode a numerical feature '+str(colname), result2exp, 'Data processing')
                        return
                    

                    label_encoder = preprocessing.LabelEncoder() 
                    write_log('Encoding-> '+colname+' (train) current classes: '+str(Xtrain_df[colname].unique()), result2exp, 'Data processing')
                    label_encoder.fit(Xtrain_df[colname]) 
                    Xtrain_df[colname] = label_encoder.transform(Xtrain_df[colname]) # train 
                    #Xtrain_df[colname] = label_encoder.fit_transform(Xtrain_df[colname]) # train 
                    write_log('Encoding-> '+colname+' (train) after labeling classes: '+str(Xtrain_df[colname].unique()), result2exp, 'Data processing')
                    Xtrain_df[colname] = Xtrain_df[colname].apply(np.int64)
                    
                    write_log('Encoding-> '+colname+' (test) current classes: '+str(Xtest_df[colname].unique()), result2exp, 'Data processing')
                    Xtest_df[colname] = label_encoder.fit_transform(Xtest_df[colname]) # test
                    write_log('Encoding-> '+colname+' (test) after labeling classes: '+str(Xtest_df[colname].unique()), result2exp, 'Data processing')
                    Xtest_df[colname] = Xtest_df[colname].apply(np.int64)
    
                    self.logger.add_action(['DataProcessing', 'LabelEncoding'], [colname])
                else: 
                    write_log('Encoding-> '+colname+', target feature..', result2exp, 'Data processing')
                    
            if encodingtype == "One Hot Encoding":

                if colname in Xtrain_df.columns:

                    if Xtrain_df[colname].dtype in ['float64','int64','int32']:
                        write_log('Encoding-> Attempt to encode a numerical feature '+str(colname), result2exp, 'Data processing')
                        return
                    
                    categorical_columns = [colname]

                    
    
                    Xtrain_df = pd.concat([Xtrain_df.drop(categorical_columns, axis = 1), pd.get_dummies(Xtrain_df[categorical_columns])], axis=1)
                    Xtest_df = pd.concat([Xtest_df.drop(categorical_columns, axis = 1), pd.get_dummies(Xtest_df[categorical_columns])], axis=1)
    
                    
                    write_log('One Hot Encoding-> (train) after one-hot features: '+str(Xtrain_df.columns), result2exp, 'Data processing')
                    write_log('One Hot Encoding-> (train) after one-hot size: '+str(len(Xtrain_df)), result2exp, 'Data processing')
    
                    write_log('One Hot Encoding-> (test) after one-hot features: '+str(Xtest_df.columns), result2exp, 'Data processing')
                    write_log('One Hot Encoding-> (test) after one-hot size: '+str(len(Xtest_df)), result2exp, 'Data processing')
    
                    self.logger.add_action(['DataProcessing', 'OneHotEncoding'], [colname])
                else: 
                    write_log('Encoding-> '+colname+', target feature..', result2exp, 'Data processing')
                
            self.main_model.set_XTest(Xtest_df)
            self.main_model.set_XTrain(Xtrain_df)
            self.main_model.set_YTrain(ytrain_df.squeeze())
            self.main_model.set_YTest(ytest_df.squeeze())
        else: #before split

            curr_df = self.main_model.get_curr_df()

            if curr_df[colname].dtype in ['float64','int64','int32']:
                write_log('Encoding-> Attempt to encode a numerical feature '+str(colname), result2exp, 'Data processing')
                return
              
            write_log('Encoding-> '+encodingtype+', '+str(encodingacts.value == "One Hot Encoding"), result2exp, 'Data processing')
            write_log('Encoding-> '+colname+' current classes: '+str(len(curr_df)), result2exp, 'Data processing')
            
            if encodingtype == "Label Encoding":
                label_encoder = preprocessing.LabelEncoder() 
                write_log('Encoding-> '+colname+' current classes: '+str(curr_df[colname].unique()), result2exp, 'Data processing')
                label_encoder.fit(curr_df[colname]) 
                curr_df[colname] = label_encoder.transform(curr_df[colname]) 
                #curr_df[colname] = label_encoder.fit_transform(curr_df[colname]) 
                write_log('Encoding-> '+colname+' after labeling classes: '+str([cls for cls in curr_df[colname].unique()]), result2exp, 'Data processing')
                self.logger.add_action(['DataProcessing', 'LabelEncoding'], [colname])
                curr_df[colname] = curr_df[colname].apply(np.int64)
                
            if encodingacts.value == "One Hot Encoding":
                write_log('Encoding-> '+colname+' current classes: '+str(curr_df[colname].unique()), result2exp, 'Data processing')
                categorical_columns = [colname]
                encoder = preprocessing.OneHotEncoder(sparse_output=False)  # Initialize OneHotEncoder
                one_hot_encoded = encoder.fit_transform(curr_df[categorical_columns])  # Fit and transform the categorical columns          
                one_hot_df = pd.DataFrame(one_hot_encoded,columns=encoder.get_feature_names_out(categorical_columns)) # Create a DataFrame
                curr_df = pd.concat([curr_df.drop(categorical_columns, axis=1), one_hot_df], axis=1)
                write_log('One Hot Encoding-> after one-hot features: '+str(curr_df.columns), result2exp, 'Data processing')
                self.logger.add_action(['DataProcessing', 'OneHotEncoding'], [colname])

      
            self.main_model.set_curr_df(curr_df)
        
        
        return

    def savedata(self, dataFolder, datasetname):
        datasetname = os.path.splitext(os.path.basename(datasetname))[0]
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = dataFolder + '/' + datasetname + '_' + current_datetime
        shutil.copy('output.log', filename + '.txt')
        self.main_model.curr_df.to_csv(filename + '.csv')

        
        version = 0