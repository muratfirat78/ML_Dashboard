from sklearn import preprocessing 
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

class DataProcessingModel:
    def __init__(self, main_model, logger):
        self.main_model = main_model
        self.logger = logger
     
    def remove_outliers(self): 
        self.main_model.curr_df[self.main_model.curr_df["outlier"] == False]
        self.main_model.curr_df.drop(["outlier"], axis=1)
        logging.info('Data preprocessing, outlier detection and removal')
        self.logger.add_action(['DataProcessing', 'outlier'], 'All columns')
        
        return
    ##################################################################################

    def assign_target(self,trg_lbl,dt_features,prdtsk_lbl,result2exp,trg_btn,predictiontask):

        self.main_model.targetcolumn = dt_features.value

        trg_lbl.value = "Target: ["+self.main_model.targetcolumn+"]"
        trg_btn.disabled = True

        curr_df = self.main_model.curr_df
        target_column = self.main_model.targetcolumn

        if (curr_df[self.main_model.targetcolumn].dtype == 'float64') or (curr_df[target_column].dtype == 'int64'):
            predictiontask = "Regression"
        else:
            predictiontask = "Classification" 

        prdtsk_lbl.value = "Prediction type: "+predictiontask 
        write_log('Target assigned: '+target_column, result2exp, 'Data processing')
        self.logger.add_action(['ModelDevelopment', 'AssignTarget'], target_column)
        

        return 
    ##################################################################################

    def NormalizeColumn(self,df,colname):
        logging.info('Data preprocessing, feature scaling: normalization of column '+ colname)
        self.logger.add_action(['DataProcessing', 'Normalization'], [colname])
        col_min = min(df[colname])
        col_max = max(df[colname])
        
        if col_max == col_min: 
            return
        
        df[colname] = (df[colname]- col_min)/(col_max-col_min)
        
        return df

    ############################################################################################################    

    def make_scaling(self,dt_features,FeatPage,scalingacts,result2exp):
                    
        write_log('Scaling-> '+scalingacts.value, result2exp, 'Data processing')
        curr_df = self.main_model.get_curr_df()
        Xtest_df = self.main_model.get_XTest()
        Xtrain_df = self.main_model.get_XTrain()
        ytrain_df = self.main_model.getYtrain().to_frame()
        ytest_df = self.main_model.get_YTest().to_frame()
      

        colname = dt_features.value
      
        if colname is None:
            return

        write_log('Scaling-> '+scalingacts.value+': '+colname, result2exp, 'Data processing')
        if (curr_df[colname].dtype == 'object') or (curr_df[colname].dtype== 'string'):
            
            with FeatPage:
                clear_output()
                display.display('Selected column is not a numerical type..')
            return

        
        if scalingacts.value == 'Standardize':
            if self.main_model.datasplit:
                # use parameters of training data for scaling
                write_log('Scaling (split)-> '+scalingacts.value+': '+colname, result2exp, 'Data processing')
                
                
                if colname in Xtrain_df.columns:
                    colmean = Xtrain_df[colname].mean(); colstd = Xtrain_df[colname].std()
                    Xtest_df[colname] = (Xtest_df[colname]- colmean)/colstd
                    Xtrain_df[colname] = (Xtrain_df[colname]- colmean)/colstd
                
                if colname in ytrain_df.columns:
                    colmean = ytrain_df[colname].mean(); colstd = ytrain_df[colname].std()
                    ytrain_df[colname] = (ytrain_df[colname]- colmean)/colstd
                    ytest_df[colname] = (ytest_df[colname]- colmean)/colstd
            else:
                # standardization before splitting data
                colmean = curr_df[colname].mean();colstd = curr_df[colname].std()
                curr_df[colname] = (curr_df[colname]- colmean)/colstd

            write_log('Scaling (split) done-> '+colname+str(len(ytrain_df.columns)), result2exp, 'Data processing')
            self.logger.add_action(['DataProcessing', 'Standardize'], [colname])
            logging.info('Data preprocessing, feature scaling: standardization of column '+ colname)


        if scalingacts.value == 'Normalize':

            if self.main_model.datasplit:
                if colname in Xtrain_df.columns:
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
                if colname in Xtrain_df.columns:

                    sns.boxplot(x=colname,data=Xtrain_df, ax=axbox)
                    axbox.set_title('Box plot (train)') 
                    sns.distplot(Xtrain_df[colname],ax=axhist)
                    axhist.set_title('Histogram (train)') 
                    plt.legend(['Mean '+str(round(Xtrain_df[colname].mean(),2)),'Stdev '+str(round(Xtrain_df[colname].std(),2))], bbox_to_anchor=(0.6, 0.6))
                    plt.show()
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

        self.main_model.set_curr_df(curr_df)
        self.main_model.set_XTest(Xtest_df)
        self.main_model.set_XTrain(Xtrain_df)
        self.main_model.set_YTrain(ytrain_df.squeeze())
        self.main_model.set_YTest(ytest_df.squeeze())
        
    
        return
    #################################################################################################################
    def make_balanced(self,features2,balncacts,ProcssPage):  
        curr_df = self.main_model.curr_df
        colname = features2.value

        if balncacts.value == 'Upsample':
            
            if len(curr_df[colname].unique()) == 2: # binary detection
                colvals = curr_df[colname].unique()
                ColmFirst = curr_df[curr_df[colname] == colvals[0]]
                ColmOther = curr_df[curr_df[colname] == colvals[1]]
            
                if len(ColmFirst) < len(ColmOther):
                    upsampled_First = resample(ColmFirst, replace=True, n_samples=len(ColmOther), random_state=27) 
                    self.main_model.curr_df = pd.concat([ColmOther, upsampled_First])
                else:
                    upsampled_Other= resample(ColmOther, replace=True, n_samples=len(ColmFirst), random_state=27) 
                    self.main_model.curr_df = pd.concat([ColmFirst, upsampled_Other])
                    
                with ProcssPage:
                    clear_output()
                    plt.figure(figsize=(6, 2))
                    ax = sns.countplot(x=colname,data=self.main_model.curr_df, palette="cool_r")
                    for p in ax.patches:
                        ax.annotate("{:.1f}".format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
                    plt.show()
        logging.info('Data preprocessing, checking and handling unbalancedness')
        self.logger.add_action(['DataProcessing', 'Unbalancedness'], [colname])
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
        self.main_model.Xtrain_df,self.main_model.Xtest_df, self.main_model.ytrain_df, self.main_model.ytest_df = train_test_split(X, y, test_size=ratio_percnt/100, random_state=16)
        splt_txt.layout.visibility = 'hidden'
        splt_txt.layout.display = 'none'
        splt_btn.layout.visibility = 'hidden'
        splt_btn.disabled = True
        splt_btn.layout.display = 'none'

        write_log('Split, XTrain size: '+str(len(self.main_model.Xtrain_df)), result2exp, 'Data processing')
        write_log('Split, XTest size: '+str(len(self.main_model.Xtest_df)), result2exp, 'Data processing')
        write_log('Split, yTrain size: '+str(len(self.main_model.ytrain_df)), result2exp, 'Data processing')
        write_log('Split, yTest size: '+str(len(self.main_model.ytest_df)), result2exp, 'Data processing')
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
                    write_log('Encoding-> '+colname+', target features..', result2exp, 'Data processing')
                    
            if encodingtype == "One Hot Encoding":
                categorical_columns = [colname]
                
                encoder = preprocessing.OneHotEncoder(sparse_output=False)  # Initialize OneHotEncoder
                
                one_hot_encoded = encoder.fit_transform(Xtrain_df[categorical_columns])  # Fit and transform the categorical columns          
                one_hot_df = pd.DataFrame(one_hot_encoded,columns=encoder.get_feature_names_out(categorical_columns)) # Create a DataFrame
                Xtrain_df = pd.concat([Xtrain_df.drop(categorical_columns, axis=1), one_hot_df], axis=1)
                write_log('One Hot Encoding-> (train) after one-hot features: '+str(Xtrain_df.columns), result2exp, 'Data processing')

                encoder = preprocessing.OneHotEncoder(sparse_output=False)  # Initialize OneHotEncoder
                one_hot_encoded = encoder.fit_transform(Xtest_df[categorical_columns])  # Fit and transform the categorical columns          
                one_hot_df = pd.DataFrame(one_hot_encoded,columns=encoder.get_feature_names_out(categorical_columns)) # Create a DataFrame 
                Xtest_df = pd.concat([Xtest_df.drop(categorical_columns, axis=1), one_hot_df], axis=1)
                write_log('One Hot Encoding-> (test) after one-hot features: '+str(Xtest_df.columns), result2exp, 'Data processing')

               

                features2.options = [col+'('+str(Xtrain_df[col].isnull().sum())+')' for col in Xtrain_df.columns]
                self.logger.add_action(['DataProcessing', 'OneHotEncoding'], [colname])
        else: #before split

            curr_df = self.main_model.get_curr_df()
              
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

            features2.options = [col for col in curr_df.columns]

        self.main_model.set_curr_df(curr_df)
        self.main_model.set_XTest(Xtest_df)
        self.main_model.set_XTrain(Xtrain_df)
        self.main_model.set_YTrain(ytrain_df.squeeze())
        self.main_model.set_YTest(ytest_df.squeeze())
        
        return

    def savedata(self, dataFolder, datasetname):
        datasetname = os.path.splitext(os.path.basename(datasetname))[0]
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = dataFolder + '/' + datasetname + '_' + current_datetime
        shutil.copy('output.log', filename + '.txt')
        self.main_model.curr_df.to_csv(filename + '.csv')

        
        version = 0