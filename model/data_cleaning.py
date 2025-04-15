from log import *

class DataCleaningModel:
    def __init__(self, main_model, logger):
        self.main_model = main_model
        self.logger = logger

    def make_cleaning(self,featurescl,result2aexp,missacts,dt_features,params): 

        colname = featurescl.value
        handling = missacts.value

        if self.main_model.datasplit:
            Xtest_df = self.main_model.get_XTest()
            Xtrain_df = self.main_model.get_XTrain()
            ytrain_df = self.main_model.getYtrain().to_frame()
            ytest_df = self.main_model.get_YTest().to_frame()
            removals_tr = None
            removals_ts = None

            if colname in Xtrain_df.columns:
                write_log('col (split) '+colname+', action '+handling+', coltype '+str(Xtrain_df[colname].dtype), result2aexp, 'Data cleaning')
                prev_size = 0
                final_size = 0
                if handling == 'Edit Range':
                    prev_size = len(Xtrain_df)
                    minval = params[0].value; maxval = params[1].value
                    if (Xtrain_df[colname].dtype == 'int64'):
                        minval =int(minval); maxval =int(maxval); 
                    if (Xtrain_df[colname].dtype == 'float64'):
                        minval =float(minval); maxval =float(maxval); 
           
                    removals_tr = Xtrain_df[(Xtrain_df[colname]<minval) | (Xtrain_df[colname]>maxval)]
                    removals_ts = Xtest_df[(Xtest_df[colname]<minval) | (Xtest_df[colname]>maxval)]
                    
        
                    write_log('Edit Range (split) is selected.. ',  result2aexp, 'Data cleaning')
                    
                elif handling == 'Drop Column':
                    del Xtrain_df[colname]
                    del Xtest_df[colname]
                    prev_size = len(Xtrain_df)

                else:  
                    if (Xtrain_df[colname].dtype == 'float64') or (Xtrain_df[colname].dtype == 'int64'):
                        if handling in ['Replace-Mean','Replace-Median','Remove-Missing']:
                            if handling == 'Replace-Mean': 
                                Xtrain_df[colname].fillna(Xtrain_df[colname].mean(), inplace=True)
                                Xtest_df[colname].fillna(Xtrain_df[colname].mean(), inplace=True)
                            if handling == 'Replace-Median': 
                                Xtrain_df[colname].fillna(Xtrain_df[colname].median(), inplace=True)
                                Xtest_df[colname].fillna(Xtrain_df[colname].median(), inplace=True)
            
                            if handling == 'Remove-Missing': 
                                Xtrain_df = Xtrain_df.dropna(subset = [colname])
                                Xtest_df = Xtest_df.dropna(subset = [colname])
                        else:
                            write_log('Improper action is selected.. ',  result2aexp, 'Data cleaning')
                            return
                    
               
            if  colname == self.main_model.targetcolumn:
                write_log('col (split) '+colname+', action '+handling+', coltype '+str(ytrain_df[colname].dtype), result2aexp, 'Data cleaning')
                
                if handling == 'Edit Range':
                    prev_size = len(ytrain_df)
                    minval = params[0].value; maxval = params[1].value
                    if (ytrain_df[colname].dtype == 'int64'):
                        minval =int(minval); maxval =int(maxval); 
                    if (ytrain_df[colname].dtype == 'float64'):
                        minval =float(minval); maxval =float(maxval); 
                    
                    removals_tr = ytrain_df[(ytrain_df[colname]<minval) | (ytrain_df[colname]>maxval)]
                    write_log('Edit Range (split) is selected.. ',  result2aexp, 'Data cleaning')
                    removals_ts = ytest_df[(ytest_df[colname]<minval) | (ytest_df[colname]>maxval)]
                    
                    
                elif handling == 'Drop Column':
                    write_log('Drop target (split) ?? ',  result2aexp, 'Data cleaning')

                else:  
                    if (Xtrain_df[colname].dtype == 'float64') or (Xtrain_df[colname].dtype == 'int64') or (Xtrain_df[colname].dtype == 'int32'):
                        if handling in ['Replace-Mean','Replace-Median','Remove-Missing']:
                            if handling == 'Replace-Mean': 
                                Xtrain_df[colname].fillna(Xtrain_df[colname].mean(), inplace=True)
                                Xtest_df[colname].fillna(Xtrain_df[colname].mean(), inplace=True)
                            if handling == 'Replace-Median': 
                                Xtrain_df[colname].fillna(Xtrain_df[colname].median(), inplace=True)
                                Xtest_df[colname].fillna(Xtrain_df[colname].median(), inplace=True)
            
                            if handling == 'Remove-Missing': 
                                Xtrain_df = Xtrain_df.dropna(subset = [colname])
                                Xtest_df = Xtest_df.dropna(subset = [colname])
                        else:
                            write_log('Improper action is selected.. ',  result2aexp, 'Data cleaning')
                            return
                    else:
                        
                        if handling == 'Replace-Mode': 
                            Xtrain_df[colname].fillna(Xtrain_df[colname].mode()[0], inplace=True)
                            Xtest_df[colname].fillna(Xtrain_df[colname].mode()[0], inplace=True)
                            write_log('mode (split) . '+str(Xtrain_df[colname].mode()[0]), result2aexp, 'Data cleaning')
                        if handling == 'Remove-Missing': 
                            Xtrain_df = Xtrain_df.dropna(subset = [colname])
                            Xtest_df = Xtest_df.dropna(subset = [colname])
                         
            
            if handling == 'Edit Range':      
                Xtrain_df = Xtrain_df.drop(removals_tr.index)
                ytrain_df = ytrain_df.drop(removals_tr.index)
                Xtest_df = Xtest_df.drop(removals_ts.index)
                ytest_df = ytest_df.drop(removals_ts.index)
                
           
            final_size = len(Xtrain_df)
            self.main_model.set_XTrain(Xtrain_df)
            self.main_model.set_YTrain(ytrain_df.squeeze())
            self.main_model.set_XTest(Xtest_df)
            self.main_model.set_YTest(ytest_df.squeeze())
            write_log('Data size (split) '+str(prev_size)+"->"+str(final_size),  result2aexp, 'Data cleaning')

        
        else:
            curr_df = self.main_model.get_curr_df()
        
           
            write_log('col '+colname+', action '+handling+', coltype '+str(curr_df[colname].dtype), result2aexp, 'Data cleaning')
            write_log('Initial data size'+str(len(curr_df)),  result2aexp, 'Data cleaning')  
    

            if handling == 'Edit Range':
                self.logger.add_action(['DataCleaning', handling + '(' + params[0].value + '-' + params[1].value +')'], [colname])
            else:
                self.logger.add_action(['DataCleaning', handling], [colname])
    
            if handling == 'Edit Range':
    
                prev_size = len(curr_df)
                
                minval = params[0].value; maxval = params[1].value
                
                if (curr_df[colname].dtype == 'int64'):
                    minval =int(minval); maxval =int(maxval); 
    
                if (curr_df[colname].dtype == 'float64'):
                    minval =float(minval); maxval =float(maxval); 
                
                curr_df = curr_df[(curr_df[colname] >= minval) & (curr_df[colname] <= maxval)]
    
                write_log('Edit Range is selected.. ',  result2aexp, 'Data cleaning')
         
                
            elif handling == 'Drop Column':
                del curr_df[colname]
                
            else:    
                if (curr_df[colname].dtype == 'float64') or (curr_df[colname].dtype == 'int64') or (curr_df[colname].dtype == 'int32'):
                    if handling in ['Replace-Mean','Replace-Median','Remove-Missing']:
                        if handling == 'Replace-Mean': 
                            curr_df[colname].fillna(curr_df[colname].mean(), inplace=True)
                        if handling == 'Replace-Median': 
                            curr_df[colname].fillna(curr_df[colname].median(), inplace=True)
        
                        if handling == 'Remove-Missing': 
                            curr_df = curr_df.dropna(subset = [colname])
                    else:
                        write_log('Improper action is selected.. ',  result2aexp, 'Data cleaning')
                        return
                else: 
                    write_log('mode.. '+str(curr_df[colname].mode()[0]), result2aexp, 'Data cleaning')
                    if handling == 'Replace-Mode': 
                        curr_df[colname].fillna(curr_df[colname].mode()[0], inplace=True)
                    if handling == 'Remove-Missing': 
                        curr_df = curr_df.dropna(subset = [colname])
                    
                        
                        
            self.main_model.set_curr_df(curr_df)
            write_log('Cleaning action done..'+str(self.main_model.get_curr_df().columns),  result2aexp, 'Data cleaning') 
            write_log('Final data size'+str(len(self.main_model.get_curr_df())),  result2aexp, 'Data cleaning')  

        return