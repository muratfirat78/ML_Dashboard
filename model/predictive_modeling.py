from log import *
from sklearn import tree,neighbors,linear_model,ensemble,svm
from sklearn.metrics import accuracy_score,mean_squared_error,confusion_matrix,roc_curve,precision_score,recall_score,mean_absolute_error,r2_score
import statsmodels.api as sm

class MLModel:
    def __init__(self,target,tasktype,mytype,report,myname,params,xtrain,ytrain):
  
        self.modelsetting = dict()
        self.performance = dict()
        self.roc = dict()
        self.Type = mytype
        self.myTask = tasktype[tasktype.find(":")+2:]
        self.PythonObject = None
        self.PreprocessingSteps = [] 
        self.Name = myname
        self.ConfMatrix = None
        self.modelinfo = dict()
        self.ROCFPR = None
        self.ROCTPR = None
        self.Predictions = None
        
        
        if self.Type == 'Decision Tree':
            if self.myTask == 'Classification': 
                write_log('DT: Classification', report, 'Predictive modeling')
                self.PythonObject = tree.DecisionTreeClassifier(criterion=params[2],max_depth=int(params[0]))
                self.PythonObject.fit(xtrain,ytrain) 
            if self.myTask == 'Regression': 
                write_log('DT: Regression', report, 'Predictive modeling')
                self.PythonObject = tree.DecisionTreeRegressor(random_state = 0)
                self.PythonObject.fit(xtrain,ytrain) 
        if self.Type == 'KNN':
            if self.myTask == 'Classification': 
                self.PythonObject = neighbors.KNeighborsClassifier(n_neighbors=int(params[0]))
                self.PythonObject.fit(xtrain,ytrain) 
            if self.myTask == 'Regression': 
                self.PythonObject = neighbors.KNeighborsRegressor(n_neighbors=int(params[0]))
                self.PythonObject.fit(xtrain,ytrain) 
        if self.Type == 'Linear Model':
            if self.myTask == 'Classification': 
                self.PythonObject = linear_model.SGDClassifier()       
                self.PythonObject.fit(xtrain,ytrain) 
            if self.myTask == 'Regression': 
                self.PythonObject = linear_model.LinearRegression()
                reg = self.PythonObject.fit(xtrain,ytrain) 
               

        if self.Type == 'Random Forest':
            if self.myTask == 'Classification': 
                self.PythonObject = ensemble.RandomForestClassifier(n_estimators=int(params[0]))      
                self.PythonObject.fit(xtrain,ytrain) 
            if self.myTask == 'Regression': 
                self.PythonObject = ensemble.RandomForestRegressor(n_estimators=int(params[0]), random_state=0,)   # try before standardization..
                self.PythonObject.fit(xtrain,ytrain) 
                
        if self.Type == 'SVM':
            if self.myTask == 'Classification': 
                self.PythonObject = svm.SVC(C=float(params[0]),kernel=params[1], gamma='auto',probability = True)
                self.PythonObject.fit(xtrain,ytrain) 
            if self.myTask == 'Regression': 
                self.PythonObject = svm.SVR(C=float(params[0]),kernel=params[1])
                self.PythonObject.fit(xtrain,ytrain) 
        if self.Type == 'Logistic Regression':
            self.PythonObject = linear_model.LogisticRegression(random_state=16)   # Initialize the model object 
            self.PythonObject.fit(xtrain,ytrain) 
            
       
        return

    def getPredictions(self):
        return self.Predictions

    def setPredictions(self,iytm):
        self.Predictions = iytm   
        return

    def getROCTPR(self):
        return self.ROCTPR

    def setROCTPR(self,iytm):
        self.ROCTPR = iytm   
        return
     
    def setConfMatrix(self,myitm):
        self.ConfMatrix = myitm
        return
      
    def getConfMatrix(self):
        return self.ConfMatrix

    def setROC(self,myroc):
        self.roc = myroc
        return

    def getROC(self):
        return self.roc 

    
    def GetPredictions(self,xtest):
        if self.myTask == 'Classification':
            return self.PythonObject.predict(xtest) 
        if self.myTask == 'Regression':
            if self.Type == 'Logistic Regression': 
                return self.PythonObject.predict_proba(xtest)
            else:
                return self.PythonObject.predict(xtest)
      
    def getSkLearnModel(self):
        return self.PythonObject

    def getName(self):
        return self.Name
        
    def getTask(self):
        return self.myTask

    def getType(self):
        return self.Type

    def getModelInfo(self):
        return self.modelinfo
    

    def GetPerformanceDict(self):
        return self.performance

class PredictiveModelingModel: 

    def __init__(self, main_model, controller, logger):
        self.main_model = main_model
        self.trainedModels = []
        self.controller = controller
        self.logger = logger

    def train_Model(self,tasktype,mytype,results,trmodels,params):
        
        #data = [self.main_model.Xtrain_df,self.main_model.ytrain_df,self.main_model.Xtest_df,self.main_model.ytest_df]
        write_log('Train Model-> '+ mytype,results,'Predictive modeling')
        

        Xtest_df = self.main_model.get_XTest()
        Xtrain_df = self.main_model.get_XTrain()
        ytrain_df = self.main_model.getYtrain()
        ytest_df = self.main_model.get_YTest()

        models = [1 for mdl in self.trainedModels if mdl.getName().find(mytype) > -1]

        write_log('Train Model-> task'+str(tasktype[tasktype.find(":")+2:]),results,'Predictive modeling')

        success = False
        try: 
            if tasktype[tasktype.find(":")+2:] == 'Regression':
                if mytype == 'Logistic Regression':
                   if (ytrain_df.to_frame()[self.main_model.targetcolumn].max() > 1) or (ytrain_df.to_frame()[self.main_model.targetcolumn].min() < 0):
                       write_log('Train Model-> No proper feature range for logit',results,'Predictive modeling')
                       return
                    
            if tasktype[tasktype.find(":")+2:] == 'Classification':
                if mytype == 'Logistic Regression':
                    if (len(ytrain_df.to_frame()[self.main_model.targetcolumn].unique()) > 2):
                        write_log('Train Model-> No binary class case for logit',results,'Predictive modeling')
                        return
                       
            mymodel = MLModel(self.main_model.targetcolumn,tasktype,mytype,results,mytype+"_"+str(len(models)),params,Xtrain_df,ytrain_df)


            y_pred = mymodel.GetPredictions(Xtest_df)
           
            mymodel.setPredictions(y_pred)

    
            write_log('>>Train Model-> predcts'+str(len(y_pred))+"??"+str(len(ytest_df))+", "+tasktype,results,'Predictive modeling')
    
            if tasktype[tasktype.find(":")+2:] == 'Classification': 
                mymodel.setConfMatrix(confusion_matrix(ytest_df,y_pred))      
                
                if len(ytrain_df.to_frame()[self.main_model.targetcolumn].unique()) == 2:
                    mymodel.GetPerformanceDict()['True-Positive'] = mymodel.getConfMatrix()[1][1]
                    mymodel.GetPerformanceDict()['False-Positive'] = mymodel.getConfMatrix()[0][1]
                    mymodel.GetPerformanceDict()['True-Negative'] = mymodel.getConfMatrix()[0][0]
                    mymodel.GetPerformanceDict()['False-Negative'] = mymodel.getConfMatrix()[1][0]
                    mymodel.GetPerformanceDict()['Accuracy'] = accuracy_score(ytest_df, y_pred)
                    mymodel.GetPerformanceDict()['Precision'] = precision_score(ytest_df, y_pred)
                    mymodel.GetPerformanceDict()['Recall'] = recall_score(ytest_df, y_pred)
                  
                    fpr,tpr,thrs = roc_curve(ytest_df, y_pred)
                    mymodel.GetPerformanceDict()['ROCFPR'] = fpr
                    mymodel.GetPerformanceDict()['ROCTPR'] = tpr
                   
                else:
                    mymodel.GetPerformanceDict()['Accuracy'] = accuracy_score(ytest_df, y_pred)
                    mymodel.GetPerformanceDict()['Precision'] = precision_score(ytest_df, y_pred,average ='weighted')
                    mymodel.GetPerformanceDict()['Recall'] = recall_score(ytest_df, y_pred,average ='weighted')
   
                
            
            if tasktype[tasktype.find(":")+2:] == 'Regression': 
                mymodel.GetPerformanceDict()['MSE'] = mean_squared_error(ytest_df, y_pred)
                mymodel.GetPerformanceDict()['MAE'] = mean_absolute_error(ytest_df, y_pred)
                mymodel.GetPerformanceDict()['RSquared'] = r2_score(ytest_df, y_pred)
              

            self.trainedModels.append(mymodel)

            performance = []
            for prf,val in mymodel.GetPerformanceDict().items():
                performance += [(prf, val)]
            
            print(ytrain_df.size)
            print(ytrain_df.dtype)
            print(ytrain_df.isnull().sum())

            
            performance += [("data_size", ytrain_df.size)]
            performance += [("missing_values", ytrain_df.isnull().sum())]
            performance += [("type", ytrain_df.dtype)]
            performance += [("range", str(ytrain_df.min()) + "-" + str(ytrain_df.max()))]
                
            self.logger.add_action(['ModelDevelopment', 'ModelPerformance'], (mytype + str(params).replace('[', '(').replace(']', ')'), performance))
            
            trmodels.options = [mdl.getName() for mdl in self.trainedModels]
            write_log('Train Model-> '+mytype+' is trained.',results,'Predictive modeling')
            
        except Exception as e: 
            write_log('Train Model-> exception raised \"'+str(e)+'\"',results,'Predictive modeling')
            write_log('Train Model-> unsuccessful trial',results,'Predictive modeling')
        
        self.controller.upload_log()
        self.controller.update_learning_path()
    
    def get_trained_models(self):
        return self.trainedModels