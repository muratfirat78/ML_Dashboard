from log import *
from sklearn import tree,neighbors,linear_model,ensemble,svm
from sklearn.metrics import accuracy_score,mean_squared_error

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
            
        write_log('Model.. Type '+str(type(self.PythonObject)), report, 'Predictive modeling')
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

class PredictiveModelingModel: 

    def __init__(self, main_model):
        self.main_model = main_model
        self.trainedModels = []

    def train_Model(self,tasktype,mytype,results,trmodels):
        data = [self.main_model.Xtrain_df,self.main_model.ytrain_df,self.main_model.Xtest_df,self.main_model.ytest_df]

        mymodel = MLModel(data,self.main_model.targetcolumn,tasktype,mytype,results)
    
        model = mymodel.getSkLearnModel().fit(data[0], data[1]) 

        y_pred = mymodel.GetPredictions()

        if tasktype == 'Classification': 
            mymodel.GetPerformanceDict()['Accuracy'] = accuracy_score(data[3], y_pred)
        
        if tasktype == 'Regression': 
            mymodel.GetPerformanceDict()['MSE'] = mean_squared_error(data[3], y_pred)

        self.trainedModels.append(mymodel)

        write_log('Train Model-> '+ mytype, results, 'Predictive modeling')
        for prf,val in mymodel.GetPerformanceDict().items():
            write_log('Model Performance-> '+prf+': '+str(val), results, 'Predictive modeling')

        trmodels.options = [mdl.getType() for mdl in self.trainedModels]
    

        return 
    
    def get_trained_models(self):
        return self.trainedModels