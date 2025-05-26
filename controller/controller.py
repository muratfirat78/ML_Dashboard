from model.convert_performance_to_task import ConvertPerformanceToTask
from model.learning_manager import LearningManagerModel
from model.task import TaskModel
from model.data_cleaning import DataCleaningModel
from model.data_processing import DataProcessingModel
from model.data_selection import DataSelectionModel
from model.learning_path import LearningPathModel
from model.local_drive import GoogleDrive
from model.logger import Logger
from model.login import LoginModel
from model.predictive_modeling import PredictiveModelingModel
from model.task_selection import TaskSelectionModel
from view.data_cleaning import DataCleaningView
from view.data_processing import DataProcessingView
from view.data_selection import DataSelectionView
from model.main_model import MainModel
from view.learning_path import LearningPathView
from view.main_view import MainView
from view.predictive_modeling import PredictiveModelingView
from view.login import LoginView
from view.task import TaskView
from view.task_selection import TaskSelectionView

class Controller:
    def __init__(self, drive, online_version):
        self.monitored_mode = None
        self.main_model = MainModel(online_version)
        self.main_view = MainView()
        self.logger = Logger(self)
        self.login_view = LoginView(self)
        self.login_model = LoginModel(self)
        self.data_selection_view = DataSelectionView(self, self.main_view)
        self.data_selection_model = DataSelectionModel(self.main_model, self.logger)
        self.data_cleaning_view = DataCleaningView(self, self.main_view)
        self.data_cleaning_model = DataCleaningModel(self.main_model, self.logger)
        self.data_processing_view = DataProcessingView(self, self.main_view)
        self.data_processing_model = DataProcessingModel(self.main_model, self.logger)
        self.predictive_modeling_view = PredictiveModelingView(self, self.main_view)
        self.predictive_modeling_model = PredictiveModelingModel(self.main_model, self, self.logger)
        self.task_model = TaskModel(self)
        self.task_view = TaskView(self)
        self.learning_path_model = LearningPathModel(self)
        self.learning_path_view = LearningPathView(self)
        self.learning_manager_model = LearningManagerModel(self)
        self.task_selection_model = TaskSelectionModel(self)
        self.task_selection_view = TaskSelectionView(self)
        self.learning_path = None
        self.convertPerformanceToTask = ConvertPerformanceToTask()
        # performance = {'General': {}, 'SelectData': {'DataSet': ('titanic.csv', 0)}, 'DataCleaning': {'Drop Column': [(['Fare'], 1), (['Ticket'], 2), (['Parch'], 3), (['SibSp'], 4), (['Name'], 5), (['Pclass'], 6), (['PassengerId'], 7), (['Cabin'], 9)], 'Remove-Missing': (['Embarked'], 8), 'Replace-Median': (['Age'], 10)}, 'DataProcessing': {'LabelEncoding': [(['Embarked'], 11), (['Sex'], 12)], 'ConvertToBoolean': ('Survived', 13), 'AssignTarget': ('Survived', 14), 'Split': ('20%', 15)}, 'ModelDevelopment': {'ModelPerformance': (('Logistic Regression()', [('True-Positive', 56), ('False-Positive', 16), ('True-Negative', 86), ('False-Negative', 20), ('Accuracy', 0.797752808988764), ('Precision', 0.7777777777777778), ('Recall', 0.7368421052631579), ('ROCFPR', ([0.        , 0.15686275, 1.        ])), ('ROCTPR',([0.        , 0.73684211, 1.        ]))]), 16)}}
        # self.convertPerformanceToTask.convert_performance_to_task(performance)
        if drive != None:
            self.drive = drive
        else:
            self.drive = GoogleDrive()

    def get_tab_set(self):
        tab_1 = self.data_selection_view.get_data_selection_tab()
        tab_2 = self.data_cleaning_view.get_data_cleaning_tab()
        tab_3 = self.data_processing_view.get_data_processing_tab()
        tab_4 = self.predictive_modeling_view.get_predictive_modeling_tab()
        tab_5 = self.learning_path_view.get_learning_path_tab()

        self.main_view.set_tabs(tab_1, tab_2, tab_3, tab_4, tab_5)
        tab_set = self.main_view.get_tabs()
        
        return tab_set

    def train_Model(self,tasktype,mytype,results,trmodels,params):
        self.predictive_modeling_model.train_Model(tasktype,mytype,results,trmodels,params)
        # print(self.logger.get_result())
        # print(self.convertPerformanceToTask.convert_performance_to_task(self.logger.get_result(), 'todo', 'todo'))
    
    def make_cleaning(self,featurescl,result2aexp,missacts,dt_features,params):
         self.data_cleaning_model.make_cleaning(featurescl,result2aexp,missacts,dt_features,params)

    def assign_target(self,trg_lbl,dt_features,prdtsk_lbl,result2exp,trg_btn,predictiontask):
        self.data_processing_model.assign_target(trg_lbl,dt_features,prdtsk_lbl,result2exp,trg_btn,predictiontask)  

    def make_balanced(self,features2,balncacts,ProcssPage,result2exp):
        self.data_processing_model.make_balanced(features2,balncacts,ProcssPage,result2exp)

    def make_encoding(self,features2,encodingacts,result2exp):
        self.data_processing_model.make_encoding(features2,encodingacts,result2exp)

    def make_featconvert(self,dt_features,result2exp):
        self.data_processing_model.make_featconvert(dt_features,result2exp)

    def make_scaling(self,dt_features,FeatPage,scalingacts,result2exp):
        self.data_processing_model.make_scaling(dt_features,FeatPage,scalingacts,result2exp)

    def showCorrHeatMap(self,ProcssPage,fxctingacts,result2exp):
        self.data_processing_model.showCorrHeatMap(ProcssPage,fxctingacts,result2exp)

    def ApplyPCA(self,features2,pca_features,result2exp):
        self.data_processing_model.ApplyPCA(features2,pca_features,result2exp)
        
    def make_split(self,splt_txt,splt_btn,result2exp):
        # self.main_view.close_tab(0)
        # self.main_view.close_tab(1)
        self.data_processing_model.make_split(splt_txt,splt_btn,result2exp)

    def remove_outliers(self,dt_features,result2exp):
        self.data_processing_model.remove_outliers(dt_features,result2exp)
        self.refresh_data_processing()

    def savedata(self,dataFolder, datasetname):
        self.data_processing_model.savedata(dataFolder, datasetname)

    def file_Click(self,foldername,filename,wsheets):
        self.data_selection_model.file_Click(self.get_online_version(),foldername,filename,wsheets)

    def on_submitfunc(self,foldername,datasets):
        self.data_selection_model.on_submitfunc(self.get_online_version(),foldername,datasets)

    def read_data_set(self,foldername,filename,sheetname):
        self.data_selection_model.read_data_set(self.get_online_version(),foldername,filename,sheetname)

    def get_curr_df(self):
        return self.main_model.curr_df

    def set_curr_df(self,mydf):
        self.main_model.curr_df = mydf
        return

    def get_XTrain(self):
        return self.main_model.Xtrain_df

    def set_XTrain(self,mfdf):
        self.main_model.Xtrain_df = mydf
        return

    def get_curr_info(self):
        return self.main_model.currinfo
    
    def get_trained_models(self):
        return self.predictive_modeling_model.get_trained_models()
    
    def get_datafolder(self):
        return self.data_selection_model.get_datafolder()
    
    def set_datafolder(self, datafolder):
        self.data_selection_model.set_datafolder(datafolder)
    
    def get_online_version(self):
        return self.main_model.get_online_version()
    
    def upload_log(self):
        self.drive.upload_log(self.logger.get_result(), self.login_model.get_userid(), self.logger.get_timestamp())

    def login(self, userid, terms_checkbox):
        if terms_checkbox:
            if self.login_model.login_correct(userid, self.drive):
                self.login_view.disable_login_button()
                self.login_view.show_loading()
                self.drive.get_performances(userid)
                self.update_learning_path()
                self.login_view.hide_login()
                self.main_view.set_title(4, 'Log (userid:' + str(userid) + ')')
                self.task_selection_view.show_task_selection()

            else:
                print("login incorrect")
        else:
            print("You must agree to the terms before continuing")
    
    def set_task_model(self,task, monitored_mode):
        self.monitored_mode = monitored_mode
        self.task_view.set_monitored_mode(monitored_mode)
        if self.monitored_mode:
            task = self.convertPerformanceToTask.convert_performance_to_task(self.logger.get_result(), task["title"],task["description"])
            self.task_model.set_current_task(task)
            self.task_view.set_task(self.task_model.get_current_task())
            self.task_model.update_statusses_and_set_current_tasks()
            self.task_view.set_active_accordion()
        else:
            self.task_model.set_current_task(task)
            self.task_view.set_task(self.task_model.get_current_task())
            self.task_model.update_statusses_and_set_current_tasks()
            self.task_view.update_task_statuses(self.task_model.get_current_task())
            self.task_view.set_active_accordion()

    def hide_task_selection_and_show_tabs(self):
        self.task_selection_view.hide_task_selection()
        self.task_view.show_task()
        self.main_view.show_tabs()
    
    def register(self):
        print(self.drive.register())
    
    def get_ui(self):
        login_view = self.login_view.get_login_view()
        task_selection_view = self.task_selection_view.get_task_selection_view()
        task_view = self.task_view.get_task_view()
        tabs = self.get_tab_set()
        return self.main_view.get_ui(login_view, tabs, task_view, task_selection_view)
    
    def update_percentage_done(self, percentage):
        self.login_view.update_percentage_done(percentage)

    def get_list_of_actions(self):
        return self.logger.get_list_of_actions()
    
    def update_log_view(self):
        self.learning_path_view.update_actions()
    
    def refresh_data_processing(self):
        self.data_processing_view.featurepr_click(None)
    
    def get_graph_data(self):
        return self.learning_path_model.get_graph_data()
    
    def update_learning_path(self):
        userid = self.login_model.get_userid()
        self.learning_manager_model.set_learning_path(userid)
        # self.learning_path_model.set_performance_data()
        self.learning_manager_model.set_skill_vectors()
        self.learning_path_view.set_graph_data(self.get_graph_data())

    def update_task_view(self, action, value):
        if self.monitored_mode:
            task = self.convertPerformanceToTask.convert_performance_to_task(self.logger.get_result(), self.task_model.get_title(), self.task_model.get_description())
            self.task_model.set_current_task(task)
            self.task_view.set_task(self.task_model.get_current_task())
        else:
            self.task_model.perform_action(action, value)
            self.task_model.update_statusses_and_set_current_tasks()
            self.task_view.update_task_statuses(self.task_model.get_current_task())
            self.task_view.set_active_accordion()

    def read_dataset_view(self, dataset):
        self.set_datafolder("DataSets")
        self.main_view.datasets.options = [dataset]
        self.main_view.datasets.value = dataset
        self.data_selection_view.read_dataset(None)

    def show_completion_popup(self):
        self.task_view.show_completion_popup()

    def get_learning_path_view(self):
        return self.learning_path_view.get_learning_path_tab()
    
    def get_tasks_data(self):
        return self.main_model.get_tasks_data()
    
    def convert_performance_to_task(self,performance, title, description):
        return self.convertPerformanceToTask.convert_performance_to_task(performance, title, description)
    
    def get_filtered_tasks(self, tasks, guided_mode, all_tasks):
        current_skill_vector = self.learning_path_model.get_current_skill_vector()
        return self.task_selection_model.get_filtered_tasks(tasks, guided_mode, all_tasks, current_skill_vector)
    
    def get_target_task(self, task):
        return self.task_model.get_target(task)
    
    def get_dataset_task(self, task):
        return self.task_model.get_dataset(task)
    
    def get_reference_task(self, target_column, dataset):
        return self.main_model.get_reference_task(target_column, dataset)
    
    def get_model_performance(self, task):
        return self.task_model.get_model_performance(task)
    
    def get_learning_path(self):
        return self.learning_manager_model.get_learning_path()
    
    def add_skill_vector(self, skill_vector):
        self.learning_path_model.add_skill_vector(skill_vector)