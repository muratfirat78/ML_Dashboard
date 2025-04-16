from model.data_cleaning import DataCleaningModel
from model.data_processing import DataProcessingModel
from model.data_selection import DataSelectionModel
from model.learning_path import LearningPathModel
from model.local_drive import GoogleDrive
from model.logger import Logger
from model.login import LoginModel
from model.predictive_modeling import PredictiveModelingModel
from view.data_cleaning import DataCleaningView
from view.data_processing import DataProcessingView
from view.data_selection import DataSelectionView
from model.main_model import MainModel
from view.learning_path import LearningPathView
from view.main_view import MainView
from view.predictive_modeling import PredictiveModelingView
from view.login import LoginView

class Controller:
    def __init__(self, drive, online_version):
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
        self.learning_path_model = LearningPathModel()
        self.learning_path_view = LearningPathView(self)
        self.learning_path = None
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
                self.learning_path_model.set_learning_path(userid)
                # self.learning_path.get_scores()
                self.login_view.hide_login()
                self.main_view.show_tabs()
            else:
                print("login incorrect")
        else:
            print("You must agree to the terms before continuing")
    
    def get_ui(self):
        login_view = self.login_view.get_login_view()
        tabs = self.get_tab_set()
        return self.main_view.get_ui(login_view, tabs)
    
    def update_percentage_done(self, percentage):
        self.login_view.update_percentage_done(percentage)

    def get_list_of_actions(self):
        return self.logger.get_list_of_actions()
    
    def update_log_view(self):
        self.learning_path_view.update_actions()
    
    def refresh_data_processing(self):
        self.data_processing_view.featurepr_click(None)