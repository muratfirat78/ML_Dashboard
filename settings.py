import pandas as pd

def init():
    global curr_df
    global Xtrain_df
    global Xtest_df
    global ytrain_df
    global ytest_df
    global clean_df
    global original_df
    global online_version
    global trg_lbl
    global f_box
    global tab1
    global tab2
    global tab3
    global tab4
    global FeatPage
    global ProcssPage
    global DFPage
    global RightPage
    global processtypes
    global dt_features
    global dt_ftslay
    global featurescl
    global ftlaycl
    global DataFolder
    global trainedModels
    
    curr_df = pd.DataFrame()
    Xtrain_df = pd.DataFrame()
    Xtest_df = pd.DataFrame()
    ytrain_df = pd.DataFrame()
    ytest_df = pd.DataFrame()
    clean_df = pd.DataFrame()
    original_df = pd.DataFrame()
    online_version = False
    trainedModels = []