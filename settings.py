import pandas as pd

def init():
    global curr_df
    global Xtrain_df
    global Xtest_df
    global ytrain_df
    global ytest_df
    global clean_df
    global original_df
    
    curr_df = pd.DataFrame()
    Xtrain_df = pd.DataFrame()
    Xtest_df = pd.DataFrame()
    ytrain_df = pd.DataFrame()
    ytest_df = pd.DataFrame()
    clean_df = pd.DataFrame()
    original_df = pd.DataFrame()