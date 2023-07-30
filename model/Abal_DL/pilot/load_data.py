import pandas as pd


from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import ADASYN

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.regularizers import L1L2

from timeit import default_timer as timer

import json

import joblib


def load_data(csv_path='/Users/leetaeryeong/Project1/model/abalone_model/2nd/Regression_data.csv', scaler=None, upsampling=None, TEST_SIZE=0.2, VAL_SIZE=0.2, RANDOM_STATE=42):

    # 글로벌로 만들어서 저장하기
    global df, X, y, X_train, X_val, X_test, y_train, y_val, y_test
    
    df = pd.read_csv(csv_path)

    # 이상치 전처리 
    df_origin = df.copy()
    df_origin.iloc[:, 1:8] *= 200

    selected_rows = df_origin[df_origin['Diameter'] == 46.0]
    selected_rows_removed = selected_rows.drop(3996)
    df_origin.loc[3996, 'Height'] = selected_rows_removed['Height'].mean()

    selected_rows = df_origin[df_origin['Diameter'] == 71.0]
    selected_rows_removed = selected_rows.drop(2051)
    df_origin.loc[2051, 'Height'] = selected_rows_removed['Height'].mean()

    selected_rows = df_origin[df_origin['Diameter'] == 68.0]
    selected_rows_removed = selected_rows.drop(1257)
    df_origin.loc[1257, 'Height'] = selected_rows_removed['Height'].mean()

    selected_rows = df_origin[df_origin['Length'] == 141.0]
    selected_rows_removed = selected_rows.drop(1417)
    df_origin.loc[1417, 'Height'] = selected_rows_removed['Height'].mean()

    condition = (df_origin['Whole weight'] > 12.0) & (df_origin['Whole weight'] < 14.0)
    selected_rows = df_origin[condition]
    selected_rows_removed = selected_rows.drop(3522)
    df_origin.loc[3522, 'Height'] = selected_rows_removed['Height'].mean()

    df_clean = df_origin.copy()
    df_clean.iloc[:, 1:8] /= 200

    # 원핫인코딩
    df_clean = pd.get_dummies(df_clean,columns=['Sex'])
    
    # 데이터 나누기
    X = df_clean.drop('Rings', axis=1)
    y = df_clean['Rings'].astype('float32')

    # 업스케일링
    # 스모트(SMOTE) 대신에 아다신(ADASYN) 사용된 이유는 좀 더 랜덤하게 업스케일링이 되게 하게 위해 사용
    if upsampling:
        adasyn = ADASYN(random_state=RANDOM_STATE)
        X, y = adasyn.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    

    # MinMaxScaler - 0, StandardScaler - 1, , MaxAbsScaler - 2, RobustScaler - 3, Normalizer - 4
    if scaler == 0:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    elif scaler == 1:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    elif scaler == 2:
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    elif scaler == 3:
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    elif scaler == 4:
        scaler = Normalizer()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        pass
    
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_val = pd.DataFrame(X_val, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler