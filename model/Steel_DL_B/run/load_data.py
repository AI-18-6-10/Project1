import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN
import numpy as np

from sklearn import metrics
from scikeras.wrappers import KerasClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers as Layer
from tensorflow.keras.metrics import Recall, Precision, BinaryAccuracy, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.regularizers import L1L2
from timeit import default_timer as timer
import json

def load_data(csv_path='https://raw.githubusercontent.com/urmd1f/Project1/main/data/steel.csv', scaler=None, upsampling=None, TEST_SIZE=0.2, VAL_SIZE=0.2, RANDOM_STATE=42):

    # 글로벌로 만들어서 저장하기
    global df, X, y, X_train, X_val, X_test, y_train, y_val, y_test
    
    df = pd.read_csv(csv_path)

    # 이상치 전처리 
    # 일단 카피 
    df_clear = df.copy()
    # 타겟특성 빼고 변수에 저장
    df_clear = df_clear.drop(df_clear.iloc[:,-7:].columns, axis=1)
    # Area라는 컬럼 생성
    df_clear['Area'] = (df['X_Maximum'] - df['X_Minimum']) * (df['Y_Maximum'] - df['Y_Minimum'])
    # Area컬럼을 생성하는데 사용된 피쳐 제거(X_Maximum, X_Minimum, Y_Maximum, Y_Minimum)
    df_clear = df_clear.drop(df.iloc[:, :4].columns, axis=1)
    # A300 제거(300 or 400이라 하나 제거 후 컬럼명 어떻게 할지 고민)
    df_clear = df_clear.drop('TypeOfSteel_A300', axis=1)
    # 일단 TypeOfSteel로만 변경해서 설명란에 0은 300 1은 400으로 해보기로
    df_clear.rename(columns={'TypeOfSteel_A400':'TypeOfSteel'}, inplace=True)
    # Log_X_index, Log_Y_index 제거 LogOFAreas가 합친결과값으로 판단되어 제거하기로 함
    df_clear = df_clear.drop(['Log_X_Index', 'Log_Y_Index'], axis=1)
    df_clear = df_clear.drop('Outside_Global_Index', axis=1)
    # 정규분포를 위한 log 변환

    log_list = [
        'X_Perimeter',
        'Y_Perimeter',
        'Steel_Plate_Thickness',
        # 'Edges_Index', # <- 0이 들어있어서 로그변환이 안됨 zerodivision_error
        'Outside_X_Index',
        'Area',
        'Edges_Y_Index',
        'Pixels_Areas',
        'Sum_of_Luminosity'
        ]

    for i in log_list:
        df_clear[i] = np.log(df_clear[i])

    y_list = list(df.iloc[:,-7:].columns)
    df_target = df.copy()
    df_target["Type"] = df_target.loc[:,y_list][y_list].idxmax(axis=1)
    df_target = df_target.drop(columns=y_list)

    # 이진분류를 위한 'Type' 컬럼 값 수정
    df_target1 = df_target.copy()
    df_target1['Type'] = df_target1['Type'].apply(lambda x: 1 if x != 'Other_Faults' else 0)
    df_target1['Type'].value_counts(normalize=True)

    ## 다중분류를 위한 'Type' 컬럼 값 수정
    #df_target2 = df_target.copy()
    ## 'Other_Faults'를 제외한 인덱스 가져오기
    #indices_to_remove = df_target2[df_target2['Type'] == 'Other_Faults'].index
    ## 해당 인덱스들을 제거하여 새로운 데이터프레임 생성
    #df_clear2 = df_clear.copy()
    #df_clear2 = df_clear.drop(indices_to_remove)
    ## 'Other_Faults' 행을 제외하고 결함이 있는 것만 남기기
    #df_target2 = df_target2[df_target2['Type'] != 'Other_Faults']
    #encoder = LabelEncoder()
    #df_target2['Type'] = encoder.fit_transform(df_target2['Type'])

    X = df_clear
    y = df_target1['Type']

    # 업스케일링
    # 스모트(SMOTE) 대신에 아다신(ADASYN) 사용된 이유는 좀 더 랜덤하게 업스케일링이 되게 하게 위해 사용
    if upsampling:
        adasyn = ADASYN(random_state=RANDOM_STATE)
        X, y = adasyn.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train)
    

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