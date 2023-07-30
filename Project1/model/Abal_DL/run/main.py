import joblib
import numpy as np
import warnings
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

# 사용자 지정 함수 불러오기
from eval import get_Accuracy_tf, total_score


warnings.filterwarnings("ignore", category=UserWarning)

def predict_abalone_rings():
    # load_model()로 모델 불러오기 -> 버그인지 평가값이 달라짐
    model = keras.models.load_model('model_layers', custom_objects={"get_Accuracy_tf": get_Accuracy_tf })

    # 2. scaler 불러오기
    scaler = joblib.load('joblib/MinMaxScaler.joblib')
    
    # 입력값 받기
    Sex = input("Enter Sex(F, M, I): ")
    Length = input("Enter Length: ")
    Diameter = input("Enter Diameter: ")
    Height = input("Enter Height: ")
    Whole_weight = input("Enter Whole weight: ")
    Shucked_weight = input("Enter Shucked weight: ")
    Viscera_weight = input("Enter Viscera weight: ")
    Shell_weight = input("Enter Shell weight: ")
    Rings = input("Enter Real Rings to compare with Prediction(Target): ")
    Sex_F, Sex_I, Sex_M = 0, 0, 0
    
    if Sex == 'F':
        Sex_F = 1
    
    if Sex == 'I':
        Sex_I = 1

    if Sex == 'M':
        Sex_M = 1


    # 리스트로 변형
    input_list = [Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight, Sex_F, Sex_I, Sex_M]
    
    # 입력 데이터를 2차원 배열로 변환하여 스케일링
    input_data_2d = np.array(input_list).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data_2d)

    # 예측
    prediction = model.predict(input_data_scaled)
    print(prediction, Rings)

predict_abalone_rings()