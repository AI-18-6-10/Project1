import numpy as np
import joblib 
from tensorflow import keras

def predict_steel_faults_b():

    # 1. 모델 불러오기
    model = keras.models.load_model('model_layers')
    # scaler 불러오기
    scaler = joblib.load('joblib/StandardScaler.joblib')

    # 입력값 받기
    Pixels_Areas = input("Enter Pixels_Areas: ")
    X_Perimeter = input("Enter X_Perimeter: ")
    Y_Perimeter = input("Enter Y_Perimeter: ")
    Sum_of_Luminosity = input("Enter Sum_of_Luminosity: ")
    Minimum_of_Luminosity = input("Enter Minimum_of_Luminosity: ")
    Maximum_of_Luminosity = input("Enter Maximum_of_Luminosity: ")
    Length_of_Conveyer = input("Enter Length_of_Conveyer: ")
    TypeOfSteel = input("Enter TypeOfSteel: ")
    Steel_Plate_Thickness = input("Enter Steel_Plate_Thickness: ")
    Edges_Index = input("Enter Edges_Index: ")
    Empty_Index = input("Enter Empty_Index: ")
    Square_Index = input("Enter Square_Index: ")
    Outside_X_Index = input("Enter Outside_X_Index: ")
    Edges_X_Index = input("Enter Edges_X_Index: ")
    Edges_Y_Index = input("Enter Edges_Y_Index: ")
    LogOfAreas = input("Enter LogOfAreas: ")
    Orientation_Index = input("Enter Orientation_Index: ")
    Luminosity_Index = input("Enter Luminosity_Index: ")
    SigmoidOfAreas = input("Enter SigmoidOfAreas: ")
    Area = input("Enter Area: ")
    # Area : ('X_Maximum' - 'X_Minimum') * ('Y_Maximum' - 'Y_Minimum')
    # 나중에 따로 웹에 구상하면 설명을 써줘야함
    
    # 리스트로 변형
    input_list = [Pixels_Areas, X_Perimeter, Y_Perimeter, Sum_of_Luminosity, Minimum_of_Luminosity,
                  Maximum_of_Luminosity, Length_of_Conveyer, TypeOfSteel, Steel_Plate_Thickness,
                  Edges_Index, Empty_Index, Square_Index, Outside_X_Index, Edges_X_Index, Edges_Y_Index,
                  LogOfAreas, Orientation_Index, Luminosity_Index, SigmoidOfAreas, Area]
    
    # 입력 데이터를 2차원 배열로 변환하여 스케일링
    input_data_2d = np.array(input_list).reshape(1, -1)

    # 입력 데이터 스케일링
    input_data_scaled = scaler.transform(input_data_2d)

    # 예측
    prediction = model.predict(input_data_scaled)

    if prediction[0] == 0:
        return 'Other_Faults'
    else:
        return 'Faults'

# 결과 예측
result = predict_steel_faults_b()

# 결과 출력
print("Result:", result)