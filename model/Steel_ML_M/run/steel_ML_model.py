import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def predict_neutron_star():

    # 1. 모델 불러오기
    with open('ML_steel_pikling/xgb_model_multy.pkl', 'rb') as file:
        model = pickle.load(file)

    # scaler 불러오기
    with open('ML_steel_pikling/xgb_scaler_multy.pkl', 'rb') as file:
        scaler = pickle.load(file)

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

    '''
    0 -> Bumps
    1 -> Dirtiness
    2 -> K_Scatch
    3 -> Pastry
    4 -> Stains
    5 -> Z_Scratch
    '''

    if prediction[0] == 0:
        return 'Bumps 결함입니다.' 
    elif prediction[0] == 1:
        return 'Dirtiness 결함입니다.'
    elif prediction[0] == 2:
        return 'K_Scatch 결함입니다.'
    elif prediction[0] == 3:
        return 'Pastry 결함입니다.'
    elif prediction[0] == 4:
        return 'Stains 결함입니다.'
    else:
        return 'Z_Scratch 결함입니다.'

# 결과 예측
result = predict_neutron_star()

# 결과 출력
print("Result:", result)