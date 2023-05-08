import numpy as np
from sklearn.linear_model import LinearRegression

def l_reg(input):
    x_data = np.array([115, 117, 140, 115, 116, 127, 121, 121, 115, 103,
                114, 90, 24, 98, 110, 119, 121, 108, 141, 78, 93, 98, 88, 141, 84])
    y_data = np.array([168, 164, 192, 136, 146, 142, 132, 199, 127, 131,
                157, 115, 0, 162, 168, 131, 112, 183, 119, 59, 176, 187, 93, 226, 154])

    # Linear Regression 모델 객체 생성
    model = LinearRegression()

    # 모델에 데이터 학습
    model.fit(x_data.reshape(-1, 1), y_data)


    # 주어진 x에 대해 y 예측
    x = float(input)
    y = model.predict(np.array([[x]]))

    # 결과 출력
    #print("x :", x)
    #print("y :", int(y[0]))
    count = int(y[0])

    return count
