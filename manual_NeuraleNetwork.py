import pandas as pd

# reading data
data = pd.read_csv('insurance_data.csv')

# data splitting
def split_data(data):
    xtrain, xtest = pd.concat([data['age'][:-12], data['affordibility'][:-12]], axis='columns'), pd.concat([data['age'][-12:-1], data['affordibility'][-12:-1]], axis='columns')
    ytrain, ytest =  data['bought_insurance'][:-12], data['bought_insurance'][-12:-1]
    return xtrain, xtest, ytrain, ytest
xtrain, xtest, ytrain, ytest = split_data(data)

#scalling data
def scalling_data(xtrain, xtest):
    xtrain_scaled = xtrain.copy()
    xtrain_scaled['age'] = xtrain_scaled['age'] / 100

    xtest_scaled = xtest.copy()
    xtest_scaled['age'] = xtest_scaled['age'] / 100
    return xtrain_scaled, xtest_scaled
xtrain_scaled, xtest_scaled = scalling_data(xtrain, xtest)



# modeling
from tensorflow import keras
import math
model = keras
def model_(model, xtrain_scaled, ytrain):
    model = model.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.fit(xtrain_scaled,  ytrain, epochs=5000)

    # model score
    score_ = model.evaluate(xtest_scaled, ytest)
    print("Model Score Evaluate: ", score_)

    # model coeficient & intercept
    coef, intercept = model.get_weights()
    print('coeficient & intercept: ', coef, intercept)

    # prediction func
    weighted_sum = coef[0]*xtrain['age'] + coef[1]*xtrain['affordibility'] + intercept

    # sigmoid func
    pred_ = []
    for i in range(len(weighted_sum)):
        sigmoid = 1 / (1 + math.exp(-weighted_sum[i]))
        pred_.append(sigmoid)
    pred_ = pd.DataFrame(pred_)
    print(xtrain, pred_)

model_(model, xtrain_scaled, ytrain)

# Model Score Evaluate:  [0.5021059513092041, 0.9090909361839294]
# coeficient & intercept:  [[5.271008 ][2.1496265]], [-3.5346916]
'''
 age  affordibility                ||    Result                                                               
1    25              0             ||     =1.0
2    47              1             ||     =1.0
3    52              0             ||     =1.0
4    46              1             ||     =1.0
5    56              1             ||     =1.0
6    55              0             ||     =1.0
7    60              0             ||     =1.0
8    62              1             ||     =1.0
9    61              1             ||     =1.0
10   18              1             ||     =1.0
11   28              1             ||     =1.0
12   27              0             ||     =1.0
13   29              0             ||     =1.0
14   49              1             ||     =1.0
15   55              1             ||     =1.0
'''