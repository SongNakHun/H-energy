import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def ensemble_model(training, estimation):
    train_x = training.drop(['amount'], axis=1)
    train_y = training[['amount']]

    test_x = estimation.drop(['amount'], axis=1)
    test_y = estimation[['amount']]

    scaler = RobustScaler()
    scaled_train_x = scaler.fit_transform(train_x)
    scaled_test_x = scaler.transform(test_x)

    model_list = [RandomForestRegressor(), XGBRegressor(), LGBMRegressor(), MLPRegressor()]

    predictions = np.zeros(shape=(len(scaled_test_x), len(model_list)))
    for idx, model in enumerate(model_list):
        print(f'Now Model {model} Training ... ')
        print(' - ' * 100)
        model.fit(scaled_train_x, train_y['amount'].ravel())
        pred_test = model.predict(scaled_test_x)

        predictions[:, idx] = pred_test

    predictions = np.mean(predictions, axis=1)

    return predictions


def ensemble_model_remain(training, estimation, training_target, estimation_target):
    training = pd.concat([training, estimation], axis=0)
    training = pd.concat([training, training_target], axis=0)
    train_x = training.drop(['amount'], axis=1)
    train_y = (training['amount'] / training['capacity']).ravel()
    test_x = estimation_target.drop(['amount'], axis=1)
    scaler = RobustScaler()
    scaled_train_x = scaler.fit_transform(train_x)
    scaled_test_x = scaler.transform(test_x)

    model_list = [RandomForestRegressor(), XGBRegressor(), LGBMRegressor(), MLPRegressor()]

    predictions = np.zeros(shape=(len(scaled_test_x), len(model_list)))
    for idx, model in enumerate(model_list):
        print(f'Now Model {model} Training ... ')
        print(' - ' * 100)
        model.fit(scaled_train_x, train_y.ravel())
        pred_test = model.predict(scaled_test_x)

        predictions[:, idx] = pred_test * estimation_target['capacity']

    predictions = np.mean(predictions, axis=1)

    return predictions

def ensemble_machine(ensemble_list):
    predictions = np.zeros(shape=(len(ensemble_list[0]), len(ensemble_list)))
    for idx, ensemble_result in enumerate(ensemble_list):
        predictions[:, idx] = ensemble_list[idx]

    weight_list = [0.4, 0.2, 0.4]

    predictions_result = np.zeros(shape=(len(ensemble_list[0]), len(ensemble_list)))
    for idx, weight in enumerate(weight_list):
        temp_pred = predictions[:, idx]
        weight = weight_list[idx]

        temp_res = temp_pred * weight
        predictions_result[:, idx] = temp_res

    predictions_result = np.sum(predictions_result, axis=1)

    return predictions_result
        
