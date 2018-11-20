# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from ultimate.mlp import MLP
import time
import gc
gc.enable()


def load_data(file_path, dtypes):
    print('loading data......')
    df = pd.read_csv(file_path, dtype=dtypes)
    # Delete rows that contain null values
    df = df.dropna()
    print('loaded finally')
    return df


def save_csv(pred, test):
    print('saving......')
    pd_data = pd.DataFrame(pred)
    pd_data.columns = ['winPlacePerc']
    pd_data = pd.concat([test['Id'], pd_data], axis=1)
    pd_data.to_csv('submission.csv', index=False)
    print('saved OK')


if __name__ == '__main__':
    start_time = time.time()
    features = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'killPlace', 'killPoints',
                'kills', 'killStreaks', 'longestKill', 'maxPlace', 'revives', 'roadKills', 'weaponsAcquired',
                'rideDistance', 'swimDistance', 'walkDistance']
    # Specify the type in advance to save memory
    dtypes = {
        'Id': 'object',
        'groupId': 'object',
        'matchId': 'object',
        'assists': 'int8',
        'boosts': 'int8',
        'damageDealt': 'float32',
        'DBNOs': 'int8',
        'headshotKills': 'int8',
        'heals': 'int8',
        'killPlace': 'int8',
        'killPoints': 'int16',
        'kills': 'int8',
        'killStreaks': 'int8',
        'longestKill': 'float32',
        'matchDuration': 'int16',
        'matchType': 'object',
        'maxPlace': 'int16',
        'numGroups': 'int16',
        'revives': 'int8',
        'rideDistance': 'float32',
        'roadKills': 'int8',
        'swimDistance': 'float32',
        'teamKills': 'int16',
        'vehicleDestroys': 'int8',
        'walkDistance': 'float32',
        'weaponsAcquired': 'int16',
        'winPoints': 'int32',
    }
    features_target = ['winPlacePerc']
    train = load_data('../input/train_V2.csv', dtypes)
    test = load_data('../input/test_V2.csv', dtypes)
    print(train.info())
    X_train = train[features]
    y_train = train[features_target]
    X_test = test[features]
    del train
    gc.collect()

    X_train['totalDistance'] = X_train['walkDistance'] + X_train['rideDistance'] + X_train['swimDistance']
    X_test['totalDistance'] = X_test['walkDistance'] + X_test['rideDistance'] + X_test['swimDistance']
    del X_train['walkDistance'], X_train['rideDistance'], X_train['swimDistance']
    del X_test['walkDistance'], X_test['rideDistance'], X_test['swimDistance']
    gc.collect()
    scaler = preprocessing.MinMaxScaler(copy=False).fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp_param = {
        'loss_type': 'mse',
        'layer_size': [X_train.shape[1], 8, 8, 8, 1],
        'output_range': [0, 1],
        'importance_out': True,
        'rate_init': 0.02,
        'epoch_train': 5,
        'epoch_decay': 10,
        'verbose': 1,
    }
    mlp = MLP(mlp_param)
    mlp.fit(X_train, y_train)
    print("feature importances:", mlp.feature_importances_)
    pred = mlp.predict(X_test)
    save_csv(pred, test)
    end_time = time.time()
    print('The total of time is', end_time - start_time)


