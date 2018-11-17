# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


def load_data(file_path):
    print('loading data......')
    df = pd.read_csv(file_path)
    df = df.dropna()
    print('loaded finally')
    return df


def wash_data(X_train, y_train):
        print(X_train.head())
        print(X_train.info())
        print(X_train.isnull().any())
        print(y_train.isnull().any())
        # X_train['DBNOs'].replace([0], X_train['DBNOs'].mean())
        # X_train['assists'].replace([0], X_train['assists'].mean())
        # X_train['boosts'].replace([0], X_train['boosts'].mean())
        # X_train['damageDealt'].replace([0], X_train['damageDealt'].mean())
        # X_train['headshotKills'].replace([0], X_train['headshotKills'].mean())
        # X_train['killStreaks'].replace([0], X_train['killStreaks'].mean())
        # X_train['kills'].replace([0], X_train['kills'].mean())
        # X_train['longestKill'].replace([0], X_train['longestKill'].mean())
        # X_train['roadKills'].replace([0], X_train['roadKills'].mean())
        # X_train['weaponsAcquired'].replace([0], X_train['weaponsAcquired'].mean())
        # X_train['maxPlace'].replace([0], X_train['maxPlace'].mean())
        # y_train['winPlacePerc'].replace([0], y_train['winPlacePerc'].mean())
        print(X_train.isnull().any())


def train_LR_model(X_train, y_train):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    return linear_model


def predict(model, X_test):
    print('predicting......')
    predict = model.predict(X_test)
    return predict


def save_csv(predict, test):
    print('saving......')
    pd_data = pd.DataFrame(predict)
    pd_data.columns = ['winPlacePerc']
    pd_data = pd.concat([test['Id'], pd_data], axis=1)
    pd_data.to_csv('submission.csv', index=False)


def normalized(X_train):
    X_train['DBNOs'] = (X_train['DBNOs'] - X_train['DBNOs'].min()) / (X_train['DBNOs'].max() - X_train['DBNOs'].min())
    return X_train


if __name__ == '__main__':
    features = ['DBNOs', 'assists', 'boosts', 'damageDealt', 'headshotKills', 'killStreaks', 'kills', 'longestKill',
                'roadKills', 'weaponsAcquired', 'maxPlace']
    features_target = ['winPlacePerc']
    train = load_data('../input/train_V2.csv')
    test = load_data('../input/test_V2.csv')
    X_train = train[features]
    y_train = train[features_target]
    X_test = test[features]
    # wash_data(X_train, y_train)
    # X_train = normalized(X_train)
    # X_test = normalized(X_test)
    X_train = (X_train -X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()
    y_train = (y_train - y_train.mean()) / y_train.std()
    # lab_enc = preprocessing.LabelEncoder()
    # y_train = lab_enc.fit_transform(y_train)
    model = train_LR_model(X_train, y_train)
    predict = predict(model, X_test)
    save_csv(predict, test)
