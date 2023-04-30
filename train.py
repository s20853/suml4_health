import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def main():
    train = pd.read_csv("DSP_13.csv", sep=";")
    train = train.fillna(train.mean())
    X = train.drop('zdrowie', axis=1)
    y = train['zdrowie']

    X_train, X_test, y_train, y_test = train_test_split(train.drop('zdrowie', axis=1),
                                                        train['zdrowie'], test_size=0.2,
                                                        random_state=103)

    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    filename = 'model.sv'
    pickle.dump(model, open(filename, 'wb'))

    score = accuracy_score(y_test, y_pred)
    print(score)

    print(min(train["leki"]))
    print(max(train["leki"]))

if __name__ == '__main__':
    main()