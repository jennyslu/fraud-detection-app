import json
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

if __name__ == '__main__':
    with open('data/data.json') as f:
        data = json.load(f)
    #classify data
    for row in data:
        if row['acct_type'].startswith('fraud'):
            row['fraud'] = 1
            row['spam'] = 0
            row['locked'] = 0
        elif row['acct_type'].startswith('spam'):
            row['fraud'] = 0
            row['spam'] = 1
            row['locked'] = 0
        elif row['acct_type'] == 'premium':
            row['fraud'] = 0
            row['spam'] = 0
            row['locked'] = 0
        else:
            row['fraud'] = 0
            row['spam'] = 0
            row['locked'] = 1

    df = pd.DataFrame(data)
    y = df[['fraud', 'spam', 'locked']]
    df.drop(['acct_type', 'fraud', 'spam', 'locked'], axis=1, inplace=True)
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                                 random_state=42)
    X_train.to_json('data/X_train.json', orient='records')
    X_test.to_json('data/X_test.json', orient='records')
    y_test.to_json('data/y_test.json', orient='records')
    y_train.to_json('data/y_train.json', orient='records')
