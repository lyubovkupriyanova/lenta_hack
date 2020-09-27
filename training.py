import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

X = pd.read_parquet('X.parquet')
y = np.load('y.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

cat_features_names = ['gender', 'client_id', 'city', 'top_1_group', 'top_2_group', 'top_3_group', 'top_1']

catboost_params = {'iterations':200, 'learning_rate':0.1, 
                                    'depth':10, 'l2_leaf_reg':10, 'random_state':7,
                                    'custom_metric':['AUC'], 'thread_count':10,  'od_type': "Iter",'od_wait':10}

model = CatBoostClassifier(train_dir='/tmp/catboost', **catboost_params)
model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=True, verbose=False, cat_features=cat_features_names)
