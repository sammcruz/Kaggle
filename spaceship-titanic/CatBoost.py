#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:56:42 2022

@author: samantha
"""

# Source: https://www.kaggle.com/code/jasoninzana/spaceship-titanic/notebook

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import choices, seed

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

from catboost import CatBoostClassifier


# Load datasets
train = pd.read_csv("data/spaceship-titanic/train.csv")
test =  pd.read_csv("data/spaceship-titanic/test.csv")

spend_vars = train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']]
sns.pairplot(spend_vars, hue='Transported', diag_kind='hist');

_, ax = plt.subplots(1,2, figsize=(15,5))
plt.sca(ax[0])
sns.countplot(data=train[['HomePlanet', 'Destination', 'Transported']], x='HomePlanet', hue='Transported');
plt.sca(ax[1])
sns.countplot(data=train[['HomePlanet', 'Destination', 'Transported']], x='Destination', hue='Transported');

_, ax = plt.subplots(1,2, figsize=(15,5))
plt.sca(ax[0])
sns.countplot(data=train[['Transported', 'CryoSleep']], x='CryoSleep', hue='Transported');
plt.sca(ax[1])
sns.countplot(data=train[['Transported', 'VIP']], x='VIP', hue='Transported');

train[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = train.Cabin.str.extract('(\w+)/(\d+)/(\w+)')
_, ax = plt.subplots(1,2, figsize=(15,5))
plt.sca(ax[0])
sns.countplot(data=train[['Transported', 'Cabin_deck']], x='Cabin_deck', hue='Transported');
plt.sca(ax[1])
sns.countplot(data=train[['Transported', 'Cabin_side']], x='Cabin_side', hue='Transported');

# Split up the training dataset for preliminary testing
X_train, X_test, y_train, y_test = train_test_split(train.drop('Transported', axis=1), train.Transported, random_state=0)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

X_train.info()

# Create functions for feature engineering

def Impute_from_Group(df, column):
    # If they are in the same travel group, impute the missing last name
    Missing = df[[column, 'Group']].loc[df[column].isnull()]
    NotMissing = df[[column, 'Group']].loc[~df[column].isnull()]
    Update = Missing.reset_index().merge(NotMissing, on='Group', how='inner', suffixes=['_old', '']).drop_duplicates()
    Update.set_index('PassengerId', inplace=True)
    df.loc[Update.index, column] = Update[column]
    return df

def Feature_Eng(df_in):
    
    df = df_in.copy()

    # Extract group from Passenger ID
    df['Group'] = df.PassengerId.str.extract('(\d{4})_\d{2}')
    df.set_index('PassengerId', inplace=True)

    # Extract last name from name
    df['LastName'] = df.Name.str.extract('[\w+]\s(\w+)')
    df.drop(['Name'], axis=1, inplace=True)
    
    # Get the cabin deck and cabin side, leave out the room number
    df[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = df.Cabin.str.extract('(\w+)/(\d+)/(\w+)')
    df.drop(['Cabin', 'Cabin_num'], axis=1, inplace=True)
    
    # If they are in the same travel group, impute the home planet, destination, cabin deck, and cabin side from the same group (if available)
    df = Impute_from_Group(df, 'HomePlanet')
    df = Impute_from_Group(df, 'Destination')
    df = Impute_from_Group(df, 'Cabin_deck')
    df = Impute_from_Group(df, 'Cabin_side')
    
    # For still missing cabin values, impute the cabin side randomly, but weighted to make the two sides balanced
    idx2impute = df.loc[df.Cabin_side.isnull(), 'Cabin_side'].index.tolist()
    Imbalance = (round(len(df)/2) - df.Cabin_side.value_counts()['S']) / (round(len(df)/2) - df.Cabin_side.value_counts()['P'])
    seed(1)
    df.loc[idx2impute, 'Cabin_side'] = choices(['S', 'P'], weights=[Imbalance, 1], k=len(idx2impute))
    
    # Spending will be 0 and VIP will be False if CryoSleep = True
    true_idx = df[df.CryoSleep == True].index
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df.loc[true_idx, spend_cols] = 0
    df.loc[true_idx, 'VIP'] = False
    
    # Create a new feature of total $ spent
    df['Total'] = df[spend_cols].sum(axis=1)
    
    # For CryoSleep, if VIP = True, CryoSleep = False or if spending > 0, CryoSleep = False
    true_idx = df[((df.Total == 0) & (df.VIP == False)) & df.CryoSleep.isnull()].index
    false_idx = df[((df.Total > 0) | (df.VIP == True)) & df.CryoSleep.isnull()].index
    df.loc[true_idx, 'CryoSleep'] = True
    df.loc[false_idx, 'CryoSleep'] = False
    
    # Repeat - Spending will be 0 and VIP will be False if CryoSleep = True
    true_idx = df[df.CryoSleep == True].index
    df.loc[true_idx, spend_cols] = 0
    df.loc[true_idx, 'VIP'] = False
    
    # Drop unwanted columns
    cols2drop = ['LastName', 'Group']
    df.drop(cols2drop, axis=1, inplace=True)
    
    return df


X_train_2 = Feature_Eng(X_train)
X_test_2 =  Feature_Eng(X_test)
X_train_2.info()

# Divide the features by type, for different preprocessing methods
cat_feats = X_train_2.dtypes[X_train_2.dtypes == object].index.tolist()
num_feats = X_train_2.dtypes[X_train_2.dtypes == float].index.tolist()

# Setup preprocessor pipelines, dependent on feature type
ord_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                           ('ordEncode', OrdinalEncoder(dtype=int))])

num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                           ('scaler', StandardScaler())])

cat_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                           ('onehot', OneHotEncoder(handle_unknown='error', drop='if_binary'))])

preprocessor = ColumnTransformer(transformers=[('num', num_pipe, num_feats),
                                               ('cat', cat_pipe, cat_feats)])

# ! VERIFY MAX AND MIN FOR THE PARAM_GRID !
# CatBoost Grid Search
param_grid = {'classifier__iterations': [50, 100, 150], 
              'classifier__depth': [4, 5, 6], 
              'classifier__learning_rate':  [0.1, 0.15, 0.2]}
cat_model = CatBoostClassifier(silent=True)
cat_clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier',  cat_model)])

grid_pipe = GridSearchCV(cat_clf, param_grid, n_jobs=4)

''' ----------------------------------------------------------
#########################   FIT  #############################
 ---------------------------------------------------------- '''
 
grid_pipe.fit(X_train_2, y_train)

''' ----------------------------------------------------------
#########################   FIT  #############################
 ---------------------------------------------------------- '''
 
print('Best params:')
print(grid_pipe.best_params_)
print('Best CV score:')
print(grid_pipe.best_score_)

cat_clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier',  CatBoostClassifier(silent=True,
                                                         iterations=grid_pipe.best_params_['classifier__iterations'],
                                                         depth=grid_pipe.best_params_['classifier__depth'], 
                                                         learning_rate=grid_pipe.best_params_['classifier__learning_rate']))])

# CatBoost with best params
cat_clf.fit(X_train_2, y_train)

print('Train: {:.3f}'.format(cat_clf.score(X_train_2, y_train)))
print('Test: {:.3f}'.format(cat_clf.score(X_test_2, y_test)))

# Plot feature importance
col_names = num_feats + np.ndarray.tolist(preprocessor.transformers_[1][1]\
                                                      .named_steps['onehot'].get_feature_names(cat_feats))
feat_importance = cat_clf.named_steps['classifier'].get_feature_importance(type='PredictionValuesChange')
plt.figure(figsize=(15,4))
pd.Series(feat_importance, index=col_names).sort_values(ascending=False).plot.bar();
plt.title('Prediction Values Change', fontdict={'weight': 'bold', 'size':20});
plt.ylabel('Relative Importance', fontdict={'weight': 'bold', 'size':16});
plt.xlabel('Features', fontdict={'weight': 'bold', 'size':16});

# Create Submission
test_2 = Feature_Eng(test)
result = pd.DataFrame(list(zip(test_2.index, cat_clf.predict(test_2))), columns=['PassengerId', 'Transported'])
result.Transported = result.Transported.astype(bool)
result.to_csv('submission.csv', index=False)






