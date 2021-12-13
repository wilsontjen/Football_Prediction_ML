from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

st.title('Football Match Prediction Using Machine Learning')
st.write(""" 
#### By: Wilson Tjendana 
""")
st.write('##')

from PIL import Image
image = Image.open('Leceister.jpeg')
st.image(image)

# add a select-box
classifier_name = st.sidebar.selectbox('Select Classifier Type', ('Logistic Regression', 'Support Vector Machine', 'Random Forest Classifier', 'Gradient Boosting Classifier'))

def get_dataset():
    df = pd.read_csv('final_dataset.csv')
    FTR_history = df[['HomeTeam','AwayTeam','FTR']]
    df = df[df.MW > 3]
    df = df[['FTR','HTP','ATP','HM1','HM2','HM3','AM1','AM2','AM3','HTGD','ATGD','DiffFormPts','DiffLP']]
    X = df.drop(['FTR'],axis=1)
    y = df['FTR']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    X_train_categorical = X_train[['HM1','HM2','HM3','AM1','AM2','AM3']]
    X_test_categorical = X_test[['HM1','HM2','HM3','AM1','AM2','AM3']]
    X_train_numerical = X_train[['HTP','ATP','HTGD','ATGD','DiffFormPts','DiffLP']]
    X_test_numerical = X_test[['HTP','ATP','HTGD','ATGD','DiffFormPts','DiffLP']]
    X_train_categorical = pd.get_dummies(X_train_categorical,drop_first=True)
    X_test_categorical = pd.get_dummies(X_test_categorical,drop_first=True)
    scaler = StandardScaler()
    X_train_numerical = scaler.fit_transform(X_train_numerical) 
    X_test_numerical = scaler.transform(X_test_numerical)
    X_train_numerical_df = pd.DataFrame(X_train_numerical, columns=['HTP','ATP','HTGD','ATGD','DiffFormPts','DiffLP'])
    X_test_numerical_df = pd.DataFrame(X_test_numerical, columns=['HTP','ATP','HTGD','ATGD','DiffFormPts','DiffLP'])
    X_train = pd.merge(X_train_numerical_df, X_train_categorical, left_index=True, right_index=True)
    X_test = pd.merge(X_test_numerical_df, X_test_categorical, left_index=True, right_index=True)
    return df, FTR_history, X_train, X_test, y_train, y_test

df, FTR_history, X_train, X_test, y_train, y_test = get_dataset()

st.write('##')
st.subheader('About EPL (English Premiere League)')
st.write('There are 380 matches for every season of the English Premier League.')
st.write('The season run from August to May, and each team is playing 38 matches (19 home and 19 away games) for every season.')
st.write('The purpose of this project is to find whether the winner will be the Home Team, the Away Team or Draw.')
st.write('Note that I am using the historical data from season 2005/06 - 2020/21.')

st.write('##')
st.subheader('Full Datasets')
st.dataframe(df)
st.write('Shape of train datasets', X_train.shape)
st.write('Shape of test datasets', X_test.shape)
st.write('Number of classes', len(np.unique(y_train)))
st.write('FTR = Full Time Result')
st.write('HTP = Home Team Point (divided by MW)')
st.write('ATP = Away Team Point (divided by MW)')
st.write('HMx = match result from the last x game of the home team')
st.write('AMx = match restul from the last x game of the away team')
st.write('HTGD = Home Team Goal Difference (divided by MW)')
st.write('ATGD = Away Team Goal Difference (divided by MW)')
st.write('DiffFormPts = Difference of home team points and away team points (divided by MW)')
st.write('DiffLP = Difference in the last year position of the home team and the away team')

st.write('##')
st.subheader('Exploratory Data Analysis')
total_match = len(df)
no_of_homewins = len(df[df.FTR == 'H'])
no_of_awaywins = len(df[df.FTR == 'A'])
no_of_draw = len(df[df.FTR == 'D'])
home_win_rate = no_of_homewins / total_match
st.write(f'The winning rate for the home team is {round(home_win_rate*100,2)}%.')
bar_label = ['Home', 'Away', 'Draw']
bar_value = [no_of_homewins, no_of_awaywins, no_of_draw]
plt.figure(figsize=(10,5))
plt.bar(bar_label,bar_value,color=['blue','red','green'])
plt.title('Aggregrate Result Statistics (season 2005/06 - 2020/21)')
plt.ylabel('Frequency')
st.pyplot(plt)
st.write('We could clearly see a home-field advantage in a football match.')
st.write('Some of the possible explanations:')
st.write('1) They are more familiar with the pitch.')
st.write('2) Less travelling time, which gives every player more energy to play.')
st.write('3) The cheer from the crowd for the home team (or the booing for the away team).')

home_team_win_dict = {}
for team in FTR_history.HomeTeam.unique():
    home_team_win_dict[team] = len(FTR_history[(FTR_history.HomeTeam == team) & (FTR_history.FTR == 'H')])
home_team_win_dict = dict(sorted(home_team_win_dict.items(), key=lambda item: item[1]))
keys = home_team_win_dict.keys()
values = home_team_win_dict.values()
st.write('##')
plt.figure(figsize=(12,7))
plt.bar(keys,values,color='blue')
plt.ylabel('Number of Home Wins')
plt.xticks(rotation='vertical')
st.write('Number of Home Wins for every team from Season 2005/06 - 2020/21.')
st.pyplot(plt)

away_team_win_dict = {}
for team in FTR_history.AwayTeam.unique():
    away_team_win_dict[team] = len(FTR_history[(FTR_history.AwayTeam == team) & (FTR_history.FTR == 'A')])
away_team_win_dict = dict(sorted(away_team_win_dict.items(), key=lambda item: item[1]))
keys = away_team_win_dict.keys()
values = away_team_win_dict.values()
plt.figure(figsize=(12,7))
plt.bar(keys,values,color='red')
plt.ylabel('Number of Away Wins')
plt.xticks(rotation='vertical')
st.write('Number of Away Wins for every team from Season 2005/06 - 2020/21.')
st.pyplot(plt)

st.write('##')
st.write("Let's visualize how our features are interacting with each other:")
st.dataframe(df.corr())
pair_plot = sns.pairplot(df)
st.pyplot(pair_plot)

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'Logistic Regression':
        C = st.sidebar.select_slider('C', options=[0.001, 0.009, 0.01, 0.09, 1, 5, 10, 25])
        max_iter = st.sidebar.select_slider('max_iter', options=[1, 10, 100, 1000])
        params['C'] = C
        params['max_iter'] = max_iter

    elif clf_name == 'Support Vector Machine':
        C = st.sidebar.select_slider('C', options=[1, 10, 20, 50])
        gamma = st.sidebar.select_slider('gamma', options=[0.0001, 0.0005, 0.001, 0.005])
        kernel = st.sidebar.radio('kernel', options=['rbf', 'linear'])
        params['C'] = C
        params['gamma'] = gamma
        params['kernel'] = kernel
    
    elif clf_name == 'Random Forest Classifier':
        n_estimators = st.sidebar.slider('n_estimators', min_value=500, max_value=1000, step=100)
        max_features = st.sidebar.radio('max_features', options=['auto', 'sqrt'])
        max_depth = st.sidebar.slider('max_depth', min_value=2, max_value=4)
        min_samples_split = st.sidebar.slider('min_samples_split', min_value=2, max_value=5)
        min_samples_leaf = st.sidebar.slider('min_samples_leaf', min_value=1, max_value=2)
        bootstrap = st.sidebar.radio('bootstrap', options=[True, False])
        params['n_estimators'] = n_estimators
        params['max_features'] = max_features
        params['max_depth'] = max_depth
        params['min_samples_split'] = min_samples_split
        params['min_samples_leaf'] = min_samples_leaf
        params['bootstrap'] = bootstrap
    #Gradient Boosting Classifier
    else:
        learning_rate = st.sidebar.select_slider('learning_rate', options=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2])
        min_samples_split = st.sidebar.slider('min_samples_split', min_value=0.1, max_value=0.5, step=0.03333)
        min_samples_leaf = st.sidebar.slider('min_samples_leaf', min_value=1, max_value=2)
        max_depth = st.sidebar.slider('max_depth', min_value=3, max_value=8)
        max_features = st.sidebar.radio('max_features', options=['log2', 'sqrt'])
        subsample = st.sidebar.select_slider('subsample', options=[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0])
        n_estimators = st.sidebar.slider('n_estimators', min_value=1, max_value=10)
        params['learning_rate'] = learning_rate
        params['min_samples_split'] = min_samples_split
        params['min_samples_leaf'] = min_samples_leaf
        params['max_depth'] = max_depth
        params['max_features'] = max_features
        params['subsample'] = subsample
        params['n_estimators'] = n_estimators

    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == 'Logistic Regression':
        clf = LogisticRegression(
            C = params['C'], 
            max_iter = params['max_iter'])

    elif clf_name == 'Support Vector Machine':
        clf = SVC(
            C = params['C'],
            gamma = params['gamma'],
            kernel = params['kernel']
        )

    elif clf_name == 'Random Forest Classifier':
        clf = RandomForestClassifier(
            n_estimators = params['n_estimators'],
            max_features = params['max_features'],
            max_depth = params['max_depth'],
            min_samples_split = params['min_samples_split'],
            min_samples_leaf = params['min_samples_leaf'],
            bootstrap = params['bootstrap']
        )

    else:
        clf = GradientBoostingClassifier(
            learning_rate = params['learning_rate'],
            min_samples_split = params['min_samples_split'],
            min_samples_leaf = params['min_samples_leaf'],
            max_depth = params['max_depth'],
            max_features = params['max_features'],
            criterion = 'friedman_mse',
            subsample = params['subsample'],
            n_estimators = params['n_estimators']
        )

    return clf

clf = get_classifier(classifier_name,params)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test,y_pred)
report = (classification_report(y_test,y_pred))

if classifier_name == 'Random Forest Classifier':
    importance = clf.feature_importances_
    features = X_train.columns
    sorted_idx = importance.argsort()
    plt.figure(figsize=(10,5))
    sns.barplot(importance[sorted_idx], features[sorted_idx])
    plt.xlabel('Random Forest Feature Importance')
    plt.ylabel('Features')
    st.write('##')
    st.subheader('Feature Importance')
    st.write('Random Forest algorithm has built-in feature importance. They are computed as the mean and standard-deviation of accumulation of the impurity decrease within each tree.')
    st.write('Be aware that impurity-based feature importances can be misleading for high cardinality features (many unique values).')
    st.pyplot(plt)

st.write('#')
st.subheader(f'Machine Learning with {classifier_name} Model')
st.write(f'Accuracy Score = {round(acc,4)}')
st.text('Classification Report:\n ' + report)
st.write('Confusion Matrix')
plot_confusion_matrix(clf,X_test,y_test)
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write('##')
st.subheader('Conclusion')
st.write('As we can see from the accuracy score and the confusion matrix above, our results were not too impressive. Predicting the winner of a football match is not an easy task by any means.')
st.write('All of the classifiers achieved around 53% of accuracy, but this is much better than the probability of success for a random guess (33.33% for Win, Draw or Loss).')

st.write('##')
st.subheader('Possible Improvements')
st.write('The accuracy of the prediciton can further increase by adding more data from previous seasons, the play specific health stats, or even the Sentiment Analysis from Twitter.')
st.write('We could also investigate and find more features that could be useful for prediction, such as injuries or the individual player forms.')