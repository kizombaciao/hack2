import numpy as np 
import pandas as pd 

from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)

import warnings
warnings.simplefilter(action='ignore')

import time
import streamlit as st
############################## !

############################## !

st.title('Risovid')
st.header('Back To The Living Room')
st.subheader('subHeader !!!')

st.sidebar.header("Sidebar Header")
st.sidebar.text("Sidebar Text")

zipcode = st.sidebar.selectbox(
    'Location',
    ('Manhattan', 'Queens', 'Bronx')
)
st.write(zipcode)

gender = st.sidebar.selectbox(
    'Gender',
    ('Female', 'Male')
)
st.write(gender)

age = st.sidebar.slider("Age", 0, 80)
st.write(age)

ethnic = st.sidebar.selectbox(
    'Ethnicity',
    ('Black', 'Hispanic', 'White')
)
st.write(ethnic)

condition = st.sidebar.multiselect("Prior Medical Condition", ("Hypertension", "Obesity", "Chronic Lung Disease", "Diabetes", "Cardiovascular Disease"))
st.write("You Selected: ", len(condition), "condition")

if len(condition) > 0:
  my_travelalone = 1 
else:
  my_travelalone = 0
my_travelalone

if ethnic == 'Black':
    my_pclass_1 = 0
    my_pclass_2 = 0
elif ethnic == 'Hispanic':
    my_pclass_1 = 0
    my_pclass_2 = 1
else:
    my_pclass_1 = 1
    my_pclass_2 = 0    
st.write(my_pclass_1, my_pclass_2)

if zipcode == 'Manhattan':
    my_embarked_c = 1
    my_embarked_s = 0
elif zipcode == 'Queens':
    my_embarked_c = 0
    my_embarked_s = 0
else:
    my_embarked_c = 0
    my_embarked_s = 1
st.write(my_embarked_c, my_embarked_s)

if gender == 'Male':
    sex_male = 1
else:
    sex_male = 0

if age <= 16:
    my_isminor = 1
else:
    my_isminor = 0

input_data = np.array([[age, my_travelalone, my_pclass_1, my_pclass_2, my_embarked_c, my_embarked_s, sex_male, my_isminor]])
st.write(input_data)


############################## !

if st.sidebar.button("Update"):

    train_df = pd.read_csv("train.csv", sep=',')
    test_df = pd.read_csv("test.csv", sep=',')
    #train_df
    
    #st.write('The number of samples into the train data is {}.'.format(train_df.shape[0]))
    #st.write('The number of samples into the test data is {}.'.format(test_df.shape[0]))
    
    #st.write('check missing values in train data')
    #st.write(train_df.isnull().sum())
    
    #st.write('percent of missing "Age"')
    #st.write('Percent of missing "Age" records is %.2f%%' %((train_df['Age'].isnull().sum()/train_df.shape[0]) *100))
    
    # mean age
    #st.write('The mean of "Age" is %.2f' %(train_df["Age"].mean(skipna=True)))
    # median age
    #st.write('The median of "Age" is %.2f' %(train_df["Age"].median(skipna=True)))
    
    # percent of missing "Cabin" 
    #st.write('Percent of missing "Cabin" records is %.2f%%' %((train_df['Cabin'].isnull().sum()/train_df.shape[0]  )*100))
    
    # percent of missing "Embarked" 
    #st.write('Percent of missing "Embarked" records is %.2f%%' %((train_df['Embarked'].isnull().sum()/train_df.    shape[0])*100))
    
    #st.write('Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)   :')
    #st.write(train_df['Embarked'].value_counts())
    #st.bar_chart(train_df['Embarked'])
    
    # set missing Age with median Age
    # set missing Embarked with 'S', the most common one
    # drop Cabin
    train_data = train_df.copy()
    train_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
    train_data["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)
    train_data.drop('Cabin', axis=1, inplace=True)
    
    #st.write('check missing values in adjusted train data')
    #st.write(train_data.isnull().sum())
    
    # preview adjusted train data
    #train_data
    
    ## Create categorical variable for traveling alone
    train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)
    train_data.drop('SibSp', axis=1, inplace=True)
    train_data.drop('Parch', axis=1, inplace=True)
    #train_data
    
    #create categorical variables and drop some variables
    training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])
    training.drop('Sex_female', axis=1, inplace=True)
    training.drop('PassengerId', axis=1, inplace=True)
    training.drop('Name', axis=1, inplace=True)
    training.drop('Ticket', axis=1, inplace=True)
    final_train = training
    #final_train
    
    #st.write('Now, apply the same changes to the test data.')
    #st.write(test_df.isnull().sum())
    
    test_data = test_df.copy()
    test_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
    test_data["Fare"].fillna(train_df["Fare"].median(skipna=True), inplace=True)
    test_data.drop('Cabin', axis=1, inplace=True)
    
    test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)
    
    test_data.drop('SibSp', axis=1, inplace=True)
    test_data.drop('Parch', axis=1, inplace=True)
    
    testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
    testing.drop('Sex_female', axis=1, inplace=True)
    testing.drop('PassengerId', axis=1, inplace=True)
    testing.drop('Name', axis=1, inplace=True)
    testing.drop('Ticket', axis=1, inplace=True)
    
    final_test = testing
    #final_test
    
    st.header('EDA')
    final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)
    final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)
    #final_test
    
    st.header('Logistic Regression')
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import RFE
    
    cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
    X = final_train[cols]
    y = final_train['Survived']
    # Build a logreg and compute the feature importances
    model = LogisticRegression()
    # create the RFE model and select 8 attributes
    rfe = RFE(model, 8)
    rfe = rfe.fit(X, y)
    # summarize the selection of the attributes
    st.write('Selected features: %s' % list(X.columns[rfe.support_]))
    
    from sklearn.feature_selection import RFECV
    # Create the RFE object and compute a cross-validated score.
    # The "accuracy" scoring is proportional to the number of correct classifications
    rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
    rfecv.fit(X, y)
    st.write("Optimal number of features: %d" % rfecv.n_features_)
    st.write('Selected features: %s' % list(X.columns[rfecv.support_]))
    
    Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C', 
                         'Embarked_S', 'Sex_male', 'IsMinor']
    X = final_train[Selected_features]
    
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
    
    # create X (features) and y (response)
    X = final_train[Selected_features]
    y = final_train['Survived']
    
    # use train/test split with different random_state values
    # we can change the random_state values that changes the accuracy scores
    # the scores change a lot, this is why testing scores is a high-variance estimate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    # check classification scores of logistic regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_pred_proba = logreg.predict_proba(X_test)[:, 1]
    [fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
    st.write('Train/Test split results:')
    st.write(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
    st.write(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
    st.write(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))
    
    idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95
    idx
    
    # 10-fold cross-validation logistic regression
    logreg = LogisticRegression()
    # Use cross_val_score function
    # We are passing the entirety of X and y, not X_train or y_train, it takes care of splitting the data
    # cv=10 for 10 folds
    # scoring = {'accuracy', 'neg_log_loss', 'roc_auc'} for evaluation metric - althought they are many
    scores_accuracy = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
    scores_log_loss = cross_val_score(logreg, X, y, cv=10, scoring='neg_log_loss')
    scores_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc')
    st.write('K-fold cross-validation results:')
    st.write(logreg.__class__.__name__+" average accuracy is %2.3f" % scores_accuracy.mean())
    st.write(logreg.__class__.__name__+" average log_loss is %2.3f" % -scores_log_loss.mean())
    st.write(logreg.__class__.__name__+" average auc is %2.3f" % scores_auc.mean())
    
    from sklearn.model_selection import cross_validate
    
    scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc'}
    
    modelCV = LogisticRegression()
    
    results = cross_validate(modelCV, X, y, cv=10, scoring=list(scoring.values()), 
                             return_train_score=False)
    
    st.write('K-fold cross-validation results:')
    for sc in range(len(scoring)):
        st.write(modelCV.__class__.__name__+" average %s: %.3f (+/-%.3f)" % (list(scoring.keys())[sc], -results ['test_%s' % list(scoring.values())[sc]].mean()
                                   if list(scoring.values())[sc]=='neg_log_loss' 
                                   else results['test_%s' % list(scoring.values())[sc]].mean(), 
                                   results['test_%s' % list(scoring.values())[sc]].std()))
    
    # What happens when we add the feature "Fare"?
    cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"]
    X = final_train[cols]
    
    scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc'}
    
    modelCV = LogisticRegression()
    
    results = cross_validate(modelCV, final_train[cols], y, cv=10, scoring=list(scoring.values()), 
                             return_train_score=False)
    
    st.write('K-fold cross-validation results:')
    for sc in range(len(scoring)):
        st.write(modelCV.__class__.__name__+" average %s: %.3f (+/-%.3f)" % (list(scoring.keys())[sc], -results ['test_%s' % list(scoring.values())[sc]].mean()
                                   if list(scoring.values())[sc]=='neg_log_loss' 
                                   else results['test_%s' % list(scoring.values())[sc]].mean(), 
                                   results['test_%s' % list(scoring.values())[sc]].std()))
    
    from sklearn.model_selection import GridSearchCV
    
    X = final_train[Selected_features]
    
    param_grid = {'C': np.arange(1e-05, 3, 0.1)}
    scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}
    
    gs = GridSearchCV(LogisticRegression(), return_train_score=True,
                      param_grid=param_grid, scoring=scoring, cv=10, refit='Accuracy')
    
    gs.fit(X, y)
    results = gs.cv_results_
    
    st.write('='*20)
    st.write("best params: " + str(gs.best_estimator_))
    st.write("best params: " + str(gs.best_params_))
    st.write('best score:', gs.best_score_)
    st.write('='*20)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import Pipeline
    
    #Define simple model
    ###############################################################################
    C = np.arange(1e-05, 5.5, 0.1)
    scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}
    log_reg = LogisticRegression()
    
    #Simple pre-processing estimators
    ###############################################################################
    std_scale = StandardScaler(with_mean=False, with_std=False)
    #std_scale = StandardScaler()
    
    #Defining the CV method: Using the Repeated Stratified K Fold
    ###############################################################################
    
    n_folds=5
    n_repeats=5
    
    rskfold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=2)
    
    #Creating simple pipeline and defining the gridsearch
    ###############################################################################
    
    log_clf_pipe = Pipeline(steps=[('scale',std_scale), ('clf',log_reg)])
    
    log_clf = GridSearchCV(estimator=log_clf_pipe, cv=rskfold,
                  scoring=scoring, return_train_score=True,
                  param_grid=dict(clf__C=C), refit='Accuracy')
    
    log_clf.fit(X, y)
    results = log_clf.cv_results_
    
    st.write('='*20)
    st.write("best params: " + str(log_clf.best_estimator_))
    st.write("best params: " + str(log_clf.best_params_))
    st.write('best score:', log_clf.best_score_)
    st.write('='*20)
    
    final_test['Survived'] = log_clf.predict(final_test[Selected_features])
    final_test['PassengerId'] = test_df['PassengerId']
    
    submission = final_test[['PassengerId','Survived']]
    
    submission.to_csv("submission.csv", index=False)
    
    st.write(submission.tail())
    final_test[Selected_features]
    
    ############################## !
    #Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C', 
    #                     'Embarked_S', 'Sex_male', 'IsMinor']
    
    df2 = pd.DataFrame(input_data, columns=Selected_features)
    df2
    y_pred_proba = log_clf.predict_proba(df2[Selected_features])
    #st.write(dir(log_clf))
    df2['Survived'] = log_clf.predict(df2[Selected_features])
    df2
    st.write('RISK SCORE IS:  ')
    st.write(y_pred_proba[:, 0])
    
    
    # note, privacy text