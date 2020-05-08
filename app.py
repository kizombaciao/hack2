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
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torch.autograd import Variable
import torch.nn.functional as F
import sklearn.linear_model as lm
from sklearn.model_selection import GridSearchCV
############################## !

st.title('Riskovid')
st.header('Back To The Living Room')
st.subheader('subHeader !!!')

st.sidebar.header("Sidebar Header")
st.sidebar.text("Sidebar Text")

zipcode = st.sidebar.selectbox(
    'Location',
    ('Manhattan', 'Queens', 'Bronx')
)
#st.write(zipcode)

gender = st.sidebar.selectbox(
    'Gender',
    ('Female', 'Male')
)
#st.write(gender)

age = st.sidebar.slider("Age", 0, 80)
#st.write(age)

ethnic = st.sidebar.selectbox(
    'Ethnicity',
    ('Black', 'Hispanic', 'White')
)
#st.write(ethnic)

condition = st.sidebar.multiselect("Prior Medical Condition", ("Hypertension", "Obesity", "Chronic Lung Disease", "Diabetes", "Cardiovascular Disease"))
#st.write("You Selected: ", len(condition), "condition")

if len(condition) > 0:
  my_travelalone = 1 
else:
  my_travelalone = 0
#my_travelalone

if ethnic == 'Black':
    my_pclass_1 = 0
    my_pclass_2 = 0
elif ethnic == 'Hispanic':
    my_pclass_1 = 0
    my_pclass_2 = 1
else:
    my_pclass_1 = 1
    my_pclass_2 = 0    
#st.write(my_pclass_1, my_pclass_2)

if zipcode == 'Manhattan':
    my_embarked_c = 1
    my_embarked_s = 0
elif zipcode == 'Queens':
    my_embarked_c = 0
    my_embarked_s = 0
else:
    my_embarked_c = 0
    my_embarked_s = 1
#st.write(my_embarked_c, my_embarked_s)

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
    #test_df = pd.read_csv("test.csv", sep=',')

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
    
    #test_data = test_df.copy()
    #test_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
    #test_data["Fare"].fillna(train_df["Fare"].median(skipna=True), inplace=True)
    #test_data.drop('Cabin', axis=1, inplace=True)
    
    #test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)
    
    #test_data.drop('SibSp', axis=1, inplace=True)
    #test_data.drop('Parch', axis=1, inplace=True)
    
    #testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
    #testing.drop('Sex_female', axis=1, inplace=True)
    #testing.drop('PassengerId', axis=1, inplace=True)
    #testing.drop('Name', axis=1, inplace=True)
    #testing.drop('Ticket', axis=1, inplace=True)
    
    #final_test = testing
    #final_test
    
    #st.header('EDA')
    final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)
    #final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)
    #final_test
    
    cols = ["Age","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
    X = final_train[cols]
    y = final_train['Survived']    

    n_samples, n_features = X.shape
    #(n_samples, n_features)

    X_train = X.to_numpy()
    y_train = y.to_numpy()
    #X_test = final_test[cols].to_numpy()

    X_train = torch.from_numpy(X_train.astype(np.float32))
    #X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    #y_test = torch.from_numpy(y_test.astype(np.float32))

    y_train = y_train.view(y_train.shape[0], 1)

    class Model(nn.Module):
        def __init__(self, n_input_features):
            super(Model, self).__init__()
            self.linear = nn.Linear(n_input_features, 1)

        def forward(self, x):
            y_pred = torch.sigmoid(self.linear(x))
            return y_pred

    model = Model(n_features)
    
    # 2) Loss and optimizer
    num_epochs = 40000
    learning_rate = 0.0045
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 3) Training loop
    for epoch in range(num_epochs):
        # Forward pass and loss
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # zero grad before new step
        optimizer.zero_grad()

        if (epoch+1) % 10000 == 0:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

    df2 = pd.DataFrame(input_data, columns=cols)

    my_test = df2[cols].to_numpy()
    my_test = torch.from_numpy(my_test.astype(np.float32))
    my_predict = model(my_test)

    st.write('RISK SCORE IS:  ')
    #my_predict[0]
    1 - my_predict.tolist()[0][0]

    st.write('From 0 to 0,3: « You’re safe to go but don’t forget the barrier gestures »')
    st.write('From 0,4 to 0,7: « You’re okay so be careful if you have to go outside »')
    st.write('From 0,8 to 1: « You’re highly at risks you’re better stay at home »')
