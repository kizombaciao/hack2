import streamlit as st

import numpy as np 
import pandas as pd

############################## !
def family(x):
    if x < 2:
        return 'Single'
    elif x == 2:
        return 'Couple'
    elif x <= 4:
        return 'InterM'
    else:
        return 'Large'
    
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

    dataset = pd.read_csv("train.csv", sep=',')
    X_test = pd.read_csv("test.csv", sep=',')
    X_test

    dataset_title = [i.split(',')[1].split('.')[0].strip() for i in dataset['Name']]
    dataset['Title'] = pd.Series(dataset_title)
    dataset['Title'].value_counts()
    dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess', 'Countess', 'Capt',    'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Ms', 'Mme', 'Mlle'],   'Rare')

    dataset_title = [i.split(',')[1].split('.')[0].strip() for i in X_test['Name']]
    X_test['Title'] = pd.Series(dataset_title)
    X_test['Title'].value_counts()
    X_test['Title'] = X_test['Title'].replace(['Lady', 'the Countess', 'Countess', 'Capt',  'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Ms', 'Mme', 'Mlle'],     'Rare')

    dataset['FamilyS'] = dataset['SibSp'] + dataset['Parch'] + 1
    X_test['FamilyS'] = X_test['SibSp'] + X_test['Parch'] + 1

    dataset
    X_test

    dataset['FamilyS'] = dataset['FamilyS'].apply(family)
    X_test['FamilyS'] = X_test['FamilyS'].apply(family)

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    X_test['Embarked'].fillna(X_test['Embarked'].mode()[0], inplace=True)
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    X_test['Age'].fillna(X_test['Age'].median(), inplace=True)
    X_test['Fare'].fillna(X_test['Fare'].median(), inplace=True)

    dataset = dataset.drop(['PassengerId', 'Cabin', 'Name', 'SibSp', 'Parch', 'Ticket'],    axis=1)
    X_test_passengers = X_test['PassengerId']
    X_test = X_test.drop(['PassengerId', 'Cabin', 'Name', 'SibSp', 'Parch', 'Ticket'], axis=1)

    dataset
    X_test

    X_train = dataset.iloc[:, 1:9].values
    Y_train = dataset.iloc[:, 0].values
    X_test = X_test.values

    # Converting the remaining labels to numbers
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X_train[:, 1] = labelencoder_X_1.fit_transform(X_train[:, 1])
    X_train[:, 4] = labelencoder_X_1.fit_transform(X_train[:, 4])
    X_train[:, 5] = labelencoder_X_1.fit_transform(X_train[:, 5])
    X_train[:, 6] = labelencoder_X_1.fit_transform(X_train[:, 6])

    labelencoder_X_2 = LabelEncoder()
    X_test[:, 1] = labelencoder_X_2.fit_transform(X_test[:, 1])
    X_test[:, 4] = labelencoder_X_2.fit_transform(X_test[:, 4])
    X_test[:, 5] = labelencoder_X_2.fit_transform(X_test[:, 5])
    X_test[:, 6] = labelencoder_X_2.fit_transform(X_test[:, 6])

    X_train
    X_test

    import sklearn
    
    # Converting categorical values to one-hot representation
    one_hot_encoder = OneHotEncoder(categorical_features = [0, 1, 4, 5, 6])
    X_train = one_hot_encoder.fit_transform(X_train).toarray()
    X_test = one_hot_encoder.fit_transform(X_test).toarray()

    from sklearn.model_selection import train_test_split

    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.1)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(19, 270)
            self.fc2 = nn.Linear(270, 2)

        def forward(self, x):
            x = self.fc1(x)
            x = F.dropout(x, p=0.1)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.sigmoid(x)

            return x

    net = Net()
