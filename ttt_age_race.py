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
st.title('Riskovid')
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
    
    # age versus race versus survival

############################## !

    import plotly.figure_factory as ff

    all_df = pd.concat([final_train, final_test], sort=False)
    all_df

    all_df[all_df['Survived'] == 1]
    age_survived = all_df[all_df['Survived'] == 1]['Age']
    age_not_survived = all_df[all_df['Survived'] == 0]['Age']

    hist_data = [age_survived.to_numpy(), age_not_survived.to_numpy()]
    group_labels = ['age_survived', 'age_not_survived']

    #fig = ff.create_distplot(hist_data, group_labels)
    fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])

    st.plotly_chart(fig, use_container_width=True)


############################## !
'''
# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2

# Group data together
hist_data = [x1, x2, x3]
group_labels = ['Group 1', 'Group 2', 'Group 3']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])

# Plot!
st.plotly_chart(fig, use_container_width=True)

'''