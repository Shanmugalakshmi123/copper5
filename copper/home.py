from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import re
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score
url="https://docs.google.com/spreadsheets/d/18eR6DBe5TMWU9FnIewaGtsepDbV4BOyr/edit#gid=462557918"
def convert_google_sheet_url(url):
    # Regular expression to match and capture the necessary part of the URL
    pattern = r'https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)(/edit#gid=(\d+)|/edit.*)?'

    # Replace function to construct the new URL for CSV export
    # If gid is present in the URL, it includes it in the export URL, otherwise, it's omitted
    replacement = lambda m: f'https://docs.google.com/spreadsheets/d/{m.group(1)}/export?' + (f'gid={m.group(3)}&' if m.group(3) else '') + 'format=csv'

    # Replace using regex
    new_url = re.sub(pattern, replacement, url)

    return new_url

def clean_columns(df):
    df2=df.drop('id',axis=1)
    df2=df2[df2['application'].notnull()]
    df2=df2.drop('material_ref',axis=1)
    df2=df2[df2['delivery date'].notnull()]
    df2=df2[df2['item_date'].notnull()]
    df2=df2[df2['country'].notnull()]
    df2=df2[df2['status'].notnull()]
    df2=df2[df2['customer'].notnull()]
    df2=df2[df2['thickness'].notnull()]
    df2=df2[df2['selling_price'].notnull()]
    df3=df2[df2['selling_price']<100000000]
    df4=df3[df3['selling_price']<70000]
    df5=df4[df4['status']=="Won"]
    df6=df4[df4['status']=="Lost"]
    df7=pd.concat([df5,df6])
    
    return df7
def smot_sample(df4,x_train,y_train):
    sm=SMOTE(random_state=0,k_neighbors=6)
    x_train_sm,y_train_sm=sm.fit_resample(x_train,y_train)
    return x_train_sm,y_train_sm
def sample_data(df4):
    won,lost,not_lost_for_am,revised,to_be_approved,draft,offered,offerable,wonderful=df4.status.value_counts()
    df_won=df4[df4['status']=='Won']
    df_lost=df4[df4['status']=='Lost']
    df_not_lost_for_am=df4[df4['status']=='Not lost for AM']
    df_revised=df4[df4['status']=='Revised']
    df_to_be_approved=df4[df4['status']=='To be approved']
    df_draft=df4[df4['status']=='Draft']
    df_offered=df4[df4['status']=='Offered']
    df_offerable=df4[df4['status']=='Offerable']
    df_wonderful=df4[df4['status']=='Wonderful']

    df_won=df_won.sample(lost)
    #df_lost=df_lost.sample(lost)
    df_not_lost_for_am=df_not_lost_for_am.sample(lost,replace=True)
    df_revised=df_revised.sample(lost,replace=True)
    df_to_be_approved=df_to_be_approved.sample(lost,replace=True)
    df_draft=df_draft.sample(lost,replace=True)
    df_offered=df_offered.sample(lost,replace=True)
    df_offerable=df_offerable.sample(lost,replace=True)
    df_wonderful=df_wonderful.sample(lost,replace=True)


    new_df=pd.concat([df_won,df_lost,df_not_lost_for_am,df_revised,df_to_be_approved,df_draft,df_offered,df_offerable,df_wonderful],axis=0)
    return new_df
def status_classifier(new_df):
    x=new_df.iloc[:,[0,2,6,7,8,9,10,11]]
    y=new_df.iloc[:,[4]]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    classifier=RandomForestClassifier()
    classifier.fit(x_train,y_train)
    return classifier,x_train,y_train,x_test,y_test
def des_tree(new_df):
    x=new_df.iloc[:,[0,2,6,7,8,9,10,11]]
    y=new_df.iloc[:,[4]]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    classifier=DecisionTreeClassifier()
    classifier.fit(x_train,y_train)
    return classifier,x_train,y_train,x_test,y_test
def ada_boost(new_df):
    x=new_df.iloc[:,[0,2,6,7,8,9,10,11]]
    y=new_df.iloc[:,[4]]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    classifier=AdaBoostClassifier()
    classifier.fit(x_train,y_train)
    return classifier,x_train,y_train,x_test,y_test
def grad_boost(new_df):
    x=new_df.iloc[:,[0,2,6,7,8,9,10,11]]
    y=new_df.iloc[:,[4]]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    classifier=GradientBoostingClassifier()
    classifier.fit(x_train,y_train)
    return classifier,x_train,y_train,x_test,y_test
def knn(new_df):
    x=new_df.iloc[:,[0,2,6,7,8,9,10,11]]
    y=new_df.iloc[:,[4]]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    classifier=KNeighborsClassifier()
    classifier.fit(x_train,y_train)
    return classifier,x_train,y_train,x_test,y_test
def predict_status(classifier,x_test,y_test):
    y_pred=classifier.predict(x_test)
    cm=confusion_matrix(y_test,y_pred)
    acc=accuracy_score(y_test,y_pred)
    return cm,acc,y_pred


def predict_status1(classifier,x_test):
    y_pred=classifier.predict(x_test)
    return y_pred
def sp_identifier(new_df):
    x=new_df.iloc[:,[0,2,6,7,8,9,10]]
    y=new_df.iloc[:,[11]] 
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
    regressor=LinearRegression()
    regressor.fit(x_train,y_train)
    return regressor,x_train,y_train,x_test,y_test
def log_regressor(new_df):
    x=new_df.iloc[:,[0,2,6,7,8,9,10]]
    y=new_df.iloc[:,[11]] 
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
    regressor=LogisticRegression()
    regressor.fit(x_train,y_train)
    return regressor,x_train,y_train,x_test,y_test
def predict_sp(regressor,x_test,y_test):
   y_pred=regressor.predict(x_test)
   cutoff = 0.7                              # decide on a cutoff limit
   y_pred_classes = np.zeros_like(y_pred)    # initialise a matrix full with zeros
   y_pred_classes[y_pred > cutoff] = 1       # add a 1 if the cutoff was breached
   y_test_classes = np.zeros_like(y_pred)
   y_test_classes[y_test > cutoff] = 1
   cm=confusion_matrix(y_test_classes,y_pred_classes)
   acc=accuracy_score(y_test_classes,y_pred_classes)
   return cm,acc,y_pred 
def predict_sp1(regressor,x_test):
    y_pred=regressor.predict(x_test)
    return y_pred

original_title = '<p style="font-family:Courier; color:Orange; font-size: 60px;">Copper Modelling</p>'
st.markdown(original_title, unsafe_allow_html=True)
#st.write("Copper Modelling",)
c1,c2,c3=st.columns(3)
item_date1=c1.text_input('item_date',"20200702")
customer1=c2.text_input('customer',"30200854")
#country1=c3.text_input('country')
c4,c5,c6=st.columns(3)
application1=c4.text_input('application',"41")
thickness1=c5.text_input('thickness',"0.71")
width1=c6.text_input('width',"1240")
c7,c8,c9=st.columns(3)
product_ref1=c7.text_input('product_ref',"164141591")
delivery_date1=c8.text_input('delivery date',"20200701")
selling_price1=c9.text_input('selling_price',"607")
c10,c11=st.columns(2)

typ=c10.selectbox('Type',('Random Forest','Decision Tree','Ada Boost','Gradient Boost','KNN'))
test=c10.button('Predict status')
if test:
    new_url=convert_google_sheet_url(url)
    df=pd.read_csv(new_url)
    df4=clean_columns(df)
    #new_df=sample_data(df4)
    #new_df=smot_sample(df4)
    if typ=="Random Forest":
        classifier,x_train,y_train,x_test,y_test=status_classifier(df4)
    if typ=="Decision Tree":
        classifier,x_train,y_train,x_test,y_test=des_tree(df4)
    if typ=="Ada Boost":
        classifier,x_train,y_train,x_test,y_test=ada_boost(df4)
    if typ=="Gradient Boost":
        classifier,x_train,y_train,x_test,y_test=grad_boost(df4)
    if typ=="KNN":
        classifier,x_train,y_train,x_test,y_test=knn(df4)
    classifier,x_train,y_train,x_test,y_test=status_classifier(df4)
    x_train,y_train=smot_sample(df4,x_train,y_train)
    cm,acc,y_pred=predict_status(classifier,x_test,y_test)
    print(cm)
    st.write("Accuracy: ",acc)
    x_test_data=[item_date1,customer1,application1,thickness1,width1,product_ref1,delivery_date1,selling_price1]
    x_test_data=pd.DataFrame([x_test_data])
    x_test_data.columns=['item_date','customer','application','thickness','width','product_ref','delivery date','selling_price']
    y_pred=predict_status1(classifier,x_test_data)
    y_pred1=y_pred[0]
    #y_pred1=str(y_pred)
    #y_pred1.replace('[','')
    #y_pred1.replace(']','')
    
    #st.write(y_pred)
    if y_pred1=="Won":
        original_title = '<p style="font-family:Courier; color:Green; font-size: 60px;">'+y_pred1+'</p>'
        c10.markdown(original_title, unsafe_allow_html=True)
        #c10.write("Won")
    else:
        original_title = '<p style="font-family:Courier; color:Red; font-size: 60px;">'+y_pred1+'<p>'
        c10.markdown(original_title, unsafe_allow_html=True)
        #st.write("Lost")
#reg_typ=c11.selectbox('Type',('Linear Regression'))
test1=c11.button('Predict selling price')
if test1:
    new_url=convert_google_sheet_url(url)
    df=pd.read_csv(new_url)
    df4=clean_columns(df)
    #new_df=sample_data(df4)
    #if reg_typ=="Linear Regression":
    regressor,x_train,y_train,x_test,y_test=sp_identifier(df4)
    #if reg_typ=="Logistic Regression":
       # regressor,x_train,y_train,x_test,y_test=log_regressor(df4)
    #x_train,y_train=smot_sample(df4,x_train,y_train)
    
    cm,acc,y_pred=predict_sp(regressor,x_test,y_test)
    print(cm)
    print(acc)
    
    # cm,acc,y_pred=predict_sp(regressor,x_train,y_train)
    # print(cm)
    # print(acc)
    x_test_data=[item_date1,customer1,application1,thickness1,width1,product_ref1,delivery_date1]
    x_test_data=pd.DataFrame([x_test_data])
    x_test_data.columns=['item_date','customer','application','thickness','width','product_ref','delivery date']
    y_pred=predict_sp1(regressor,x_test_data)
    y_pred1=str(round(y_pred[0][0],2))
    y_pred1.replace("[","")
    y_pred1.replace("]","")
    original_title = '<p style="font-family:Courier; color:Pink; font-size: 60px;">'+y_pred1+'</p>'
    c11.markdown(original_title, unsafe_allow_html=True)
    #c12,c13=st.columns(2)
    st.write("Accuracy: ",acc)
    #st.write(y_pred)
