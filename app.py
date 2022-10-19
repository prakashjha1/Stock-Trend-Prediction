import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model
from datetime import date
import streamlit as st
import models 

st.title('Stock Trend Prediction')

user_input = st.text_input("Enter the Stock Ticker", 'AAPL')
def fetch_data(user_input):
    start_date = st.text_input("Enter the Start Date in yy-mm-dd Format", '2010-01-01')
    end_date = st.text_input("Enter the Start Date in yy-mm-dd Format", date.today().isoformat())
    df = data.DataReader(user_input,'yahoo',start_date,end_date)

    return df,start_date,end_date

df,start,end = fetch_data(user_input)


def get_training_data(df,feature,days):

    data_training = pd.DataFrame(df.iloc[0:int(len(df)*0.80)][feature])
    data_testing = pd.DataFrame(df.iloc[int(len(df)*0.80):int(len(df))][feature])

    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)
    x_train = []
    y_train = []

    for i in range(days,data_training_array.shape[0]):
        x_train.append(data_training_array[i-days:i])
        y_train.append(data_training_array[i,0])
        
    x_train,y_train = np.array(x_train),np.array(y_train)

    return x_train,y_train,scaler

# Describing Data
st.subheader('Data from {}  to  {}'.format(start,end))
st.write(df.describe())


st.subheader('Stock Trend Analysis')

MA1 = st.text_input("Enter the First Moving Average", '5')
MA2 = st.text_input("Enter the Second Moving Average", '10')

col1,col2 = st.columns(2)
with col1:
    st.subheader("Close Trend Analysis with {} & {} Moving Average".format(MA1,MA2))
    ma1 = df.Close.rolling(int(MA1)).mean()
    ma2 = df.Close.rolling(int(MA2)).mean()
    fig = plt.figure(figsize=(7,5))
    plt.plot(ma1)
    plt.plot(ma2)
    plt.plot(df.Close)
    plt.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Open Trend Analysis with {} & {} Moving Average".format(MA1,MA2))
    ma3 = df.Open.rolling(int(MA1)).mean()
    ma4 = df.Open.rolling(int(MA2)).mean()
    fig = plt.figure(figsize=(7,5))
    plt.plot(ma3)
    plt.plot(ma4)
    plt.plot(df.Open)
    plt.legend()
    st.pyplot(fig)

col3,col4 = st.columns(2)
with col3:
    st.subheader("High Trend Analysis with {} & {} Moving Average".format(MA1,MA2))
    ma5 = df.High.rolling(int(MA1)).mean()
    ma6 = df.High.rolling(int(MA2)).mean()
    fig = plt.figure(figsize=(7,5))
    plt.plot(ma5)
    plt.plot(ma6)
    plt.plot(df.High)
    plt.legend()
    st.pyplot(fig)

with col4:
    st.subheader("Low Trend Analysis with {} & {} Moving Average".format(MA1,MA2))
    ma7 = df.Low.rolling(int(MA1)).mean()
    ma8 = df.Low.rolling(int(MA2)).mean()
    fig = plt.figure(figsize=(7,5))
    plt.plot(ma7)
    plt.plot(ma8)
    plt.plot(df.Low)
    plt.legend()
    st.pyplot(fig)

# Model Training
st.title("Model Training")
model_list = ['Open Trend Model','Close Trend Model','High Trend Model','Low Trend Model']
opt_list = ['sgd','adam','RMSprop','Adadelta','Adagrad','Nadam']
# c1,c2,c3,c4 = st.columns(4)

# with c1:
selected_model = st.selectbox("Choose Model to Train",model_list)

# Unit inputs 
U1 = st.text_input("Enter the Nodes in layer 1", '50',key='m1u1')
U2 = st.text_input("Enter the Nodes in layer 2", '60',key='m1u2')
U3 = st.text_input("Enter the Nodes in layer 3", '80',key='m1u3')
U4 = st.text_input("Enter the Nodes in layer 4", '120',key='m1u4')

# Dropout Input
D1 = st.text_input("Enter the value of dropout in layer 1", '0.2',key='m1d1')
D2 = st.text_input("Enter the value of dropout in layer 2", '0.3',key='m1d2')
D3 = st.text_input("Enter the value of dropout in layer 3", '0.4',key='m1d3')
D4 = st.text_input("Enter the value of dropout in layer 4", '0.5',key='m1d4')

# Epoch Input
EP1 = st.text_input("Enter the number of epochs", '10')

# Optimizer Input
OPT1 = st.selectbox("Select Optimizer",opt_list)

day = st.text_input("Enter number of previous days",'5')
day = int(day)

if st.button("Train"):


    if selected_model == 'Open Trend Model':

        x_train_open,y_train_open,open_scaler = get_training_data(df=df,feature='Open',days=day)
        m1 = models.model_1_train()
        open_output = m1.train_open_trend(x_train_open,y_train_open,int(EP1),OPT1,int(U1),int(U2),int(U3),int(U4),float(D1),float(D2),float(D3),float(D4))

        m2 = models.model_1_predict()
        result = m2.predict_open_trend(open_scaler,day,df)

        st.write("Loss of Open Trained Model : {}".format(open_output['loss']))

        st.write("Prediction : {}".format(result['output']))

        st.write("Correct Prediction : {}".format(result['correct']))

        st.write("Incorrect Prediction : {}".format(result['incorrect']))

        st.write("Total : {}".format(result['total']))

        st.write(pd.DataFrame({'Actual':result['actual'],'Predicted':result['predicted']}))

        st.subheader("Actual Vs Prediction Chart")
        fig = plt.figure(figsize=(12,6))
        plt.plot(result['actual'])
        plt.plot(result['predicted'])
        st.pyplot(fig)


    if selected_model == 'Close Trend Model':

        x_train_close,y_train_close,close_scaler = get_training_data(df=df,feature='Close',days=day)
        
        m1 = models.model_1_train()
        close_output = m1.train_close_trend(x_train_close,y_train_close,int(EP1),OPT1,int(U1),int(U2),int(U3),int(U4),float(D1),float(D2),float(D3),float(D4))

        m2 = models.model_1_predict()
        result = m2.predict_close_trend(close_scaler,day,df)

        st.write("Loss of Close Trained Model : {}".format(close_output['loss']))

        st.write("Prediction : {}".format(result['output']))

        st.write("Correct Prediction : {}".format(result['correct']))

        st.write("Incorrect Prediction : {}".format(result['incorrect']))

        st.write("Total : {}".format(result['total']))

        st.write(pd.DataFrame({'Actual':result['actual'],'Predicted':result['predicted']}))

        st.subheader("Actual Vs Prediction Chart")
        fig = plt.figure(figsize=(12,6))
        plt.plot(result['actual'])
        plt.plot(result['predicted'])
        st.pyplot(fig)
        




    if selected_model == 'High Trend Model':
        x_train_high,y_train_high,high_scaler = get_training_data(df=df,feature='High',days=day)
        
        m1 = models.model_1_train()
        high_output = m1.train_high_trend(x_train_high,y_train_high,int(EP1),OPT1,int(U1),int(U2),int(U3),int(U4),float(D1),float(D2),float(D3),float(D4))

        m2 = models.model_1_predict()
        result = m2.predict_high_trend(high_scaler,day,df)

        st.write("Loss of High Trained Model : {}".format(high_output['loss']))

        st.write("Prediction : {}".format(result['output']))

        st.write("Correct Prediction : {}".format(result['correct']))

        st.write("Incorrect Prediction : {}".format(result['incorrect']))

        st.write("Total : {}".format(result['total']))

        st.write(pd.DataFrame({'Actual':result['actual'],'Predicted':result['predicted']})) 

        st.subheader("Actual Vs Prediction Chart")
        fig = plt.figure(figsize=(12,6))
        plt.plot(result['actual'])
        plt.plot(result['predicted'])
        st.pyplot(fig)


    if selected_model == 'Low Trend Model':

        x_train_low,y_train_low,low_scaler = get_training_data(df=df,feature='Low',days=day)
        m1 = models.model_1_train()
        low_output = m1.train_low_trend(x_train_low,y_train_low,int(EP1),OPT1,int(U1),int(U2),int(U3),int(U4),float(D1),float(D2),float(D3),float(D4))

        m2 = models.model_1_predict()
        result = m2.predict_low_trend(low_scaler,day,df)

        st.write("Loss of Open Trained Model : {}".format(low_output['loss']))

        st.write("Prediction : {}".format(result['output']))

        st.write("Correct Prediction : {}".format(result['correct']))

        st.write("Incorrect Prediction : {}".format(result['incorrect']))

        st.write("Total : {}".format(result['total']))

        st.write(pd.DataFrame({'Actual':result['actual'],'Predicted':result['predicted']}))

        st.subheader("Actual Vs Prediction Chart")
        fig = plt.figure(figsize=(12,6))
        plt.plot(result['actual'])
        plt.plot(result['predicted'])
        st.pyplot(fig)


