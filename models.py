import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model

class model_1_train :

    def train_open_trend(self,x_train,y_train,EP,OPT,U1,U2,U3,U4,D1,D2,D3,D4):

        # Create model
        model = Sequential()
        model.add(LSTM(units= U1, activation = 'relu', return_sequences=True, input_shape= (x_train.shape[1],1)))
        model.add(Dropout(D1))

        model.add(LSTM(units= U2, activation = 'relu', return_sequences=True))
        model.add(Dropout(D2))

        model.add(LSTM(units= U3, activation = 'relu', return_sequences=True))
        model.add(Dropout(D3))

        model.add(LSTM(units= U4, activation = 'relu'))
        model.add(Dropout(D4))

        model.add(Dense(units=1))

        # Train Model
        model.compile(optimizer=OPT,loss='mean_squared_error',metrics=['accuracy'])
        history = model.fit(x_train,y_train,epochs=EP)
        model.save("model_1_open.h5")
        loss = round(history.history['loss'][-1],4)
        acc = round(history.history['accuracy'][-1],4)

        return {'loss':loss,'accuracy':acc}
        



    def train_close_trend(self,x_train,y_train,EP,OPT,U1,U2,U3,U4,D1,D2,D3,D4):

        # Create model
        model = Sequential()
        model.add(LSTM(units= U1, activation = 'relu', return_sequences=True, input_shape= (x_train.shape[1],1)))
        model.add(Dropout(D1))

        model.add(LSTM(units= U2, activation = 'relu', return_sequences=True))
        model.add(Dropout(D2))

        model.add(LSTM(units= U3, activation = 'relu', return_sequences=True))
        model.add(Dropout(D3))

        model.add(LSTM(units= U4, activation = 'relu'))
        model.add(Dropout(D4))

        model.add(Dense(units=1))

        # Train Model
        model.compile(optimizer=OPT,loss='mean_squared_error',metrics=['accuracy'])
        history = model.fit(x_train,y_train,epochs=EP)
        model.save("model_1_close.h5")
        loss = round(history.history['loss'][-1],4)
        acc = round(history.history['accuracy'][-1],4)

        return {'loss':loss,'accuracy':acc}



    def train_high_trend(self,x_train,y_train,EP,OPT,U1,U2,U3,U4,D1,D2,D3,D4):

        # Create model
        model = Sequential()
        model.add(LSTM(units= U1, activation = 'relu', return_sequences=True, input_shape= (x_train.shape[1],1)))
        model.add(Dropout(D1))

        model.add(LSTM(units= U2, activation = 'relu', return_sequences=True))
        model.add(Dropout(D2))

        model.add(LSTM(units= U3, activation = 'relu', return_sequences=True))
        model.add(Dropout(D3))

        model.add(LSTM(units= U4, activation = 'relu'))
        model.add(Dropout(D4))

        model.add(Dense(units=1))

        # Train Model
        model.compile(optimizer=OPT,loss='mean_squared_error',metrics=['accuracy'])
        history = model.fit(x_train,y_train,epochs=EP)
        model.save("model_1_high.h5")
        loss = round(history.history['loss'][-1],4)
        acc = round(history.history['accuracy'][-1],4)

        return {'loss':loss,'accuracy':acc}

    def train_low_trend(self,x_train,y_train,EP,OPT,U1,U2,U3,U4,D1,D2,D3,D4):

        # Create model
        model = Sequential()
        model.add(LSTM(units= U1, activation = 'relu', return_sequences=True, input_shape= (x_train.shape[1],1)))
        model.add(Dropout(D1))

        model.add(LSTM(units= U2, activation = 'relu', return_sequences=True))
        model.add(Dropout(D2))

        model.add(LSTM(units= U3, activation = 'relu', return_sequences=True))
        model.add(Dropout(D3))

        model.add(LSTM(units= U4, activation = 'relu'))
        model.add(Dropout(D4))

        model.add(Dense(units=1))

        # Train Model
        model.compile(optimizer=OPT,loss='mean_squared_error',metrics=['accuracy'])
        history = model.fit(x_train,y_train,epochs=EP)
        model.save("model_1_low.h5")
        loss = round(history.history['loss'][-1],4)
        acc = round(history.history['accuracy'][-1],4)

        return {'loss':loss,'accuracy':acc}


class model_1_predict :

    def predict_close_trend(self,scaler,MA,df):
        
        model = load_model("model_1_close.h5")
        prediction = pd.DataFrame(np.array(list(df['Close'].values[-MA:])).reshape(MA,1),columns=['Close'])
        pred = scaler.fit_transform(prediction)
        res = scaler.inverse_transform([[model.predict(pred.reshape(1,MA,1))[0][0]]])[0][0]

        actual = []
        predicted = []

        for i in range(1,20):
            actual.append(df['Close'].values[-i])
            prediction = pd.DataFrame(np.array(list(df['Close'].values[(-i-MA):-i])).reshape(MA,1),columns=['Close'])
            pred = scaler.fit_transform(prediction)
            r = scaler.inverse_transform([[model.predict(pred.reshape(1,MA,1))[0][0]]])[0][0]
            predicted.append(r)

        correct = 0
        incorrect = 0
        count =0
        for i in range(1,18):
            count +=1
            if ((actual[-i-1] - actual[-i-2])<0) == ((predicted[-i] - predicted[-i-1])<0):
                correct +=1
            else:
                incorrect +=1



        return {'output':res, 'actual':actual, 'predicted':predicted,'correct':correct,'incorrect':incorrect,'total':count}

    def predict_open_trend(self,scaler,MA,df):

        model = load_model("model_1_open.h5")
        prediction = pd.DataFrame(np.array(list(df['Close'].values[-MA:])).reshape(MA,1),columns=['Open'])
        pred = scaler.fit_transform(prediction)
        res = scaler.inverse_transform([[model.predict(pred.reshape(1,MA,1))[0][0]]])[0][0]

        actual = []
        predicted = []

        for i in range(1,20):
            actual.append(df['Open'].values[-i])
            prediction = pd.DataFrame(np.array(list(df['Open'].values[(-i-MA):-i])).reshape(MA,1),columns=['Open'])
            pred = scaler.fit_transform(prediction)
            r = scaler.inverse_transform([[model.predict(pred.reshape(1,MA,1))[0][0]]])[0][0]
            predicted.append(r)

        correct = 0
        incorrect = 0
        count =0
        for i in range(1,18):
            count +=1
            if ((actual[-i-1] - actual[-i-2])<0) == ((predicted[-i] - predicted[-i-1])<0):
                correct +=1
            else:
                incorrect +=1



        return {'output':res, 'actual':actual, 'predicted':predicted,'correct':correct,'incorrect':incorrect,'total':count}

    def predict_high_trend(self,scaler,MA,df):

        model = load_model("model_1_high.h5")
        prediction = pd.DataFrame(np.array(list(df['Close'].values[-MA:])).reshape(MA,1),columns=['High'])
        pred = scaler.fit_transform(prediction)
        res = scaler.inverse_transform([[model.predict(pred.reshape(1,MA,1))[0][0]]])[0][0]

        actual = []
        predicted = []

        for i in range(1,20):
            actual.append(df['Open'].values[-i])
            prediction = pd.DataFrame(np.array(list(df['Open'].values[(-i-MA):-i])).reshape(MA,1),columns=['High'])
            pred = scaler.fit_transform(prediction)
            r = scaler.inverse_transform([[model.predict(pred.reshape(1,MA,1))[0][0]]])[0][0]
            predicted.append(r)

        correct = 0
        incorrect = 0
        count =0
        for i in range(1,18):
            count +=1
            if ((actual[-i-1] - actual[-i-2])<0) == ((predicted[-i] - predicted[-i-1])<0):
                correct +=1
            else:
                incorrect +=1



        return {'output':res, 'actual':actual, 'predicted':predicted,'correct':correct,'incorrect':incorrect,'total':count}

    def predict_low_trend(self,scaler,MA,df):

        model = load_model("model_1_low.h5")
        prediction = pd.DataFrame(np.array(list(df['Close'].values[-MA:])).reshape(MA,1),columns=['Low'])
        pred = scaler.fit_transform(prediction)
        res = scaler.inverse_transform([[model.predict(pred.reshape(1,MA,1))[0][0]]])[0][0]

        actual = []
        predicted = []

        for i in range(1,20):
            actual.append(df['Open'].values[-i])
            prediction = pd.DataFrame(np.array(list(df['Open'].values[(-i-MA):-i])).reshape(MA,1),columns=['Low'])
            pred = scaler.fit_transform(prediction)
            r = scaler.inverse_transform([[model.predict(pred.reshape(1,MA,1))[0][0]]])[0][0]
            predicted.append(r)

        correct = 0
        incorrect = 0
        count =0
        for i in range(1,18):
            count +=1
            if ((actual[-i-1] - actual[-i-2])<0) == ((predicted[-i] - predicted[-i-1])<0):
                correct +=1
            else:
                incorrect +=1



        return {'output':res, 'actual':actual, 'predicted':predicted,'correct':correct,'incorrect':incorrect,'total':count}