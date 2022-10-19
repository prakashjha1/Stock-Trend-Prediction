# Stock-Trend-Prediction
Deep learning for predicting stock market prices and trends has become even more popular than before. I have used yahoo finance to collect the data and LSTM to build the stock trend model. I have streamlit to build web UI. 

# Python Modules used in this project

- numpy
- pandas
- matplotlib
- pandas_datareader
- scikit-learn
- tensorflow
- datetime
- streamlit

# Steps to train the Model
- fetch the data from start date to end date.
- choose the target variable (High or Low or Open or Close)
- create dataframe with input and variable. 
- input is previous 10 days or 5 days stock price based on user choice.
- output is next day stock price.
- split the data into training and testing set.
- Scale the data input data.
- Build the LSTM model using keras module of tensorflow.
- Train the model using training data set.
- Save the model.

# Step for Prediction
- Take user input (previous 10 days or 5 days stock price data).
- Scale the data.
- Load Saved model.
- Predict the Ouput



