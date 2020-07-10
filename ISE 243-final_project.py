# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:31:53 2019

@author: vivek
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pylab as plt

bike_rental = pd.read_csv('C:\ISE_fall_2019\ISE_243_Supply_Chain_Analytics\Project\day.csv') 
bike_rental.head()

#Check dimensions
bike_rental.shape
#(731, 16), 731 rows and 16 columns

#Examining the first/last 5 rows to get a better idea
bike_rental.head()
bike_rental.tail()

#Checking data type of each feature
bike_rental.dtypes
#Out[9]: 
#instant         int64
#dteday         object
#season          int64
#yr              int64
#mnth            int64
#holiday         int64
#weekday         int64
#workingday      int64
#weathersit      int64
#temp          float64
#atemp         float64
#hum           float64
#windspeed     float64
#casual          int64
#registered      int64
#cnt             int64
#dtype: object

#Checking the Statistical Summary for numerical features
u = bike_rental.describe()
#         instant      season  ...   registered          cnt
#count  731.000000  731.000000  ...   731.000000   731.000000
#mean   366.000000    2.496580  ...  3656.172367  4504.348837
#std    211.165812    1.110807  ...  1560.256377  1937.211452
#min      1.000000    1.000000  ...    20.000000    22.000000
#25%    183.500000    2.000000  ...  2497.000000  3152.000000
#50%    366.000000    3.000000  ...  3662.000000  4548.000000
#75%    548.500000    3.000000  ...  4776.500000  5956.000000
#max    731.000000    4.000000  ...  6946.000000  8714.000000

#Data visualization

# scatter plot matrix - interactions between the features
from pandas.plotting import scatter_matrix
scatter_matrix(bike_rental,figsize=(20,20))

##############################################Preparing the Data#####################################################################
#Are all the data useful? - eliminate redundant data to only keep

bike_rental_df = bike_rental[['season','yr','mnth','holiday','weekday','workingday','weathersit','atemp','hum','windspeed','casual','registered','cnt']]
bike_rental_df


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#preprocessing is the module used to do some cleaning/scaling of data prior to machine learning, 
#and cross_ validation is used in training and testing. 
#we'll be using LinearRegression algorithm from Scikit-learn as our machine learning algorithms to demonstrate results


#Defined X (features), as our entire dataframe EXCEPT for the target column, converted to a numpy array.
X = np.array(bike_rental_df.drop(['cnt'], 1)) 
X
#Generally, you want your features in machine learning to be in a range of -1 to 1(it is optional). 
#This may do nothing, but it usually speeds up processing and can also help with accuracy.
X = preprocessing.scale(X)
X
len(X)

#731
#define y variable(target), as simply the target column of the dataframe, converted to a numpy array.
y = np.array(bike_rental_df['cnt'])

#The lenth of X and y should be the same. Make sure to double check
len(X)
len(y)
# 731
#Let's generate out X_train, X_test, y_train, y_test using cross_validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#The return is the training set of features, testing set of features, training set of targets, and testing set of targets. 
len(X_train) #584
len(X_test)  #147
len(y_train) #584
len(y_test) #147
#let's perform regression on our stock price data
clf = LinearRegression()
clf

model=clf.fit(X_train, y_train)
pred_test=model.predict(X_test)
pred_test
import matplotlib.pylab as plt

ts_test = pd.DataFrame(y_test)
ts_pred_test = pd.DataFrame(pred_test)

plt.plot(ts_test, label='Test')
plt.plot(ts_pred_test, label='Linear Regression')
plt.legend(loc=2)

confidence = model.score(X_test, y_test)
confidence
#0.99

import math 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
rmse = math.sqrt(mean_squared_error(y_test, pred_test))
rmse
#2.5993682127776676
mae =mean_absolute_error(y_test, pred_test)
mae
#Check the coefficents of the model
model.coef_
model.intercept_
dfnew= bike_rental_df.drop(['cnt'], 1)
dfnew.columns
x = pd.DataFrame(zip(dfnew.columns, model.coef_), columns=['features','estimated_Coefficients'])
x
#     features  estimated_Coefficients
#0       season           -3.275378e-13
#1           yr           -5.903935e-13
#2         mnth           -1.374541e-14
#3      holiday            4.566130e-13
#4      weekday           -1.972128e-13
#5   workingday           -7.362632e-13
#6   weathersit           -2.684193e-13
#7        atemp           -4.587952e-13
#8          hum           -5.714962e-13
#9    windspeed            1.033341e-12
#10      casual            6.861527e+02
#11  registered            1.559189e+03

###################### Holt-Winter Model

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

dat = pd.read_csv('C:\ISE_fall_2019\ISE_243_Supply_Chain_Analytics\Project\day.csv', parse_dates=['dteday'], index_col='dteday',date_parser=dateparse)

dat = dat[['cnt']]

dat.sort_index(ascending = True, inplace = True)
dat
#            cnt
#dteday          
#2011-01-01   985
#2011-01-02   801
#2011-01-03  1349
#2011-01-04  1562
#2011-01-05  1600
#         ...

len(dat)
#731

# Decomposing
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(dat)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
#3-digit integer or three separate integers in subplot() describing the position of the subplot in nrows, ncols, and index in order.
plt.subplot(411)
plt.plot(dat, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


train= dat.iloc[:584, 0]
train
#dteday
#2011-01-01     985
#2011-01-02     801
#2011-01-03    1349
#2011-01-04    1562
#2011-01-05    1600
#
#2012-08-02    7261
#2012-08-03    7175
#2012-08-04    6824
#2012-08-05    5464
#2012-08-06    7013
#Name: cnt, Length: 584, dtype: int64

test = dat.iloc[584:, 0]



from statsmodels.tsa.holtwinters import ExponentialSmoothing
#Subsetting the dataset, can't use loc() to slice indexing on <class 'pandas.core.indexes.datetimes.DatetimeIndex>

model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=90).fit()
test.index
#DatetimeIndex(['2012-08-07', '2012-08-08', '2012-08-09', '2012-08-10',
#               '2012-08-11', '2012-08-12', '2012-08-13', '2012-08-14',
#               '2012-08-15', '2012-08-16',
#               ...
#               '2012-12-22', '2012-12-23', '2012-12-24', '2012-12-25',
#               '2012-12-26', '2012-12-27', '2012-12-28', '2012-12-29',
#               '2012-12-30', '2012-12-31'],
#              dtype='datetime64[ns]', name='dteday', length=147, freq=None)

pred = model.predict(start=test.index[0], end=test.index[-1])
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc=2)

import math 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
rmse = math.sqrt(mean_squared_error(test, pred))
rmse
#2182.79
mae =mean_absolute_error(test, pred)
mae



###ARIMA
plt.plot(dat)

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
h = dat.iloc[:, 0]
dftest = adfuller(h)
dftest
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
dfoutput
#Out[107]: 
#Test Statistic                  -1.877448
#p-value                          0.342743
#Lags Used                      13.000000
#Number of Observations Used    717.000000
#dtype: float64

data_cnt = np.log(dat['cnt'])
plt.plot(data_cnt)
moving_avg_org = pd.Series(data_cnt).rolling(window=15).mean()
moving_avg_org 
data_moving_avg_diff = data_cnt - moving_avg_org
data_moving_avg_diff

#Out[113]: 
#dteday
#2011-01-01         NaN
#2011-01-02         NaN
#2011-01-03         NaN
#2011-01-04         NaN
#2011-01-05         NaN
#  
#2012-12-27   -0.250670
#2012-12-28    0.169254
#2012-12-29   -0.571698
#2012-12-30   -0.210669
#2012-12-31    0.229529
#Name: cnt, Length: 731, dtype: float64

data_moving_avg_diff.dropna(inplace=True)
plt.plot(data_moving_avg_diff)
adfuller(data_moving_avg_diff)
#(-18.375193842604727,
# 2.214463538905168e-30,
# 0,
# 716,
 #{'1%': -3.439516060164992,
 # '5%': -2.8655850998755263,
 # '10%': -2.5689240826597173},
 #449.96554633280584)
#Since the p value is now smaller than 0.05 the data is stationary

data_moving_avg_diff_train = data_moving_avg_diff[0:584]
data_moving_avg_diff_train

data_moving_avg_diff_train.dropna(inplace=True)
adfuller(data_moving_avg_diff_train)

data_moving_avg_diff_test = data_moving_avg_diff[584:]


plt.plot(data_moving_avg_diff_train)
adfuller(data_moving_avg_diff_train)

plt.plot(data_moving_avg_diff_test)
adfuller(data_moving_avg_diff_test)

#Since the p value is now smaller than 0.05 the data is stationary

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf3 = acf(data_moving_avg_diff_train, nlags=20)
lag_pacf3 = pacf(data_moving_avg_diff_train, nlags=20, method='ols')
#Plot ACF:

plt.subplot(121)
plt.plot(lag_acf3, '^k:')
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_moving_avg_diff_train)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_moving_avg_diff_train)),linestyle='--',color='gray')
plt.grid()
plt.title('Autocorrelation Function')
#Choose p=2
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf3, '^k:')
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_moving_avg_diff_train)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_moving_avg_diff_train)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
#Choose q=2

data_moving_ave_ARIMA = ARIMA(data_moving_avg_diff_train, order=(2, 0, 2))
data_moving_ave_ARIMA_fitted= data_moving_ave_ARIMA.fit(disp=-1)
data_moving_ave_ARIMA_fittedvalue=data_moving_ave_ARIMA_fitted.fittedvalues
data_moving_ave_ARIMA_fittedvalue
data_moving_avg_diff_train
plt.plot(data_moving_avg_diff_train.index, data_moving_avg_diff_train, label='Train Stationary')
plt.plot(data_moving_avg_diff_train.index, data_moving_ave_ARIMA_fittedvalue, label='Fitted Values ARIMA',color='red')
plt.title('RSS: %.4f' % sum((data_moving_ave_ARIMA_fittedvalue-data_moving_avg_diff_train)**2))
plt.legend(loc=2)


history1 = [x for x in data_moving_avg_diff_train]
predictions1 = list()
for t in range(len(data_moving_avg_diff_test)):
	model = ARIMA(history1, order=(2,0,2))
	model_fit = model.fit(disp=-1)
	output = model_fit.forecast() #forecast() will transpform data to its origin domain
	yhat1 = output[0]
	predictions1.append(yhat1)
	obs1 = data_moving_avg_diff_test[t]
	history1.append(obs1)
	print('predicted=%f, expected=%f' % (yhat1, obs1))
rmse2 =math.sqrt(mean_squared_error(data_moving_avg_diff_test, predictions1))
rmse2
# 0.5490598550975089



# Uni-Variate Neural Network  
import theano

import tensorflow

import keras

import scipy
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
from numpy import array

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

dat = pd.read_csv('C:\ISE_fall_2019\ISE_243_Supply_Chain_Analytics\Project\day.csv', parse_dates=['dteday'], index_col='dteday',date_parser=dateparse)

dat = dat[['cnt']]
series = dat
X = series.values
#Divide your data into input (X) and output (y) components with time steps=1
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
 
supervised = timeseries_to_supervised(X, 1)
supervised.head()
#     0     0
#0     0.0   985
#1   985.0   801
#2   801.0  1349
#3  1349.0  1562
#4  1562.0  1600
series.head()
#  cnt
#dteday          
#2011-01-01   985
#2011-01-02   801
#2011-01-03  1349
#2011-01-04  1562
#2011-01-05  1600

#Transform Time Series to Stationary
#The bike rental dataset is not stationary.
#Stationary data is easier to model and will very likely result in more skillful forecasts.
# create a differenced series and then divide data into input (X) and output (y) components with time steps=1

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 # transform to be stationary
diff_values = difference(X, 1)
diff_values.head()
#Out[172]: 
#0    [-184]
#1     [548]
#2     [213]
#3      [38]
#4       [6]
#dtype: object

adfuller(diff_values)
#(array([-11.77880907]),
 #array([1.04677975e-21]),
# 12,
# 717,
# {'1%': -3.439503230053971,
#  '5%': -2.8655794463678346,
#  '10%': -2.5689210707289982},
# array([11726.34373306]))
#We also need to invert this process in order to take forecasts made on the differenced series back into their original scale.

#The function below, called inverse_difference(), inverts this operation

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# test out the invert transform function
inverted = list()
for i in range(len(diff_values)):
	value = inverse_difference(X, diff_values[i], len(X)-i)
	inverted.append(value)
inverted = Series(inverted)
inverted.head()

#Out[180]: 
#0     [801]
#1    [1349]
#2    [1562]
#3    [1600]
#4    [1606]
#dtype: object

# Divide the transformed data into input (X) and output (y) components with time steps=1
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
supervised_values
#Out[183]: 
#array([[0, array([-184], dtype=int64)],
#       [array([-184], dtype=int64), array([548], dtype=int64)],
#       [array([548], dtype=int64), array([213], dtype=int64)],
#       ...,
#       [array([981], dtype=int64), array([-1754], dtype=int64)],
#       [array([-1754], dtype=int64), array([455], dtype=int64)],
#       [array([455], dtype=int64), array([933], dtype=int64)]],
#      dtype=object)

# split data into train and test sets, use 147 of them as testing set and the rest as training 
train, test = supervised_values[0:-147], supervised_values[-147:]

#Transform Time Series to Scale
#Like other neural networks, LSTMs expect data to be within the scale of the 
#activation function used by the network.

#The default activation function for LSTMs is the hyperbolic tangent (tanh), 
#which outputs values between -1 and 1. This is the preferred range for the time series data.
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(X)
scaled_X = scaler.transform(X)
scaled_series = Series(scaled_X[:, 0])
scaled_series.head()
#Out[193]: 
#0   -0.778417
#1   -0.820755
#2   -0.694662
#3   -0.645651
#4   -0.636908
#dtype: float64

#Again, we must invert the scale on forecasts to return the values back to the 
#original scale so that the results can be interpreted and a comparable error 
#score can be calculated.
# invert transform
inverted_X = scaler.inverse_transform(scaled_X)
inverted_series = Series(inverted_X[:, 0])
inverted_series.head()
#Out[199]: 
#0     985.0
#1     801.0
#2    1349.0
#3    1562.0
#4    1600.0
#dtype: float64

#We can define a function to scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled
# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

X, y = train_scaled[:, 0:-1], train_scaled[:, -1]
X = X.reshape(X.shape[0], 1, X.shape[1])



# fit the model

#The shape of the input data must be specified in the LSTM layer using the “batch_input_shape” 
#argument as a tuple that specifies the expected number of observations to read each batch, 
#the number of time steps, and the number of features.
#The batch size is often much smaller than the total number of samples. It, along with the number 
#of epochs, defines how quickly the network learns the data (how often the weights are updated).

#The final import parameter in defining the LSTM layer is the number of neurons, also called the 
#number of memory units or blocks. This is a reasonably simple problem and a number between 1 
#and 5 should be sufficient.
#Ser the parameters:
neurons=4
batch_size=1
nb_epoch=1500
#The line below creates a single LSTM hidden layer that also specifies the expectations of the input layer via the “batch_input_shape” argument.

layer = LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True)
#The network requires a single neuron in the output layer with a linear activation to predict 
#the number of rental bikes at the next time step.

#Once the network is specified, it must be compiled into an efficient symbolic representation 
#using a backend mathematical library, such as TensorFlow or Theano.

#In compiling the network, we must specify a loss function and optimization algorithm. 
#We will use “mean_squared_error” as the loss function as it closely matches RMSE that we will are interested in, and the efficient ADAM optimization algorithm.

#Using the Sequential Keras API to define the network, the below snippet creates and compiles the network.

model = Sequential()
model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#Once compiled, it can be fit to the training data. Because the network is stateful, 
#we must control when the internal state is reset. Therefore, we must manually manage the training process one 
#epoch at a time across the desired number of epochs.

#By default, the samples within an epoch are shuffled prior to being exposed to the network. 
#Again, this is undesirable for the LSTM because we want the network to build up state as 
#it learns across the sequence of observations. We can disable the shuffling of samples by 
#setting “shuffle” to “False“.
#Below is a loop that manually fits the network to the training data.
for i in range(nb_epoch):
	model.fit(X, y, epochs=1, batch_size=batch_size, shuffle=False)
	model.reset_states()
#The batch_size must be set to 1. This is because it must be a factor of the size of the training 
#and test datasets.

#The predict() function on the model is also constrained by the batch size; there it must be set 
#to 1 because we are interested in making one-step forecasts on the test data.

#We will not tune the network parameters in this tutorial 
    
#As an extension to this tutorial, you might like to explore different model parameters and see 
#if you can improve performance: Consider trying 1500 epochs and 1 neuron, the performance may be better!

#LSTM Forecast
#To make a forecast, we can call the predict() function on the model. This requires a 
#3D NumPy array input as an argument. In this case, it will be an array of one value, 
#the observation at the previous time step.

#The predict() function returns an array of predictions, one for each input row provided. 
#Because we are providing a single input, the output will be a 2D NumPy array with one value.
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(series.values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = series.values[len(train) + i + 1]
	print('Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(series.values[-147:], predictions))
print('Test RMSE: %.3f' % rmse)


# line plot of observed vs predicted
pyplot.plot(series.values[-147:])
pyplot.plot(predictions)
pyplot.show()

from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pandas import Series
from pandas import array

from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from math import sqrt
import numpy as np
import pandas as pd
from numpy import concatenate

# load dataset
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

dat = pd.read_csv('C:\ISE_fall_2019\ISE_243_Supply_Chain_Analytics\Project\day.csv', parse_dates=['dteday'], index_col='dteday',date_parser=dateparse)
dataset = dat[['cnt','season','yr','mnth','holiday','weekday','workingday','weathersit','atemp','hum', 'windspeed','casual','registered']]
values = dataset.values
type(values)

DataFrame(values).head()
DataFrame(values).describe()

# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7,8,9,10,11,12]
i=1
# plot each column
#pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
#We will frame the supervised learning problem as predicting the pollution 
#at the current hour (t) given the pollution measurement and weather 
#conditions at the prior time step.
    
#A supervised learning problem is comprised of input patterns (X) and 
#output patterns (y), such that an algorithm can learn how to predict the output 
#patterns from the input patterns.    
    
df = DataFrame()
df['t'] = [x for x in range(10)]
df  
df['t-1'] = df['t'].shift(1)
df
df['t'] = [x for x in range(10)]
df['t+1'] = df['t'].shift(-1)
print(df)
##LSTM Data Preparation

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence t-1
	for i in range(n_in, 0, -1): #range(start, stop, step)
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        #the first col name is var1(t-1), the second is var2(t-1) ...
	# forecast sequence 
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            #the first col name is var1(t), the second is var2(t) ...
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg   
 
supervised = series_to_supervised(values,1,1)
# integer encode direction
from sklearn.preprocessing import LabelEncoder 
#encode categorical features using a encoding scheme to fit the 
#requirements of machine leatrning algorithm
from sklearn.preprocessing import MinMaxScaler
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
values

# ensure all data is float
values = values.astype('float32')
values
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[14,15,16,17,18,19,20,21,22,23,24,25]], axis=1, inplace=True)
reframed.head() 
values = reframed.values
n_train_days = 365
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D array [samples, timesteps(to make each single row observation when you make timestamp = 1), features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# design network
#We will define the LSTM with 50 neurons in the first hidden layer and 1 neuron 
#in the output layer for predicting pollution. The input shape will be 1 time step with 8 features.

#We will use the Mean Absolute Error (MAE) loss function and the efficient 
#Adam version of stochastic gradient descent.

#The model will be fit for 50 training epochs with a batch size of 1.
train_X.shape[1]
train_X.shape[2]
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=500, batch_size=1, validation_data=(test_X, test_y), verbose=2, shuffle=False) #shuffle helps to shuffle the data but in time series the data needs to be ordered so shuffle = false
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
#make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
 

