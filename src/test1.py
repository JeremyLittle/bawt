import urllib.request, json
import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import tensorflow
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD, RMSprop
import tensorflow.keras
from keras.callbacks import CSVLogger


# import sqlite3
#with urllib.request.urlopen("https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1566288000&end=9999999999&period=300") as url:
 #   data = json.loads(url.read().decode())
  #  print(data[1]['date'])
print('here?')
TIME_STEPS = 3
BATCH_SIZE = 3

def loadData(coin1,coin2,startTime,endTime,period):
    url1 = "https://poloniex.com/public?command=returnChartData&currencyPair=" + coin1 + "_" + coin2 + "&start=" + str(startTime) + "&end=" + str(endTime) + "&period=" + str(period)

    with urllib.request.urlopen(url1) as url:
        data = json.loads(url.read().decode())
        print(data[1]['date'])

    # open a file for writing
    csvtest_data = open('testData.csv', 'w')

    # create the csv writer object
    # csvwriter = csv.writer(csvtest_data)
    # count = 0
    # for entry in data:
    #     if count == 0:
    #         header = entry.keys()
    #         csvwriter.writerow(header)
    #         count += 1
    #     csvwriter.writerow(entry.values())
    # csvtest_data.close()
    # return url

#from tutorial
def build_timeseries(mat, y_col_index):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    
    #changed tqdm_notebook to mat, this -3 could be wrong
    #for i in mat(range(dim_0)):
    for i in range((mat.shape[0])-3):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y

#from tutorial
def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat

print("start")
df = pd.read_csv('testData.csv')

# First code block from tutorial: https://towardsdatascience.com/predicting-stock-price-with-lstm-13af86a74944
train_cols = ["high","low","open","close","volume","quoteVolume","weightedAverage"]
df_train, df_test = train_test_split(df, train_size=0.8, test_size=0.2, shuffle=False)
print("Train and Test size", len(df_train), len(df_test))
# scale the feature MinMax, build array
x = df_train.loc[:,train_cols].values
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])

print("end")
# second code block from tutorial
x_t, y_t = build_timeseries(x_train, 6)
x_t = trim_dataset(x_t, BATCH_SIZE)
y_t = trim_dataset(y_t, BATCH_SIZE)
x_temp, y_temp = build_timeseries(x_test, 6)
x_val, x_test_t = np.array_split(trim_dataset(x_temp, BATCH_SIZE),2)
y_val, y_test_t = np.array_split(trim_dataset(y_temp, BATCH_SIZE),2)

print("second block done")
# third code block from tutorial
lstm_model = Sequential()
lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful=True,     kernel_initializer='random_uniform'))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(20,activation='relu'))
lstm_model.add(Dense(1,activation='sigmoid'))
#optimizer = optimizers.RMSprop(lr=lr)
lstm_model.compile(loss='mean_squared_error', optimizer='sgd')
print("3rd block done")

# 4th code block
csv_logger = CSVLogger('training.log', append=True)

history = lstm_model.fit(x_t, y_t, epochs=13, verbose=2, batch_size=BATCH_SIZE,
                    shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                    trim_dataset(y_val, BATCH_SIZE)), callbacks=[csv_logger])

#prediction = lstm_model.predict_on_batch(trim_dataset(x_val, BATCH_SIZE))
prediction = lstm_model.predict(trim_dataset(x_val, BATCH_SIZE), batch_size=BATCH_SIZE, verbose = 0, steps=None, callbacks=None)
print(prediction)
print("1")
print(prediction.shape)
matchtest = trim_dataset(y_val, BATCH_SIZE)
right = 0
wrong = 0
for i in range(1,152):
    if (matchtest[i]-prediction[i+1])*(matchtest[i]-matchtest[i+1]) > -0.00000000001 :
        right = right + 1
    else:
        wrong = wrong + 1
print(wrong)
print(right)
# print("1")
# print(y_t)
# print("2")
# print(x_val)
# print("3")
# print(y_val)

#url1 = loadData("USDT","BTC",1566288000,9999999999,300)

#messing

#print(df)
# print(df['date'][1:9])
# inputsDF=(df.drop(columns=['date']))
# print(inputsDF['high'][1])
# dateDF = (df['date'])
# print(dateDF)
# print(df.shape[1])
#print (price_matrix_creator(df))

# plt.figure()
# plt.plot(df["high"])
# plt.plot(df["low"])
# plt.plot(df["weightedAverage"])
# plt.title('Data')
# plt.ylabel('Price (USD)')
# plt.xlabel('time chunks')
# plt.legend(['High','Low','Avg'], loc='upper left')
# plt.show()



