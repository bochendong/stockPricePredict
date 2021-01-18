import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# import the data set
df = pd.read_csv("NSE-Tata-Global-Beverages-Limited.csv")

df.index = df["Date"]
df = df.sort_index(ascending=True, axis=0)
print(df.head())

# Gernerate a new dataset that only contains Date and Close price.
new_dataset = pd.DataFrame(index=range(0,len(df)), columns=['Date','Close'])
for i in range(0, len(df)):
    new_dataset["Date"][i]=df["Date"][i]
    new_dataset["Close"][i]=df["Close"][i]

new_dataset.index = new_dataset["Date"]
new_dataset.drop("Date",axis=1,inplace=True)


# Normalize the new filtered dataset
close_value = new_dataset.to_numpy()

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_value)

# |train_data| : |valid_data| = 4 : 1
len_data = len(close_value)
len_training_data = int((len_data / 5) * 4)
train_data = close_value[0: len_training_data - 1]
valid_data = close_value[len_training_data - 1: ]

# x_train_data =
#  [[first 60 items in scaled_data], [1st to 61st items in scaled_data] , ...]
# y_train_data =
#  [[60th item in scaled_data], [61th item in scaled_data] , ... ]
x_train_data, y_train_data=[],[]
for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])
    
# List object to numpy list object (easy to reshape)
x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)
# x_train_data reshape to (927, 60, 1)
x_train_data = np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(x_train_data, y_train_data,epochs=1, batch_size=1, verbose=2)


inputs_data = new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data = inputs_data.reshape(-1,1)
inputs_data = scaler.transform(inputs_data)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

closing_price = lstm_model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

closing_list = []
for i in closing_price:
    closing_list.append(i[0])

lstm_model.save("saved_lstm_model.h5")


print(valid_data)
train_data = new_dataset[:987]
valid_data = new_dataset[987:]
valid_data["pred"] = closing_list
plt.plot(train_data ["Close"], 'r-')
plt.plot(valid_data ["Close"], 'y-')
plt.plot(valid_data ["pred"], 'b-')
plt.show()
