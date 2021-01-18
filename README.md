# Stock Price Predict

## Prepare the database:

#### Load the database
```Python
df = pd.read_csv("NSE-Tata-Global-Beverages-Limited.csv")
```

#### Gernerate a new dataset that only contains Date and Close price.
```Python
new_dataset = pd.DataFrame(index=range(0,len(df)), columns=['Date','Close'])
for i in range(0, len(df)):
    new_dataset["Date"][i]=df["Date"][i]
    new_dataset["Close"][i]=df["Close"][i]

new_dataset.index = new_dataset["Date"]
new_dataset.drop("Date",axis=1,inplace=True)
```

#### Normalize the new filtered dataset

```Python
close_value = new_dataset.to_numpy()

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_value)
```

#### |train_data| : |valid_data| = 4 : 1

```Python
len_data = len(close_value)
len_training_data = int((len_data / 5) * 4)
train_data = close_value[0: len_training_data - 1]
valid_data = close_value[len_training_data - 1: ]
```

## Build and train Model:

```Python
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(x_train_data, y_train_data,epochs=1, batch_size=1, verbose=2)


inputs_data = new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data = inputs_data.reshape(-1,1)
inputs_data = scaler.transform(inputs_data)
```

## Show Prediction Result:
```Python
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
```

