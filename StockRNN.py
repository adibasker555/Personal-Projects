import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn import preprocessing
from collections import deque
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


df = pd.read_csv("dlydata.csv")
cleaned_data = pd.read_csv("C:/Users/Adhithya/Desktop/cleaned.csv")
df.rename(columns={'A': 'Symbol', '2014-10-20': 'Date', '52.0700': 'Starting', '52.3800': 'High', '50.8600': 'Low',
             '52.3500': 'Closing', '4942200': 'Volume', '201443': 'Week'}, inplace=True)


SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 1
EPOCHS = 5
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


def preprocess_df(df):
    df = df.drop('future',1)
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df = df.replace([np.inf, -np.inf], np.nan)
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq,target])
        elif target == 1:
            buys.append([seq,target])

    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

def new_preprocess(df):
    sequential_data = []
    for symbol in cleaned_data['Symbol'].unique():
        prev_days = deque(maxlen=SEQ_LEN)
        df = cleaned_data[cleaned_data["Symbol"] == symbol]
        for i, row in df.iterrows():
            volume = row['Scaled Volumes']
            snp = row['snp_change']
            nyse = row['nyse_change']
            nasdaq = row['nasdaq_change']
            two_hundred = row['Scaled 200']
            ninety = row['Scaled 90']
            thirty = row['Scaled 30']
            sector = row['Sector Change']
            prev_days.append([volume, two_hundred, ninety, thirty, snp, nyse, nasdaq])
            if len(prev_days) == SEQ_LEN:
                sequential_data.append([np.array(prev_days), row['Prev Target']])

    buys = []
    sells = []

    for seq, target in sequential_data:
        if int(target) == 0:
            sells.append([seq, target])
        elif int(target) == 1:
            buys.append([seq, target])

    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y
    
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def filter_date(date):
    lowest_date = datetime.strptime("2017-01-01", "%Y-%m-%d")
    highest_date = datetime.strptime("2018-01-01", "%Y-%m-%d")
    this_date = datetime.strptime(date, "%Y-%m-%d")
    bool = (this_date >= lowest_date) and (this_date <= highest_date)
    return bool

main_df = pd.DataFrame()


cleaned_data.set_index("Date", inplace=True)
cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
cleaned_data.dropna(inplace=True)

times = sorted(cleaned_data.index.values, key = lambda x: datetime.strptime(x,"%Y-%m-%d"))
last_5pct = times[-int(0.15 * len(times))]
last_5pct = datetime.strptime(last_5pct,"%Y-%m-%d")

validation_main_df = cleaned_data[cleaned_data.index.map(lambda x: datetime.strptime(x,"%Y-%m-%d") >= last_5pct)]
main_df = cleaned_data[cleaned_data.index.map(lambda x: datetime.strptime(x,"%Y-%m-%d") < last_5pct)]

train_x, train_y = new_preprocess(main_df)
validation_x, validation_y = new_preprocess(validation_main_df)


model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation="tanh", return_sequences = True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, activation="tanh", return_sequences = True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, activation="tanh"))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
tensorboard = TensorBoard(log_dir=f'logs\{NAME}')

filepath = "RNN_Final-{epoch:02d}"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode='max'))

print(train_x.shape)

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint]
)









