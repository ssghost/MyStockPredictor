import time
import tensorflow as tf
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr

class Predictor:
    def __init__(self, symbol, key):
        self.dataset = []
        self.key = key
        self.tset = []
        self.vset = []
        self.model = None
        self.mset = []
        self.symbol = symbol
        self.result = []
        self.callback = None
        self.window_size = 10
        self.batch_size = 32
        self.buffer_size = 100
    
    def fetch_dataset(self):
        start_date = time.strftime("%Y-%m-%d %H:%M:%S", time.time() - 100*24*3600)[:10]
        with self.key as os.environ["IEX_API_KEY"]:
            for s in self.symbol:
                price,step = [],[]
                try:
                    price.append(float(pdr.DataReader(s, "iex", start=start_date)['open']))
                    step.append(float(range(len(pdr.DataReader(s, "iex", start=start_date)))))
                    self.dataset.append([price,step])
                except: 
                    print('Symbol not found.')

    def arrange_dataset(self):
        for ds in self.dataset:
            series = np.array(ds[0])
            smin = np.min(series)
            smax = np.max(series)
            series -= smin
            series /= smax
            steps = np.array(ds[1])
            step_train = steps[:90]
            x_train = series[:90]
            step_valid = steps[90:]
            x_valid = series[90:]
            self.tset.append([x_train, step_train])
            self.vset.append([x_valid, step_valid])

        
    def windowed_dataset(series, self):
        series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(self.window_size + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.window_size + 1))
        ds = ds.shuffle(self.buffer_size)
        ds = ds.map(lambda w: (w[:-1], w[1:]))
        return ds.batch(self.batch_size).prefetch(1)


    def create_model(self):
        self.model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                               strides=1, padding="causal",
                               activation="relu",
                               input_shape=[None, 1]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(1)
    ])

    def train(self):
        optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
        model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
        for i in range(len(self.tset)):
            train_set = windowed_dataset(self.tset[i][0])
            valid_set = windowed_dataset(self.vset[i][0])
            self.mset.append(self.model.fit(train_set, epochs=3000, validation_data= valid_set, callbacks = [self.callback]))
    
    def predict(self):
        start_date = time.strftime("%Y-%m-%d %H:%M:%S", time.time() - 10*24*3600)[:10]
        with self.key as os.environ["IEX_API_KEY"]:
            test =[]
            for s in self.symbol:
                try:
                    test.append(float(pdr.DataReader(s, "iex", start=start_date)['open']))
                except: 
                    print('Symbol not found.')
            for model in self.mset:
                self.result.append(model.predict(test.pop(0)))

    def callback(self):
        class acc_clip(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if (logs.get('mae') < 0.2):
                    self.model.stop_training = True
        self.callback = acc_clip()
