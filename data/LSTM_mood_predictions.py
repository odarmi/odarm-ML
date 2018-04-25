import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from matplotlib import pyplot

from ML_mood_predictions import extract_features

# fix random seed for reproducibility
np.random.seed(7)

# load the dataset 
df = pd.read_csv('tung_hist_jan_mar_weather_nolocomotion_mood.csv')
#df = pd.read_table('data.csv', sep=",", usecols=range(108))
df = extract_features(df, True)
#print df


# Rescale it - should help network performance
#scaler = MinMaxScaler(feature_range=(0,1))
#X = scaler.fit_transform(df)
X = df.drop('Mood', axis=1)
#print X.shape
#print X

y = df['Mood']
#y = y.replace(5, 4)
#y = y.replace(1, 2)
#y = y.replace(3, 4)
y = y - 1
#print y.shape
#print y

# Just so we can have a validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
print X_train.shape[0]
print X_train.shape[1]
print X_test.shape
y_train = y_train.reshape((len(y_train),1))
y_test = y_test.reshape((len(y_test),1))
print y_train.shape
print y_test.shape

# This next part is because keras is picky with how they want data
X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

# truncate and pad input sequences
# create the model
model = Sequential()
model.add(LSTM(25, input_shape=(X_train.shape[1], X_train.shape[2])))
# Since it seems to be a categorical problem, use softmax activation instead of linear
model.add(Dense(5, activation='softmax')) # 16 possible key values, activation, 16keys*4training-traces, 197us sigmoid
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) #rmsprop
print(model.summary())

# diagnosis
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split = 0.33)
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()


scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print(model.predict(X_test))
