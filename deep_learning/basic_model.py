#@title imports
import math
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense




# To match the Input shape (1,5) and our x_train shape is very important. 

def train_model(Optimizer, x_train, y_train, x_test, y_test):
  model = Sequential()
  model.add(LSTM(50, input_shape=(1, 5)))
  model.add(Dense(1))

  model.compile(loss="mean_squared_error", optimizer=Optimizer, metrics =["accuracy"])
  scores = model.fit(x=x_train,y=y_train, batch_size=1, epochs = 100, validation_data = (x_test, y_test))

  return scores, model


#@title LSTM with SGD, RMSprop, Adam optimizers, epochs = 100
#SGD_score, SGD_model = train_model(Optimizer = "sgd", x_train=x_train, y_train = y_train, x_test =x_test, y_test=y_test)
#RMSprop_score, RMSprop_model = train_model(Optimizer = "RMSprop", x_train=x_train, y_train = y_train, x_test =x_test, y_test=y_test)
#Adam_score, Adam_model = train_model(Optimizer = "adam", x_train=x_train, y_train = y_train, x_test =x_test, y_test=y_test)

#title Plot result

def plot_result(score, optimizer_name, label = "loss"):
  plt.figure(figsize=(18, 8))
  plt.plot(range(1, 101), score.history["loss"], label ="Training Loss")
  plt.plot(range(1,101), score.history["val_loss"], label="Validation Loss")
  plt.axis([1,100, 0, 7])
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Train and Validation Loss using "+optimizer_name + "optimizer")
  plt.legend()
  plt.show()

#title Plot predictions
def plot_predict(model, x_train, x_test, y_train, y_test):
  train_predict = RMSprop_model.predict(x_train)
  test_predict = RMSprop_model.predict(x_test)

  # Calculate root mean squared error.
  trainScore = math.sqrt(mean_squared_error(y_train, train_predict))
  print('Train Score: %.2f RMSE' % (trainScore))
  testScore = math.sqrt(mean_squared_error(y_test, test_predict))
  print('Test Score: %.2f RMSE' % (testScore))

  plt.figure(figsize=(18, 8))
  plt.plot(train_predict)
  plt.plot(y_train)
  plt.show()
  
  plt.figure(figsize=(18, 8))
  plt.plot(test_predict)
  plt.plot(y_test)
  plt.show()