# imports
import math
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Multi-step LSTM model, change the input_shape and Dense layer parameter to fit the train data shape and test data shape.

def train_multi_step_model(Optimizer, x_train, y_train, x_test, y_test):
  model = Sequential()
  model.add(LSTM(50, input_shape=(12, 4)))
  model.add(Dense(6))

  model.compile(loss="mean_squared_error", optimizer=Optimizer, metrics =["accuracy"])
  scores = model.fit(x=x_train,y=y_train, batch_size=1, epochs = 200, validation_data = (x_test, y_test))

  return scores, model


#Plot Multi-step, Multi-feature predictions.
def predict_plot(df, y_predict, targets):
  """ Plot the multi-step, multi-result predictions.

  :param: df, pd.DataFrame, e.g. df = pd.read_excel("USMacroData.xls", "All")
  :param: y_predict, 2-dim np.array, the model-predicted values, in each row, it has look_forward*(number of target features) elements.
          In our example, look_forward = 3, number of target features =2 ("Inflation", "Unemployment").
  :param: targets, list, target features, e.g ["Inflation", "Unemployment"]
  """

  y_predict = de_vectorize(y_predict, 2, 3)
  assert y_predict.shape[1] == len(targets), "Incompatible size of targets and dataset"
  assert df.shape[0] == y_predict.shape[0], "Incompatible original data rows and y_predict rows"
  
  look_forward = y_predict.shape[2]

  for index, target in enumerate(targets):
    plt.figure(figsize=(17, 8))
    plt.plot(df[0:12][target])
    for i in range(len(y_predict)):
      y = list(y_predict[i][index])
      x = list(df.index[i: i+look_forward])
      data = pd.DataFrame(list(zip(x, y)), columns =[df.index.name, target]) 
      data = data.sort_values(df.index.name)
      data.set_index(df.index.name, inplace=True)

      if i < 12:
        plt.plot(df[i: i+look_forward][target])
        plt.plot(data)
        plt.xlabel("Date")
        plt.ylabel(target)
        plt.title("3-month predictions of " + target)
    plt.show()
