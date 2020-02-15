#@title ```basic.py```

#import numpy, pandas and matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


#Basic checks: find null values and fill, set index, etc.

def basic_check(df, index_name = "Month"):
  """Find the null values and set index of a given DataFrame.
  :param: df, pd.DataFrame, the data, e.g. df = pd.read_excel("USMacroData.xls", "All")
  :param: index_name, str, name of the index, must be one of the column names, e.g. index_name ="Month"
  :rtype: pd.DataFrame
  """
  df = df.sort_values(index_name)
  df.set_index(index_name, inplace=True)

  #check for null entries
  print("Null values summary:\n")
  print(df.isnull().sum())

  return df

def plot_column(df, feature):
    """Plot the resampled column of df, e.g. plot_column(df, "Inflation") plots the "Inflation" column
    
    :param: df, pandas.DataFrame, the data, e.g. df = pd.read_excel("USMacroData", "All")
    :param: feature, str, name of column to be plotted. 
    """
    y = df[feature].resample('MS').mean()
    y.plot(figsize=(18, 8))
    plt.xlabel('Date')
    plt.ylabel(feature)
    plt.show()