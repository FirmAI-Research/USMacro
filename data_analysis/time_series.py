#@title ```time_series.py```


def plot_column(df, feature):
    """Plot the resampled column of df, e.g. plot_column(df, "Inflation") plots the "Inflation" column
    
    :param: df, pandas.DataFrame, the data, e.g. df = pd.read_excel("USMacroData", "All")
    :param: feature, str, name of column to be plotted. 
    """
    y = df[feature].resample('MS').mean()
    y.plot(figsize=(15, 6))
    plt.show()


def plot_component(df, feature):
    """Decompose the time series data into trend, seasonal, and residual components.
    
    :param: df, pd.DataFram.
    :param: feature, str,column name/feature name we want to decompose
    :rtype: None
    """
    decomposition = sm.tsa.seasonal_decompose(df[feature].resample("MS").mean(), model='additive')
    fig = decomposition.plot()
    plt.show()






###### This section uses ARIMA to analyze the data and make predictions.########################################

# Grid search to find the best ARIMA parameters 
def arima_parameters(df, feature, search_range=2):
    """Grid search for the optimal parameters of the Arima model for given data (df) and feature.
    :param: df, pdf.DataFrame, data
    :param: feature, str, feature name.
    :param: search_range, int, the range for the search of the parameters, the default value is 2
    """
    p = d = q = range(0, search_range)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    minimal_aic = 0
    optimal_param =[]
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(df[feature].resample('MS').mean(),order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                if results.aic < minimal_aic:
                    optimal_param = [param, param_seasonal]
                    minimal_aic = results.aic
                    print(minimal_aic)
            except:
                continue
    print('\n Optimal parameters ARIMA{}x{}12 - Minimal AIC:{}'.format(optimal_param[0], optimal_param[1], minimal_aic))
    return optimal_param[0], optimal_param[1]



def arima_train(df, feature):
    """Train the arima model with the optimal parameters computed for df and feature.
    """
    order, seasonal_order =  arima_parameters(df, feature)
    mod = sm.tsa.statespace.SARIMAX(df[feature].resample('MS').mean(),
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()
    return results

def arima_diagonostics(results):
    results.plot_diagnostics(figsize=(16, 8))
    plt.show()

def arima_table(results):
    print(results.summary().tables[1])

def arima_predict(results, df, feature, init_date = "2009-01-01", start_date = "2012-01-01"): 
    pred = results.get_prediction(start=pd.to_datetime(start_date), dynamic=False)
    pred_ci = pred.conf_int()
    y = df[feature].resample("MS").mean() 
    ax = y[init_date:].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel(feature)
    plt.legend()
    plt.show()
    y_forecasted = pred.predicted_mean
    y_truth = y['2012-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}\n'.format(round(mse, 20)))
    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))    


def arima_forcast(results, df, feature):
    pred_uc = results.get_forecast(steps=100)
    pred_ci = pred_uc.conf_int()
    y = df[feature].resample("MS").mean() 
    ax = y.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel(feature)
    plt.legend()
    plt.show()

