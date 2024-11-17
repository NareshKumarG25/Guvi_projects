import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def predict_quantity(df, pizza_id, forecast_periods):
    pizza_data = df[df['pizza_name_id'] == pizza_id].set_index('order_date')

    model = ARIMA(pizza_data['quantity'], order=(5, 1, 0))  
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=forecast_periods)
    
    forecast_dates = pd.date_range(pizza_data.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
    forecast_df = pd.DataFrame({
        'order_date': forecast_dates,
        'pizza_name_id': pizza_id,
        'forecasted_quantity': forecast
    })

    return forecast_df