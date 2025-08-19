import yfinance as yf
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

st.title('Stock Price Forecasting using ARIMA')

# Stock symbol input
stock = st.text_input('Enter stock symbol (e.g., AAPL, GOOGL, MSFT):', 'AAPL')
days = st.number_input('Enter number of days to forecast:', min_value=1, value=1)

if stock:
    df = yf.download(stock, period='1y', interval='1d')
    if not df.empty:
        data = df['Close'].dropna()
        st.line_chart(data)

        model = ARIMA(data, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=int(days))

        st.write(f'Forecast for next {days} day(s):')
        st.write(forecast)

        # Plot forecast
        plt.figure(figsize=(10,4))
        plt.plot(data.index, data, label='Historical')
        plt.plot(pd.date_range(data.index[-1], periods=int(days)+1)[1:], forecast, label='Forecast', color='red')
        plt.legend()
        st.pyplot(plt)
    else:
        st.error('No data found for this symbol.')