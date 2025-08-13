import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="NIFTY 50 Dashboard History 2025",
    layout="wide",
)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\THARUNBALAAJE\Downloads\NIFTY 50-01-01-2025-to-13-08-2025.csv")
        df['Date'] = pd.to_datetime(df['Date '], format='%d-%b-%Y')
        df = df.drop(columns=['Date '])
        df = df.rename(columns={
            'Open ': 'Open',
            'High ': 'High',
            'Low ': 'Low',
            'Close ': 'Close',
            'Shares Traded ': 'Shares Traded',
            'Turnover (₹ Cr)': 'Turnover (Cr)'
        })
        df.set_index('Date', inplace=True)
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please check the file path.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# ---------- MAIN FUNCTION ----------
def main():
    # ----- LOGIN -----
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        st.sidebar.header("Admin Login")
        with st.sidebar.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Log In")

        if submit:
            if username == "Tharun" and password == "Nifty50":
                st.session_state['logged_in'] = True
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    if st.session_state['logged_in']:
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()

        st.title("NIFTY 50 Stock Analysis Dashboard (2025) for 01-01-2025 to 13-08-2025")
        st.markdown("---")

        df = load_data()
        if df.empty:
            return

        # ----- SIDEBAR -----
        st.sidebar.header("Filters")
        min_date, max_date = df.index.min(), df.index.max()

        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        start_date, end_date = date_range if len(date_range) == 2 else (date_range[0], max_date)
        filtered_df = df.loc[start_date:end_date]

        # Theme
        theme = st.sidebar.radio("Chart Theme", ["plotly_white", "plotly_dark"])
        chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line", "Area"])
        show_volume = st.sidebar.checkbox("Show Volume Chart", True)

        st.download_button(
            label="Download Filtered Data",
            data=filtered_df.to_csv().encode('utf-8'),
            file_name="nifty50_filtered.csv",
            mime="text/csv"
        )

        if filtered_df.empty:
            st.warning("No data in selected range.")
            return

        # ----- KPIs -----
        st.subheader("Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)

        current_close = filtered_df['Close'].iloc[-1]
        prev_close = filtered_df['Close'].iloc[-2] if len(filtered_df) > 1 else None
        delta = current_close - prev_close if prev_close else 0
        col1.metric("Current Close", f"₹ {current_close:,.2f}", f"{delta:,.2f}")
        col2.metric("Highest Price", f"₹ {filtered_df['High'].max():,.2f}")
        col3.metric("Lowest Price", f"₹ {filtered_df['Low'].min():,.2f}")
        col4.metric("Total Turnover", f"₹ {filtered_df['Turnover (Cr)'].sum():,.2f} Cr")

        # ----- MOVING AVERAGES -----
        filtered_df['SMA_20'] = filtered_df['Close'].rolling(20).mean()
        filtered_df['SMA_50'] = filtered_df['Close'].rolling(50).mean()

        st.markdown("---")
        st.subheader("Interactive Stock Chart")
        fig = go.Figure()

        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=filtered_df.index,
                open=filtered_df['Open'],
                high=filtered_df['High'],
                low=filtered_df['Low'],
                close=filtered_df['Close'],
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
        elif chart_type == "Line":
            fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Close'], mode='lines', name='Close', line=dict(color='orange')))
        elif chart_type == "Area":
            fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Close'], fill='tozeroy', name='Close', line=dict(color='orange')))

        fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', dash='dot')))
        fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='purple', dash='dot')))
        fig.update_layout(template=theme, xaxis_title="Date", yaxis_title="Price (₹)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # ----- VOLUME -----
        if show_volume:
            st.subheader("Shares Traded Volume")
            fig_vol = px.bar(filtered_df, x=filtered_df.index, y="Shares Traded", title="Daily Shares Traded", template=theme)
            st.plotly_chart(fig_vol, use_container_width=True)

        # ----- MONTHLY TURNOVER -----
        st.subheader("Turnover Share by Month")
        turnover_monthly = filtered_df.copy()
        turnover_monthly['Month'] = turnover_monthly.index.month_name()
        monthly_sum = turnover_monthly.groupby('Month')['Turnover (Cr)'].sum().reset_index()
        fig_pie = px.pie(monthly_sum, names='Month', values='Turnover (Cr)', title="Monthly Turnover Share", template=theme, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

        # ----- EXPONENTIAL SMOOTHING FORECAST -----
        st.subheader("Exponential Smoothing Forecast (Next 10 Days)")
        try:
            df_2024 = filtered_df[filtered_df.index >= "2024-01-01"]
            model = ExponentialSmoothing(filtered_df['Close'], trend="add", seasonal=None)
            fit = model.fit()
            forecast = fit.forecast(10)
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Close'], mode='lines', name='Historical'))
            fig_forecast.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))
            st.plotly_chart(fig_forecast, use_container_width=True)
        except Exception as e:
            st.error(f"Forecasting error: {e}")

        # ----- AUTOCORRELATION & PARTIAL AUTOCORRELATION -----
        st.subheader("Autocorrelation & Partial Autocorrelation")
        fig_acf, ax_acf = plt.subplots()
        plot_acf(filtered_df['Close'], ax=ax_acf, lags=20)
        st.pyplot(fig_acf)

        fig_pacf, ax_pacf = plt.subplots()
        plot_pacf(filtered_df['Close'], ax=ax_pacf, lags=20, method='ywm')
        st.pyplot(fig_pacf)

        # ----- AUTOREGRESSIVE MODEL -----
        st.subheader("Autoregressive Model Prediction (AR)")
        try:
            model_ar = AutoReg(filtered_df['Close'], lags=5)
            model_fit = model_ar.fit()
            pred = model_fit.predict(start=len(filtered_df), end=len(filtered_df)+9)
            st.line_chart(pred)
        except Exception as e:
            st.error(f"AR model error: {e}")

        # ----- LEAD & LAG INDICATORS -----
        st.subheader("Lead & Lag Indicators")
        filtered_df['Close_Lag1'] = filtered_df['Close'].shift(1)
        filtered_df['Close_Lead1'] = filtered_df['Close'].shift(-1)
        st.dataframe(filtered_df[['Close', 'Close_Lag1', 'Close_Lead1']].tail(10))


if __name__ == "__main__":
    main()