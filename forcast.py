import pandas as pd
from neuralprophet import NeuralProphet
import streamlit as st
import plotly.graph_objects as go

def plot_forecast_plotly(df_history, forecast_df, buffer_percent):
    forecast_df['Recommended_Balance'] = forecast_df['yhat1'] * (1 + buffer_percent / 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_history['ds'], y=df_history['y'],
                             mode='markers+lines', name='Historical',
                             marker=dict(color='blue'), line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat1'],
                             mode='lines', name='Forecast', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['Recommended_Balance'],
                             mode='lines', name='Recommended (+buffer)',
                             line=dict(color='green', dash='dash')))
    fig.update_layout(title='Weekly Forecast (NeuralProphet)',
                      xaxis_title='Week', yaxis_title='SumValue',
                      legend=dict(x=0, y=1), template='plotly_white')
    return fig

st.set_page_config(page_title="Weekly Forex Balance Forecast (AI)", layout="wide")
st.title("🧠 AI-Powered Weekly Forex Balance Forecast")

uploaded_file = st.file_uploader("📂 Upload CSV (Date, Currency, SumValue)", type=["csv"])
buffer_percent = st.slider("Select buffer percentage (%)", 0, 50, 10)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'Currency', 'SumValue'])
        df['SumValue'] = pd.to_numeric(df['SumValue'], errors='coerce')
        df = df.dropna(subset=['SumValue'])

        st.subheader("📊 Cleaned Data")
        st.dataframe(df.head())

        results = []
        forecast_append = []

        for currency in df['Currency'].unique():
            st.markdown(f"### 🔮 AI Weekly Forecast for {currency}")
            df_currency = df[df['Currency'] == currency][['Date', 'SumValue']]
            df_currency = df_currency.resample('W', on='Date').sum().reset_index()
            df_currency = df_currency.rename(columns={'Date': 'ds', 'SumValue': 'y'})

            if len(df_currency) < 5:
                st.warning(f"Not enough weekly data for {currency}. Skipping.")
                continue

            model = NeuralProphet(yearly_seasonality=False, weekly_seasonality=True, learning_rate=0.01)
            metrics = model.fit(df_currency, freq='W')
            future = model.make_future_dataframe(df_currency, periods=4, n_historic_predictions=False)
            forecast = model.predict(future)

            last_actual = df_currency['y'].iloc[-1]
            forecast['Last_Actual'] = last_actual
            forecast['Recommended_Balance'] = forecast['yhat1'] * (1 + buffer_percent / 100)

            forecast_display = forecast[['ds', 'Last_Actual', 'yhat1', 'yhat1_lower', 'yhat1_upper', 'Recommended_Balance']].copy()
            forecast_display = forecast_display.rename(columns={
                'ds': 'Week_Start', 'yhat1': 'Predicted',
                'yhat1_lower': 'Lower_Bound', 'yhat1_upper': 'Upper_Bound'
            }).round(2)

            st.dataframe(forecast_display)

            results.append({"Currency": currency,
                            "Total_Recommended_Balance_4weeks": round(forecast['Recommended_Balance'].sum(), 2)})

            fig = plot_forecast_plotly(df_currency, forecast, buffer_percent)
            st.plotly_chart(fig, use_container_width=True)

        if results:
            results_df = pd.DataFrame(results)
            st.subheader("📌 Total Recommended Balance (Next 4 Weeks)")
            st.dataframe(results_df)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a CSV to start forecasting.")
