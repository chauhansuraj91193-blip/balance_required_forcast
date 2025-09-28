import pandas as pd
from prophet import Prophet
import streamlit as st
import plotly.graph_objects as go

# ========== Helper function for interactive forecast plot ==========
def plot_forecast_plotly(df_history, forecast_df, buffer_percent):
    forecast_df['Recommended_Balance'] = forecast_df['yhat'] * (1 + buffer_percent / 100)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_history['ds'], y=df_history['y'],
                             mode='markers+lines', name='Historical',
                             marker=dict(color='blue'), line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'],
                             mode='lines', name='Forecast', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['Recommended_Balance'],
                             mode='lines', name='Recommended Balance (+buffer)',
                             line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'].tolist() + forecast_df['ds'][::-1].tolist(),
        y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'][::-1].tolist(),
        fill='toself', fillcolor='rgba(255, 165, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip",
        showlegend=True, name='Confidence Interval'))

    fig.update_layout(title='Forecast with Confidence Interval and Recommended Balance',
                      xaxis_title='Date', yaxis_title='SumValue',
                      legend=dict(x=0, y=1), template='plotly_white', hovermode="x unified")
    return fig


# ========== Streamlit UI ==========
st.set_page_config(page_title="Forex Balance Forecast", layout="wide")
st.title("💱 Forex Currency Balance Forecast")

uploaded_file = st.file_uploader("📂 Upload CSV file (Date, Currency, SumValue)", type=["csv"])
buffer_percent = st.slider("Select buffer percentage (%)", 0, 50, 10)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Cleaning
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'Currency', 'SumValue'])
        df['SumValue'] = pd.to_numeric(df['SumValue'], errors='coerce')
        df = df.dropna(subset=['SumValue'])

        st.subheader("📊 Preview of Cleaned Data")
        st.dataframe(df.head())

        results = []
        forecast_append = []

        for currency in df['Currency'].unique():
            st.markdown(f"### 🔮 Forecast for {currency}")
            df_currency = df[df['Currency'] == currency][['Date', 'SumValue']].rename(columns={'Date': 'ds', 'SumValue': 'y'})

            if len(df_currency) < 4:
                st.warning(f"Not enough data to forecast for {currency}. Skipping.")
                continue

            # Add logistic growth constraints
            df_currency['cap'] = df_currency['y'].max() * 1.5
            df_currency['floor'] = 0

            model = Prophet(growth='logistic', weekly_seasonality=True, yearly_seasonality=False)
            model.fit(df_currency)

            future = model.make_future_dataframe(periods=7, freq='D')
            future['cap'] = df_currency['cap'].iloc[0]
            future['floor'] = 0

            forecast = model.predict(future)

            # Clip any negative predictions
            forecast['yhat'] = forecast['yhat'].clip(lower=0)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

            forecast_7days = forecast[forecast['ds'] > df_currency['ds'].max()].copy()
            forecast_7days['Recommended_Balance'] = forecast_7days['yhat'] * (1 + buffer_percent / 100)

            forecast_display = forecast_7days[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'Recommended_Balance']].copy()
            forecast_display = forecast_display.rename(columns={
                'ds': 'Date', 'yhat': 'Predicted',
                'yhat_lower': 'Lower_Bound', 'yhat_upper': 'Upper_Bound'
            })
            forecast_display = forecast_display.round(2)

            st.write(f"Forecast table for next 7 days for {currency}:")
            st.dataframe(forecast_display)

            for _, row in forecast_7days.iterrows():
                forecast_append.append({
                    "Date": row['ds'],
                    "Currency": currency,
                    "SumValue": None,
                    "Forecast_Balance": round(row['Recommended_Balance'], 2)
                })

            total_recommended = forecast_7days['Recommended_Balance'].sum()
            results.append({
                "Currency": currency,
                "Total_Recommended_Balance_7days": round(total_recommended, 2)
            })

            fig = plot_forecast_plotly(df_currency, forecast, buffer_percent)
            st.plotly_chart(fig, use_container_width=True)

        if results:
            results_df = pd.DataFrame(results)
            st.subheader("📌 Summary: Total Recommended Balance Over Next 7 Days (Per Currency)")
            st.dataframe(results_df)

            grand_total = results_df['Total_Recommended_Balance_7days'].sum()
            st.markdown(f"## 🏦 **Total Recommended Balance for Next 7 Days:** {grand_total:,.2f}")

            df['Forecast_Balance'] = None
            forecast_df = pd.DataFrame(forecast_append)
            combined_df = pd.concat([df, forecast_df], ignore_index=True)

            st.download_button(
                label="💾 Download Full CSV (Actual + 7-day Forecast)",
                data=combined_df.to_csv(index=False),
                file_name="forecast_appended.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a CSV to start forecasting.")
