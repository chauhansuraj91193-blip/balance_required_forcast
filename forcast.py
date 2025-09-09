import pandas as pd
from prophet import Prophet
import streamlit as st
import plotly.graph_objects as go

# ========== Helper function for interactive forecast plot ==========

def plot_forecast_plotly(df_history, forecast_df, buffer_percent):
    """
    df_history: DataFrame with historical data columns ['ds', 'y']
    forecast_df: DataFrame from Prophet with forecast columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    buffer_percent: float, e.g., 10 for 10%
    """

    # Calculate recommended balance with buffer
    forecast_df['Recommended_Balance'] = forecast_df['yhat'] * (1 + buffer_percent / 100)

    fig = go.Figure()

    # Historical actual values
    fig.add_trace(go.Scatter(
        x=df_history['ds'], y=df_history['y'],
        mode='markers+lines',
        name='Historical',
        marker=dict(color='blue'),
        line=dict(color='blue')
    ))

    # Forecast predicted
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'], y=forecast_df['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='orange')
    ))

    # Recommended balance (forecast * (1 + buffer))
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'], y=forecast_df['Recommended_Balance'],
        mode='lines',
        name='Recommended Balance (+buffer)',
        line=dict(color='green', dash='dash')
    ))

    # Confidence interval fill (between yhat_lower and yhat_upper)
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'].tolist() + forecast_df['ds'][::-1].tolist(),
        y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)',  # orange, transparent
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Confidence Interval'
    ))

    fig.update_layout(
        title='Forecast with Confidence Interval and Recommended Balance',
        xaxis_title='Date',
        yaxis_title='SumValue',
        legend=dict(x=0, y=1),
        template='plotly_white',
        hovermode="x unified"
    )

    return fig

# ========== Streamlit UI ==========

st.set_page_config(page_title="Forex Balance Forecast", layout="wide")
st.title("💱 Forex Currency Balance Forecast")

st.markdown("Upload your **CSV with columns: Date, Currency, SumValue** to forecast next 7-day balance requirements.")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

buffer_percent = st.slider("Select buffer percentage (%)", 0, 50, 10)

if uploaded_file:
    try:
        # Load Data
        df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_cols = {'Date', 'Currency', 'SumValue'}
        if not required_cols.issubset(df.columns):
            st.error(f"CSV missing required columns: {required_cols}. Found columns: {df.columns.tolist()}")
            st.stop()

        # Convert Date column to datetime, coerce errors to NaT
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isnull().any():
            st.warning("Some 'Date' values could not be parsed and were set to missing. These rows will be dropped.")
        df = df.dropna(subset=['Date'])

        # Drop rows with missing Currency or SumValue
        df = df.dropna(subset=['Currency', 'SumValue'])

        # Convert SumValue to numeric, coerce errors to NaN then drop
        df['SumValue'] = pd.to_numeric(df['SumValue'], errors='coerce')
        if df['SumValue'].isnull().any():
            st.warning("Some 'SumValue' entries could not be converted to numeric and were dropped.")
        df = df.dropna(subset=['SumValue'])

        if df.empty:
            st.error("No valid data left after cleaning. Please check your CSV file.")
            st.stop()

        st.subheader("📊 Preview of Cleaned Data")
        st.dataframe(df.head())

        results = []

        # Forecast per currency
        for currency in df['Currency'].unique():
            st.markdown(f"### 🔮 Forecast for {currency}")

            df_currency = df[df['Currency'] == currency][['Date', 'SumValue']].rename(columns={'Date': 'ds', 'SumValue': 'y'})

            # Drop NaNs
            df_currency = df_currency.dropna(subset=['ds', 'y'])

            # Need at least 2 records to fit Prophet
            if len(df_currency) < 2:
                st.warning(f"Not enough data to forecast for {currency} (need at least 2 valid records). Skipping.")
                continue

            # Fit model
            model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
            model.fit(df_currency)

            # Make future dataframe for 7 days
            future = model.make_future_dataframe(periods=7, freq='D')
            forecast = model.predict(future)

            # Forecast only next 7 days (exclude historical)
            forecast_7days = forecast[forecast['ds'] > df_currency['ds'].max()].copy()
            forecast_7days['Recommended_Balance'] = forecast_7days['yhat'] * (1 + buffer_percent / 100)

            # Prepare forecast display table
            forecast_display = forecast_7days[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'Recommended_Balance']].copy()
            forecast_display = forecast_display.rename(columns={
                'ds': 'Date',
                'yhat': 'Predicted',
                'yhat_lower': 'Lower_Bound',
                'yhat_upper': 'Upper_Bound'
            })
            forecast_display[['Predicted', 'Lower_Bound', 'Upper_Bound', 'Recommended_Balance']] = forecast_display[
                ['Predicted', 'Lower_Bound', 'Upper_Bound', 'Recommended_Balance']].round(2)

            st.write(f"Forecast table for next 7 days for {currency}:")
            st.dataframe(forecast_display)

            # Append summary for total recommended balance over 7 days
            total_recommended = forecast_display['Recommended_Balance'].sum()
            results.append({
                "Currency": currency,
                "Total_Recommended_Balance_7days": round(total_recommended, 2)
            })

            # Plot interactive forecast chart
            fig = plot_forecast_plotly(df_currency, forecast, buffer_percent)
            st.plotly_chart(fig, use_container_width=True)

        if results:
            results_df = pd.DataFrame(results)
            st.subheader("📌 Summary: Total Recommended Balance Over Next 7 Days")
            st.dataframe(results_df)

            # Download button for summary CSV
            st.download_button(
                label="💾 Download Summary CSV",
                data=results_df.to_csv(index=False),
                file_name="7day_balance_forecast_summary.csv",
                mime="text/csv"
            )
        else:
            st.info("No forecasts generated. Check your data and filters.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to begin forecasting.")
