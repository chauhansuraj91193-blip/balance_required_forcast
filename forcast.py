import pandas as pd
from prophet import Prophet
import streamlit as st

# ========== UI ==========

st.set_page_config(page_title="Forex Balance Forecast", layout="wide")
st.title("💱 Forex Currency Balance Forecast")

st.markdown("Upload your **CSV with columns: Date, Currency, SumValue** to forecast next 7-day balance requirements.")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

buffer_percent = st.slider("Select buffer percentage (%)", 0, 50, 10)

if uploaded_file:
    try:
        # === Load Data ===
        df = pd.read_csv(uploaded_file)

        # Validate required columns exist
        required_cols = {'Date', 'Currency', 'SumValue'}
        if not required_cols.issubset(df.columns):
            st.error(f"CSV missing required columns: {required_cols}. Found columns: {df.columns.tolist()}")
            st.stop()

        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isnull().any():
            st.warning("Some 'Date' values could not be parsed and were set to NaT (missing). These rows will be dropped.")
        df = df.dropna(subset=['Date'])

        # Drop rows with missing Currency or SumValue
        df = df.dropna(subset=['Currency', 'SumValue'])

        # Make sure SumValue is numeric
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

        # Forecast per Currency
        for currency in df['Currency'].unique():
            st.markdown(f"### 🔮 Forecast for {currency}")

            df_currency = df[df['Currency'] == currency][['Date', 'SumValue']].rename(columns={'Date': 'ds', 'SumValue': 'y'})

            # Drop NaNs in ds or y
            df_currency = df_currency.dropna(subset=['ds', 'y'])

            # Prophet needs at least 2 rows
            if len(df_currency) < 2:
                st.warning(f"Not enough data to forecast for {currency} (need at least 2 valid records). Skipping.")
                continue

            # Build and fit model
            model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
            model.fit(df_currency)

            # Forecast next 7 days
            future = model.make_future_dataframe(periods=7, freq='D')
            forecast = model.predict(future)

            # Filter forecast for only future 7 days (not including historical)
            forecast_7days = forecast[forecast['ds'] > df_currency['ds'].max()]

            # Calculate recommended balance with buffer per day
            forecast_7days['Recommended_Balance'] = forecast_7days['yhat'] * (1 + buffer_percent / 100)

            # Display forecast table for next 7 days
            forecast_display = forecast_7days[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'Recommended_Balance']]
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

            # Append summary info: sum of recommended balances over 7 days (optional)
            total_recommended = forecast_display['Recommended_Balance'].sum()
            results.append({
                "Currency": currency,
                "Total_Recommended_Balance_7days": round(total_recommended, 2)
            })

            # Plot forecast with confidence intervals using built-in Prophet plot (matplotlib)
            fig = model.plot(forecast)
            st.pyplot(fig)

        if results:
            results_df = pd.DataFrame(results)
            st.subheader("📌 Summary: Total Recommended Balance Over Next 7 Days")
            st.dataframe(results_df)

            # Download button for summary
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
