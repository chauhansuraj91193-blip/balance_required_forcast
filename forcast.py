import pandas as pd
from prophet import Prophet
import streamlit as st

# ========== UI ==========
st.set_page_config(page_title="Forex Balance Forecast", layout="wide")
st.title("💱 Forex Currency Balance Forecast")

st.markdown("Upload your **CSV with Date, Currency, SumValue** to forecast next-day balance requirements.")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

buffer_percent = st.slider("Select buffer percentage (%)", 0, 50, 10)

if uploaded_file:
    # === Load Data ===
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    st.subheader("📊 Preview of Data")
    st.dataframe(df.head())

    results = []
    plots = {}

    # === Forecast per Currency ===
    for currency in df['Currency'].unique():
        st.markdown(f"### 🔮 Forecast for {currency}")

        df_currency = df[df['Currency'] == currency][['Date', 'SumValue']]
        df_currency = df_currency.rename(columns={'Date': 'ds', 'SumValue': 'y'})

        # Build model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        model.fit(df_currency)

        # Forecast next day
        future = model.make_future_dataframe(periods=1, freq='D')
        forecast = model.predict(future)

        next_day = forecast.tail(1).iloc[0]
        predicted = next_day['yhat']
        lower = next_day['yhat_lower']
        upper = next_day['yhat_upper']
        with_buffer = predicted * (1 + buffer_percent/100)

        results.append({
            "Currency": currency,
            "Predicted_NextDay": round(predicted, 2),
            "Lower_Bound": round(lower, 2),
            "Upper_Bound": round(upper, 2),
            "Recommended_Balance": round(with_buffer, 2)
        })

        # Plot
        fig = model.plot(forecast)
        st.pyplot(fig)

    # === Show Results Table ===
    results_df = pd.DataFrame(results)
    st.subheader("📌 Next-Day Balance Forecast")
    st.dataframe(results_df)

    # Download option
    st.download_button(
        label="💾 Download Forecast CSV",
        data=results_df.to_csv(index=False),
        file_name="next_day_forecast.csv",
        mime="text/csv"
    )
