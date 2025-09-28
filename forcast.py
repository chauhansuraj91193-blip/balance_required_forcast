import pandas as pd
from neuralprophet import NeuralProphet
import streamlit as st
import plotly.graph_objects as go

# ================= Helper function =================
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
    
    if 'yhat1_lower' in forecast_df.columns and 'yhat1_upper' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'].tolist() + forecast_df['ds'][::-1].tolist(),
            y=forecast_df['yhat1_upper'].tolist() + forecast_df['yhat1_lower'][::-1].tolist(),
            fill='toself', fillcolor='rgba(255,165,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip",
            showlegend=True, name='Confidence Interval'
        ))
        
    fig.update_layout(title='Weekly Forecast with Recommended Balance',
                      xaxis_title='Week', yaxis_title='SumValue',
                      template='plotly_white', hovermode='x unified')
    return fig

# ================= Streamlit UI =================
st.set_page_config(page_title="Weekly Forex Balance Forecast", layout="wide")
st.title("🧠 AI-Powered Weekly Forex Balance Forecast")

uploaded_file = st.file_uploader("📂 Upload CSV (Date, Currency, SumValue[, Forecast_Balance])", type=["csv"])
buffer_percent = st.slider("Select buffer percentage (%)", 0, 50, 10)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
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
            df_currency = df[df['Currency'] == currency][['Date', 'SumValue']].copy()
            df_currency = df_currency.resample('W', on='Date').sum().reset_index().rename(columns={'Date':'ds','SumValue':'y'})

            if len(df_currency) < 4:
                st.warning(f"Not enough weekly data for {currency}. Skipping.")
                continue

            # ---------- SELF-LEARNING BIAS ----------
            adjustment_factor = 1.0
            if 'Forecast_Balance' in df.columns:
                df_forecast_compare = df[df['Currency']==currency].dropna(subset=['Forecast_Balance'])
                df_forecast_compare = df_forecast_compare[df_forecast_compare['Forecast_Balance']>0] # avoid div0
                if not df_forecast_compare.empty:
                    df_forecast_compare['error'] = (df_forecast_compare['SumValue'] - df_forecast_compare['Forecast_Balance']) / df_forecast_compare['Forecast_Balance']
                    mean_bias = df_forecast_compare['error'].mean()
                    if pd.notna(mean_bias) and mean_bias not in [float('inf'), float('-inf')]:
                        adjustment_factor = 1 + mean_bias
                        st.info(f"📈 Historical bias detected for {currency}: {mean_bias:.2%}. Adjusting new forecast by {adjustment_factor:.2f}x.")
                    else:
                        st.warning(f"⚠️ Skipping bias adjustment for {currency} due to invalid forecast data.")

            # ---------- MODEL FITTING ----------
            model = NeuralProphet(yearly_seasonality=False, weekly_seasonality=True, uncertainty_samples=1000)
            model.fit(df_currency, freq='W')

            future = model.make_future_dataframe(df_currency, periods=4, n_historic_predictions=False)
            forecast = model.predict(future)

            # Apply bias adjustment
            forecast['yhat1'] *= adjustment_factor
            if 'yhat1_lower' in forecast.columns and 'yhat1_upper' in forecast.columns:
                forecast['yhat1_lower'] *= adjustment_factor
                forecast['yhat1_upper'] *= adjustment_factor

            # Recommended Balance
            forecast['Recommended_Balance'] = forecast['yhat1'] * (1 + buffer_percent / 100)
            forecast['Last_Actual'] = df_currency['y'].iloc[-1]

            forecast_display = forecast[['ds','Last_Actual','yhat1','yhat1_lower','yhat1_upper','Recommended_Balance']].copy()
            forecast_display = forecast_display.rename(columns={'ds':'Week_Start','Last_Actual':'Actual_Used','yhat1':'Predicted','yhat1_lower':'Lower_Bound','yhat1_upper':'Upper_Bound'})
            forecast_display = forecast_display.round(2)
            st.dataframe(forecast_display)

            # Append for CSV
            for _, row in forecast.iterrows():
                forecast_append.append({
                    "Date": row['ds'],
                    "Currency": currency,
                    "SumValue": None,
                    "Forecast_Balance": round(row['Recommended_Balance'],2)
                })

            results.append({
                "Currency": currency,
                "Total_Recommended_Balance_4weeks": round(forecast['Recommended_Balance'].sum(),2)
            })

            fig = plot_forecast_plotly(df_currency, forecast, buffer_percent)
            st.plotly_chart(fig, use_container_width=True)

        if results:
            results_df = pd.DataFrame(results)
            st.subheader("📌 Total Recommended Balance (Next 4 Weeks, per Currency)")
            st.dataframe(results_df)

            grand_total = results_df['Total_Recommended_Balance_4weeks'].sum()
            st.markdown(f"## 🏦 **Grand Total Recommended Balance (4 Weeks): {grand_total:,.2f}**")

            df['Forecast_Balance'] = None
            forecast_df = pd.DataFrame(forecast_append)
            combined_df = pd.concat([df, forecast_df], ignore_index=True)

            st.download_button(
                label="💾 Download Full CSV (Actual + 4-Week Forecast)",
                data=combined_df.to_csv(index=False),
                file_name="forecast_weekly.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a CSV to start forecasting.")
