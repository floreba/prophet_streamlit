import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go

# Title of the app
import streamlit as st

st.header("Forecasting made easy 📈")
st.write("This app provides an easy to use interface to generate accurate time series forecasts using Facebooks Prophet model.")
st.divider()


# Instructions in the Sidebar
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Upload your time historic series data file in CSV or Excel format. It should contain a column with dates and one with numeric values that you want to forecast.
2. Select the appropriate columns.
3. (Optional) If you want to include public holidays in your forecasting model, select the option and enter the country code.
4. Choose the number of days you want to forecast.
5. View the forecasted results and analyze the components of the forecast.
""")
st.sidebar.warning("Looking to integrate a more customizable forecasting model into your data pipeline?")
st.sidebar.subheader("Let's connect:")
st.sidebar.write("weiterleitung.duplex110@passinbox.com")


# File uploader for CSV or Excel
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file:
    # Reading the main data file
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("Data loaded successfully! Here is a preview of your data:")
        st.write(df.head())

        st.info("Now, let's select the columns we'll use for the forecasting.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Checking columns
    columns = df.columns.tolist()
    if not columns:
        st.error("The uploaded file has no columns. Please check your file.")
    else:
        ds_column = st.selectbox("Select the date column", columns)
        y_column = st.selectbox("Select the target column (the values you want to predict)", columns)

        # Check if date and value columns are the same
        if ds_column == y_column:
            st.error("The date column and the target column cannot be the same. Please select different columns.")
            st.stop()

        # Ensuring correct data types
        try:
            # Convert date column to datetime
            df[ds_column] = pd.to_datetime(df[ds_column], errors='coerce')

            # Convert target column to numeric
            df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
        except Exception as e:
            st.error(f"Error converting data types: {e}")
            st.stop()

        # Preparing the dataframe for Prophet
        try:
            df_prophet = df[[ds_column, y_column]]
            df_prophet.rename(columns={ds_column: 'ds', y_column: 'y'}, inplace=True)
            st.success("Data preparation is complete. You can now optionally include public holidays in your forecasting model.")
        except KeyError as e:
            st.error(f"Error in selecting columns: {e}")
            st.stop()

        # Option to include Public Holidays using a Radio Button
        include_holidays = st.radio(
            "Would you like to include public holidays in the model?",
            ("No", "Yes")
        )

        if include_holidays == "Yes":
            country = st.text_input("Enter the country code for holidays (e.g., 'US' for United States, 'DE' for Germany)", value='DE')

        # Slider to select the number of days to forecast
        periods = st.slider("Select the number of days to forecast", 1, 365, 30)

        # Stop the script here if the user is selecting or deselecting the public holiday option
        if st.button("Create Forecast"):
            # Setting up the Prophet model
            model = Prophet()

            if include_holidays == "Yes" and country:
                model.add_country_holidays(country_name=country)

            # Fitting the model
            try:
                model.fit(df_prophet)
            except Exception as e:
                st.error(f"Error fitting the model: {e}")
                st.stop()

            # Making future predictions
            future = model.make_future_dataframe(periods=periods, freq="D")
            try:
                forecast = model.predict(future)
                st.success("Forecast complete! Review the results below.")
            except Exception as e:
                st.error(f"Error in prediction: {e}")
                st.stop()

            # Plotting the forecast using Plotly
            st.write("Forecasted Data:")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            # Plotly forecast plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                name='Actual',
                x=df_prophet['ds'],
                y=df_prophet['y'],
                mode='markers',
            ))

            fig.add_trace(go.Scatter(
                name='Forecast',
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
            ))

            fig.add_trace(go.Scatter(
                name='Upper Bound',
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                name='Lower Bound',
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                mode='lines',
                fill='tonexty',
                line=dict(width=0),
                showlegend=False
            ))

            fig.update_layout(
                title="Forecast Plot",
                xaxis_title="Date",
                yaxis_title="Value",
            )

            st.plotly_chart(fig)
            st.write("Components Plot:")
            try:
                st.info("Here you can see how different components like trend, weekly seasonality, and holidays influence the forecast.")
                st.write(model.plot_components(forecast))
            except Exception as e:
                st.error(f"Error generating components plot: {e}")

else:
    st.write("Please upload a file to begin.")
