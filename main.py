from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import seaborn as sns
import statsmodels as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna


from statsmodels.tsa.arima.model import ARIMA

from streamlit_extras.metric_cards import style_metric_cards




st.set_page_config(
    page_title="Prophet",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

style_metric_cards(
    background_color="#0990AA",
    border_size_px=3,
    border_color="#ccc",
    border_radius_px=20,
    box_shadow="0px 4px 8px rgba(0, 0, 0, 0.1)"
)
sns.set_theme(style = None, rc = {"axes.facecolor":"none","figure.facecolor":"none","axes.grid":True,"grid.color":"#5881af","axes.edgecolor":'black'})
st.sidebar.image("images.png", caption='Prophet Model',use_container_width=True)

file = st.sidebar.file_uploader("Upload the CSV file", type="CSV")
st.title("Prophet Model")
st.markdown("### Dataset")

if file is not None:

    axis=pd.read_csv(file)
    axis.columns = axis.columns.str.strip().str.lower()
    st.write(axis)
    a,b = st.columns(2)
    with a:
        st.write("Shape of the Data:",axis.shape)

    axis = axis[['date', 'open', 'high', 'low',"close","volume"]]
    axis.columns = ['Date', 'Open', 'High', 'Low',"Close","Volume"]

    axis["Date"]=pd.to_datetime(axis['Date'])
    with b:
        st.write(f"The data is from {axis.Date[0].date()} to {axis.Date[len(axis)-1].date()}")
    axis.set_index("Date",inplace=True)

    axis[["Open", "High", "Low", "Close", "Volume"]] = (
    axis[["Open", "High", "Low", "Close", "Volume"]]
    .replace({",": ""}, regex=True)  # Remove commas
    .apply(pd.to_numeric, errors="coerce")  # Convert to numeric
    )


    n = len(axis)
    data = axis.Close[:n-25]




    st.sidebar.subheader("Provide Input Features")

    n_changepoints = st.sidebar.number_input("Select n_changepoints", min_value=1, max_value=100, step=1, value =15)
    seasonality_mode = st.sidebar.selectbox("Select model type",["additive",'multiplicative'],index = 1)
    yearly_seasonality = st.sidebar.slider("Select yearly Seasonality",min_value = 1.0,max_value = 30.0, step = 0.1, value = 5.74)
    changepoint_prior_scale = st.sidebar.slider("Select Changepoint prior scale",min_value = 0.01, max_value = 20.0, step = 0.01,value = 0.70)
    seasonality_prior_scale = st.sidebar.slider("Select seasonality prior scale",min_value = 0.1, max_value = 20.0, step = 0.1,value = 2.61)
    holiday_prior_scale = st.sidebar.slider("Select Holiday prior scale",min_value = 0.1, max_value = 10.0, step = 0.1,value = 6.6)


    data = pd.DataFrame(data)
    data["ds"] = data.index
    data.columns  =["y","ds"]
    data.reset_index(inplace = True)
    data.drop("Date",inplace=True,axis = 1)



    def objective(trial):
        changepoint_prior_scale = trial.suggest_loguniform("changepoint_prior_scale", 0.01, 4)
        seasonality_prior_scale = trial.suggest_loguniform("seasonality_prior_scale", 0.01, 20)
        holidays_prior_scale = trial.suggest_loguniform("holidays_prior_scale", 0.01, 10)
        yearly_seasonality = trial.suggest_loguniform("yearly_seasonality", 2, 20)
        n_changepoints = trial.suggest_uniform("n_changepoints", 1, 20)
        seasonality_mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])


        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,

            yearly_seasonality=yearly_seasonality,
            seasonality_mode=seasonality_mode
            )




        model.fit(data)

        future = model.make_future_dataframe(periods=25)
        forecast = model.predict(future)

        y_true = axis.Close[n-25:].values
        y_pred = forecast["yhat"][n-25:].values
        return mean_absolute_error(y_true, y_pred)

    button = st.button("Tune using Optuna")


    if button:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        st.session_state.params = study.best_params
    #if "params" in st.session_state:
        #st.write(st.session_state.params)

    if "params" not in st.session_state:

        model = Prophet(daily_seasonality=True,n_changepoints=n_changepoints,seasonality_mode=seasonality_mode,
                yearly_seasonality = yearly_seasonality, changepoint_prior_scale = changepoint_prior_scale)
    else:
        if st.session_state.params != None:
            st.markdown("**Parameter tuned using Optuna**")
            st.write(st.session_state.params)
            model = Prophet(daily_seasonality=True, n_changepoints=int(st.session_state.params["n_changepoints"]), seasonality_mode=st.session_state.params["seasonality_mode"],
                        yearly_seasonality=st.session_state.params["yearly_seasonality"], changepoint_prior_scale=st.session_state.params["changepoint_prior_scale"])
            st.session_state.params = None
        else:
            model = Prophet(daily_seasonality=True, n_changepoints=n_changepoints, seasonality_mode=seasonality_mode,
                            yearly_seasonality=yearly_seasonality, changepoint_prior_scale=changepoint_prior_scale)

    model.fit(data)



    future = model.make_future_dataframe(periods=25)




    forecast  = model.predict(future)



    # fig, ax = plt.subplots(figsize=(10, 5))
    #
    # Plot actual values
    # ax.plot(axis.Close.reset_index(drop=True), label='Actual')
    # ax.plot(forecast.yhat[n-25:], label='Forecast')
    # ax.set_title("Prophet Model Actual vs Forecast(for 25 days)")
    # ax.legend()
    # st.pyplot(fig)

    actual_y = axis.Close.reset_index(drop=True)
    actual_x = axis.index  # Should be datetime index

    # Forecast 1
    forecast_y = forecast.yhat.tail(25).reset_index(drop=True)
    forecast_x = axis.tail(25).index  # Should be datetime index

    # Forecast 2


    # Create the figure
    fig = go.Figure()

    # Actual values line
    fig.add_trace(go.Scatter(
        x=actual_x,
        y=actual_y,
        mode='lines',
        name='Actual',
        line=dict(color='royalblue')
    ))

    # Forecast 1 line
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast_y,
        mode='lines',
        name='Forecast 1',
        line=dict(color='orange')
    ))



    # Layout customization
    fig.update_layout(
        title='Prophet Model Actual vs Forecast (for 25 days)',
        xaxis=dict(
                title='Time',
                showgrid=True,  # Enable vertical grid lines
                gridcolor='royalblue'  # Optional: soft gridline color
            ),
            yaxis=dict(
                title='Stock Price',
                showgrid=True,  # Enable horizontal grid lines
                gridcolor='royalblue'
            ),
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0)'),
        width=900,
        height=500
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)


    mae = mean_absolute_error(axis.Close[n-25:], forecast.yhat[n-25:])


    mse = mean_squared_error(axis.Close[n-25:], forecast.yhat[n-25:])
    rmse = np.sqrt(mse)




    mape = np.mean(np.abs((np.array(forecast.yhat[n-25:]) - np.array(axis.Close[n-25:])) / axis.Close[n-25:])) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label = "Mean Absolute Error",value = round(mae,2))
    with col2:
        st.metric(label="Root Mean Square Error", value=round(rmse,2))

    with col3:
        st.metric(label="Mean Absolute Percentage Error", value=round(mape,2))

    forecast = st.button("Forecast using total data")

    if forecast:

        data2 = axis.Close

        data2 = pd.DataFrame(data2)
        data2["ds"] = data2.index
        data2.columns = ["y", "ds"]
        data2.reset_index(inplace=True)
        data2.drop("Date", inplace=True, axis=1 )




        if "params" not in st.session_state:

            model2 = Prophet(daily_seasonality=True, n_changepoints=n_changepoints, seasonality_mode=seasonality_mode,
                            yearly_seasonality=yearly_seasonality, changepoint_prior_scale=changepoint_prior_scale)
        else:
            if st.session_state.params != None:


                model2 = Prophet(daily_seasonality=True, n_changepoints=int(st.session_state.params["n_changepoints"]),
                                seasonality_mode=st.session_state.params["seasonality_mode"],
                                yearly_seasonality=st.session_state.params["yearly_seasonality"],
                                changepoint_prior_scale=st.session_state.params["changepoint_prior_scale"])
                st.session_state.params = None
            else:
                model2 = Prophet(daily_seasonality=True, n_changepoints=n_changepoints,
                                seasonality_mode=seasonality_mode,
                                yearly_seasonality=yearly_seasonality, changepoint_prior_scale=changepoint_prior_scale)
        model2.fit(data2)
        future2 = model2.make_future_dataframe(periods=25)

        forecast2 = model2.predict(future2)

        forecast2.index = forecast2.ds

        last_date = data2['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=25, freq='B')  # 'B' = business day frequency

        # Convert to DataFrame as Prophet expects
        future2 = pd.DataFrame({'ds': future_dates})

        forecast2['ds'] = list(forecast2.ds[:len(forecast2)-25]) + list(future2.ds)
        forecast2.index = forecast2.ds
        forecast2 = forecast2[["ds", "yhat_lower", "yhat_upper", "yhat"]]
        forecast2.columns = ["Date", 'Lower limit', "upper limit", "Forecast"]
        forecast2['Date'] = forecast2["Date"].dt.date
        forecast2.reset_index(inplace = True, drop =True)
        st.markdown("**Following is the forecasted Dataframe**")
        st.write(forecast2.tail(25))



        actual_values = axis.Close
        predicted_values = forecast2.Forecast

        # Create Plotly figure
        fig = go.Figure()

        # Add actual values line
        fig.add_trace(go.Scatter(
            y=actual_values,
            mode='lines',
            name='Actual',
            line=dict(color='royalblue', width=2)
        ))

        # Add predicted values line
        fig.add_trace(go.Scatter(
            y=predicted_values,
            mode='lines',
            name='Predicted',
            line=dict(color='orange', width=2)
        ))

        # Customize layout
        fig.update_layout(
            title=f'📈 Prophet Model Forecast for given Stocks (25 days after {last_date.date()})',
            xaxis=dict(
                title='Time',
                showgrid=True,  # Enable vertical grid lines
                gridcolor='royalblue'  # Optional: soft gridline color
            ),
            yaxis=dict(
                title='Stock Price',
                showgrid=True,  # Enable horizontal grid lines
                gridcolor='royalblue'
            ),
            xaxis_title='Time',
            yaxis_title='Stock Price',
            legend=dict(x=0, y=1),
            template='plotly_white',
            hovermode='x unified',
            margin=dict(l=40, r=40, t=60, b=40),
            height=400
        )

        # Show plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**This Dataframe shows forecasted value of given stock data for next 25 days using the choosen prophet model parameters**")









