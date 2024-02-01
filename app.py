from flask import Flask, render_template, request
import yfinance as yf
from prophet import Prophet
import pandas as pd

app = Flask(__name__)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    dates = []
    actual_prices = []
    predicted_prices = []
    predicted_date = None
    stock_symbol = None
    current_price = None

    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        try:
            data = yf.download(stock_symbol, start="2022-01-01", end="2024-01-01")
            if not data.empty:
                data.reset_index(inplace=True)
                data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
                
                dates = data['ds'].dt.strftime('%Y-%m-%d').tolist()
                actual_prices = data['y'].tolist()
                
                model = Prophet()
                model.fit(data)
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)

                predicted_prices = [None] * len(actual_prices) + forecast['yhat'][-30:].tolist()
                prediction = forecast.iloc[-1]['yhat']
                predicted_date = forecast.iloc[-1]['ds'].strftime('%Y-%m-%d')
                current_price = data.iloc[-1]['y']
        except Exception as e:
            print(f"Error: {e}")

    return render_template('index.html', prediction=prediction, predicted_date=predicted_date, 
                           stock_symbol=stock_symbol, dates=dates, actual_prices=actual_prices, 
                           predicted_prices=predicted_prices, current_price=current_price)

if __name__ == '__main__':
    app.run(debug=True)
