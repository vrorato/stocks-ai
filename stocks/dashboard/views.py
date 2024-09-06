# views.py
from django.shortcuts import render
import yfinance as yf
from .forms import StockSymbolForm
from .models import StockData
import json
from .utils import predict_stock_price, predict_stock_price_lstm, predict_stock_price_random_forest

def fetch_yahoo_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    historical_data = stock.history(start=start_date, end=end_date)
    
    stock_data = []
    for index, row in historical_data.iterrows():
        data = {
            'date': index.strftime('%Y-%m-%d'),  # Format date as string
            'close_price': row['Close']
        }
        stock_data.append(data)

        # Save to database if needed (optional)
        # StockData.objects.create(...)

    return stock_data

def stock_dashboard(request):
    form = StockSymbolForm(request.POST or None)
    stock_data = []
    predictions = {
        'linear_regression': None,
        'lstm' : None,
        'random_forest': None
    }

    if request.method == 'POST' and form.is_valid():
        symbol = form.cleaned_data['symbol']
        start_date = form.cleaned_data['start_date']
        end_date = form.cleaned_data['end_date']
        
        stock_data = fetch_yahoo_stock_data(symbol, start_date, end_date)
        stock = yf.Ticker(symbol)
        historical_data = stock.history(start=start_date, end=end_date)


        #predicitions
        predictions['linear_regression'] = predict_stock_price(historical_data)
        predictions['lstm'] = predict_stock_price_lstm(historical_data)
        predictions['random_forest'] = predict_stock_price_random_forest(historical_data)

    # Assuming 'stock_data' is a list of dictionaries with 'date' and 'close_price' keys
    stock_data_json = json.dumps(stock_data)

    return render(request, 'dash.html', {'form': form, 'stock_data': stock_data, 'stock_data_json': stock_data_json,'predictions': predictions})