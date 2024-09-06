# forms.py
from django import forms

class StockSymbolForm(forms.Form):
    symbol = forms.CharField(label='Enter Stock Symbol', max_length=10)
    start_date = forms.DateField(label='Start Date', widget=forms.DateInput(attrs={'type': 'date'}))
    end_date = forms.DateField(label='End Date', widget=forms.DateInput(attrs={'type': 'date'}))