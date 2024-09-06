# models.py
from django.db import models

class StockData(models.Model):
    symbol = models.CharField(max_length=10)
    date = models.DateField()
    open_price = models.FloatField()
    close_price = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    volume = models.IntegerField()

    def __str__(self):
        return f"{self.symbol} - {self.date}"