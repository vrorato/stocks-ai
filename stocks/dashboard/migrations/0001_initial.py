# Generated by Django 4.2.7 on 2023-12-04 15:47

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="StockData",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("symbol", models.CharField(max_length=10)),
                ("date", models.DateField()),
                ("open_price", models.FloatField()),
                ("close_price", models.FloatField()),
                ("high", models.FloatField()),
                ("low", models.FloatField()),
                ("volume", models.IntegerField()),
            ],
        ),
    ]
