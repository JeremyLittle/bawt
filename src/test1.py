import urllib.request, json
import csv
with urllib.request.urlopen("https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1566288000&end=9999999999&period=300") as url:
    data = json.loads(url.read().decode())
    print(data)

