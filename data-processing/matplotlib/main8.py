import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import urllib

def graph_data(stock):
  # ! Link is invalid
  stock_price_url = 'http://chartapi.finance.yahoo.com/instrument/1.0'+stock+'/chartdata;type=quote;range=10y/csv'
  source_code = urllib.request.urlopen(stock_price_url).read().decode()
  stock_data = []
  split_source = source_code.split('\n')
  plt.show()

graph_data('TSLA')