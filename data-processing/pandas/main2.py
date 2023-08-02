import pandas as pd


'''
header가 없을때는?
header가 첫번째 row 부터 시작되지 않는다면?
일부의 데이터만 사용하려 할때는? 행?열?
결국 중요한건 데이터를 주의 깊게 분석하고, 필요한 데이터를 선별하고, 예외처리를 하는것.
가공된 데이터는 또한 다시 저장하여 활용할 수 있도록.
'''

df = pd.read_csv('stock_data.csv', header=1, index_col=False, usecols=['tickers', 'price'])
print(df.head(5))

# try to write
# remove index column
# remove headers
# choose specific columns
df.to_csv('new.csv')

#def convert_peopel_cell(cell):
#    if cell == 'n.a.':
#        return 'sam walton'
#    return cell
#
#def convert_eps_cell(cell):
#    if cell == 'not available':
#        return None
#    return cell
#
## converters={'people':convert_people_cell}
#df2 = pd.read_excel('stock_data.xlsx', sheet_name='')
#
#
#pd.to_excel('new.xlsx')
#
#
## two DataFrames to one single excel file with two different sheets
#stock_df = pd.DataFrame()
#weather_df = pd.DataFrame()
#with pd.ExcelWriter as writer:
#    stock_df.to_excel(writer, sheet_name='')
#    weather_df.to_excel(writer, sheet_name='')