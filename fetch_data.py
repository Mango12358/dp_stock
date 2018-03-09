import tushare as ts
import sys

# 获取连接备用
cons = ts.get_apis()

freq = 'D'
start_date = '2017-03-31'
end_date = '2018-02-27'
ma = [5, 10, 20, 30, 60]


def fetch_index(code='000001'):
    df = ts.bar(code=code, conn=cons, freq=freq, asset='INDEX', start_date=start_date, end_date=end_date, ma=ma)
    columns = ['close', 'high', 'low', 'open', 'vol', 'amount', 'ma5', 'ma10', 'ma20', 'ma30', 'ma60']
    df = df.loc[:, columns]
    df = df.sort_index(axis=0, ascending=True)
    df.to_csv('data/index.csv')
    return df


def fetch_code(code='000001'):
    df = ts.bar(code=code, conn=cons, freq=freq, start_date=start_date, end_date=end_date, ma=ma)
    columns = ['close', 'high', 'low', 'open', 'vol', 'amount', 'ma5', 'ma10', 'ma20', 'ma30', 'ma60']
    df = df.loc[:, columns]
    df = df.sort_index(axis=0, ascending=True)
    df.to_csv('data/code.csv')
    return df


code = '000001'
if len(sys.argv) > 1:
    code = sys.argv[1]
print(code)
if code.startswith('60'):
    fetch_index(code='000001')
    fetch_code(code=code)
print('End')
