import pickle
import pandas as pd

path1 = "d:\\MyTHGNN\\data\\futures.pkl"
df1 = pickle.load(open(path1, 'rb'), encoding='utf-8')
prev_date_num = 20
date_unique = df1['dt'].unique()
stock_trade_data = date_unique.tolist()
stock_trade_data.sort()

# 分析月份
months = {}
for date in stock_trade_data:
    date_str = str(date)[:10]  # 确保是字符串并获取前10个字符
    year_month = date_str[:7]  # 如 "2022-11"
    if year_month not in months:
        months[year_month] = []
    months[year_month].append(date_str)

print("所有月份:")
for year_month, dates in sorted(months.items()):
    print(f"{year_month}: {len(dates)}天")

dt = []
for year_month, dates in months.items():
    if len(dates) >= prev_date_num:  # 确保有足够的数据
        dates.sort()
        dt.append(dates[-1])  # 添加月末日期

print("\n符合条件的月末日期:")
for date in sorted(dt):
    print(date)
