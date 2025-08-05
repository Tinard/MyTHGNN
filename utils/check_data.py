import pickle
import pandas as pd

path1 = "d:\\MyTHGNN\\data\\futures.pkl"
df1 = pickle.load(open(path1, 'rb'), encoding='utf-8')

# 打印前10行数据
print("前10行数据:")
print(df1.head(10))

# 显示日期的格式
print("\n日期类型:", type(df1['dt'].iloc[0]))

# 查看代码列表
print("\n合约代码列表:")
print(df1['code'].unique())
