import pandas as pd
import numpy as np
import os

# 设置随机种子以便结果可复现
np.random.seed(42)

# 读取原始CSV文件
input_file = "d:\\MyTHGNN\\data\\prediction\\test_pred.csv"
output_file = "d:\\MyTHGNN\\data\\prediction\\random.csv"

# 读取数据
df = pd.read_csv(input_file)
print(f"原始文件行数: {len(df)}")

# 生成0-1之间的随机数替换score列
df['score'] = np.random.random(len(df))

# 保存新的CSV文件
df.to_csv(output_file, index=False)
print(f"已生成随机score的文件: {output_file}")
print(f"新文件行数: {len(df)}")

# 显示部分结果
print("\n随机生成的前5行数据:")
print(df.head(5))
