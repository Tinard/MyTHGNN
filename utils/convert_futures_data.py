import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# 定义输入输出路径
FUTURES_DATA_DIR = "d:\\MyTHGNN\\futures_data"  # 原始期货数据目录
OUTPUT_FILE = "d:\\MyTHGNN\\data\\futures.pkl"   # 输出pickle文件

# 新的特征列
feature_cols = ['open', 'high', 'low', 'close', 'oi', 'volume']

def process_futures_data():
    """
    处理期货数据，转换为模型所需的输入格式
    输出格式: DataFrame包含['code', 'dt', 'open', 'high', 'low', 'close', 'oi', 'volume', 'label']
    """
    print("开始处理期货数据...")
    
    # 创建空的结果DataFrame
    result_df = pd.DataFrame()
    
    # 处理训练集、验证集和测试集
    for dataset_name in ['train_set', 'valid_set', 'test_set']:
        dataset_path = os.path.join(FUTURES_DATA_DIR, f"{dataset_name}.csv")
        
        if not os.path.exists(dataset_path):
            print(f"警告: {dataset_path} 不存在，跳过")
            continue
            
        print(f"处理 {dataset_name}...")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(dataset_path)
            
            # 确保所需列存在
            required_cols = ['date', 'symbol'] + feature_cols
            available_cols = df.columns.tolist()
            
            # 检查列名是否匹配，调整列名映射
            col_mapping = {'date': 'dt', 'symbol': 'code'}
            
            # 检查哪些必要列不存在
            missing_cols = []
            for col in required_cols:
                if col not in available_cols:
                    # 检查是否有替代列
                    if col == 'oi' and 'OI' in available_cols:
                        col_mapping['OI'] = 'oi'
                    elif col == 'volume' and 'Volume' in available_cols:
                        col_mapping['Volume'] = 'volume'
                    else:
                        missing_cols.append(col)
            
            if missing_cols:
                print(f"警告: {dataset_name}.csv 缺少必要的列: {missing_cols}")
                print(f"可用的列: {available_cols}")
                continue
            
            # 重命名列以匹配模型期望的格式
            df = df.rename(columns=col_mapping)
            
            # 确保日期格式正确
            df['dt'] = pd.to_datetime(df['dt'])
            
            # 计算label (二分类标签: 1表示上涨, 0表示下跌或持平)
            # 这里使用下一天的收盘价与今天的收盘价的涨跌情况作为标签
            df = df.sort_values(by=['code', 'dt'])
            if 'return' in df.columns:
                # 如果已有return列，转换为二分类标签
                df['label'] = (df['return'] > 0).astype(int)
            else:
                # 否则计算收益率，再转换为二分类标签
                returns = df.groupby('code')['close'].shift(-1) / df['close'] - 1
                df['label'] = (returns > 0).astype(int)
            
            # 删除缺失标签的行
            df = df.dropna(subset=['label'])
            
            # 选择并排序所需的列
            df = df[['code', 'dt'] + feature_cols + ['label']]
            
            # 添加到结果DataFrame
            result_df = pd.concat([result_df, df], ignore_index=True)
            
        except Exception as e:
            print(f"处理 {dataset_name}.csv 时出错: {str(e)}")
    
    # 按日期排序
    if not result_df.empty:
        result_df = result_df.sort_values(by=['dt', 'code']).reset_index(drop=True)
        
        # 保存为pickle文件
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(result_df, f)
        
        print(f"数据处理完成，保存到 {OUTPUT_FILE}")
        print(f"总记录数: {len(result_df)}")
        print(f"期货合约数: {result_df['code'].nunique()}")
        print(f"日期范围: {result_df['dt'].min()} 至 {result_df['dt'].max()}")
    else:
        print("警告: 没有找到任何数据，无法生成futures.pkl文件")
        print("请确保futures_data目录中包含正确格式的CSV文件")
    
    return result_df

if __name__ == "__main__":
    process_futures_data()
