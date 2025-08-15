import pickle
import pandas as pd
import os
import numpy as np
import torch

def view_pickle_file(file_path, num_rows=5):
    """
    查看pickle文件的内容
    
    参数:
        file_path: pickle文件路径
        num_rows: 显示的行数
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    try:
        print(f"\n正在读取文件: {file_path}...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, pd.DataFrame):
            print(f"\n数据类型: DataFrame")
            print(f"形状: {data.shape} (行数, 列数)")
            print(f"列名: {data.columns.tolist()}")
            print(f"数据类型:\n{data.dtypes}")
            
            print(f"\n前 {num_rows} 行数据:")
            print(data.head(num_rows))
            
            print(f"\n数据统计:")
            print(data.describe())
            
            if 'code' in data.columns:
                print(f"\n期货/股票代码统计:")
                code_counts = data['code'].value_counts().head(10)
                print(f"共有 {data['code'].nunique()} 个不同的代码")
                print(f"前10个代码出现次数:\n{code_counts}")
            
            if 'dt' in data.columns:
                print(f"\n日期范围:")
                print(f"最早日期: {data['dt'].min()}")
                print(f"最晚日期: {data['dt'].max()}")
                print(f"共有 {data['dt'].nunique()} 个不同的日期")
        elif isinstance(data, dict):
            print(f"\n数据类型: Dictionary")
            print(f"键列表: {list(data.keys())}")
            
            for key, value in data.items():
                print(f"\n键: {key}")
                print(f"  值类型: {type(value)}")
                
                if isinstance(value, torch.Tensor):
                    print(f"  张量形状: {value.shape}")
                    print(f"  张量数据类型: {value.dtype}")
                    print(f"  张量设备: {value.device}")
                    print(f"  数值范围: 最小值={value.min().item():.4f}, 最大值={value.max().item():.4f}, 平均值={value.mean().item():.4f}")
                    
                    if value.dim() <= 2 and value.numel() < 100:
                        print(f"  张量内容:\n{value}")
                    else:
                        print(f"  张量太大，只显示一小部分...")
                        if value.dim() == 1:
                            print(f"  前{min(5, len(value))}个元素: {value[:5]}")
                        elif value.dim() == 2:
                            print(f"  左上角5x5区域(如果可能):\n{value[:5, :5]}")
                        else:
                            print(f"  高维张量，无法简洁显示")
                            
                elif isinstance(value, list):
                    print(f"  列表长度: {len(value)}")
                    if len(value) > 0:
                        print(f"  列表元素类型: {type(value[0])}")
                        # 对于mask，显示全部元素而不仅仅是前5个
                        if key == 'mask':
                            print(f"  全部元素: {value}")
                        else:
                            print(f"  前{min(5, len(value))}个元素: {value[:5]}")
                elif isinstance(value, np.ndarray):
                    print(f"  数组形状: {value.shape}")
                    print(f"  数组数据类型: {value.dtype}")
                    print(f"  数值范围: 最小值={np.min(value):.4f}, 最大值={np.max(value):.4f}, 平均值={np.mean(value):.4f}")
                    
                    if value.ndim <= 2 and value.size < 100:
                        print(f"  数组内容:\n{value}")
                    else:
                        print(f"  数组太大，只显示一小部分...")
                        if value.ndim == 1:
                            print(f"  前{min(5, len(value))}个元素: {value[:5]}")
                        elif value.ndim == 2:
                            print(f"  左上角5x5区域(如果可能):\n{value[:5, :5]}")
                        else:
                            print(f"  高维数组，无法简洁显示")
                else:
                    print(f"  值内容: {value}")
        else:
            print(f"\n数据类型: {type(data)}")
            print(f"数据内容: {data}")
    
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")

def view_graph_data_pickle(file_path):
    """专门用于查看图结构数据的函数"""
    view_pickle_file(file_path)

def list_pickle_files(directory):
    """列出目录中的所有pickle文件"""
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return []
    
    pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    return pickle_files

if __name__ == "__main__":
    # 直接检查数据目录
    base_dir = "d:\\MyTHGNN\\data\\data_train_predict"
    
    for dataset in ['test', 'train', 'valid']:
        dir_path = os.path.join(base_dir, dataset)
        if not os.path.exists(dir_path):
            continue
            
        pkl_files = list_pickle_files(dir_path)
        if not pkl_files:
            continue
            
        print(f"\n在 {dir_path} 中找到 {len(pkl_files)} 个pickle文件")
        print(f"显示第一个文件: {pkl_files[0]}")
        first_file = os.path.join(dir_path, pkl_files[0])
        view_graph_data_pickle(first_file)
        break  # 只显示一个数据集的第一个文件
