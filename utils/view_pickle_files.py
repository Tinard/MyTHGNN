import pickle
import pandas as pd
import os

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
        else:
            print(f"\n数据类型: {type(data)}")
            print(f"数据内容: {data}")
    
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")

if __name__ == "__main__":
    # 查看csi300.pkl文件
    csi_file = "d:\\MyTHGNN\\data\\csi300.pkl"
    view_pickle_file(csi_file)
    
    # 查看futures.pkl文件(如果存在)
    futures_file = "d:\\MyTHGNN\\data\\futures.pkl"
    view_pickle_file(futures_file)
