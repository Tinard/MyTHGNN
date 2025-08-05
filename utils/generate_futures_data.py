import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx
import pandas as pd
from torch.autograd import Variable

# 修改特征列以适应期货数据
feature_cols = ['open', 'high', 'low', 'close', 'oi', 'volume']

# 加载期货数据
path1 = "d:\\MyTHGNN\\data\\futures.pkl"
df1 = pickle.load(open(path1, 'rb'), encoding='utf-8')
df1['dt'] = df1['dt'].astype('datetime64[ns]')

# 读取训练、验证和测试集时间范围
futures_data_dir = "d:\\MyTHGNN\\futures_data"
datasets = {}

# 加载训练、验证和测试集数据
for dataset_name in ['train_set', 'valid_set', 'test_set']:
    file_path = os.path.join(futures_data_dir, f"{dataset_name}.csv")
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data['date'] = pd.to_datetime(data['date'])
        datasets[dataset_name] = {
            'data': data,
            'start_date': data['date'].min(),
            'end_date': data['date'].max()
        }
        print(f"{dataset_name}: {data['date'].min()} 至 {data['date'].max()}, {len(data)} 条记录")

# 确保输出目录存在
os.makedirs("d:\\MyTHGNN\\data\\data_train_predict", exist_ok=True)
os.makedirs("d:\\MyTHGNN\\data\\daily_stock", exist_ok=True)

# 获取所有交易日期并排序
date_unique = df1['dt'].unique()
stock_trade_data = date_unique.tolist()
stock_trade_data.sort()

def process_data_by_relation(relation_file, dataset_name, df1):
    """处理指定关系文件对应的数据"""
    prev_date_num = 20
    relation_dt = os.path.basename(relation_file).replace('.csv', '')
    print(f"\n处理 {dataset_name} 中的 {relation_dt}")
    
    # 读取关系文件
    adj_all = pd.read_csv(relation_file, index_col=0)
    adj_stock_set = list(adj_all.index)
    
    # 构建正相关图
    pos_g = nx.Graph(adj_all > 0.1)
    pos_adj = nx.adjacency_matrix(pos_g).toarray()
    pos_adj = pos_adj - np.diag(np.diag(pos_adj))
    pos_adj = torch.from_numpy(pos_adj).type(torch.float32)
    
    # 构建负相关图
    neg_g = nx.Graph(adj_all < -0.1)
    neg_adj = nx.adjacency_matrix(neg_g)
    neg_adj.data = np.ones(neg_adj.data.shape)
    neg_adj = neg_adj.toarray()
    neg_adj = neg_adj - np.diag(np.diag(neg_adj))
    neg_adj = torch.from_numpy(neg_adj).type(torch.float32)
    
    print(f'正负相关图构建完成，形状: {neg_adj.shape}')
    
    # 找到对应的日期
    end_data = None
    for date in stock_trade_data:
        date_str = str(date)[:10]
        # 标准化关系文件中的日期格式
        if relation_dt == date_str or relation_dt == pd.Timestamp(date).strftime('%Y-%m-%d'):
            end_data = date
            break
    
    if end_data is None:
        print(f"无法找到 {relation_dt} 对应的日期，跳过")
        return
    
    # 确保有足够的历史数据
    end_data_index = stock_trade_data.index(end_data)
    if end_data_index < prev_date_num - 1:
        print(f"跳过 {relation_dt}，没有足够的历史数据")
        return
    
    # 获取开始日期
    start_data = stock_trade_data[end_data_index - (prev_date_num - 1)]
    
    # 确保日期类型一致
    end_date_dt = pd.to_datetime(end_data) if not isinstance(end_data, pd.Timestamp) else end_data
    start_date_dt = pd.to_datetime(start_data) if not isinstance(start_data, pd.Timestamp) else start_data
    
    # 筛选时间范围内的数据
    df2 = df1.loc[df1['dt'] <= end_date_dt]
    df2 = df2.loc[df2['dt'] >= start_date_dt]
    
    # 准备特征和标签
    code = adj_stock_set
    feature_all = []
    mask = []
    labels = []
    day_last_code = []
    
    # 在处理每个股票代码的循环中
    for j in range(len(code)):
        df3 = df2.loc[df2['code'] == code[j]]
        
        # 如果不是根据日期排序的，先排序
        df3 = df3.sort_values(by='dt')
        
        # 提取特征
        dts = df3['dt'].values
        y = df3[feature_cols].values
        
        if y.T.shape[1] == prev_date_num:
            one = []
            feature_all.append(y)
            mask.append(True)
            
            # 打印调试信息
            print(f"处理代码 {code[j]} 的标签...")
            
            # 获取标签，添加更多错误检查和调试信息
            try:
                # 打印具体日期，方便调试
                end_date_str = pd.Timestamp(end_date_dt).strftime('%Y-%m-%d')
                print(f"查找日期 {end_date_str} 的标签")
                
                # 检查df3中是否有end_date_dt日期的数据
                matching_rows = df3.loc[df3['dt'] == end_date_dt]
                if len(matching_rows) > 0:
                    # 确认该行有label列
                    if 'label' in matching_rows.columns:
                        label_value = matching_rows['label'].values[0]
                        print(f"找到标签: {label_value}")
                        labels.append(label_value)
                    else:
                        print(f"警告: {code[j]} 在 {end_date_str} 的数据中没有label列")
                        # 使用0作为默认标签
                        labels.append(0.0)
                        print(f"使用默认标签: 0.0")
                else:
                    print(f"警告: 在df3中找不到日期 {end_date_str} 的数据")
                    # 如果是测试集且接近最后一天，可以使用最后一个可用的标签
                    if dataset_name == 'test_set':
                        last_label = df3['label'].iloc[-1] if len(df3['label']) > 0 else 0.0
                        labels.append(last_label)
                        print(f"使用最后可用标签: {last_label}")
                    else:
                        # 使用0作为默认标签
                        labels.append(0.0)
                        print(f"使用默认标签: 0.0")
                    
                one.append(code[j])
                one.append(end_date_str)  # 使用字符串形式的日期
                day_last_code.append(one)
            except Exception as e:
                print(f"处理标签时出错: {str(e)}")
                # 继续处理下一个代码，而不是中断整个过程
                continue
    
    if len(feature_all) == 0:
        print(f"跳过 {relation_dt}，没有有效的特征数据")
        return
    
    # 转换为模型输入格式
    feature_all = np.array(feature_all)
    features = torch.from_numpy(feature_all).type(torch.float32)
    mask = [True] * len(labels)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    # 准备结果字典
    result = {
        'pos_adj': Variable(pos_adj), 
        'neg_adj': Variable(neg_adj),  
        'features': Variable(features),
        'labels': Variable(labels), 
        'mask': mask
    }
    
    # 保存结果
    end_data_str = str(end_data)[:10]
    # 检查格式是否正确
    if not end_data_str[4:5] == '-' or not end_data_str[7:8] == '-':
        # 如果不是标准日期格式，尝试转换
        try:
            end_data_str = pd.Timestamp(end_data).strftime('%Y-%m-%d')
        except:
            print(f"警告: 无法将 {end_data} 转换为标准日期格式")
    
    # 创建数据集特定的目录
    dataset_dir = f"d:\\MyTHGNN\\data\\data_train_predict\\{dataset_name}"
    os.makedirs(dataset_dir, exist_ok=True)
    
    daily_stock_dir = f"d:\\MyTHGNN\\data\\daily_stock\\{dataset_name}"
    os.makedirs(daily_stock_dir, exist_ok=True)
    
    # 保存结果文件
    with open(f"{dataset_dir}/{end_data_str}.pkl", 'wb') as f:
        pickle.dump(result, f)
    
    # 同时保存到原始目录（兼容模式）
    with open(f"d:\\MyTHGNN\\data\\data_train_predict\\{end_data_str}.pkl", 'wb') as f:
        pickle.dump(result, f)
    
    # 保存日期和代码映射
    df = pd.DataFrame(columns=['code', 'dt'], data=day_last_code)
    df.to_csv(f"{daily_stock_dir}/{end_data_str}.csv", header=True, index=False, encoding='utf_8_sig')
    
    # 同时保存到原始目录（兼容模式）
    df.to_csv(f"d:\\MyTHGNN\\data\\daily_stock\\{end_data_str}.csv", header=True, index=False, encoding='utf_8_sig')
    
    print(f"已处理 {end_data_str}: 特征形状 {features.shape}, 标签数量 {len(labels)}")

# 主函数：处理每个数据集的关系文件
def main():
    # 检查每个数据集的关系文件目录
    for dataset_name in ['train_set', 'valid_set', 'test_set']:
        # 检查数据集特定的关系目录
        relation_dir = f"d:\\MyTHGNN\\data\\relation\\{dataset_name}"
        
        if not os.path.exists(relation_dir):
            print(f"警告: {relation_dir} 不存在，跳过")
            continue
            
        # 获取该数据集的所有关系文件
        relation_files = []
        for file in os.listdir(relation_dir):
            if file.endswith('.csv'):
                relation_files.append(os.path.join(relation_dir, file))
                
        if not relation_files:
            print(f"警告: {relation_dir} 中没有找到关系文件，请先运行 generate_futures_relation.py")
            continue
            
        print(f"\n处理 {dataset_name} 的 {len(relation_files)} 个关系文件...")
        
        # 处理每个关系文件
        for relation_file in sorted(relation_files):
            process_data_by_relation(relation_file, dataset_name, df1)
    
    # 兼容模式：处理主关系目录中的文件
    main_relation_dir = "d:\\MyTHGNN\\data\\relation"
    main_relation_files = []
    
    for file in os.listdir(main_relation_dir):
        if file.endswith('.csv') and os.path.isfile(os.path.join(main_relation_dir, file)):
            main_relation_files.append(os.path.join(main_relation_dir, file))
    
    if main_relation_files:
        print(f"\n兼容模式: 处理主关系目录中的 {len(main_relation_files)} 个文件...")
        
        # 尝试确定每个文件属于哪个数据集
        for relation_file in sorted(main_relation_files):
            file_date = os.path.basename(relation_file).replace('.csv', '')
            
            # 查找该日期属于哪个数据集
            assigned_dataset = None
            for dataset_name, dataset_info in datasets.items():
                start_date = dataset_info['start_date']
                end_date = dataset_info['end_date']
                
                try:
                    file_timestamp = pd.Timestamp(file_date)
                    if file_timestamp >= start_date and file_timestamp <= end_date:
                        assigned_dataset = dataset_name
                        break
                except:
                    pass
            
            if assigned_dataset:
                print(f"文件 {file_date} 属于 {assigned_dataset}")
                process_data_by_relation(relation_file, assigned_dataset, df1)
            else:
                print(f"无法确定 {file_date} 属于哪个数据集，作为未分类处理")
                process_data_by_relation(relation_file, "uncategorized", df1)
    
    print("\n数据处理完成!")

if __name__ == "__main__":
    main()
