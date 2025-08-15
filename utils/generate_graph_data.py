"""
THGNN 期货图结构数据生成工具

此脚本使用之前生成的归一化互信息矩阵构建图结构数据，为THGNN模型提供训练和预测输入。
主要功能:
1. 读取归一化互信息矩阵文件
2. 构建正相关和负相关图结构
3. 提取对应日期的特征数据
4. 准备模型输入格式（特征、标签、掩码）
5. 保存为pickle格式，供模型训练和预测使用

重要参数:
- FEATURE_COLS: 从generate_relation.py导入的动量因子特征列表
- PREV_DATE_NUM: 从generate_relation.py导入的历史天数参数
"""
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch.autograd import Variable
from tqdm import tqdm

# 从generate_relation.py导入一些公共设置
from utils.generate_relation import FEATURE_COLS, DATA_OUTPUT_DIR, PREV_DATE_NUM, ensure_directories, load_dataset

def process_data_by_relation(relation_file, dataset_name, df):
    """
    处理指定关系文件对应的数据，生成模型输入
    
    参数:
        relation_file: 归一化互信息矩阵文件路径
        dataset_name: 数据集名称 (train/valid/test)
        df: 包含特征和标签的DataFrame
        
    返回:
        无返回值，结果直接保存到文件
    """
    relation_dt = os.path.basename(relation_file).replace('.csv', '')
    print(f"\n处理 {dataset_name} 中的 {relation_dt}")
    
    # 读取关系文件
    adj_all = pd.read_csv(relation_file, index_col=0)
    adj_stock_set = list(adj_all.index)
    
    # 1. 提取互信息矩阵
    adj_matrix = adj_all.values.astype(np.float32)
    n_nodes = adj_matrix.shape[0]
    
    # 2. 图稀疏化：为每个节点保留互信息分数最高的10%的边（不包括自环）
    sparse_adj = np.zeros_like(adj_matrix)
    
    for i in range(n_nodes):
        # 获取当前节点与所有其他节点的互信息分数
        node_scores = adj_matrix[i, :].copy()
        # 将自环设为-1，确保不参与排序
        node_scores[i] = -1
        
        # 找出前10%的边索引（排除自环）
        k = int(0.1 * (n_nodes - 1))  # 计算保留的边数
        top_indices = np.argsort(node_scores)[-k:] if k > 0 else []
        
        # 为当前节点保留这些连接
        sparse_adj[i, top_indices] = 1
    
    # 3. 添加自环：将对角线元素设为1
    np.fill_diagonal(sparse_adj, 1.0)
    
    # 构建稀疏化的正相关图
    pos_adj = torch.from_numpy(sparse_adj).type(torch.float32)
    print(f'正相关图构建完成，形状: {pos_adj.shape}, 平均每节点连接数: {sparse_adj.sum()/n_nodes:.2f}')
    
    # 转换日期格式
    end_date = pd.to_datetime(relation_dt)
    
    # 找出该日期前PREV_DATE_NUM天的数据
    all_dates = sorted(df['date'].unique())
    
    # 找到end_date在所有日期中的位置
    try:
        end_date_idx = all_dates.index(end_date)
    except ValueError:
        # 如果精确日期不存在，找最接近的日期
        for i, date in enumerate(all_dates):
            if date > end_date:
                end_date_idx = i - 1
                end_date = all_dates[end_date_idx]
                print(f"未找到精确日期，使用最接近的日期: {pd.Timestamp(end_date).strftime('%Y-%m-%d')}")
                break
        else:
            print(f"无法找到合适的日期，跳过")
            return
    
    # 对于整个数据集，我们确保有足够的历史数据用于填充
    # 即使end_date_idx < PREV_DATE_NUM - 1，我们也不再跳过，而是使用可用数据并在必要时进行填充
    start_date_idx = max(0, end_date_idx - (PREV_DATE_NUM - 1))
    start_date = all_dates[start_date_idx]
    
    # 计算可用的历史天数
    available_days = end_date_idx - start_date_idx + 1
    
    # 筛选时间范围内的数据
    period_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    # 准备特征和标签
    code_list = adj_stock_set
    feature_all = []
    labels = []
    day_last_code = []  # 记录日期和代码对应关系
    
    # 处理每个期货代码
    for code in code_list:
        # 检查该代码是否在数据中
        code_data = period_data[period_data['symbol'] == code]
        # 确保数据按日期排序
        code_data = code_data.sort_values('date')
        
        # 检查该代码的数据是否不足PREV_DATE_NUM天
        if len(code_data) < PREV_DATE_NUM:
            # 计算需要补齐的天数
            pad_len = PREV_DATE_NUM - len(code_data)
            print(f"代码 {code} 的历史数据不足 {PREV_DATE_NUM} 天，只有 {len(code_data)} 天，补充 {pad_len} 天数据")
            
            # 构造补齐的DataFrame
            # 如果有数据，使用最早一天的前pad_len天作为日期
            if len(code_data) > 0:
                pad_dates = pd.date_range(end=code_data['date'].min() - pd.Timedelta(days=1), periods=pad_len)
            else:
                # 如果完全没有数据，使用开始日期前的日期
                pad_dates = pd.date_range(end=start_date - pd.Timedelta(days=1), periods=pad_len)
            
            # 创建填充数据字典，所有特征值为0
            pad_dict = {col: 0 for col in FEATURE_COLS}
            pad_dict['symbol'] = code
            pad_dict['label'] = 0  # 默认标签为0
            
            # 创建填充DataFrame
            pad_df = pd.DataFrame([{**pad_dict, 'date': d} for d in pad_dates])
            
            # 将填充数据与原始数据合并
            code_data = pd.concat([pad_df, code_data], ignore_index=True)
            print(f"补充后数据长度: {len(code_data)} 天")
        
        # 如果数据超过PREV_DATE_NUM天，只保留最近的PREV_DATE_NUM天
        if len(code_data) > PREV_DATE_NUM:
            code_data = code_data.iloc[-PREV_DATE_NUM:]
        
        # 提取特征，再次检查 NaN 值（以防万一）
        # 检查每个动量因子特征是否存在
        feature_list = []
        for feature in FEATURE_COLS:
            if feature in code_data.columns:
                feature_values = code_data[feature].values
                if np.isnan(feature_values).any():
                    print(f"警告: {code} 的 {feature} 特征中发现 NaN 值，替换为0")
                    feature_values = np.nan_to_num(feature_values, nan=0.0)
                feature_list.append(feature_values)
            else:
                print(f"警告: {code} 缺少 {feature} 特征，使用零向量代替")
                # 使用零向量替代缺失特征
                feature_list.append(np.zeros(len(code_data)))
        
        # 将所有特征列堆叠为一个二维数组
        features = np.stack(feature_list, axis=1)
        if np.isnan(features).any():
            print(f"警告: {code} 的特征矩阵中仍有 NaN 值，替换为0")
            features = np.nan_to_num(features, nan=0.0)
        
        feature_all.append(features)
        
        # 获取最后一天的标签
        try:
            end_day_data = code_data[code_data['date'] == end_date]
            if not end_day_data.empty:
                label = end_day_data['label'].values[0]
                labels.append(label)
                
                # 记录日期和代码
                day_last_code.append([code, pd.Timestamp(end_date).strftime('%Y-%m-%d')])
            else:
                print(f"警告: {code} 在 {pd.Timestamp(end_date).strftime('%Y-%m-%d')} 没有数据，使用倒数第二天的数据")
                # 尝试使用倒数第二天的数据
                if len(code_data) > 0:
                    label = code_data.iloc[-1]['label']
                    labels.append(label)
                    day_last_code.append([code, pd.Timestamp(code_data.iloc[-1]['date']).strftime('%Y-%m-%d')])
        except Exception as e:
            print(f"处理标签时出错: {str(e)}")
            # 继续处理下一个代码
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
        'features': Variable(features),
        'labels': Variable(labels),
        'mask': mask
    }
    
    # 保存结果
    end_date_str = pd.Timestamp(end_date).strftime('%Y-%m-%d')
    
    # 创建数据集特定的目录
    dataset_dir = os.path.join(DATA_OUTPUT_DIR, "data_train_predict", dataset_name)
    daily_stock_dir = os.path.join(DATA_OUTPUT_DIR, "daily_stock", dataset_name)
    
    # 直接保存结果文件到目标目录
    with open(os.path.join(dataset_dir, f"{end_date_str}.pkl"), 'wb') as f:
        pickle.dump(result, f)
    
    # 保存日期和代码映射
    df_map = pd.DataFrame(columns=['symbol', 'date'], data=day_last_code)
    df_map.to_csv(os.path.join(daily_stock_dir, f"{end_date_str}.csv"), 
                 header=True, index=False, encoding='utf_8_sig')
    
    print(f"已处理 {end_date_str}: 特征形状 {features.shape}, 标签数量 {len(labels)}")
    print(f"使用的特征数量: {features.shape[2]}, 特征列表: {FEATURE_COLS}")

def generate_graph_data(datasets):
    """
    处理每个数据集的关系文件，生成图结构数据
    
    参数:
        datasets: 包含所有数据集的字典 {dataset_name: dataframe}
        
    返回:
        无返回值，结果直接保存到文件
    """
    # 统一从 merged 目录读取互信息矩阵
    relation_dir = os.path.join(DATA_OUTPUT_DIR, "relation", "merged")
    if not os.path.exists(relation_dir):
        print(f"警告: {relation_dir} 不存在，跳过")
        return
    relation_files = []
    for file in os.listdir(relation_dir):
        if file.endswith('.csv'):
            relation_files.append(os.path.join(relation_dir, file))
    if not relation_files:
        print(f"警告: {relation_dir} 中没有找到关系文件")
        return
    print(f"\n处理所有数据集的 {len(relation_files)} 个关系文件...")
    for dataset_name, df in datasets.items():
        for relation_file in sorted(relation_files):
            process_data_by_relation(relation_file, dataset_name, df)

def main(datasets=None):
    """
    生成图结构数据的主函数
    
    参数:
        datasets: 可选，包含所有数据集的字典 {dataset_name: dataframe}
                如果为None，将尝试从关系矩阵文件恢复数据集信息
                
    返回:
        bool: 处理是否成功
    """
    print("开始生成图结构数据...")
    
    # 确保所有目录存在
    ensure_directories()
    
    # 如果没有提供数据集，则从relation模块导入它们
    if datasets is None:
        # 尝试从生成的关系文件中恢复数据集信息
        print("未提供数据集，尝试从相关性矩阵文件恢复数据集信息...")
        
        # 1. 首先加载所有数据集并合并
        print("加载并合并所有数据集...")
        all_df = None
        datasets = {}
        
        for dataset_name in ['train', 'valid', 'test']:
            df = load_dataset(dataset_name)
            if df is not None:
                print(f"从文件加载 {dataset_name} 数据集: {len(df)} 条记录")
                # 添加数据集标记，便于后续分割
                df['dataset'] = dataset_name
                
                # 合并到全局数据集
                if all_df is None:
                    all_df = df
                else:
                    all_df = pd.concat([all_df, df], ignore_index=True)
                
                # 保留原始数据集引用，以便兼容旧代码
                datasets[dataset_name] = df
        
        if all_df is not None:
            print(f"合并后的完整数据集: {len(all_df)} 条记录, 日期范围: {all_df['date'].min()} 至 {all_df['date'].max()}")
            
            # 2. 处理全局数据集的归一化互信息矩阵
            # 按日期分割，为每个日期创建一个包含全部代码的关系矩阵
            # 注意：这部分代码只有在需要重新计算互信息矩阵时取消注释
            """
            print("为完整数据集生成归一化互信息矩阵...")
            from utils.generate_relation import calculate_mutual_information, stock_mutual_info_matrix
            
            # 获取所有日期和代码
            all_dates = sorted(all_df['date'].unique())
            
            # 为每个日期生成关系矩阵
            for i in range(PREV_DATE_NUM - 1, len(all_dates)):
                end_date = all_dates[i]
                end_date_str = pd.Timestamp(end_date).strftime('%Y-%m-%d')
                
                # 获取该日期对应的数据集类型
                dataset_type = all_df[all_df['date'] == end_date]['dataset'].iloc[0]
                
                # 处理互信息矩阵
                # ...此处添加互信息矩阵计算代码...
            """
            
            # 3. 重新分割数据集
            for dataset_name in ['train', 'valid', 'test']:
                dataset_df = all_df[all_df['dataset'] == dataset_name]
                datasets[dataset_name] = dataset_df
                print(f"重新分割后的 {dataset_name} 数据集: {len(dataset_df)} 条记录")
    
    if not datasets:
        print("错误: 没有找到任何数据集，无法继续处理")
        return False
    
    # 生成图结构数据
    generate_graph_data(datasets)
    
    print("\n图结构数据生成完成!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("THGNN 期货图结构数据生成工具")
    print("=" * 60)
    
    # 输出当前配置信息
    print(f"使用输出目录: {DATA_OUTPUT_DIR}")
    print(f"使用的特征 ({len(FEATURE_COLS)}个): {FEATURE_COLS}")
    print("注意: 本工具使用generate_relation.py生成的归一化互信息矩阵")
    print("=" * 60)
    
    main()
