"""
THGNN 期货归一化互信息矩阵生成工具

此脚本用于计算金融时间序列数据之间的归一化互信息 (NMI)，建立品种间关联强度矩阵。
归一化互信息是一种统计方法，可以测量两个随机变量之间的相互依赖性，不受线性关系的限制。
公式: NMI(X,Y) = MI(X,Y)/sqrt(H(X)*H(Y))，其中MI是互信息，H是熵。

主要功能:
1. 读取金融时间序列数据
2. 使用'return'列计算品种间的归一化互信息
3. 生成品种间关联强度矩阵
4. 将结果保存为CSV文件，供后续图结构生成使用

重要参数:
- FEATURE_COLS: 只保留动量因子相关特征
- PREV_DATE_NUM: 计算互信息的历史天数
"""
import os
import pickle
import shutil
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KernelDensity

# 定义输入输出路径
FUTURES_DATA_DIR = "d:\\MyTHGNN\\futures_data"  # 原始期货数据目录
DATA_OUTPUT_DIR = "d:\\MyTHGNN\\data"           # 输出根目录

# 特征列定义 - 修改为仅包含动量因子相关特征，并添加return列
FEATURE_COLS = ["ROC_5", "ROC_10", "MACD", "MACD_Signal", "MACD_Hist", "RSI", "return"]

# 相关性计算的天数
PREV_DATE_NUM = 20  # 修改为使用20个交易日计算互信息

def calculate_mutual_information(x, y, bins=10):
    """
    计算两个时间序列的归一化互信息值
    
    参数:
        x: 第一个时间序列的收益率数据
        y: 第二个时间序列的收益率数据
        bins: 离散化时的箱数
    
    返回:
        归一化互信息值: NMI(X,Y) = MI(X,Y)/sqrt(H(X)*H(Y))
    """
    # 检查输入是否包含 NaN，如果有则替换为 0
    x = np.nan_to_num(x, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    # 确保数据是二维的，mutual_info_regression需要二维输入
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    try:
        # 计算X的熵 (使用KNN估计器)
        # 使用高斯核密度估计
        kde_x = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x)
        log_dens_x = kde_x.score_samples(x)
        # 估计熵: H(X) = -E[log p(x)]
        h_x = -np.mean(log_dens_x)
        
        # 计算Y的熵
        kde_y = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(y)
        log_dens_y = kde_y.score_samples(y)
        h_y = -np.mean(log_dens_y)
        
        # 使用mutual_info_regression计算互信息
        mi_xy = mutual_info_regression(x, y.ravel(), discrete_features=False, n_neighbors=3)
        mi_yx = mutual_info_regression(y, x.ravel(), discrete_features=False, n_neighbors=3)
        
        # 取两个方向互信息的平均值以确保对称性
        mi = (mi_xy[0] + mi_yx[0]) / 2
        
        # 计算归一化互信息 NMI(X,Y) = MI(X,Y)/sqrt(H(X)*H(Y))
        # 保持NMI为[0,1]，不做线性映射
        if h_x > 0 and h_y > 0:
            nmi = mi / np.sqrt(h_x * h_y)
        else:
            nmi = 0.0
        # 防止NaN和Inf值
        if np.isnan(nmi) or np.isinf(nmi):
            return 0.0
        return nmi
    except Exception as e:
        print(f"计算归一化互信息时出错: {str(e)}")
        return 0.0

def calculate_mi_matrix(xs, yss):
    """计算一个时间序列与多个时间序列的归一化互信息值"""
    result = []
    for name in yss:
        ys = yss[name]
        
        # 只使用'return'列计算互信息
        # 寻找return列的位置
        return_idx = -1
        for i, feat in enumerate(FEATURE_COLS):
            if feat == 'return':
                return_idx = i
                break
        
        # 使用return列计算互信息
        if return_idx >= 0 and return_idx < len(xs):
            return_x = xs[return_idx]
            return_y = ys[return_idx]
            nmi = calculate_mutual_information(return_x, return_y)
            result.append(nmi)
        else:
            # 如果找不到return列，直接报错
            raise ValueError("未找到return列，无法计算互信息。请检查FEATURE_COLS和数据格式。")
    
    return np.array(result)

def stock_mutual_info_matrix(ref_dict, codes, processes=1):
    """计算股票/期货之间的归一化互信息矩阵"""
    # 单进程版本
    data = np.zeros([len(codes), len(codes)])
    
    for i in tqdm(range(len(codes))):
        try:
            # 使用归一化互信息计算关联性
            nmi_values = calculate_mi_matrix(ref_dict[codes[i]], ref_dict)
            # 确保没有 NaN 值
            nmi_values = np.nan_to_num(nmi_values, nan=0.0)
            data[i, :] = nmi_values
        except Exception as e:
            print(f"计算 {codes[i]} 的归一化互信息时出错: {str(e)}")
            # 出错时使用零值
            data[i, :] = 0
    
    # 最终检查，确保整个矩阵没有 NaN 值
    if np.isnan(data).any():
        print("警告: 归一化互信息矩阵中仍存在 NaN 值，替换为 0")
        data = np.nan_to_num(data, nan=0.0)
    
    return pd.DataFrame(data=data, index=codes, columns=codes)

def ensure_directories():
    """确保所有需要的目录都存在"""
    dirs = [
        os.path.join(DATA_OUTPUT_DIR, "data_train_predict"),
        os.path.join(DATA_OUTPUT_DIR, "daily_stock"),
        os.path.join(DATA_OUTPUT_DIR, "relation", "merged"),
        os.path.join(DATA_OUTPUT_DIR, "prediction")
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"确保目录存在: {directory}")

def load_dataset(dataset_name):
    """加载指定数据集"""
    global FEATURE_COLS
    
    file_path = os.path.join(FUTURES_DATA_DIR, f"{dataset_name}_features.csv")
    if not os.path.exists(file_path):
        print(f"警告: {file_path} 不存在，跳过")
        return None
        
    print(f"加载 {dataset_name}_features.csv...")
    df = pd.read_csv(file_path)
    
    # 确保日期列格式正确
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # 检查所需的基本列
    basic_cols = ['symbol', 'date', 'label']
    missing_basic = [col for col in basic_cols if col not in df.columns]
    
    if missing_basic:
        print(f"警告: {dataset_name} 缺少必要的基本列: {missing_basic}")
        return None
    
    # 检查特征列
    feature_cols_found = [col for col in FEATURE_COLS if col in df.columns]
    
    if len(feature_cols_found) < len(FEATURE_COLS):
        missing_features = [col for col in FEATURE_COLS if col not in feature_cols_found]
        print(f"警告: {dataset_name} 缺少部分特征列: {missing_features}")
        
        # 为缺失的特征添加默认值
        for col in missing_features:
            print(f"为缺失特征 {col} 添加默认值 (0)")
            df[col] = 0
            feature_cols_found.append(col)
        
        print(f"已添加缺失特征，总特征数: {len(feature_cols_found)}")
    
    # 使用所有找到的特征列
    FEATURE_COLS = feature_cols_found
    print(f"最终使用的特征列: {FEATURE_COLS}")
    
    if not FEATURE_COLS:
        print(f"错误: {dataset_name} 没有可用的特征列")
        return None
    
    # 检查并处理 NaN 值
    nan_counts = df[FEATURE_COLS].isna().sum()
    if nan_counts.sum() > 0:
        print(f"警告: 发现 NaN 值，按列统计: \n{nan_counts[nan_counts > 0]}")
        print(f"将所有 NaN 值填充为 0")
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
        print(f"NaN 值处理完成")
    
    # 将label转换为二分类标签（label为正时为1，为负时为0）
    df['label'] = (df['label'] > 0).astype(int)
    
    return df

def generate_relation_files(datasets):
    """为每个数据集生成归一化互信息矩阵文件"""
    for dataset_name, df in datasets.items():
        print(f"\n开始处理 {dataset_name} 的归一化互信息矩阵...")
        # 只保留 merged 目录
        relation_dir = os.path.join(DATA_OUTPUT_DIR, "relation", "merged")
        os.makedirs(relation_dir, exist_ok=True)
        # 遍历所有可用日期，为每一天生成相关性矩阵
        all_dates = sorted(df['date'].unique())
        if len(all_dates) < PREV_DATE_NUM:
            print(f"{dataset_name} 没有足够的数据天数，跳过")
            continue
        print(f"{dataset_name} 共 {len(all_dates)} 个交易日，将为每一天生成归一化互信息矩阵")

        for i in range(PREV_DATE_NUM - 1, len(all_dates)):
            end_date = all_dates[i]
            end_date_str = pd.Timestamp(end_date).strftime('%Y-%m-%d')
            
            # 检查是否已经存在该日期的矩阵文件
            output_file = os.path.join(relation_dir, f"{end_date_str}.csv")
            if os.path.exists(output_file) and dataset_name != 'merged':
                print(f"跳过 {end_date_str}，矩阵文件已存在")
                continue
                
            print(f"\n处理 {end_date_str} 的归一化互信息矩阵")

            # 获取历史数据的起始日期
            start_date_idx = max(0, i - PREV_DATE_NUM + 1)  # 确保不会越界
            start_date = all_dates[start_date_idx]
            
            # 计算实际可用的历史天数
            available_days = i - start_date_idx + 1
            if available_days < PREV_DATE_NUM:
                print(f"注意: {end_date_str} 只有 {available_days} 天历史数据可用，少于要求的 {PREV_DATE_NUM} 天")
                # 我们继续处理，但会在后续步骤中填充缺失的历史数据

            # 筛选时间范围内的数据
            period_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

            # 按期货代码整理特征数据
            code_features = {}
            for code in period_data['symbol'].unique():
                code_data = period_data[period_data['symbol'] == code]
                # 确保数据按日期排序
                code_data = code_data.sort_values('date')
                
                # 如果数据天数不足，需要填充
                if len(code_data) < PREV_DATE_NUM:
                    # 计算需要补齐的天数
                    pad_len = PREV_DATE_NUM - len(code_data)
                    print(f"代码 {code} 的历史数据不足 {PREV_DATE_NUM} 天，只有 {len(code_data)} 天，计算互信息时将考虑数据填充影响")
                
                # 提取特征
                features = []
                for feature in FEATURE_COLS:
                    if feature in code_data.columns:
                        feat_values = code_data[feature].values
                        if np.isnan(feat_values).any():
                            print(f"警告: {code} 的 {feature} 中包含 NaN 值，替换为 0")
                            feat_values = np.nan_to_num(feat_values, nan=0.0)
                        features.append(feat_values)
                    else:
                        # 如果缺少特征，使用零向量
                        print(f"警告: {code} 缺少 {feature} 特征，使用零向量代替")
                        # 获取返回向量长度来创建零向量
                        if 'return' in code_data.columns:
                            features.append(np.zeros_like(code_data['return'].values))
                        else:
                            features.append(np.zeros(len(code_data)))
                
                code_features[code] = np.array(features)

            # 至少需要2个合约才能计算互信息
            if len(code_features) < 2:
                print(f"跳过 {end_date_str}，没有足够的合约数据 (只有 {len(code_features)} 个合约)")
                continue

            # 计算归一化互信息矩阵
            t1 = time.time()
            print(f"计算 {len(code_features)} 个合约之间的归一化互信息")
            correlation_matrix = stock_mutual_info_matrix(code_features, list(code_features.keys()), processes=1)
            correlation_matrix = correlation_matrix.fillna(0)

            # 设置对角线为1
            for k in range(correlation_matrix.shape[0]):
                correlation_matrix.iloc[k, k] = 1

            t2 = time.time()
            print(f'计算归一化互信息矩阵耗时 {t2 - t1:.2f} 秒')

            # 保存互信息矩阵
            correlation_matrix.to_csv(output_file)
            print(f"已保存归一化互信息矩阵到 {output_file}")

def main():
    """
    主函数，用于生成品种间归一化互信息矩阵
    """
    # 确保所有目录存在
    ensure_directories()

    # 只加载合并数据集
    merged_df = None
    merged_path = os.path.join(FUTURES_DATA_DIR, "merged_features.csv")
    if os.path.exists(merged_path):
        print(f"加载 merged_features.csv...")
        merged_df = pd.read_csv(merged_path)
        if 'date' in merged_df.columns:
            merged_df['date'] = pd.to_datetime(merged_df['date'])
        print(f"合并数据集加载完成: {len(merged_df)} 条记录, {merged_df['symbol'].nunique()} 个合约")
        print(f"时间范围: {merged_df['date'].min()} 至 {merged_df['date'].max()}")
        print(f"特征: {FEATURE_COLS}")
        print(f"标签分布: {merged_df['label'].value_counts().to_dict()}")
    else:
        print(f"错误: {merged_path} 不存在，无法处理")
        return False

    print("\n处理合并后的完整数据集...")
    merged_df = merged_df.sort_values('date')
    generate_relation_files({'merged': merged_df})
        


    print("\n归一化互信息矩阵生成完成!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("THGNN 期货归一化互信息矩阵生成工具")
    print("=" * 60)
    
    # 输出当前配置信息
    print(f"使用输入目录: {FUTURES_DATA_DIR}")
    print(f"使用输出目录: {DATA_OUTPUT_DIR}")
    print(f"计算归一化互信息的天数: {PREV_DATE_NUM}")
    print(f"使用的特征 ({len(FEATURE_COLS)}个): {FEATURE_COLS}")
    print("=" * 60)

    main()
