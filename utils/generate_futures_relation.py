import time
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# 修改特征列以适应期货数据
feature_cols = ['open', 'high', 'low', 'close', 'oi', 'volume']

def cal_pccs(x, y, n):
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc

def calculate_pccs(xs, yss, n):
    result = []
    for name in yss:
        ys = yss[name]
        tmp_res = []
        for pos, x in enumerate(xs):
            y = ys[pos]
            tmp_res.append(cal_pccs(x, y, n))
        result.append(tmp_res)
    return np.mean(result, axis=1)

def stock_cor_matrix(ref_dict, codes, n, processes=1):
    if processes > 1:
        pool = mp.Pool(processes=processes)
        args_all = [(ref_dict[code], ref_dict, n) for code in codes]
        results = [pool.apply_async(calculate_pccs, args=args) for args in args_all]
        output = [o.get() for o in results]
        data = np.stack(output)
        return pd.DataFrame(data=data, index=codes, columns=codes)
    data = np.zeros([len(codes), len(codes)])
    for i in tqdm(range(len(codes))):
        data[i, :] = calculate_pccs(ref_dict[codes[i]], ref_dict, n)
    return pd.DataFrame(data=data, index=codes, columns=codes)

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

# prev_date_num 表示计算期货相关性的天数
prev_date_num = 20
date_unique = df1['dt'].unique()
stock_trade_data = date_unique.tolist()
stock_trade_data.sort()
stock_num = df1.code.unique().shape[0]

# 确保 relation 目录存在
os.makedirs("d:\\MyTHGNN\\data\\relation", exist_ok=True)

# 需要根据实际期货数据调整月末日期
# 为每个数据集选择月末日期
monthly_end_dates = {}

# 根据训练、验证和测试集时间范围划分月份
for dataset_name, dataset_info in datasets.items():
    # 获取数据集时间范围内的日期
    start_date = dataset_info['start_date']
    end_date = dataset_info['end_date']
    
    # 筛选该范围内的交易日
    dataset_dates = [date for date in stock_trade_data 
                     if pd.Timestamp(date) >= start_date and pd.Timestamp(date) <= end_date]
    
    if not dataset_dates:
        print(f"{dataset_name} 没有找到符合条件的交易日")
        continue
        
    # 按月份分组
    months = {}
    for date in dataset_dates:
        date_str = str(date)[:10]
        year_month = date_str[:7]  # 如 "2022-11"
        if year_month not in months:
            months[year_month] = []
        months[year_month].append(date)
    
    # 为每个月选择月末日期
    end_dates = []
    for year_month, dates in sorted(months.items()):
        if len(dates) >= prev_date_num:  # 确保有足够的数据
            dates.sort()
            end_dates.append(dates[-1])  # 添加月末日期
    
    monthly_end_dates[dataset_name] = end_dates
    print(f"{dataset_name} 找到 {len(end_dates)} 个月末日期")
    
    # 如果没有足够数据的月份，选择所有月份的最后一天
    if not end_dates:
        print(f"{dataset_name} 没有月份有足够的交易日，改为选择所有月份的最后一天")
        for year_month, dates in months.items():
            dates.sort()
            end_dates.append(dates[-1])
        monthly_end_dates[dataset_name] = end_dates
        print(f"{dataset_name} 现在有 {len(end_dates)} 个月末日期")

# 打印所有将处理的月末日期
for dataset_name, end_dates in monthly_end_dates.items():
    print(f"\n{dataset_name} 将处理的月末日期:")
    for date in sorted(end_dates):
        print(str(date)[:10])

# 处理每个数据集的月末日期
for dataset_name, end_dates in monthly_end_dates.items():
    print(f"\n开始处理 {dataset_name} 的数据...")
    
    # 创建数据集特定的关系目录
    dataset_relation_dir = f"d:\\MyTHGNN\\data\\relation\\{dataset_name}"
    os.makedirs(dataset_relation_dir, exist_ok=True)
    
    for end_data in end_dates:
        df2 = df1.copy()
        # 保存为以日期字符串命名的CSV文件
        
        end_data_str = str(end_data)[:10]  # 确保只取日期部分 YYYY-MM-DD
        # 检查格式是否正确
        if not end_data_str[4:5] == '-' or not end_data_str[7:8] == '-':
            # 如果不是标准日期格式，尝试转换
            try:
                end_data_str = pd.Timestamp(end_data).strftime('%Y-%m-%d')
            except:
                print(f"警告: 无法将 {end_data} 转换为标准日期格式")

        print(f"\n处理 {end_data_str}")
        
        # 找到end_data在stock_trade_data中的索引
        end_data_index = None
        for j, date in enumerate(stock_trade_data):
            if date == end_data:
                end_data_index = j
                break
        
        if end_data_index is None:
            print(f"跳过 {end_data_str}，找不到对应的索引")
            continue
            
        if end_data_index < prev_date_num - 1:
            print(f"跳过 {end_data_str}，没有足够的历史数据")
            continue
            
        start_data = stock_trade_data[end_data_index - (prev_date_num - 1)]
        
        # 确保日期类型一致
        end_date_dt = pd.to_datetime(end_data) if not isinstance(end_data, pd.Timestamp) else end_data
        start_date_dt = pd.to_datetime(start_data) if not isinstance(start_data, pd.Timestamp) else start_data
        
        df2 = df2.loc[df2['dt'] <= end_date_dt]
        df2 = df2.loc[df2['dt'] >= start_date_dt]
        code = sorted(list(set(df2['code'].values.tolist())))
        test_tmp = {}
        for j in tqdm(range(len(code))):
            df3 = df2.loc[df2['code'] == code[j]]
            y = df3[feature_cols].values
            if y.T.shape[1] == prev_date_num:
                test_tmp[code[j]] = y.T
                
        if len(test_tmp) < 2:
            print(f"跳过 {end_data_str}，没有足够的合约数据")
            continue
                
        t1 = time.time()
        result = stock_cor_matrix(test_tmp, list(test_tmp.keys()), prev_date_num, processes=1)
        result = result.fillna(0)
        for k in range(min(len(test_tmp), result.shape[0])):
            result.iloc[k, k] = 1
        t2 = time.time()
        print('time cost', t2 - t1, 's')
                
        save_path = f"{dataset_relation_dir}/{end_data_str}.csv"
        result.to_csv(save_path)
        print(f"已保存到 {save_path}")
        
        # 同时保存到原始目录，保持兼容性
        compat_path = f"d:\\MyTHGNN\\data\\relation\\{end_data_str}.csv"
        result.to_csv(compat_path)
        print(f"同时保存到 {compat_path} (兼容模式)")

print("\n关系文件生成完成!")
