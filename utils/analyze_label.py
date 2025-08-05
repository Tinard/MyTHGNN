import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def analyze_label_in_pickle(file_path="d:\\MyTHGNN\\data\\csi300.pkl"):
    """
    分析 pickle 文件中 label 列的含义
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    try:
        print(f"正在读取文件: {file_path}...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, pd.DataFrame):
            print(f"数据不是DataFrame类型，而是: {type(data)}")
            return
        
        if 'label' not in data.columns:
            print("数据中没有'label'列")
            return
        
        print("\n=== Label 基本统计信息 ===")
        label_stats = data['label'].describe()
        print(label_stats)
        
        # 计算收益率 (假设label可能是收益率)
        if 'close' in data.columns:
            print("\n=== 计算股票收益率并与label比较 ===")
            
            # 按照股票代码和日期排序
            data_sorted = data.sort_values(by=['code', 'dt'])
            
            # 计算下一日收益率
            data_sorted['next_day_return'] = data_sorted.groupby('code')['close'].shift(-1) / data_sorted['close'] - 1
            
            # 选择有下一日收益率的数据
            valid_data = data_sorted.dropna(subset=['next_day_return'])
            
            # 计算label与收益率的相关性
            correlation = valid_data['label'].corr(valid_data['next_day_return'])
            print(f"Label与下一日收益率的相关性: {correlation}")
            
            # 显示label和收益率的前几行进行比较
            compare_df = valid_data[['code', 'dt', 'close', 'next_day_return', 'label']].head(10)
            print("\nLabel与下一日收益率比较:")
            print(compare_df)
            
            # 检查label是否等于下一日收益率
            is_equal = np.isclose(valid_data['label'], valid_data['next_day_return'], rtol=1e-5)
            equal_percentage = is_equal.mean() * 100
            print(f"\nLabel与下一日收益率完全相同的比例: {equal_percentage:.2f}%")
            
            # 如果相关性低，检查是否为排名或其他关系
            if abs(correlation) < 0.9:
                print("\n=== 探索其他可能的label含义 ===")
                
                # 检查是否为分类标签
                unique_labels = data['label'].unique()
                print(f"Label的不同取值数量: {len(unique_labels)}")
                if len(unique_labels) <= 5:  # 如果label只有少数几个值，可能是分类
                    print(f"Label的唯一值: {sorted(unique_labels)}")
                    label_counts = data['label'].value_counts()
                    print("各类别的数量:")
                    print(label_counts)
                    
                    # 如果是二分类，检查是否与收益率方向一致
                    if len(unique_labels) == 2:
                        # 检查label是否代表收益率的正负
                        sign_match = (valid_data['label'] > 0) == (valid_data['next_day_return'] > 0)
                        sign_match_percentage = sign_match.mean() * 100
                        print(f"Label与收益率方向一致的比例: {sign_match_percentage:.2f}%")
                
                # 检查是否为收益率的分位数或排名
                # 在每个日期内，按收益率对股票排序
                data_sorted['return_rank'] = data_sorted.groupby('dt')['next_day_return'].rank(pct=True)
                
                # 查看label与收益率排名的相关性
                valid_data = data_sorted.dropna(subset=['return_rank', 'label'])
                rank_correlation = valid_data['label'].corr(valid_data['return_rank'])
                print(f"Label与收益率排名的相关性: {rank_correlation}")
                
                # 检查是否为离散化的收益率
                # 将收益率分为几个区间
                data_sorted['return_bin'] = pd.qcut(data_sorted['next_day_return'], 5, labels=False, duplicates='drop')
                valid_data = data_sorted.dropna(subset=['return_bin', 'label'])
                
                # 如果label也是分类变量，计算它与收益率分位数的一致性
                if len(unique_labels) <= 5:
                    # 转换label为数值类型以进行比较
                    valid_data['label_numeric'] = pd.Categorical(valid_data['label']).codes
                    bin_correlation = valid_data['label_numeric'].corr(valid_data['return_bin'])
                    print(f"Label与收益率分位数的相关性: {bin_correlation}")
        
        # 绘制label的分布图
        plt.figure(figsize=(10, 6))
        sns.histplot(data['label'], kde=True)
        plt.title('Label分布')
        plt.xlabel('Label值')
        plt.ylabel('频率')
        
        # 保存图像
        plt_path = os.path.join(os.path.dirname(file_path), 'label_distribution.png')
        plt.savefig(plt_path)
        print(f"\n分布图已保存到: {plt_path}")
        
        # 如果label可能是收益率，绘制label与实际收益率的散点图
        if 'next_day_return' in locals():
            plt.figure(figsize=(10, 6))
            plt.scatter(valid_data['next_day_return'], valid_data['label'], alpha=0.1)
            plt.title('Label vs 实际收益率')
            plt.xlabel('实际下一日收益率')
            plt.ylabel('Label值')
            
            # 添加对角线
            min_val = min(valid_data['next_day_return'].min(), valid_data['label'].min())
            max_val = max(valid_data['next_day_return'].max(), valid_data['label'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            scatter_path = os.path.join(os.path.dirname(file_path), 'label_vs_return.png')
            plt.savefig(scatter_path)
            print(f"对比图已保存到: {scatter_path}")
        
        # 总结分析结果
        print("\n=== 分析结论 ===")
        if 'correlation' in locals() and abs(correlation) > 0.9:
            print(f"Label很可能代表下一日收益率，相关性为 {correlation:.4f}")
        elif 'sign_match_percentage' in locals() and sign_match_percentage > 80:
            print(f"Label很可能代表收益率的方向 (上涨/下跌)，准确率为 {sign_match_percentage:.2f}%")
        elif 'rank_correlation' in locals() and abs(rank_correlation) > 0.7:
            print(f"Label很可能代表收益率的相对排名，相关性为 {rank_correlation:.4f}")
        elif 'bin_correlation' in locals() and abs(bin_correlation) > 0.7:
            print(f"Label很可能代表收益率的分位数，相关性为 {bin_correlation:.4f}")
        else:
            print("无法确定Label的准确含义，可能需要进一步分析或参考项目文档")
            
    except Exception as e:
        print(f"分析过程中出错: {str(e)}")

if __name__ == "__main__":
    analyze_label_in_pickle()
