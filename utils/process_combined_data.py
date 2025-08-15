"""
THGNN 数据处理优化工具

此脚本实现两个主要优化:
1. 历史数据补全：对于历史数据不足的合约，用0填充到所需长度
2. 数据集统一处理：将train/valid/test合并为一个完整的数据集处理，保证所有节点历史信息一致性

用法：
1. 首先生成归一化互信息矩阵: python utils/process_combined_data.py --step=relation
2. 然后生成图结构数据: python utils/process_combined_data.py --step=graph
3. 或一次性完成所有步骤: python utils/process_combined_data.py --step=all
"""
import os
import argparse
from utils.generate_relation import main as generate_relation_main
from utils.generate_graph_data import main as generate_graph_main
from utils.generate_relation import DATA_OUTPUT_DIR, ensure_directories

def process_combined_data(step="all"):
    """处理合并数据集，实现历史数据补全和数据集统一处理"""
    print("=" * 60)
    print("THGNN 数据处理优化工具")
    print("=" * 60)
    
    # 确保所有目录存在
    ensure_directories()
    
    # 步骤1: 生成归一化互信息矩阵
    if step in ["relation", "all"]:
        print("\n第1步: 生成统一处理的归一化互信息矩阵...")
        datasets = generate_relation_main()
        if not datasets:
            print("错误: 生成归一化互信息矩阵失败!")
            return False
    
    # 步骤2: 生成图结构数据
    if step in ["graph", "all"]:
        print("\n第2步: 生成具有历史数据补全的图结构数据...")
        success = generate_graph_main()
        if not success:
            print("错误: 生成图结构数据失败!")
            return False
    
    print("\n处理完成!")
    print("=" * 60)
    print("数据处理优化完成，现在数据集具有以下改进:")
    print("1. 所有节点的历史数据都已补全，确保图结构完整性")
    print("2. 数据集已统一处理，保证valid/test数据集拥有完整的历史信息")
    print("3. 节点特征保持一致的时序长度，确保模型输入标准化")
    print("=" * 60)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="THGNN 数据处理优化工具")
    parser.add_argument("--step", type=str, choices=["relation", "graph", "all"], default="all",
                        help="处理步骤: relation(只生成关系矩阵), graph(只生成图结构), all(全部步骤)")
    
    args = parser.parse_args()
    process_combined_data(args.step)
