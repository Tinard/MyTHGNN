# THGNN - 时序异构图神经网络

这是一个基于时序异构图神经网络(Temporal Heterogeneous Graph Neural Network)的期货合约价格预测系统。

## 项目结构

- `utils/generate_relation.py`: 生成归一化互信息矩阵，计算品种间关系强度
- `utils/generate_graph_data.py`: 基于归一化互信息矩阵构建图结构数据
- `process_data.py`: 数据处理主模块，调用上述两个模块完成数据处理
- `model/Thgnn.py`: 模型定义，包含THGNN的网络结构
- `trainer/trainer.py`: 训练器，包含训练和评估函数
- `data_loader.py`: 数据加载器，用于读取处理后的图数据
- `main.py`: 主程序，用于模型训练和预测

## 数据目录

- `data/`: 处理后的数据存储目录
  - `daily_stock/`: 日线数据
  - `data_train_predict/`: 训练和预测用的图数据
  - `relation/`: 品种间归一化互信息矩阵
  - `model_saved/`: 保存的模型参数
  - `prediction/`: 预测结果

## 使用方法

### 1. 数据处理

运行数据处理程序，将原始期货数据转换为模型所需的格式：

```bash
python process_data.py
```

该程序会执行以下操作：
- 生成品种间的归一化互信息矩阵
- 基于互信息矩阵构建异构图数据

也可以单独执行其中一个步骤：

```bash
# 仅生成归一化互信息矩阵
python process_data.py --only-relation

# 仅生成图结构数据（前提是互信息矩阵已存在）
python process_data.py --only-graph
```

如需修改输入输出目录或特征配置，请直接编辑 `utils/generate_relation.py` 文件中的相关变量：
- `FUTURES_DATA_DIR`: 输入数据目录
- `DATA_OUTPUT_DIR`: 输出数据目录
- `FEATURE_COLS`: 使用的特征列（当前仅保留动量因子相关特征）
- `PREV_DATE_NUM`: 计算归一化互信息的天数

### 2. 模型训练

运行主程序进行模型训练：

```bash
python main.py
```

如需修改训练参数，请直接编辑 `main.py` 文件：
- 在 `Args` 类中修改模型参数（如 `max_epochs`、`lr`、`batch_size` 等）
- 在 `if __name__ == "__main__":` 部分修改 `pre_data` 值来改变模型保存名称

训练完成后，模型会被保存到 `data/model_saved/` 目录。

### 3. 特征配置

当前使用的特征为动量因子相关特征：
- ROC_5：5日价格变化率
- ROC_10：10日价格变化率
- MACD：平滑异同移动平均线
- MACD_Signal：MACD信号线
- MACD_Hist：MACD柱状图
- RSI：相对强弱指标
- return：收益率

## 技术架构

该项目实现了一个时序异构图神经网络，用于期货价格预测：

1. 基于归一化互信息计算品种间关系强度
2. 使用GRU编码时序特征
3. 使用多头图注意力网络处理异构图结构
4. 通过语义层次的注意力机制融合不同类型的图信息
5. 支持二分类和回归任务

## 最近更新

- 将相关性计算替换为归一化互信息(NMI)计算，更好地捕捉非线性关系
- 特征选择优化为仅保留动量因子相关特征
- 使用'return'列计算互信息，更准确地表示品种间关系
- 将数据处理模块重构为两个独立模块，提高代码可维护性

## 需求环境

- Python 3.6+
- PyTorch 1.7+
- scikit-learn (用于互信息计算)
- pandas
- numpy
- networkx
- tqdm

## 安装依赖

```bash
pip install torch pandas numpy networkx tqdm scikit-learn
```
