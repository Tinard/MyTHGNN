import torch
import torch.nn as nn
import torch.nn.functional as F

def mse_loss(logits, targets):
    mse = nn.MSELoss()
    loss = mse(logits.squeeze(), targets)
    return loss


def bce_loss(logits, targets):
    bce = nn.BCELoss()
    loss = bce(logits.squeeze(), targets)
    return loss


def graph_regularization_loss(logits, adj_matrix, alpha=0.01):
    """
    计算图正则化损失，鼓励相连节点具有相似的预测
    
    参数:
    logits: 模型预测输出
    adj_matrix: 邻接矩阵，表示节点间的连接关系
    alpha: 正则化强度系数
    
    返回:
    正则化损失
    """
    # 确保logits是合适的形状
    if logits.dim() == 1:
        logits = logits.unsqueeze(1)
    
    # 计算节点间预测差异
    # 对于每对相连节点 (i,j)，计算 ||logits_i - logits_j||^2 * adj_matrix[i,j]
    n = logits.shape[0]
    if n <= 1:  # 如果只有一个节点，则无法计算图正则化
        return torch.tensor(0.0, device=logits.device)
    
    # 创建差异矩阵 D，其中 D[i,j] = (logits[i] - logits[j])^2
    logits_i = logits.repeat(n, 1)
    logits_j = logits.repeat(1, n).view(n*n, -1)
    diff = (logits_i - logits_j).pow(2).sum(dim=1).view(n, n)
    
    # 应用邻接矩阵作为权重，只考虑相连节点
    reg_loss = torch.sum(diff * adj_matrix) / 2.0  # 除以2避免重复计算
    
    return alpha * reg_loss


def bce_with_graph_reg_loss(logits, targets, adj_matrix=None, reg_weight=0.01):
    """
    结合二元交叉熵损失和图正则化损失
    
    参数:
    logits: 模型预测输出
    targets: 目标标签
    adj_matrix: 邻接矩阵
    reg_weight: 图正则化权重
    
    返回:
    组合损失
    """
    # 基础BCE损失
    bce = nn.BCELoss()
    base_loss = bce(logits.squeeze(), targets)
    
    # 如果没有提供邻接矩阵，则只返回BCE损失
    if adj_matrix is None:
        return base_loss
    
    # 图正则化损失
    graph_loss = graph_regularization_loss(logits, adj_matrix, alpha=1.0)
    
    # 组合损失
    combined_loss = base_loss + reg_weight * graph_loss
    
    return combined_loss


def evaluate(model, features, adj_pos, labels, mask, loss_func=nn.L1Loss()):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_pos)
        
    # 检查logits是否为标量
    if logits.dim() == 0:  # 如果是标量，转换为适当的形状
        logits = logits.unsqueeze(0)  # 将标量转换为1维张量
    
    # 计算损失时确保使用掩码
    if mask is not None and len(mask) > 0:
        loss = loss_func(logits[mask], labels[mask])
    else:
        loss = loss_func(logits, labels)
        
    # 计算准确率
    predictions = (logits > 0.5).float()
    if mask is not None and len(mask) > 0:
        accuracy = (predictions[mask] == labels[mask]).float().mean()
    else:
        accuracy = (predictions == labels).float().mean()
        
    return loss, logits, accuracy


def extract_data(data_dict, device):
    pos_adj = data_dict['pos_adj'].to(device).squeeze()
    features = data_dict['features'].to(device).squeeze()
    labels = data_dict['labels'].to(device).squeeze()
    mask = data_dict['mask']
    return pos_adj, features, labels, mask


def train_epoch(epoch, args, model, dataset_train, optimizer, scheduler, loss_fcn):
    model.train()
    loss_return = 0
    acc_return = 0
    samples_count = 0
    
    # 获取图正则化权重参数
    reg_weight = getattr(args, 'graph_reg_weight', 0.01)  # 默认值为0.01
    
    for batch_data in dataset_train:
        for batch_idx, data in enumerate(batch_data):
            model.zero_grad()
            pos_adj, features, labels, mask = extract_data(data, args.device)
            
            # 前向传播
            logits = model(features, pos_adj)
            
            # 检查logits是否为标量
            if logits.dim() == 0:  # 如果是标量，转换为适当的形状
                print(f"警告: 批次 {batch_idx} 的logits是标量，转换为适当的形状")
                logits = logits.unsqueeze(0)  # 将标量转换为1维张量
            
            # 判断使用哪种损失函数
            if loss_fcn.__name__ == 'bce_with_graph_reg_loss':
                # 如果使用图正则化损失函数
                if mask is not None and len(mask) > 0:
                    # 获取基础BCE损失和图正则化损失，用于记录
                    bce = nn.BCELoss()
                    base_loss = bce(logits[mask].squeeze(), labels[mask])
                    graph_loss = graph_regularization_loss(logits[mask], pos_adj[mask][:, mask], alpha=1.0)
                    # 组合损失
                    loss = base_loss + reg_weight * graph_loss
                    
                    # 记录单独的损失组件（如果需要在外部访问）
                    model.base_loss = base_loss.item()
                    model.graph_loss = graph_loss.item()
                    model.total_loss = loss.item()
                    
                    # 计算准确率
                    predictions = (logits[mask] > 0.5).float()
                    accuracy = (predictions == labels[mask]).float().mean()
                else:
                    # 获取基础BCE损失和图正则化损失，用于记录
                    bce = nn.BCELoss()
                    base_loss = bce(logits.squeeze(), labels)
                    graph_loss = graph_regularization_loss(logits, pos_adj, alpha=1.0)
                    # 组合损失
                    loss = base_loss + reg_weight * graph_loss
                    
                    # 记录单独的损失组件（如果需要在外部访问）
                    model.base_loss = base_loss.item()
                    model.graph_loss = graph_loss.item()
                    model.total_loss = loss.item()
                    
                    # 计算准确率
                    predictions = (logits > 0.5).float()
                    accuracy = (predictions == labels).float().mean()
            else:
                # 使用普通损失函数
                if mask is not None and len(mask) > 0:
                    loss = loss_fcn(logits[mask], labels[mask])
                    # 计算准确率
                    predictions = (logits[mask] > 0.5).float()
                    accuracy = (predictions == labels[mask]).float().mean()
                else:
                    loss = loss_fcn(logits, labels)
                    # 计算准确率
                    predictions = (logits > 0.5).float()
                    accuracy = (predictions == labels).float().mean()
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            # 仅当scheduler不为None时才调用step
            if scheduler is not None:
                scheduler.step()
            
            # 累计损失和准确率
            loss_return += loss.item()
            acc_return += accuracy.item()
            samples_count += 1
    
    # 计算平均损失和准确率
    if samples_count > 0:
        loss_return /= samples_count
        acc_return /= samples_count
    
    return loss_return, acc_return


def eval_epoch(args, model, dataset_eval, loss_fcn):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    samples_count = 0
    all_logits = None
    
    # 获取图正则化权重参数
    reg_weight = getattr(args, 'graph_reg_weight', 0.01)  # 默认值为0.01
    
    for batch_idx, data in enumerate(dataset_eval):
        # 处理数据加载器返回的列表
        if isinstance(data, list):
            # 如果是列表，我们只处理第一个元素
            if data:  # 确保列表不为空
                data = data[0]  # 获取列表的第一个元素
            else:
                continue
        pos_adj, features, labels, mask = extract_data(data, args.device)
        
        with torch.no_grad():
            logits = model(features, pos_adj)
            
            # 判断使用哪种损失函数
            if loss_fcn.__name__ == 'bce_with_graph_reg_loss':
                # 如果使用图正则化损失函数
                if mask is not None and len(mask) > 0:
                    loss = loss_fcn(logits[mask], labels[mask], pos_adj, reg_weight)
                    predictions = (logits[mask] > 0.5).float()
                    accuracy = (predictions == labels[mask]).float().mean()
                else:
                    loss = loss_fcn(logits, labels, pos_adj, reg_weight)
                    predictions = (logits > 0.5).float()
                    accuracy = (predictions == labels).float().mean()
            else:
                # 使用普通评估
                if mask is not None and len(mask) > 0:
                    loss = loss_fcn(logits[mask], labels[mask])
                    predictions = (logits[mask] > 0.5).float()
                    accuracy = (predictions == labels[mask]).float().mean()
                else:
                    loss = loss_fcn(logits, labels)
                    predictions = (logits > 0.5).float()
                    accuracy = (predictions == labels).float().mean()
                
        loss_sum += loss.item()
        acc_sum += accuracy.item()
        samples_count += 1
        # 保存第一个批次的logits用于返回
        if batch_idx == 0:
            all_logits = logits
    
    # 计算平均损失和准确率
    avg_loss = loss_sum / max(1, samples_count)
    avg_acc = acc_sum / max(1, samples_count)
    
    # 返回平均损失、第一个批次的logits和平均准确率
    return avg_loss, all_logits, avg_acc
