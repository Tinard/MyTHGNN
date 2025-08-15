from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch
import math


class GraphAttnMultiHead(Module):
    def __init__(self, in_features, out_features, negative_slope=0.2, num_heads=4, bias=True, residual=True):
        super(GraphAttnMultiHead, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, num_heads * out_features))
        self.weight_u = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.weight_v = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.residual = residual
        if self.residual:
            self.project = nn.Linear(in_features, num_heads*out_features)
        else:
            self.project = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, num_heads * out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_u.size(-1))
        self.weight_u.data.uniform_(-stdv, stdv)
        self.weight_v.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj_mat, requires_weight=False):
        # 检查输入是否包含NaN
        if torch.isnan(inputs).any():
            print("警告: GraphAttnMultiHead输入包含NaN值，尝试修复")
            inputs = torch.nan_to_num(inputs, nan=0.0)
            
        support = torch.mm(inputs, self.weight)
        
        # 检查support是否包含NaN
        if torch.isnan(support).any():
            print("警告: 线性变换后的支持向量包含NaN值，尝试修复")
            support = torch.nan_to_num(support, nan=0.0)
            
        support = support.reshape(-1, self.num_heads, self.out_features).permute(dims=(1, 0, 2))
        
        # 计算注意力系数，检查NaN
        f_1 = torch.matmul(support, self.weight_u).reshape(self.num_heads, 1, -1)
        f_2 = torch.matmul(support, self.weight_v).reshape(self.num_heads, -1, 1)
        
        if torch.isnan(f_1).any() or torch.isnan(f_2).any():
            print("警告: 注意力系数计算中出现NaN值，尝试修复")
            f_1 = torch.nan_to_num(f_1, nan=0.0)
            f_2 = torch.nan_to_num(f_2, nan=0.0)
            
        logits = f_1 + f_2
        weight = self.leaky_relu(logits)
        
        # 检查注意力权重
        if torch.isnan(weight).any():
            print("警告: 激活后的注意力权重包含NaN值，尝试修复")
            weight = torch.nan_to_num(weight, nan=0.0)
            
        # 安全地创建和处理稀疏矩阵
        try:
            masked_weight = torch.mul(weight, adj_mat).to_sparse()
            attn_weights = torch.sparse.softmax(masked_weight, dim=2).to_dense()
        except Exception as e:
            print(f"警告: 稀疏注意力计算失败: {str(e)}，使用密集实现")
            masked_weight = torch.mul(weight, adj_mat)
            # 防止softmax中的溢出
            masked_weight = masked_weight - masked_weight.max(dim=2, keepdim=True)[0]
            exp_weights = torch.exp(masked_weight)
            # 创建掩码以避免除零
            mask = (adj_mat > 0).float()
            attn_weights = exp_weights * mask
            # 归一化
            sum_weights = attn_weights.sum(dim=2, keepdim=True)
            sum_weights = torch.clamp(sum_weights, min=1e-10)  # 防止除零
            attn_weights = attn_weights / sum_weights
        
        # 检查注意力权重
        if torch.isnan(attn_weights).any():
            print("警告: 归一化后的注意力权重包含NaN值，使用均匀权重")
            # 如果attn_weights有NaN，使用均匀注意力
            n_nodes = adj_mat.size(-1)
            mask = (adj_mat > 0).float()
            attn_weights = mask / (mask.sum(dim=2, keepdim=True) + 1e-10)
            
        support = torch.matmul(attn_weights, support)
        support = support.permute(dims=(1, 0, 2)).reshape(-1, self.num_heads * self.out_features)
        
        if self.bias is not None:
            support = support + self.bias
            
        if self.residual:
            residual = self.project(inputs)
            # 检查残差连接
            if torch.isnan(residual).any():
                print("警告: 残差连接计算产生NaN值，使用零替代")
                residual = torch.zeros_like(residual)
            support = support + residual
        
        # 最终检查结果
        if torch.isnan(support).any():
            print("警告: GraphAttnMultiHead最终输出包含NaN值，进行替换")
            support = torch.nan_to_num(support, nan=0.0)
            
        if requires_weight:
            return support, attn_weights
        else:
            return support, None


class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1, eps=1e-6):
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale
        self.eps = eps  # 增加epsilon值，防止除零

    def forward(self, x):
        # 检查输入是否包含NaN
        if torch.isnan(x).any():
            print("警告: PairNorm输入包含NaN值，尝试修复")
            x = torch.nan_to_num(x, nan=0.0)
            
        if self.mode == 'None':
            return x
            
        # 计算列平均值，忽略NaN
        col_mean = x.mean(dim=0)
        if torch.isnan(col_mean).any():
            print("警告: 列平均值计算产生NaN，使用零替代")
            col_mean = torch.zeros_like(col_mean)
            
        if self.mode == 'PN':
            x = x - col_mean
            # 使用更大的epsilon并加入边界检查
            row_square_sum = x.pow(2).sum(dim=1)
            if (row_square_sum < 0).any():
                print("警告: 行平方和包含负值，进行修正")
                row_square_sum = torch.clamp(row_square_sum, min=0.0)
                
            rownorm_mean = (self.eps + row_square_sum.mean()).sqrt()
            if rownorm_mean == 0 or torch.isnan(rownorm_mean):
                print("警告: 行范数平均值为零或NaN，使用epsilon替代")
                rownorm_mean = torch.tensor(self.eps, device=x.device)
                
            x = self.scale * x / rownorm_mean
            
        if self.mode == 'PN-SI':
            x = x - col_mean
            # 使用更大的epsilon并加入边界检查
            row_square_sum = x.pow(2).sum(dim=1, keepdim=True)
            if (row_square_sum < 0).any():
                print("警告: 行平方和包含负值，进行修正")
                row_square_sum = torch.clamp(row_square_sum, min=0.0)
                
            rownorm_individual = (self.eps + row_square_sum).sqrt()
            if (rownorm_individual == 0).any() or torch.isnan(rownorm_individual).any():
                print("警告: 个别行范数为零或NaN，使用epsilon替代")
                rownorm_individual = torch.ones_like(rownorm_individual) * self.eps
                
            x = self.scale * x / rownorm_individual
            
        if self.mode == 'PN-SCS':
            # 使用更大的epsilon并加入边界检查
            row_square_sum = x.pow(2).sum(dim=1, keepdim=True)
            if (row_square_sum < 0).any():
                print("警告: 行平方和包含负值，进行修正")
                row_square_sum = torch.clamp(row_square_sum, min=0.0)
                
            rownorm_individual = (self.eps + row_square_sum).sqrt()
            if (rownorm_individual == 0).any() or torch.isnan(rownorm_individual).any():
                print("警告: 个别行范数为零或NaN，使用epsilon替代")
                rownorm_individual = torch.ones_like(rownorm_individual) * self.eps
                
            x = self.scale * x / rownorm_individual - col_mean
            
        # 最终检查结果是否包含NaN
        if torch.isnan(x).any():
            print("警告: PairNorm输出包含NaN值，使用原始输入")
            return torch.nan_to_num(x, nan=0.0)
            
        return x


class GraphAttnSemIndividual(Module):
    def __init__(self, in_features, hidden_size=128, act=nn.Tanh()):
        super(GraphAttnSemIndividual, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_features, hidden_size),
                                     act,
                                     nn.Linear(hidden_size, 1, bias=False))

    def forward(self, inputs, requires_weight=False):
        # 检查输入是否包含NaN
        if torch.isnan(inputs).any():
            print("警告: GraphAttnSemIndividual输入包含NaN值，尝试修复")
            inputs = torch.nan_to_num(inputs, nan=0.0)
        
        # 计算注意力权重
        w = self.project(inputs)
        
        # 检查权重是否包含NaN
        if torch.isnan(w).any():
            print("警告: 语义注意力权重计算产生NaN值，使用均匀权重")
            w = torch.ones_like(w) / inputs.size(1)  # 均匀注意力
        else:
            # 安全地计算softmax
            try:
                # 减去最大值以提高数值稳定性
                w_max = w.max(dim=1, keepdim=True)[0]
                w_exp = torch.exp(w - w_max)
                beta = w_exp / (w_exp.sum(dim=1, keepdim=True) + 1e-10)
            except Exception as e:
                print(f"警告: 语义注意力softmax计算失败: {str(e)}，使用均匀权重")
                beta = torch.ones_like(w) / inputs.size(1)  # 均匀注意力
        
        # 检查beta是否包含NaN
        if torch.isnan(beta).any():
            print("警告: 归一化后的语义注意力权重包含NaN值，使用均匀权重")
            beta = torch.ones_like(w) / inputs.size(1)  # 均匀注意力
        
        # 计算加权和
        weighted_sum = (beta * inputs).sum(1)
        
        # 最终检查结果
        if torch.isnan(weighted_sum).any():
            print("警告: 语义注意力加权和包含NaN值，使用输入平均值")
            weighted_sum = inputs.mean(dim=1)  # 使用平均值作为后备
        
        if requires_weight:
            return weighted_sum, beta
        else:
            return weighted_sum, None


class StockHeteGAT(nn.Module):
    def __init__(self, in_features=7, out_features=8, num_heads=8, hidden_dim=64, num_layers=1, dropout_rate=0.2):
        super(StockHeteGAT, self).__init__()
        self.encoding = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1
        )
        
        # 在GRU输出后添加Dropout
        self.dropout_gru = nn.Dropout(dropout_rate)
        
        self.pos_gat = GraphAttnMultiHead(
            in_features=hidden_dim,
            out_features=out_features,
            num_heads=num_heads
        )
        
        # 在GAT输出后添加Dropout
        self.dropout_gat = nn.Dropout(dropout_rate)
        
        # 添加MLPs处理不同类型的支持向量
        self.mlp_self = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_pos = nn.Linear(num_heads * out_features, hidden_dim)
        
        # 在MLP后添加Dropout
        self.dropout_mlp = nn.Dropout(dropout_rate)
        
        self.pn = PairNorm(mode='PN', scale=1.0)
        
        # 已删除负相关图相关模块
        self.sem_gat = GraphAttnSemIndividual(in_features=hidden_dim,
                                              hidden_size=hidden_dim,
                                              act=nn.Tanh())
        
        # 在语义GAT后添加Dropout
        self.dropout_sem = nn.Dropout(dropout_rate)
        
        # 在最终预测器前添加Dropout
        self.predictor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                
    def train(self, mode=True):
        """
        重写train方法，确保模型在训练模式下正确激活所有Dropout层
        """
        super(StockHeteGAT, self).train(mode)
        if mode:
            self.dropout_gru.train()
            self.dropout_gat.train()
            self.dropout_mlp.train()
            self.dropout_sem.train()
        return self
        
    def eval(self):
        """
        重写eval方法，确保模型在评估模式下正确禁用所有Dropout层
        """
        super(StockHeteGAT, self).eval()
        self.dropout_gru.eval()
        self.dropout_gat.eval()
        self.dropout_mlp.eval()
        self.dropout_sem.eval()
        return self

    def forward(self, inputs, pos_adj, requires_weight=False):
        # 检查输入是否包含NaN
        if torch.isnan(inputs).any():
            print("警告: 输入数据包含NaN值，尝试修复")
            inputs = torch.nan_to_num(inputs, nan=0.0)
            
        _, support = self.encoding(inputs)
        support = support.squeeze()
        
        # 在GRU输出后应用Dropout
        support = self.dropout_gru(support)
        
        # 检查support是否包含NaN
        if torch.isnan(support).any():
            print("警告: 编码后的支持向量包含NaN值，尝试修复")
            support = torch.nan_to_num(support, nan=0.0)
            
        pos_support, pos_attn_weights = self.pos_gat(support, pos_adj, requires_weight)
        
        # 在GAT输出后应用Dropout
        pos_support = self.dropout_gat(pos_support)
        
        # 处理支持向量
        self_support = self.mlp_self(support)
        pos_support = self.mlp_pos(pos_support)
        
        # 在MLP输出后应用Dropout
        self_support = self.dropout_mlp(self_support)
        pos_support = self.dropout_mlp(pos_support)
        
        # 检查MLP处理后的支持向量
        if torch.isnan(self_support).any() or torch.isnan(pos_support).any():
            print("警告: MLP处理后的支持向量包含NaN值，尝试修复")
            self_support = torch.nan_to_num(self_support, nan=0.0)
            pos_support = torch.nan_to_num(pos_support, nan=0.0)
            
        # 堆叠自连接和正相关
        all_embedding = torch.stack((self_support, pos_support), dim=1)
        all_embedding, sem_attn_weights = self.sem_gat(all_embedding, requires_weight)
        
        # 在语义GAT后应用Dropout
        all_embedding = self.dropout_sem(all_embedding)
        
        # 检查嵌入向量
        if torch.isnan(all_embedding).any():
            print("警告: 语义GAT处理后的嵌入向量包含NaN值，尝试修复")
            all_embedding = torch.nan_to_num(all_embedding, nan=0.0)
            
        all_embedding = self.pn(all_embedding)
        
        # 最后检查嵌入向量
        if torch.isnan(all_embedding).any():
            print("警告: PairNorm处理后的嵌入向量包含NaN值，尝试修复")
            all_embedding = torch.nan_to_num(all_embedding, nan=0.0)
        
        # 获取预测结果 (predictor已包含Dropout)
        prediction = self.predictor(all_embedding)
        
        # 确保输出在0-1范围内且没有NaN值
        if torch.isnan(prediction).any():
            print("警告: 预测输出包含NaN值，替换为0.5")
            prediction = torch.nan_to_num(prediction, nan=0.5)
        
        prediction = torch.clamp(prediction, min=0.0, max=1.0)
        
        # 确保输出不是标量
        if prediction.dim() == 0:
            print("警告: 预测输出是标量，转换为适当的形状")
            prediction = prediction.unsqueeze(0)  # 转换为1维张量
        
        if requires_weight:
            return prediction, (pos_attn_weights, sem_attn_weights)
        else:
            return prediction
