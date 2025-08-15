from trainer.trainer import *
from data_loader import *
from model.Thgnn import *
import warnings
import torch
import os
import shutil
import time
import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
t_float = torch.float64
torch.multiprocessing.set_sharing_strategy('file_system')

class Args:
    def __init__(self, gpu=0, subtask="classification_binary"):
        train_dir = "d:\\MyTHGNN\\data\\data_train_predict\\train"
        val_dir = "d:\\MyTHGNN\\data\\data_train_predict\\valid"
        self.gpu = gpu  # 新增gpu属性，默认0
        self.device = 'cpu'
        # data settings
        adj_threshold = 0.1
        self.adj_str = str(int(100*adj_threshold))
        self.pos_adj_dir = "pos_adj_" + self.adj_str
        self.feat_dir = "features"
        self.label_dir = "label"
        self.mask_dir = "mask"
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        self.pre_data = pre_data
        # epoch settings
        self.max_epochs = 20              # 设置为20个epoch
        self.epochs_eval = 10
        # learning rate settings
        self.lr = 0.0002
        self.lr_scheduler_factor = 0.8    # 学习率降低的比例因子（从0.5改为0.8，降低更慢）
        self.lr_scheduler_patience = 15   # 等待更多epoch后降低学习率（从5改为15）
        self.lr_scheduler_min_lr = 1e-6   # 学习率下限
        # 图正则化设置
        self.use_graph_reg = True         # 是否使用图正则化
        self.graph_reg_weight = 0.1       # 图正则化权重，从0.01调整为0.1
        test_dir = "d:\\MyTHGNN\\data\\data_train_predict\\test"
        test_stock_dir = "d:\\MyTHGNN\\data\\daily_stock\\test"
        self.hidden_dim = 128
        self.num_heads = 8
        self.out_features = 32
        self.dropout_rate = 0.2  # 添加dropout率参数
        self.model_name = "StockHeteGAT"
        self.batch_size = 1
        self.loss_fcn = mse_loss
        # save model settings
        self.save_path = os.path.join(os.path.abspath('.'), "data\\model_saved\\")
        self.load_path = self.save_path
        self.save_name = self.model_name + "_hidden_" + str(self.hidden_dim) + "_head_" + str(self.num_heads) + \
                         "_outfeat_" + str(self.out_features) + "_batchsize_" + str(self.batch_size) + "_adjth_" + \
                         str(self.adj_str)
        self.epochs_save_by = 50          # 每50个epoch保存一次模型（适用于200个epoch的训练）
        
        # TensorBoard设置
        self.use_tensorboard = True
        self.tensorboard_log_dir = os.path.join(os.path.abspath('.'), "logs")
        
        self.sub_task = subtask
        eval("self.{}".format(self.sub_task))()

    def regression(self):
        self.save_name = self.save_name + "_reg_rank_"
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_regression"
        self.mask_dir = self.mask_dir + "_regression"

    def regression_binary(self):
        self.save_name = self.save_name + "_reg_binary_"
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"

    def classification_binary(self):
        self.save_name = self.save_name + "_clas_binary_"
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"

    def classification_tertiary(self):
        self.save_name = self.save_name + "_clas_tertiary_"
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_threeclass"
        self.mask_dir = self.mask_dir + "_threeclass"


def fun_train_predict(data_start, data_middle, data_end, pre_data):
    # 生成时间戳和日期字符串，用于创建唯一文件名
    timestamp = int(time.time())
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    args = Args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # 使用专门的数据集目录
    train_dir = "d:\\MyTHGNN\\data\\data_train_predict\\train"
    val_dir = "d:\\MyTHGNN\\data\\data_train_predict\\valid"
    
    # 加载数据集
    dataset = AllGraphDataSampler(base_dir=train_dir, data_start=0,
                              data_middle=None, data_end=None)
    val_dataset = AllGraphDataSampler(base_dir=val_dir, mode="val", data_start=0,
                                  data_middle=None, data_end=None)
    
    # 检查数据集是否为空
    if len(dataset) == 0:
        print("错误：训练数据集为空！请检查数据路径和切片设置。")
        return
    if len(val_dataset) == 0:
        print("警告：验证数据集为空！请检查数据路径和切片设置。")
        
    print(f"训练数据集大小: {len(dataset)}, 验证数据集大小: {len(val_dataset)}")
    
    # 确保批次大小不大于数据集大小
    batch_size = min(args.batch_size, len(dataset))
    if batch_size != args.batch_size:
        print(f"警告：调整批次大小从 {args.batch_size} 到 {batch_size}")
        
    # 确保训练和验证集使用相同的collate_fn
    collate_fn = lambda x: x
    
    dataset_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=collate_fn)
    val_dataset_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True, collate_fn=collate_fn)
    
    print(f"数据加载器创建完成: 训练集 {len(dataset)} 样本, 验证集 {len(val_dataset)} 样本")
    
    # 设置Dropout率
    dropout_rate = args.dropout_rate
    print(f"使用Dropout率: {dropout_rate}")
    
    model = eval(args.model_name)(hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                                  out_features=args.out_features, dropout_rate=dropout_rate).to(args.device)

    # 初始化TensorBoard
    if args.use_tensorboard:
        # 创建日志目录，格式为 logs/YYYYMMDD_HHMMSS
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_log_dir = os.path.join(args.tensorboard_log_dir, f"{current_time}")
        
        # 只保留最近两次的日志目录
        if os.path.exists(args.tensorboard_log_dir):
            try:
                # 获取所有子目录
                subdirs = [os.path.join(args.tensorboard_log_dir, d) for d in os.listdir(args.tensorboard_log_dir) 
                          if os.path.isdir(os.path.join(args.tensorboard_log_dir, d))]
                # 按修改时间排序
                subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                
                # 如果超过1个子目录，删除最旧的（保留最新的一个+当前要创建的=两个）
                if len(subdirs) > 1:
                    for old_dir in subdirs[1:]:
                        print(f"删除旧的日志目录: {old_dir}")
                        shutil.rmtree(old_dir, ignore_errors=True)
            except Exception as e:
                print(f"清理旧日志时出错: {e}")
        
        # 确保日志目录存在
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard日志将保存到: {tb_log_dir}")
        
        # 记录模型超参数
        writer.add_text("Parameters/model_name", args.model_name)
        writer.add_text("Parameters/hidden_dim", str(args.hidden_dim))
        writer.add_text("Parameters/num_heads", str(args.num_heads))
        writer.add_text("Parameters/dropout_rate", str(dropout_rate))
    else:
        writer = None

    # train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 使用ReduceLROnPlateau替代StepLR
    # 当验证损失在'patience'个epoch内没有改善时，将学习率乘以'factor'
    lr_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',                       # 监控最小化指标(验证损失)
        factor=args.lr_scheduler_factor,  # 学习率降低倍数
        patience=args.lr_scheduler_patience,  # 等待多少个epoch无改善后降低学习率
        verbose=True,                     # 打印学习率变化信息
        threshold=0.0001,                 # 认为是改善的最小变化阈值
        min_lr=args.lr_scheduler_min_lr   # 学习率下限
    )
    
    # 保留原来的调度器以备需要
    # cold_scheduler = StepLR(optimizer=optimizer, step_size=5000, gamma=0.9, last_epoch=-1)
    # default_scheduler = cold_scheduler
    
    print('start training')
    
    # 初始化最佳模型追踪变量
    best_eval_loss = float('inf')
    best_epoch = -1
    best_model_path = os.path.join(args.save_path, today_date + "_best.dat")
    print(f"基于损失的最佳模型将保存为: {today_date}_best.dat")
    
    # 定义模型评估得分函数 - 同时考虑损失和准确率
    def compute_model_score(loss, acc, alpha=0):
        # 将损失转换为[0,1]范围内的分数，越小越好
        loss_score = max(0, 1 - loss / 2)  # 假设损失通常小于2
        # 组合分数：较低的损失和较高的准确率会产生更高的分数
        return alpha * loss_score + (1 - alpha) * acc
    
    # 选择损失函数 - 是否使用图正则化
    if args.use_graph_reg:
        print(f"使用图结构化正则化损失，权重: {args.graph_reg_weight}")
        loss_function = bce_with_graph_reg_loss
    else:
        print("使用标准BCE损失")
        loss_function = bce_loss
    # 计算预计总训练时间
    start_time = time.time()
    
    for epoch in range(args.max_epochs):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_epoch(epoch=epoch, args=args, model=model, dataset_train=dataset_loader,
                                 optimizer=optimizer, scheduler=None, loss_fcn=loss_function)
        
        eval_loss, _, eval_acc = eval_epoch(args=args, model=model, dataset_eval=val_dataset_loader, loss_fcn=loss_function)
        
        # 更新学习率调度器 - 基于验证损失
        # 当验证损失在args.lr_scheduler_patience个epoch内没有改善时，
        # 学习率会乘以args.lr_scheduler_factor
        lr_scheduler.step(eval_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录到TensorBoard
        if args.use_tensorboard and writer is not None:
            # 损失指标
            writer.add_scalar('1_Loss/train', train_loss, epoch)
            writer.add_scalar('1_Loss/validation', eval_loss, epoch)
            # 损失组件（仅图正则化时记录）
            if args.use_graph_reg and hasattr(model, 'base_loss') and hasattr(model, 'graph_loss'):
                writer.add_scalar('4_Loss_Components/base_bce', model.base_loss, epoch)
                writer.add_scalar('4_Loss_Components/graph_regularization', model.graph_loss, epoch)
                writer.add_scalar('4_Loss_Components/total_combined', model.total_loss, epoch)
            # 准确率
            writer.add_scalar('2_Accuracy/train', train_acc, epoch)
            writer.add_scalar('2_Accuracy/validation', eval_acc, epoch)
            # 学习率
            writer.add_scalar('3_Learning_rate', current_lr, epoch)
        
        # 计算已用时间和估计剩余时间
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / (epoch + 1) * args.max_epochs
        remaining_time = estimated_total_time - elapsed_time
        
        # 格式化时间显示
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02d}h:{minutes:02d}m:{seconds:02d}s"
        
        # 计算进度百分比
        progress = (epoch + 1) / args.max_epochs * 100
        
        # 输出训练信息，包含时间和进度
        print('Epoch: {}/{} ({:.1f}%) [学习率: {:.6f}], train loss: {:.6f}, train acc: {:.4f}, val loss: {:.6f}, val acc: {:.4f}'.format(
            epoch + 1, args.max_epochs, progress, current_lr, train_loss, train_acc, eval_loss, eval_acc))
        
        # 每10个epoch显示时间信息
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'时间信息 - 已用: {format_time(elapsed_time)}, 预计剩余: {format_time(remaining_time)}, 当前epoch耗时: {format_time(epoch_time)}')
        
        # 保存基于损失的最佳模型
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch + 1
            print(f"发现损失更低的模型! Epoch {best_epoch}, val loss: {best_eval_loss:.6f}, val acc: {eval_acc:.4f}")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                    'epoch': epoch + 1, 'val_loss': eval_loss, 'val_acc': eval_acc}
            torch.save(state, best_model_path)
        
        if (epoch + 1) % args.epochs_save_by == 0:
            print("保存周期性模型!")
            periodic_model_path = os.path.join(args.save_path, today_date + "_epoch_" + str(epoch + 1) + ".dat")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                    'val_loss': eval_loss, 'val_acc': eval_acc}
            torch.save(state, periodic_model_path)
            print(f"周期性模型已保存为: {today_date}_epoch_{epoch+1}.dat")

    # 计算总训练时间
    total_training_time = time.time() - start_time
    
    print(f"\n训练完成! 总共训练了 {args.max_epochs} 个epochs，用时 {format_time(total_training_time)}")
    print(f"最佳模型出现在 Epoch {best_epoch}, 验证损失: {best_eval_loss:.6f}")
    print(f"最佳模型已保存到 {best_model_path}")
    
    # 在TensorBoard中添加最终的模型比较结果
    if args.use_tensorboard and writer is not None:
        # 添加最佳模型信息
        writer.add_text("BestModels", 
            f"最佳模型: Epoch {best_epoch}, Loss: {best_eval_loss:.6f}\n"
            f"总训练时间: {format_time(total_training_time)}"
        )
        
        # 关闭TensorBoard writer
        writer.close()
        print("TensorBoard日志已保存完毕")
    
    # 评估最佳模型
    print(f"\n加载最佳模型进行评估...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model'])
    
    # 对验证集重新评估，确保结果一致性
    print("\n对验证集使用最佳模型进行完整评估...")
    val_loss, _, val_acc = eval_epoch(args=args, model=model, dataset_eval=val_dataset_loader, loss_fcn=bce_loss)
    print(f"验证集完整评估 - 损失: {val_loss:.6f}, 准确率: {val_acc:.4f}")
    
    # 如果发现不一致，给出警告
    if abs(val_loss - best_eval_loss) > 0.01:  # 允许0.01的误差
        print(f"警告: 当前验证损失 ({val_loss:.6f}) 与保存的最佳验证损失 ({best_eval_loss:.6f}) 不一致!")
        print("这可能是由于验证方法、数据处理或随机性导致的差异。")
    
    # 所选模型路径用于后续预测
    selected_model_path = best_model_path
    print(f"选定的模型: {os.path.basename(selected_model_path)}")
    
    test_dir = "d:\\MyTHGNN\\data\\data_train_predict\\test"
    test_stock_dir = "d:\\MyTHGNN\\data\\daily_stock\\test"
    
    if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
        print("找到测试集目录，将对测试集进行预测...")
        test_dataset = AllGraphDataSampler(base_dir=test_dir, mode="test", data_start=0,
                                      data_middle=None, data_end=None)
        test_dataset_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True, collate_fn=collate_fn)
        
        # 首先对整个测试集进行完整评估
        print("\n对测试集进行完整评估...")
        test_loss, _, test_acc = eval_epoch(args=args, model=model, dataset_eval=test_dataset_loader, loss_fcn=bce_loss)
        print(f"测试集完整评估 - 损失: {test_loss:.6f}, 准确率: {test_acc:.4f}")
        
        # 然后处理每个样本并保存预测结果
        test_loss_sum = 0.0
        test_acc_sum = 0.0
        test_count = 0
        
        if os.path.exists(test_stock_dir) and len(os.listdir(test_stock_dir)) > 0:
            print(f"使用测试集专用目录加载股票代码数据 ({len(os.listdir(test_stock_dir))} 个文件)...")
            data_code = os.listdir(test_stock_dir)
            data_code = sorted(data_code)
            
            if len(data_code) != len(test_dataset):
                print(f"警告: 测试集数据 ({len(test_dataset)}) 和股票代码文件 ({len(data_code)}) 数量不匹配")
                print("使用较小的数量进行预测")
                
            predict_count = min(len(data_code), len(test_dataset))
            
            df_score = pd.DataFrame()
            for i in tqdm(range(predict_count)):
                stock_file_path = os.path.join(test_stock_dir, data_code[i])
                df = pd.read_csv(stock_file_path, dtype=object)
                
                tmp_data = test_dataset[i]
                pos_adj, features, labels, mask = extract_data(tmp_data, args.device)
                model.eval()
                
                with torch.no_grad():
                    loss, logits, accuracy = evaluate(model, features, pos_adj, labels, mask, loss_func=bce_loss)
                    test_loss_sum += loss.item()
                    test_acc_sum += accuracy
                    test_count += 1
                
                result = logits.data.cpu().numpy().tolist()
                result_new = []
                for j in range(len(result)):
                    result_new.append(result[j][0])
                
                res = {"score": result_new}
                res = DataFrame(res)
                df['score'] = res
                df_score = pd.concat([df_score, df])
            
            if test_count > 0:
                avg_test_loss = test_loss_sum / test_count
                avg_test_acc = test_acc_sum / test_count
                print(f"\n测试集评估结果 - 平均损失: {avg_test_loss:.6f}, 平均准确率: {avg_test_acc:.4f}")
            
            os.makedirs("d:\\MyTHGNN\\data\\prediction", exist_ok=True)
            test_pred_filename = f"test_pred_{date_str}_{timestamp}.csv"
            output_path = os.path.join('d:\\MyTHGNN\\data\\prediction', test_pred_filename)
            try:
                df_score.to_csv(output_path)
                print(f"测试集预测完成，结果已保存到 {output_path}")
            except Exception as e:
                detailed_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = f"test_pred_{detailed_timestamp}.csv"
                backup_path = os.path.join('d:\\MyTHGNN\\data\\prediction', backup_filename)
                df_score.to_csv(backup_path)
                print(f"原保存失败 ({str(e)})，测试集预测结果已保存到备用路径: {backup_path}")
            print(df_score)
        else:
            print("未找到测试集股票代码目录，无法进行测试集预测")
    
    print("\n同时对验证集进行预测...")
    val_stock_dir = "d:\\MyTHGNN\\data\\daily_stock\\valid"
    stock_dir = "d:\\MyTHGNN\\data\\daily_stock"
    
    # 加载验证集股票数据
    data_code = os.listdir(val_stock_dir)
    data_code = sorted(data_code)
    data_code_last = data_code
    
    df_score=pd.DataFrame()
    
    val_loss_sum = 0.0
    val_acc_sum = 0.0
    val_count = 0
    
    predict_count = min(len(data_code_last), len(val_dataset))
    print(f"将对 {predict_count} 个验证集样本进行预测")
    
    for i in tqdm(range(predict_count)):
        stock_file_path = os.path.join(val_stock_dir, data_code_last[i])
        df = pd.read_csv(stock_file_path, dtype=object)
        tmp_data = val_dataset[i]
        pos_adj, features, labels, mask = extract_data(tmp_data, args.device)
        model.eval()
        
        with torch.no_grad():
            loss, logits, accuracy = evaluate(model, features, pos_adj, labels, mask, loss_func=bce_loss)
            val_loss_sum += loss.item()
            val_acc_sum += accuracy
            val_count += 1
            
        result = logits.data.cpu().numpy().tolist()
        result_new = []
        for j in range(len(result)):
            result_new.append(result[j][0])
        res = {"score": result_new}
        res = DataFrame(res)
        df['score'] = res
        df_score=pd.concat([df_score,df])
    
    if val_count > 0:
        avg_val_loss = val_loss_sum / val_count
        avg_val_acc = val_acc_sum / val_count
        print(f"\n验证集评估结果 - 平均损失: {avg_val_loss:.6f}, 平均准确率: {avg_val_acc:.4f}")
    
    try:
        val_pred_filename = f"val_pred_{date_str}_{timestamp}.csv"
        output_path = os.path.join('d:\\MyTHGNN\\data\\prediction', val_pred_filename)
        df_score.to_csv(output_path)
        print(f"验证集预测完成，结果已保存到 {output_path}")
    except Exception as e:
        detailed_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"val_pred_{detailed_timestamp}.csv"
        backup_path = os.path.join('d:\\MyTHGNN\\data\\prediction', backup_filename)
        df_score.to_csv(backup_path)
        print(f"原保存失败 ({str(e)})，验证集预测结果已保存到备用路径: {backup_path}")
    
    print(df_score)
    
if __name__ == "__main__":
    data_start = 0
    data_middle = None
    data_end = None
    # 使用当前日期作为pre_data
    pre_data = datetime.datetime.now().strftime("%Y-%m-%d")
    
    print("使用按数据集划分的目录结构进行训练和预测...")
    fun_train_predict(data_start, data_middle, data_end, pre_data)