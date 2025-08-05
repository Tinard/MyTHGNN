from trainer.trainer import *
from data_loader import *
from model.Thgnn import *
import warnings
import torch
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm

warnings.filterwarnings("ignore")
t_float = torch.float64
torch.multiprocessing.set_sharing_strategy('file_system')

class Args:
    def __init__(self, gpu=0, subtask="classification_binary"):
        # device
        self.gpu = str(gpu)
        self.device = 'cpu'
        # data settings
        adj_threshold = 0.1
        self.adj_str = str(int(100*adj_threshold))
        self.pos_adj_dir = "pos_adj_" + self.adj_str
        self.neg_adj_dir = "neg_adj_" + self.adj_str
        self.feat_dir = "features"
        self.label_dir = "label"
        self.mask_dir = "mask"
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        self.pre_data = pre_data
        # epoch settings
        self.max_epochs = 60
        self.epochs_eval = 10
        # learning rate settings
        self.lr = 0.0002
        self.gamma = 0.3
        # model settings
        self.hidden_dim = 128
        self.num_heads = 8
        self.out_features = 32
        self.model_name = "StockHeteGAT"
        self.batch_size = 1
        self.loss_fcn = mse_loss
        # save model settings
        self.save_path = os.path.join(os.path.abspath('.'), "data\\model_saved\\")
        self.load_path = self.save_path
        self.save_name = self.model_name + "_hidden_" + str(self.hidden_dim) + "_head_" + str(self.num_heads) + \
                         "_outfeat_" + str(self.out_features) + "_batchsize_" + str(self.batch_size) + "_adjth_" + \
                         str(self.adj_str)
        self.epochs_save_by = 60
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
    # 生成时间戳，用于创建唯一文件名
    import time
    timestamp = int(time.time())
    
    args = Args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # 使用专门的数据集目录，如果存在的话
    train_dir = "d:\\MyTHGNN\\data\\data_train_predict\\train_set"
    val_dir = "d:\\MyTHGNN\\data\\data_train_predict\\valid_set"
    base_dir = "d:\\MyTHGNN\\data\\data_train_predict"
    
    # 检查是否存在新的目录结构
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print("使用按数据集划分的目录结构...")
        dataset = AllGraphDataSampler(base_dir=train_dir, data_start=0,
                                  data_middle=None, data_end=None)
        val_dataset = AllGraphDataSampler(base_dir=val_dir, mode="val", data_start=0,
                                      data_middle=None, data_end=None)
    else:
        print("使用传统目录结构...")
        dataset = AllGraphDataSampler(base_dir=base_dir, data_start=data_start,
                                  data_middle=data_middle, data_end=data_end)
        val_dataset = AllGraphDataSampler(base_dir=base_dir, mode="val", data_start=data_start,
                                      data_middle=data_middle, data_end=data_end)
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=lambda x: x)
    val_dataset_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True)
    model = eval(args.model_name)(hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                                  out_features=args.out_features).to(args.device)

    # train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    cold_scheduler = StepLR(optimizer=optimizer, step_size=5000, gamma=0.9, last_epoch=-1)
    default_scheduler = cold_scheduler
    print('start training')
    
    # 初始化最佳模型追踪变量
    best_eval_loss = float('inf')
    best_epoch = -1
    best_model_path = os.path.join(args.save_path, pre_data + "_best.dat")
    print(f"最佳模型将保存为: {pre_data}_best.dat")
    
    for epoch in range(args.max_epochs):
        train_loss = train_epoch(epoch=epoch, args=args, model=model, dataset_train=dataset_loader,
                                 optimizer=optimizer, scheduler=default_scheduler, loss_fcn=bce_loss)
        
        # 每个周期都评估验证集
        eval_loss, _ = eval_epoch(args=args, model=model, dataset_eval=val_dataset_loader, loss_fcn=bce_loss)
        
        # 打印训练信息
        if epoch % args.epochs_eval == 0:
            print('Epoch: {}/{}, train loss: {:.6f}, val loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss,
                                                                              eval_loss))
        else:
            print('Epoch: {}/{}, train loss: {:.6f}, val loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss, eval_loss))
        
        # 保存最佳模型
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch + 1
            print(f"发现更好的模型! Epoch {best_epoch}, val loss: {best_eval_loss:.6f}")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                    'epoch': epoch + 1, 'val_loss': eval_loss}
            torch.save(state, best_model_path)
        
        # 定期保存模型
        if (epoch + 1) % args.epochs_save_by == 0:
            print("保存周期性模型!")
            periodic_model_path = os.path.join(args.save_path, pre_data + "_epoch_" + str(epoch + 1) + ".dat")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, periodic_model_path)
            print(f"周期性模型已保存为: {pre_data}_epoch_{epoch+1}.dat")

    print(f"\n训练完成! 最佳模型出现在 Epoch {best_epoch}, 验证损失: {best_eval_loss:.6f}")
    print(f"最佳模型已保存到 {best_model_path}")
    
    # predict - 使用最佳模型而不是最后一个模型
    print(f"加载最佳模型进行预测...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model'])
    
    # 测试集数据目录
    test_dir = "d:\\MyTHGNN\\data\\data_train_predict\\test_set"
    test_stock_dir = "d:\\MyTHGNN\\data\\daily_stock\\test_set"
    
    # 确认是否存在测试集目录
    if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
        print("找到测试集目录，将对测试集进行预测...")
        test_dataset = AllGraphDataSampler(base_dir=test_dir, mode="test", data_start=0,
                                      data_middle=None, data_end=None)
        test_dataset_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True)
        
        # 加载测试集的股票代码数据
        if os.path.exists(test_stock_dir) and len(os.listdir(test_stock_dir)) > 0:
            print(f"使用测试集专用目录加载股票代码数据 ({len(os.listdir(test_stock_dir))} 个文件)...")
            data_code = os.listdir(test_stock_dir)
            data_code = sorted(data_code)
            
            # 确保测试集数据和股票代码文件数量匹配
            if len(data_code) != len(test_dataset):
                print(f"警告: 测试集数据 ({len(test_dataset)}) 和股票代码文件 ({len(data_code)}) 数量不匹配")
                print("使用较小的数量进行预测")
                
            predict_count = min(len(data_code), len(test_dataset))
            
            # 进行预测
            df_score = pd.DataFrame()
            for i in tqdm(range(predict_count)):
                # 加载股票代码数据
                stock_file_path = os.path.join(test_stock_dir, data_code[i])
                df = pd.read_csv(stock_file_path, dtype=object)
                
                # 获取测试数据并预测
                tmp_data = test_dataset[i]
                pos_adj, neg_adj, features, labels, mask = extract_data(tmp_data, args.device)
                model.eval()  # 设置为评估模式，而不是训练模式
                with torch.no_grad():
                    logits = model(features, pos_adj, neg_adj)
                
                # 处理预测结果
                result = logits.data.cpu().numpy().tolist()
                result_new = []
                for j in range(len(result)):
                    result_new.append(result[j][0])
                
                # 添加预测分数到DataFrame
                res = {"score": result_new}
                res = DataFrame(res)
                df['score'] = res
                df_score = pd.concat([df_score, df])
            
            # 保存预测结果
            os.makedirs("d:\\MyTHGNN\\data\\prediction", exist_ok=True)
            # 使用时间戳创建唯一文件名
            test_pred_filename = f"test_pred_{timestamp}.csv"
            output_path = os.path.join('d:\\MyTHGNN\\data\\prediction', test_pred_filename)
            try:
                df_score.to_csv(output_path)
                print(f"测试集预测完成，结果已保存到 {output_path}")
            except Exception as e:
                # 如果保存失败，尝试使用更具体的时间戳
                import datetime
                detailed_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = f"test_pred_{detailed_timestamp}.csv"
                backup_path = os.path.join('d:\\MyTHGNN\\data\\prediction', backup_filename)
                df_score.to_csv(backup_path)
                print(f"原保存失败 ({str(e)})，测试集预测结果已保存到备用路径: {backup_path}")
            print(df_score)
        else:
            print("未找到测试集股票代码目录，无法进行测试集预测")
    
    # 验证集预测（原有逻辑）
    print("\n同时对验证集进行预测...")
    val_stock_dir = "d:\\MyTHGNN\\data\\daily_stock\\valid_set"
    stock_dir = "d:\\MyTHGNN\\data\\daily_stock"
    
    if os.path.exists(val_stock_dir) and len(os.listdir(val_stock_dir)) > 0:
        print("使用验证集专用目录加载股票代码数据...")
        data_code = os.listdir(val_stock_dir)
        data_code = sorted(data_code)
        data_code_last = data_code  # 所有文件都属于验证集
    else:
        print("使用传统目录结构加载股票代码数据...")
        data_code = os.listdir(stock_dir)
        data_code = sorted(data_code)
        data_code_last = data_code[data_middle:data_end]
    
    # 对验证集进行预测
    df_score=pd.DataFrame()
    
    # 确保验证集数据和股票代码文件数量匹配
    predict_count = min(len(data_code_last), len(val_dataset))
    print(f"将对 {predict_count} 个验证集样本进行预测")
    
    for i in tqdm(range(predict_count)):
        # 确定正确的股票代码文件路径
        stock_file_path = None
        if os.path.exists(val_stock_dir) and len(os.listdir(val_stock_dir)) > 0:
            stock_file_path = os.path.join(val_stock_dir, data_code_last[i])
        else:
            stock_file_path = os.path.join(stock_dir, data_code_last[i])
            
        # 加载股票代码数据
        df = pd.read_csv(stock_file_path, dtype=object)
        tmp_data = val_dataset[i]
        pos_adj, neg_adj, features, labels, mask = extract_data(tmp_data, args.device)
        model.eval()  # 设置为评估模式，而不是训练模式
        with torch.no_grad():
            logits = model(features, pos_adj, neg_adj)
        result = logits.data.cpu().numpy().tolist()
        result_new = []
        for j in range(len(result)):
            result_new.append(result[j][0])
        res = {"score": result_new}
        res = DataFrame(res)
        df['score'] = res
        df_score=pd.concat([df_score,df])

        #df.to_csv('prediction/' + data_code_last[i], encoding='utf-8-sig', index=False)
    
    try:
        # 保存验证集预测结果（使用时间戳创建唯一文件名）
        val_pred_filename = f"val_pred_{timestamp}.csv"
        output_path = os.path.join('d:\\MyTHGNN\\data\\prediction', val_pred_filename)
        df_score.to_csv(output_path)
        print(f"验证集预测完成，结果已保存到 {output_path}")
    except Exception as e:
        # 如果保存失败，尝试使用更具体的时间戳
        import datetime
        detailed_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"val_pred_{detailed_timestamp}.csv"
        backup_path = os.path.join('d:\\MyTHGNN\\data\\prediction', backup_filename)
        df_score.to_csv(backup_path)
        print(f"原保存失败 ({str(e)})，验证集预测结果已保存到备用路径: {backup_path}")
    
    print(df_score)
    
if __name__ == "__main__":
    # 当使用按数据集划分的目录结构时，以下参数可能不再需要
    # 但为了兼容性，我们保留它们
    data_start = 0  # 不再使用切片，从0开始
    data_middle = None  # 不再需要中间索引
    data_end = None  # 不再需要结束索引
    pre_data = '2022-12-29'  # 仍然需要作为模型保存名称
    
    print("使用按数据集划分的目录结构进行训练和预测...")
    fun_train_predict(data_start, data_middle, data_end, pre_data)