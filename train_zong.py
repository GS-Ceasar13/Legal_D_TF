# coding: utf-8

from LSTM import LSTM
from GRU import GRU
from RNN import RNN
from CNN import CNN
from Transformer import Transformer
from TextCNN import TextCNN
from TextRCNN import TextRCNN
from FastText import FastText
from GRU_new import GRU_new
from DPCNN import DPCNN
# from MLP_NP import MLP
# from model_final import MLP
from input_final import InputData
from collections import deque
import numpy as np
import os
import torch
from math import sqrt
import torch.nn as nn
import csv
import datetime
import time
# import gensim
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def train(data_address,out_txt_path, out_csv_path,data_name,output_dir,latest_checkpoint_file,start_record_epoch,vector_address=None, embd_dimension=2, embd_dimension_suffix=5, hidden_dim_suffix=2,
          train_splitThreshold=0.8,
          time_unit='day', batch_size=9,
          loss_type='L1Loss', optim_type='Adam', model_type='RNN', hidden_dim=10,    ######################修改model_type名，分别是RNN，LSTM和GRU
          train_type='iteration', n_layer=1, dropout=0.1, max_epoch_num=10, learn_rate_min=0.0001, path_length=8,
          train_record_folder='./train_record/', model_save_folder='./model/', result_save_folder='./result/',
          ts_type='set'):
    # 初始化数据
    out_size = 1
    learn_rate = 0.01
    epoch = 0
    learn_rate_backup = 0.01
    learn_rate_down_backup = 0.001
    loss_deque = deque(maxlen=20)
    loss_change_deque = deque(maxlen=30)
    print('User Model ' + model_type + " To Start Experience.")

    data = InputData(data_address, embd_dimension=embd_dimension)
    print("InputData Finish")

    data.encodeEvent()
    print("EncodeEvent Finish")
    data.encodeTrace()
    print("EncodeTrace Finish")
    # 通过设置固定的随机数种子以获取相同的训练集和测试集,timeEmbedding=data.timeEmbedding
    data.splitData(train_splitThreshold)
    print("SplitData Finish")
    train_singlePrefixData1, train_labelPrefixData, test_singlePrefixData1, test_labelPrefixData, \
               train_singlePrefixData_processing, test_singlePrefixData_processing = data.initBatchData_Prefix()
    print("InitBatchData Finish")
    # data.generateSingleLengthBatch(batch_size)
    temp=len(train_singlePrefixData1[0])
    # 初始化模型CrossEntropyLoss
    if model_type == 'GRU':
        model=GRU(len(train_singlePrefixData1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer ,dropout=dropout, embedding=data.embedding)
    elif model_type == 'GRU_new':
        model = GRU_new(len(train_singlePrefixData1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'RNN':
        model = RNN(len(train_singlePrefixData1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'LSTM':
        model = LSTM(len(train_singlePrefixData1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'CNN':
        model = CNN(len(train_singlePrefixData1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'TF':
        model = Transformer(len(train_singlePrefixData1[0]), nhead=1,dim_feedforward=hidden_dim,
                     num_layers=n_layer, dropout=dropout)
    elif model_type == 'TCNN':
        model = TextCNN(len(train_singlePrefixData1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'TRCNN':
        model = TextRCNN(len(train_singlePrefixData1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'FT':
        model = FastText(len(train_singlePrefixData1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'DPCNN':
        model = DPCNN(len(train_singlePrefixData1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    # print("len(train_singlePrefixData1[0]):    $$$$$$$$$", len(train_singlePrefixData1[0]))

    if loss_type == 'L1Loss':
        criterion = nn.L1Loss()
        # criterion = nn.CrossEntropyLoss()
    elif loss_type == 'MSELoss':
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    # 开始训练


    tensor_x = torch.from_numpy(np.array(train_singlePrefixData1).astype(np.float32))
    tensor_y = torch.from_numpy(np.array(train_labelPrefixData).astype(np.float32))
    my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    my_dataset_loader = DataLoader(my_dataset, batch_size, drop_last=True, shuffle=False)

    tensor1_x = torch.from_numpy(np.array(test_singlePrefixData1).astype(np.float32))
    tensor1_y = torch.from_numpy(np.array(test_labelPrefixData).astype(np.float32))
    testPre_dataset = torch.utils.data.TensorDataset(tensor1_x, tensor1_y)
    testPre_dataset_loader = DataLoader(testPre_dataset, batch_size, drop_last=True, shuffle=False)

    #     训练
    # model = MLP()
    for enpoch in range(50):
        total_loss = torch.FloatTensor([0])
        for i, (input, target) in enumerate(my_dataset_loader):
            optimizer.zero_grad()
            # print("input:              ", np.array(input).shape)

            output = model(Variable(input))
            loss = criterion(output, target)
            # optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.data
            # print(" np.array(input).shape            ", np.array(input).shape)
            # print("input:              ", input)
        ACC = 0
        F1_EVERY_CLASS = 0
        F1_MACRO = 0
        F1_MICRO = 0
        PREC_EVERY_CLASS = 0
        PREC_MACRO = 0
        PREC_MICRO = 0
        RECALL_EVERY_CLASS = 0
        RECALL_MACRO = 0
        RECALL_MICRO = 0
        ROC_AUC_EVERY_CLASS = 0
        ROC_AUC_MACRO = 0
        ROC_AUC_MICRO = 0
        count = 0
        Time_sum = 0
        predList=[]
        realList=[]
        for j, (input1, target1) in enumerate(testPre_dataset_loader):
            # print("input:              ", np.array(input).shape)
            input1=input1.float()
            target1=target1.float()
            start_time=time.time()
            output = model(input1)
            end_time = time.time()
            times = end_time - start_time

            output1 = output.detach().numpy()
            predList += [pdic1.item() for pdic1 in output1]
            realList += [pdic2.item() for pdic2 in target1]
            for line in range(len(output1)):
                if output1[line][0] < 0.5:
                    output1[line][0] = 0
                if output1[line][0] >= 0.5:
                    output1[line][0] = 1
            acc = accuracy_score(target1.detach().numpy(), output1)
            f1_every_class = f1_score(target1.detach().numpy(), output1,
                                      average=None, zero_division=1)
            f1_macro = f1_score(target1.detach().numpy(), output1,
                                average='macro', zero_division=1)
            f1_micro = f1_score(target1.detach().numpy(), output1,
                                average='micro', zero_division=1)
            prec_micro = precision_score(target1.detach().numpy(), output1, average='micro', zero_division=1)
            prec_macro = precision_score(target1.detach().numpy(), output1, average='macro', zero_division=1)
            prec_every_class = precision_score(target1.detach().numpy(), output1, average=None, zero_division=1)

            recall_micro = recall_score(target1.detach().numpy(), output1, average='micro', zero_division=1)
            recall_macro = recall_score(target1.detach().numpy(), output1, average='macro', zero_division=1)
            recall_every_class = recall_score(target1.detach().numpy(), output1, average=None, zero_division=1)

            # roc_auc_micro=roc_auc_score(target.detach().numpy(), output1,average='micro')
            # roc_auc_macro = roc_auc_score(target.detach().numpy(), output1, average='macro')
            # roc_auc_every_class = roc_auc_score(target.detach().numpy(), output1, average=None)
            # print(acc)
            Time_sum = Time_sum + times
            ACC = ACC + acc
            F1_EVERY_CLASS = F1_EVERY_CLASS + f1_every_class
            F1_MACRO = F1_MACRO + f1_macro
            F1_MICRO = F1_MICRO + f1_micro
            PREC_EVERY_CLASS = PREC_EVERY_CLASS + prec_every_class
            PREC_MACRO = PREC_MACRO + prec_macro
            PREC_MICRO = PREC_MICRO + prec_micro
            RECALL_EVERY_CLASS = RECALL_EVERY_CLASS + recall_every_class
            RECALL_MACRO = RECALL_MACRO + recall_macro
            RECALL_MICRO = RECALL_MICRO + recall_micro
            # ROC_AUC_EVERY_CLASS = ROC_AUC_EVERY_CLASS+roc_auc_every_class
            # ROC_AUC_MACRO = ROC_AUC_MACRO+roc_auc_macro
            # ROC_AUC_MICRO = ROC_AUC_MICRO+roc_auc_micro
            count += 1
        MSE = computeMSE(realList, predList)
        MAE = computeMAE(realList, predList)
        RMSE = sqrt(MSE)
        if enpoch>=start_record_epoch:
            now = datetime.datetime.now()
            latest_checkpoint_file1=latest_checkpoint_file+str(enpoch)+'.pt'
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()

                },
                os.path.join(output_dir, latest_checkpoint_file1),
            )
            f = open(out_txt_path, 'a+', encoding='utf-8')
            f.write('Epoch: %d'% (enpoch)+'\n')
            f.write(str(now)+'\n')
            f.write('MSE: %f, MAE: %f, RMSE: %f' % (MSE, MAE, RMSE)+'\n')
            f.write('MSE: %f, MAE: %f, RMSE: %f' % (MSE, MAE, RMSE) + '\n')
            f.write('\n')
            f.close()
            with open(out_csv_path, 'a+', newline='') as file:
                f_csv = csv.writer(file)
                row = []
                # row.append('3inlayer')
                row.append(ACC / count)
                row.append(F1_EVERY_CLASS / count)
                row.append(F1_MACRO / count)
                row.append(F1_MICRO / count)
                row.append(PREC_EVERY_CLASS / count)
                row.append(PREC_MACRO / count)
                row.append(PREC_MICRO / count)
                row.append(RECALL_EVERY_CLASS / count)
                row.append(RECALL_MACRO / count)
                row.append(RECALL_MICRO / count)
                row.append(Time_sum / count)
                f_csv.writerow(row)
                row.clear()
        if enpoch % 5 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9
        scheduler.step()
        # if enpoch==99:
        #     torch.save(
        #         {
        #             "model": model.state_dict(),
        #             "optim": optimizer.state_dict(),
        #             "scheduler": scheduler.state_dict(),
        #             "epoch": epoch,
        #             "lr": scheduler.get_last_lr()
        #
        #         },
        #         os.path.join(output_dir, latest_checkpoint_file),
        #     )
        # print("@@@@@@@@@@@@@@@@@@@@@@:   ", total_loss)

def computeMAE(list_a, list_b):
    MAE_temp = []
    for num in range(len(list_a)):
        MAE_temp.append(abs(list_a[num] - list_b[num]))
    MAE = sum(MAE_temp) / len(list_a)
    return MAE


def computeMSE(list_a, list_b):
    MSE_temp = []
    for num in range(len(list_a)):
        MSE_temp.append((list_a[num] - list_b[num]) * (list_a[num] - list_b[num]))
    MSE = sum(MSE_temp) / len(list_a)
    return MSE


def computeTOTAL(list_a, list_b):
    TOTAL_temp = []
    for num in range(len(list_a)):
        TOTAL_temp.append(abs(list_a[num] - list_b[num]))
    TOTAL = sum(TOTAL_temp)
    return TOTAL


def computeMEAN(list_a, list_b):
    MEAN_temp = []
    for num in range(len(list_a)):
        MEAN_temp.append(abs(list_a[num] - list_b[num]))
    MEAN = sum(MEAN_temp) / len(list_a)
    return MEAN



if __name__ == '__main__':
    # #       BPIC2017A.csv   BPIC2017O1.csv    BPIC2017W.csv
    # BPIC_2019_A11.csv   BPIC_2019_O.csv   BPIC_2019_W1.csv
    train(data_address='E:/PycharmProjects/Article2/Data/output_second_zong/train_result_10_1%.csv',
          out_txt_path='./result_baseline/FT_test_metric_result10_1%zong.txt',
          out_csv_path = './result_baseline/FT_test_metric_result10_1%zong.csv',
    # 修改数据名
          data_name='result9_30%', embd_dimension=1, ################################################## 修改数据名
          train_splitThreshold=0.8, #这是为了把所有数据都训练模型，目的是保存隐状态。
          batch_size=9, n_layer=1, hidden_dim=1,
          loss_type='L1Loss', optim_type='Adam', model_type='FT', ######################修改model_type名，分别是RNN，LSTM和GRU
          train_type='iteration',output_dir='./model_baseline',latest_checkpoint_file='FT_test_result10_5%zong',start_record_epoch=0)
    # test(data_address='./Data/result10_50%.csv',  # 修改数据名
    #       data_name='result10', embd_dimension=1,  ################################################## 修改数据名
    #       train_splitThreshold=0.8,  # 这是为了把所有数据都训练模型，目的是保存隐状态。
    #       batch_size=9, n_layer=1, hidden_dim=1,
    #       loss_type='L1Loss', optim_type='Adam', model_type='GRU',  ######################修改model_type名，分别是RNN，LSTM和GRU
    #       train_type='iteration', output_dir='./model_ablation', latest_checkpoint_file='GRU_3inlayer_result10_50%.pt')