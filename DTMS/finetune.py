import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import msssf
from tils import get_cls_map
import time
from torch.utils.data import Dataset
from tils.datasets import *

torch.autograd.set_detect_anomaly(True)

'''Hyperparameter Settings'''
epochs = 100
patch_size = 25
learn_rate = 0.001
BATCH_SIZE_TRAIN = 64
CENTER_SIZE = 7
is_load_pretrain = 1  #1 or 0

'''Equipment parameter setting'''
workers = 0                 #加载数据线程数量
ngpu = 1                    #用来运行的GPU数量
run_num = 1                 #连续运行次数
cudnn.benchmark = True      #对卷积进行加速
torch.cuda.empty_cache()

'''Global variable declaration'''
global test_ratio, class_num, pca_components, same_sample_num, randomState

'''Pattern selection of training samples'''
train_samples_type = ['ratio', 'same_num'][0]
if train_samples_type == 'same_num':
     same_samples_num = 5

'''Selection of PCA dimensionality reduction'''
pca_type = ['True', 'False'][0]

'''Dataset selection'''
dataset = ['IP', 'UP', 'SAL'][0]
if dataset == 'IP':
    class_num = 16
    pca_components = 64
    # patch_size = 21
    test_ratio = 0.90
elif dataset == 'UP':
    class_num = 9
    pca_components = 64
    # patch_size = 25
    test_ratio = 0.97
elif dataset == 'SAL':
    class_num = 16
    pca_components = 128
    # patch_size = 3
    test_ratio = 0.99
elif dataset == 'KSC':
    class_num = 13
    pca_components = 128
    # patch_size = 3
    test_ratio = 0.95

def create_data_loader():
    # 读入数据
    X, y = loadData(dataset = dataset, preprocess_scal = 'True')

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    if pca_type == 'True':
        X_pca = applyPCA(X, numComponents = pca_components)
    else:
        X_pca = X
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y_all shape: ', y_all.shape)

    print('\n... ... create train & test data ... ...')
    if train_samples_type == 'ratio':
        Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)
        print(f'train samples type : {train_samples_type}, test ratio : {test_ratio}')
    else:
        Xtrain, Xtest, ytrain, ytest = Split_Train_Test_Num_Set(X_pca, y_all, class_num = class_num, num_per_class = same_samples_num)
        print(f'train samples type : {train_samples_type}, same samples num : {same_samples_num}')
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)
    print('ytrain shape: ', ytrain.shape)
    print('ytest  shape: ', ytest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # 为了适应 pytorch 结构，数据要做 transpose1
    X = X.transpose(0, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 3, 1, 2)
    Xtest = Xtest.transpose(0, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建train_loader和 test_loader
    X = TestDataSet(X, y_all)
    trainset = TrainDataSet(Xtrain, ytrain)
    testset = TestDataSet(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=workers,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=workers,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=workers,
                                              )

    return train_loader, test_loader, all_data_loader, y

def train(train_loader, epochs):

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("cuda:0" if torch.cuda.is_available() else "cpu")

    # 网络放到GPU上
    # net = model.proposed(num_class = class_num, patch_size = patch_size, pca_components = pca_components).to(device)
    net = msssf.MSSSF(
        dim=512,
        center_dim=512,
        depth=5,
        heads=4,
        mlp_dim=4,
        num_classes=16,
        image_size=patch_size,
        center_size = CENTER_SIZE,
        dim_head = 64
    ).to(device)

    if is_load_pretrain == 1:    #读取预训练模型参数
        state_dict = torch.load('model/pretrain_IP_num50000_crop_size25_mask_ratio_0.7_DDH_42564_epoch_157_loss_0.6298663909561537.pth')  # 不带模型结构的模型参数
        net.load_state_dict(state_dict,strict=False)
        print("model load successfully!")

    #多个GPU加速训练
    # if torch.cuda.is_available() and ngpu > 1:
    #     net = nn.DataParallel(net, device_ids = list(range(ngpu)),broadcast_buffers=False)

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=learn_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer, step_size = 50, gamma = 0.1)   #IP:40  UP :50  SAL:40

    # 开始训练
    total_loss = 0

    for epoch in range(epochs):

        net.train()

        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            outputs = net(data)
            # 计算损失函数
            if len(outputs.shape) == 1:
                outputs = torch.unsqueeze(outputs, dim = 0)
            loss = criterion(outputs, target)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.5f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))

    print('Finished Training')

    return net, device

def My_test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            if len(outputs.shape) !=2:
                continue
            else:
                outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
        print('Finished Testing')

    return y_pred_test, y_test

if __name__ == '__main__':

    OA = []
    AA = []
    KAPPA = []
    Each_Accuracy = []
    train_loader, test_loader, all_data_loader, y_all = create_data_loader()



    for run in range(run_num):
        print(f'run num : {run}')

        tic1 = time.perf_counter()
        net, device = train(train_loader, epochs=epochs)

        toc1 = time.perf_counter()
        tic2 = time.perf_counter()
        y_pred_test, y_test = My_test(device, net, test_loader)
        toc2 = time.perf_counter()

        #评价指标
        classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test, dataset = dataset)
        OA.append(oa)
        AA.append(aa)
        KAPPA.append(kappa)
        Each_Accuracy.append(each_acc)
        classification = str(classification)
        Training_Time = toc1 - tic1
        Test_time = toc2 - tic2
        file_name = "results/cls_result/"+dataset+'_'+str(run)+'_'+str(test_ratio)+ '_' + str(patch_size) + '_' + str(CENTER_SIZE) + "_.txt"


        with open(file_name, 'w') as x_file:
            x_file.write('{} Training_Time (s)'.format(Training_Time))
            x_file.write('\n')
            x_file.write('{} Test_time (s)'.format(Test_time))
            x_file.write('\n')
            x_file.write('{} Overall accuracy (%)'.format(np.around(oa, decimals = 2)))
            x_file.write('\n')
            x_file.write('{} Average accuracy (%)'.format(np.around(aa, decimals = 2)))
            x_file.write('\n')
            x_file.write('{} Kappa accuracy (%)'.format(np.around(kappa, decimals = 2)))
            x_file.write('\n')
            x_file.write('{} Each accuracy (%)'.format(np.around(each_acc, decimals = 2)))
            x_file.write('\n')
            x_file.write('{}'.format(classification))
            x_file.write('\n')
            x_file.write('{}'.format(confusion))
        print('write successful')
        get_cls_map.get_cls_map(net, device, all_data_loader, y_all, dataset = dataset, encoder_num = run)

    oa_file_name = "results/cls_result/" + dataset + '_' + str(test_ratio)+ '_' + '平均'+ '_' + str(patch_size) + '_' + str(CENTER_SIZE) + "_.txt"
    with open(oa_file_name, 'w') as x_file:
        x_file.write('{}+{} Overall accuracy (%)'.format(np.around(np.mean(OA), decimals = 2), np.std(OA)))
        x_file.write('\n')
        x_file.write('{}+{} Average accuracy (%)'.format(np.around(np.mean(AA), decimals = 2), np.std(AA)))
        x_file.write('\n')
        x_file.write('{}+{} Kappa accuracy (%)'.format(np.around(np.mean(KAPPA), decimals = 2), np.std(KAPPA)))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(np.around(np.mean(Each_Accuracy, axis = 0), decimals = 2)))

        print(OA, AA,KAPPA)

