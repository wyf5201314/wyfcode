import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from tils import get_cls_map
import time
from torch.utils.data import Dataset
from tils.datasets import *
import math
from utility import output_metric
from optimizer_step import Optimizer
import msssf

from net.mae import MAEVisionTransformers as MAE
from net.mae import VisionTransfromers as MAEFinetune
from loss.mae_loss import MSELoss, build_mask_chan

torch.autograd.set_detect_anomaly(True)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse

parser = argparse.ArgumentParser("HSI")

# ---- 预训练参数设置
parser.add_argument('--is_train', default=0, type=int)
parser.add_argument('--is_load_pretrain', default=1, type=int)
parser.add_argument('--is_pretrain', default=1, type=int)
parser.add_argument('--is_test', default=0, type=int)
parser.add_argument('--model_file', default='model', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# ---- network parameter
parser.add_argument('--size_SA', default=49, type=int, help='the size of spatial attention')
parser.add_argument('--channel_num', default=64, type=int)
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--pca_num', default=64, type=int)
parser.add_argument('--mask_ratio', default=0.5, type=float)
parser.add_argument('--crop_size', default=25, type=int)
parser.add_argument('--center_size', default=7, type=int)

# ----- data
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--dataset', default='IP', type=str,help='IP,PU or SA')
parser.add_argument('--num_classes', default=16, type=int)
parser.add_argument('--pretrain_num', default=50000, type=int)

# --- vit
parser.add_argument('--patch_size', default=1, type=int)
parser.add_argument('--finetune', default=0, type=int)
parser.add_argument('--mae_pretrain', default=1, type=int)
parser.add_argument('--depth', default=4, type=int)
parser.add_argument('--head', default=4, type=int)
parser.add_argument('--dim', default=320, type=int)

# ---- train
parser.add_argument('--model_name', type=str)
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--test_interval', default=5, type=int)
parser.add_argument('--optimizer_name', default="adamw", type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--cosine', default=0, type=int)
parser.add_argument('--weight_decay', default=5e-2, type=float)
parser.add_argument('--batch_size', default=32, type=int)

args = parser.parse_args()




#finetune 参数设置
'''Hyperparameter Settings'''  # patchsize centersize为奇数
epochs = 50
patch_size = 25
learn_rate = 0.001
BATCH_SIZE_TRAIN = 32
CENTER_SIZE = 7

'''Equipment parameter setting'''
workers = 1                 #加载数据线程数量
ngpu = 1                    #用来运行的GPU数量
run_num = 5                 #连续运行次数
cudnn.benchmark = True      #对卷积进行加速
torch.cuda.empty_cache()

'''Global variable declaration'''
global test_ratio, class_num, pca_components, same_samples_num, randomState

'''Pattern selection of training samples'''
train_samples_type = ['ratio', 'same_num'][0]
if train_samples_type == 'same_num':
     same_samples_num = 5

'''Selection of PCA dimensionality reduction'''
pca_type = ['True', 'False'][0]

'''Dataset selection''' #在此选择数据集，预训练阶段无视test_ratio
dataset = ['IP', 'UP', 'SAL'][0]
if dataset == 'IP':
    class_num = 16
    pca_components = 64
    # patch_size = 21
    test_ratio = 0.9
elif dataset == 'UP':
    class_num = 9
    pca_components = 64
    # patch_size = 3
    test_ratio = 0.95
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
    Xpretrain = X
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建train_loader和 test_loader
    X = TestDataSet(X, y_all)

    pretrainset = PreTrainDataSet(Xpretrain)
    trainset = TrainDataSet(Xtrain, ytrain)
    testset = TestDataSet(Xtest, ytest)

    pretrain_loader = torch.utils.data.DataLoader(dataset=pretrainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=workers,
                                               )
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

    return pretrain_loader, train_loader, test_loader, all_data_loader, y

# pretrain_loader, train_loader, test_loader, all_data_loader, y_all = create_data_loader()

def train(train_loader, epochs):

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("cuda:0" if torch.cuda.is_available() else "cpu")

    # 网络放到GPU上
    # net = model.proposed(num_class = class_num, patch_size = patch_size, pca_components = pca_components).to(device)
    net = msssf.MSSSF(
        dim=320,
        center_dim=320,
        depth= 5,
        heads=4,
        mlp_dim=4,
        num_classes=16,
        image_size=patch_size,
        center_size = CENTER_SIZE,
        dim_head = pca_components
    ).to(device)

    # 多个GPU加速训练
    # if torch.cuda.is_available() and ngpu > 1:
    #     model = nn.DataParallel(model, device_ids = list(range(ngpu)), broadcast_buffers=False)

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer, step_size = 50, gamma = 0.1)   #IP:40  UP :50  SAL:40

    # 开始训练
    total_loss = 0

    for epoch in range(epochs):

        model.train()

        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            outputs = model(data)
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

        # y_pred_test, y_test = test(device, net, test_loader)
        # classification, oa, confusion, each_acc, aa, kappa = acc_reports(
        #     y_test, y_pred_test)
        # print('OA: ', oa)
        # print('AA: ', aa)
        # print('K: ', kappa)
        # print('best oa: ', best_oa)
        # if oa > best_oa:
        #     best_oa = oa

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

def val(
        args,
        model
):
    """Validation and get the metric
    """
    batch_acc_list = []
    count = 0
    with torch.no_grad():  # 不计算梯度
        for batch_idx, (hsi, lidar, tr_labels, hsi_pca) in enumerate(test_loader):

            hsi = hsi.to(device)
            hsi = hsi[:, 0, :, :, :]
            lidar = lidar.to(device)
            hsi_pca = hsi_pca.to(device)
            tr_labels = tr_labels.to(device)

            outputs, _ = model(hsi, lidar, hsi_pca)

            batch_accuracy, _ = accuracy(outputs, tr_labels)

            batch_acc_list.append(batch_accuracy[0])

            if count == 0:
                y_pred_test = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                gty = tr_labels.detach().cpu().numpy()
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, np.argmax(outputs.detach().cpu().numpy(), axis=1)))  #
                gty = np.concatenate((gty, tr_labels.detach().cpu().numpy()))

    OA2, AA_mean2, Kappa2, AA2 = output_metric(gty, y_pred_test)
    classification = classification_report(gty, y_pred_test, digits=4)
    print(classification)
    print("OA2=", OA2)
    print("AA_mean2=", AA_mean2)
    print("Kappa2=", Kappa2)
    print("AA2=", AA2)
    epoch_acc = np.mean(batch_acc_list)

    print("Epoch_mean_accuracy:" % epoch_acc)

    return epoch_acc

def step_learning_rate(args, epoch, batch_iter, optimizer, train_batch):
    total_epochs = args.epoch
    warm_epochs = args.warmup_epochs
    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch)
        # lr_adj = 1.
    elif epoch < int(0.6 * total_epochs):
        lr_adj = 1.
    elif epoch < int(0.8 * total_epochs):
        lr_adj = 1e-1
    elif epoch < int(1 * total_epochs):
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj


def cosine_learning_rate(args, epoch, batch_iter, optimizer, train_batch):
    """Cosine Learning rate
    """
    total_epochs = args.max_epochs
    warm_epochs = args.warmup_epochs
    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch) + 1e-6
    else:
        lr_adj = 1 / 2 * (1 + math.cos(batch_iter * math.pi /
                                       ((total_epochs - warm_epochs) * train_batch)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj

def translate_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module' in key:
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        crr = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(1 / batch_size).item()
            res.append(acc)  # unit: percentage (%)
            crr.append(correct_k)
        return res, crr

def Pretrain(args,
             scaler,
             model,
             criterion,
             optimizer,
             epoch,
             batch_iter,
             batch_size
             ):
    """Traing with the batch iter for get the metric
    """

    total_loss = 0
    n = 0
    loader_length = len(pretrain_loader)
    print("pretrain_loader-------------", loader_length)
    for batch_idx, hsi in enumerate(pretrain_loader):
        n = n + 1
        # TODO: add the layer-wise lr decay
        if args.cosine:
            # cosine learning rate
            lr = cosine_learning_rate(
                args, epoch, batch_iter, optimizer, batch_size
            )
        else:
            # step learning rate
            lr = step_learning_rate(
                args, epoch, batch_iter, optimizer, batch_size
            )

        # forward
        hsi = hsi.to(device)
        # hsi = hsi[:, 0, :, :]

        outputs_chan, mask_index_chan = model(hsi)
        mask_chan = build_mask_chan(mask_index_chan, channel_num=args.channel_num, patch_size=args.patch_size)
        losses = criterion(outputs_chan,hsi,mask_chan.unsqueeze(-1))

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss = total_loss + losses.data.item()
        batch_iter += 1

        print(
            "Epoch:", epoch,
            " batch_idx:", batch_idx,
            " batch_iter:", batch_iter,
            " losses:", losses.data.item(),
            " lr:", lr
        )
    print(
        "Epoch:", epoch,
        " losses:", total_loss / n,
        " lr:", lr
    )
    return total_loss / n, batch_iter, scaler

if args.is_pretrain == 1:  # 根据预训练还是微调选择模型
    model = MAE(
        channel_number=args.channel_num,
        img_size=args.crop_size,
        center_size=args.center_size,
        patch_size=args.patch_size,
        encoder_dim=args.dim,
        encoder_depth=args.depth,
        encoder_heads=args.head,
        decoder_dim=args.dim,
        decoder_depth=args.depth,
        decoder_heads=args.head,
        mask_ratio=args.mask_ratio,
        args=args,

    )
else:
    model = MAEFinetune(
        channel_number=args.channel_num,
        img_size=args.crop_size,
        patch_size=args.patch_size,
        embed_dim=args.dim,
        depth=args.depth,
        num_heads=args.head,
        num_classes=args.num_classes,
        args=args
    )

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     print("using " + args.device + " as device")
# else:
#     print("using cpu as device")
model.to(device)



if __name__ == '__main__':

    total_loss = 0
    max_acc = 0

    model = model.to(device)
    model.cuda(device=device)
    optimizer = Optimizer(args.optimizer_name)(
        param=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        finetune=args.finetune
    )
    scaler = torch.cuda.amp.GradScaler()


    if args.is_pretrain == 1:                       # 预训练
        criterion = MSELoss(device=device)
        print("Pretraining!!--------")
        min_loss = 1e8
        batch_iter = 0

        pretrain_loader, train_loader, test_loader, all_data_loader, y_all = create_data_loader()

        for epoch in range(args.epoch):
            # 进入训练模式
            model.train()
            n = 0
            # 它执行预训练过程。该函数可能包括数据加载、前向传播、计算损失、反向传播和参数更新等步骤
            loss, batch_iter, scaler = Pretrain(args, scaler, model, criterion, optimizer, epoch, batch_iter,
                                                args.batch_size)
            # 检查并记录最佳损失，通过这种方式，即使训练过程中被中断，也可以从最近的检查点恢复，而无需从头开始训练
            if loss < min_loss:
                # 获取模型当前的参数状态
                state_dict = translate_state_dict(model.state_dict())
                # 创建新字典
                state_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(
                    state_dict,
                    'model/' + 'pretrain_' + dataset + '_num' + str(args.pretrain_num) + '_crop_size' + str(
                        args.crop_size) + '_mask_ratio_' + str(args.mask_ratio) \
                    + '_DDH_' + str(args.depth) + str(args.dim) + str(args.head) + '_epoch_' + str(
                        epoch) + '_loss_' + str(loss) + '.pth'
                )
                min_loss = loss

    if args.is_train == 1:
        OA = []
        AA = []
        KAPPA = []
        Each_Accuracy = []
        pretrain_loader, train_loader, test_loader, all_data_loader, y_all = create_data_loader()

        optimizer = Optimizer(args.optimizer_name)(  # 优化器
            # 获取可变换参数形式
            param=model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            finetune=args.finetune
        )

        if args.is_load_pretrain == 1:    #读取预训练模型参数
            state_dict = torch.load('model/pretrain_IP_num50000_crop_size25_mask_ratio_0.7_DDH_42564_epoch_157_loss_0.6298663909561537.pth')  # 不带模型结构的模型参数
            model.load_state_dict(state_dict)
            print("model load successfully!")

        for run in range(run_num):   #运行run_num次取平均值
            print(f'run num : {run}')

            tic1 = time.perf_counter()
            model, device = train(train_loader, epochs=epochs)

            toc1 = time.perf_counter()
            tic2 = time.perf_counter()
            y_pred_test, y_test = My_test(device, model, test_loader)
            toc2 = time.perf_counter()


            # 评价指标
            classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test, dataset=dataset)
            OA.append(oa)
            AA.append(aa)
            KAPPA.append(kappa)
            Each_Accuracy.append(each_acc)
            classification = str(classification)
            Training_Time = toc1 - tic1
            Test_time = toc2 - tic2
            file_name = "results/cls_result/" + dataset + '_' + str(run) + '_' + str(test_ratio) + "_" + str(
                patch_size) + "-" + str(CENTER_SIZE) + ".txt"

            with open(file_name, 'w') as x_file:
                x_file.write('{} Training_Time (s)'.format(Training_Time))
                x_file.write('\n')
                x_file.write('{} Test_time (s)'.format(Test_time))
                x_file.write('\n')
                x_file.write('{} Overall accuracy (%)'.format(np.around(oa, decimals=2)))
                x_file.write('\n')
                x_file.write('{} Average accuracy (%)'.format(np.around(aa, decimals=2)))
                x_file.write('\n')
                x_file.write('{} Kappa accuracy (%)'.format(np.around(kappa, decimals=2)))
                x_file.write('\n')
                x_file.write('{} Each accuracy (%)'.format(np.around(each_acc, decimals=2)))
                x_file.write('\n')
                x_file.write('{}'.format(classification))
                x_file.write('\n')
                x_file.write('{}'.format(confusion))
            print('write successful')
            get_cls_map.get_cls_map(model, device, all_data_loader, y_all, dataset=dataset, encoder_num=run)

        oa_file_name = "results/cls_result/" + dataset + '_' + str(test_ratio) + '_' + '平均' + "_" + str(
            patch_size) + "-" + str(CENTER_SIZE) + ".txt"
        with open(oa_file_name, 'w') as x_file:
            x_file.write('{}+{} Overall accuracy (%)'.format(np.around(np.mean(OA), decimals=2), np.std(OA)))
            x_file.write('\n')
            x_file.write('{}+{} Average accuracy (%)'.format(np.around(np.mean(AA), decimals=2), np.std(AA)))
            x_file.write('\n')
            x_file.write('{}+{} Kappa accuracy (%)'.format(np.around(np.mean(KAPPA), decimals=2), np.std(KAPPA)))
            x_file.write('\n')
            x_file.write('{} Each accuracy (%)'.format(np.around(np.mean(Each_Accuracy, axis=0), decimals=2)))

            print(OA, AA, KAPPA)

        # pretrain_loader, train_loader, test_loader, all_data_loader, y_all = create_data_loader()
        # criterion = nn.CrossEntropyLoss()
        # batch_iter = 0
        # for epoch in range(args.epoch):
        #     model.train()
        #     n = 0
        #     loss, batch_iter, scaler = Train(args, scaler, model, criterion, optimizer, epoch, batch_iter,
        #                                      args.batch_size)
        #     if epoch % args.test_interval == 0:
        #         # For some datasets (such as Berlin), the test set is too large and the test speed is slow,
        #         # so it is recommended to split the validation set with a small sample size
        #         model.eval()
        #         acc1 = val(args, model)
        #         print("epoch:", epoch, "acc:", acc1)
        #         if acc1 > max_acc:
        #             state_dict = translate_state_dict(model.state_dict())
        #             state_dict = {
        #                 'epoch': epoch,
        #                 'state_dict': state_dict,
        #                 'optimizer': optimizer.state_dict(),
        #             }
        #             torch.save(state_dict, 'model/' + 'train_' + args.dataset + '_num' + str(
        #                 args.pretrain_num) + '_crop_size' + str(args.crop_size) + '_mask_ratio_' + str(args.mask_ratio) \
        #                        + '_DDH_' + str(args.depth) + str(args.dim) + str(args.head) + '_epoch_' + str(
        #                 epoch) + '_acc_' + str(acc1) + '.pth')
        #
        #             max_acc = acc1




    if args.is_test == 1:
        model_path = 'model/' + 'train_' + args.dataset + '_num' + str(
                        args.pretrain_num) + '_crop_size' + str(args.crop_size) + '_mask_ratio_' + str(args.mask_ratio) \
                               + '_DDH_' + str(args.depth) + str(args.dim) + str(args.head) + '_epoch_' + str(
                        epoch) + '.pth'
        checkpoint = torch.load(model_path, map_location="gpu")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        acc1 = val(args, model)



















