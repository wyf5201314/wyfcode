import os
import scipy.io as sio
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import random
from torch.utils.data import Dataset
import torch
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import h5py
from einops import rearrange


# 读入数据
def loadData(dataset='IP', preprocess_scal='False'):
    if dataset == 'IP':
        data = sio.loadmat('data/IndianPines/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat('data/IndianPines/Indian_pines_gt.mat')['indian_pines_gt']
    elif dataset == 'UP':
        data = sio.loadmat('data/PaviaU.mat')['paviaU']
        labels = sio.loadmat('data/PaviaU_gt.mat')['paviaU_gt']
    elif dataset == 'SAL':
        data = sio.loadmat('data/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat('data/Salinas_gt.mat')['salinas_gt']
    elif dataset == 'KSC':
        data = sio.loadmat('data/KSC.mat')['KSC']
        labels = sio.loadmat('data/KSC_gt.mat')['KSC_gt']
    # elif dataset == 'Houston2013':
    #     data = sio.loadmat('../data/Houston.mat')['Houston']
    #     labels = sio.loadmat('../data/Houston_gt.mat')['Houston_gt']
    # elif dataset == 'Loukia':
    #     data = sio.loadmat('../data/Loukia.mat')['Loukia']
    #     labels = sio.loadmat('../data/Loukia_GT.mat')['Loukia_GT']
    # elif dataset == 'Dioni':
    #     data = sio.loadmat('../data/Dioni.mat')['Dioni']
    #     labels = sio.loadmat('../data/Dioni_GT.mat')['Dioni_GT']

    # 数据预处理：归一化
    if preprocess_scal == 'True':
        data_scale = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
        data_scale = preprocessing.scale(data_scale)
        data = data_scale.reshape(data.shape[0], data.shape[1], data.shape[2])

    return data, labels


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)

    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype='float16')
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype='float16')
    patchIndex = 0
    # 将图像划分为一个个patch
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1

    # 去除背景，只保留地物
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1  # 第一个类别索引为0

    return patchesData, patchesLabels


# 使用sklearn库划分训练样本和测试样本
def splitTrainTestSet(X, y, testRatio, randomState=202301):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test


# 按照比例划分数据集
def Split_Train_Test_Set(X, y, class_num, test_ratio):
    random.seed(345)
    gt_reshape = np.reshape(y, [-1])
    train_rand_idx = []

    for i in range(class_num):
        idx = np.where(gt_reshape == i)[-1]
        # print(f'idx shape: {idx.shape}')              #获取每类样本数
        samplesCount = len(idx)
        rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
        rand_idx = random.sample(rand_list, np.around(samplesCount * (1 - test_ratio)).astype('int32'))
        # print(f'rand_idx shape: {len(rand_idx)}')
        rand_real_idx_per_class = idx[rand_idx]
        train_rand_idx.append(rand_real_idx_per_class)

    train_rand_idx = np.array(train_rand_idx)
    train_data_index = []
    for c in range(train_rand_idx.shape[0]):
        a = train_rand_idx[c]
        for j in range(a.shape[0]):
            train_data_index.append(a[j])
    train_data_index = np.array(train_data_index)

    train_data_index = set(train_data_index)
    all_data_index = [i for i in range(len(gt_reshape))]
    all_data_index = set(all_data_index)
    test_data_index = all_data_index - train_data_index

    test_data_index = list(test_data_index)
    train_data_index = list(train_data_index)

    Xtrain = []
    ytrain = []
    for i in range(len(train_data_index)):
        Xtrain.append(X[train_data_index[i]])
        ytrain.append(y[train_data_index[i]])
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    # print(Xtrain.shape)
    # print(ytrain.shape)

    Xtest = []
    ytest = []
    for i in range(len(test_data_index)):
        Xtest.append(X[test_data_index[i]])
        ytest.append(y[test_data_index[i]])
    Xtest = np.array(Xtest)
    ytest = np.array(ytest)
    # print(Xtest.shape)
    # print(ytest.shape)

    return Xtrain, Xtest, ytrain, ytest


# 小样本实验划分:每个类别取相同有限样本
def Split_Train_Test_Num_Set(X, y, class_num, num_per_class=5):
    random.seed(345)
    gt_reshape = np.reshape(y, [-1])
    train_rand_idx = []

    for i in range(class_num):
        idx = np.where(gt_reshape == i)[-1]
        # print(f'idx shape: {idx.shape}')
        samplesCount = len(idx)
        rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
        rand_idx = random.sample(rand_list, num_per_class)
        rand_real_idx_per_class = idx[rand_idx]
        train_rand_idx.append(rand_real_idx_per_class)

    train_rand_idx = np.array(train_rand_idx)
    train_data_index = []
    for c in range(train_rand_idx.shape[0]):
        a = train_rand_idx[c]
        for j in range(a.shape[0]):
            train_data_index.append(a[j])
    train_data_index = np.array(train_data_index)

    train_data_index = set(train_data_index)
    all_data_index = [i for i in range(len(gt_reshape))]
    all_data_index = set(all_data_index)
    test_data_index = all_data_index - train_data_index

    test_data_index = list(test_data_index)
    train_data_index = list(train_data_index)

    Xtrain = []
    ytrain = []
    for i in range(len(train_data_index)):
        Xtrain.append(X[train_data_index[i]])
        ytrain.append(y[train_data_index[i]])
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    # print(Xtrain.shape)
    # print(ytrain.shape)

    Xtest = []
    ytest = []
    for i in range(len(test_data_index)):
        Xtest.append(X[test_data_index[i]])
        ytest.append(y[test_data_index[i]])
    Xtest = np.array(Xtest)
    ytest = np.array(ytest)
    # print(Xtest.shape)
    # print(ytest.shape)

    return Xtrain, Xtest, ytrain, ytest


class PreTrainDataSet(Dataset):

    def __init__(self, Xtrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len

# 数据集设置
class TrainDataSet(Dataset):

    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


class TestDataSet(Dataset):

    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


# 评价指标计算
def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test, dataset='IP'):
    if dataset == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif dataset == 'UP':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Metal Sheets',
                        'Bare soil', 'Bitumen', 'Bricks', 'Shadows']
    elif dataset == 'SAL':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop',
                        'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif dataset == 'KSC':
        target_names = ['Scrub', 'Willow_swamp', 'CP_hammock', 'CP/Oak', 'Slash_pine', 'Oak/Broadleaf',
                        'Hardwood_swamp', 'Graminoid_marsh', 'Spartina_marsh', 'Catial_marsh', 'Salt_marsh',
                        'Mud_flats', 'Water']
    # elif dataset == 'Houston2013':
    #     target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees'
    #                     , 'Soil', 'Water', 'Residential',
    #                     'Commercial', 'Road', 'Highway', 'Railway',
    #                     'Parking Lot 1', 'Parking Lot 2', 'Tennis CourtRunning', 'Running Track']
    # elif dataset == 'Loukia':
    #     target_names = ['Water', 'Hippo grass', 'Floodplain grasses 1', 'Floodplain grasses 2'
    #                     , 'Reeds', 'Riparian', 'Firescar',
    #                     'lsland interior', 'Acacia woodlands', 'Acacia shrublands', 'Acacia grasslands',
    #                     'Short mopanc', 'Mixed mopane', 'Chalcedony']
    # elif dataset == 'Dioni':
    #     target_names = ['Water', 'Hippo grass', 'Floodplain grasses 1', 'Floodplain grasses 2'
    #                     , 'Reeds', 'Riparian', 'Firescar',
    #                     'lsland interior', 'Acacia woodlands', 'Acacia shrublands', 'Acacia grasslands',
    #                     'Short mopanc']

    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100
