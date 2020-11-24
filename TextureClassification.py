# * coding: utf8 *
import numpy as np
import cv2
import os
import scipy.io as scio
import time
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
# 快速计算距离矩阵
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
# 用于加速运算
from numba import jit

np.set_printoptions(threshold=np.inf)
# 单个类别下的样本数目
SINGLE_CLASS_SAMPLES_NUM = 81
CLASS_NUM = 10
# self-adjust
FILTER_BANK_SIZE = 13
# Adjustable
LEARNING_SAMPLES_NUM = 13
# Adjustable
CLUSTER_SIZE = 10
# Adjustable
BINS_NUM = CLUSTER_SIZE * CLASS_NUM
# Adjustable 这里可用的滤波池类型有：['S', 'LMS', 'LML', 'RFS', 'MR8']
FILTER_BANK_TYPE = 'MR8'
classes_info = []
filter_bank = []
dictionary = []


# 存储类别信息（类别序号、类别名称、包含图片的路径）
class ClassInfo:
    def __init__(self, label_index, label_name, images_path):
        self.label_index = label_index
        self.label_name = label_name
        self.images_path = images_path

    def show_info(self):
        print(self.label_index)
        print(self.label_name)
        print(self.images_path)


# 读取数据集信息，存储在classInfo对象中
def load_dataset_info(path):
    _classes_info = []
    k = 0
    class_names = os.listdir(path)
    for class_name in class_names:
        if class_name == '_exclude':
            continue
        img_names = os.listdir(os.path.join(path, class_name))
        img_paths = [os.path.join(path, class_name, img_name) for img_name in img_names]
        _classes_info.append(ClassInfo(k, class_name, img_paths))
        k = k + 1
    return len(_classes_info), _classes_info


# 读取指定类别的指定图片
def get_image(label_index, img_index):
    # z-score标准化
    def z_score_norm(matrix):
        _mean = np.mean(matrix)
        _std = np.std(matrix)
        return (matrix - _mean) / _std

    img = cv2.imread(classes_info[label_index].images_path[img_index], cv2.IMREAD_GRAYSCALE)
    # 根据原始论文，图像需要进行标准化操作
    return z_score_norm(img)


# 加载用于texton词典生成的样本
@jit
def load_learning_images(label_index):
    images = []
    # 从每个类别中随机选取 LEARNING_SAMPLES_NUM 个样本
    imgs_index = np.random.choice(np.arange(SINGLE_CLASS_SAMPLES_NUM), size=LEARNING_SAMPLES_NUM, replace=False)
    # read image
    for img_index in imgs_index:
        images.append(get_image(label_index, img_index))
    return images


# 加载滤波池
@jit
def load_filter_bank():
    # 从mat文件中加载filter bank
    data = scio.loadmat('./filter_banks/filter_banks.mat')
    # MR8的本质还是RFS，只是选取8个最大滤波响应
    fb = data['RFS'] if FILTER_BANK_TYPE == 'MR8' else data[FILTER_BANK_TYPE]
    fb = fb.swapaxes(0, 2)
    fb = fb.swapaxes(1, 2)
    filter_bank_size = len(fb)
    return filter_bank_size, fb


# 获取滤波响应
@jit
def img2filter_responses(image):
    filter_bank_size = FILTER_BANK_SIZE
    responses = np.zeros((np.size(image), filter_bank_size), dtype=np.float32)
    for k in range(filter_bank_size):
        # cv2.filter2D()计算卷积的速度更快
        convolved = cv2.filter2D(image, -1, filter_bank[k])
        responses[:, k] = np.reshape(convolved, -1)
    # 对于MR8，只保留8个最大的响应
    if FILTER_BANK_TYPE == 'MR8':
        # 调整滤波池大小为8
        filter_bank_size = 8
        responses_mr8 = np.zeros((np.size(image), filter_bank_size), dtype=np.float32)
        responses_mr8[:, 0] = np.max(responses[:, 0:5], 1)
        responses_mr8[:, 1] = np.max(responses[:, 6:11], 1)
        responses_mr8[:, 2] = np.max(responses[:, 12:17], 1)
        responses_mr8[:, 3] = np.max(responses[:, 18:23], 1)
        responses_mr8[:, 4] = np.max(responses[:, 24:29], 1)
        responses_mr8[:, 5] = np.max(responses[:, 30:35], 1)
        responses_mr8[:, 6:7] = responses[:, 36:37]
        responses = responses_mr8
    # 根据论文，每个像素点的response需要标准化
    # F(x)=F(x)*[log (1 + L(x)/0.03)] /L(x) ，其中x代表每个像素点
    # L(x) = ||F(x)||2
    lx = (np.sqrt(np.sum(responses.T ** 2, 0)))
    for k in range(filter_bank_size):
        responses[:, k] *= (np.log(1 + lx / 0.03)) / lx
    return responses


# 词典学习(return 聚类中心组成的texton词典)
@jit
def build_texton_dictionary():
    # 存储每个类别的词典
    _dictionary = None
    # 加载并构建每类图片的texton词典
    for i in range(CLASS_NUM):
        # start = time.time()
        # 加载词典构建所需图片集
        learning_images = load_learning_images(i)
        # 当前类别图片的filter responses
        responses = None
        for image in learning_images:
            # 当前图像的滤波响应
            filter_responses = img2filter_responses(image)
            responses = filter_responses if responses is None else np.vstack((responses, filter_responses))
        logger('Finish building Responses of Class {0}-{1}, now clustering...'.format(i, classes_info[i].label_name))
        # n_init代表多次执行，取最优结果，但注意n_init不能设置太大，否则time cost很大
        k_means = KMeans(n_clusters=CLUSTER_SIZE, n_init=10)
        k_means.fit(responses)
        _dictionary = k_means.cluster_centers_ if _dictionary is None else np.vstack(
            (_dictionary, k_means.cluster_centers_))
        # end = time.time()
        # print('time cost ', end - start)
        logger('Finish building Dictionary of Class {0}-{1}.'.format(i, classes_info[i].label_name))
    return _dictionary


# 获取给定图像的histogram
@jit
def img2histogram(image):
    # start = time.time()
    # 获取滤波响应
    responses = img2filter_responses(image)
    # 计算每个pixel的response到每个texton的欧氏距离
    # SciPy的c-dist计算距离矩阵的速度更快
    distances = cdist(responses, dictionary)
    # dictionary中的texton在各个像素点上的映射
    texton_map = np.argmin(distances, 1)
    # 构造直方图，大小为BINS_NUM
    [histogram, _] = np.histogram(texton_map, BINS_NUM, range=(0, BINS_NUM - 1))
    # 根据论文，对于图片大小不统一的数据集，需要对histogram归一化
    histogram = histogram / np.sum(histogram)
    return histogram


# 划分构建70%训练集和30%测试集
@jit
def build_dataset():
    # 从每个类别中随机选取70%张图片作为训练集，其余作为测试集
    training_samples_num = int(SINGLE_CLASS_SAMPLES_NUM * 0.7)
    # 训练集、测试集数据
    training_data, testing_data = None, None
    # 训练集、测试集标签
    training_label, testing_label = [], []
    for i in range(CLASS_NUM):
        whole_img_index = np.arange(0, SINGLE_CLASS_SAMPLES_NUM)
        training_img_index = np.random.choice(np.arange(SINGLE_CLASS_SAMPLES_NUM), size=training_samples_num,
                                              replace=False)
        for img_index in whole_img_index:
            # 当前图像的直方图
            histogram = img2histogram(get_image(i, img_index))
            if img_index in training_img_index:
                training_data = histogram if training_data is None else np.vstack((training_data, histogram))
                training_label.append(i)
            else:
                testing_data = histogram if testing_data is None else np.vstack((testing_data, histogram))
                testing_label.append(i)
        logger('Finish building Dataset of Class {0}-{1}.'.format(i, classes_info[i].label_name))
    return training_data, training_label, testing_data, testing_label


# 加载本地数据集
def read_from_local_dataset(path):
    training_data = np.loadtxt(path + '/train_data')
    training_label = np.loadtxt(path + '/train_label')
    testing_data = np.loadtxt(path + '/test_data')
    testing_label = np.loadtxt(path + '/test_label')
    return training_data, training_label, testing_data, testing_label


# 将划分的数据集写入本地
def write_to_local_dataset(path, training_data, training_label, testing_data, testing_label):
    np.savetxt(path + '/train_data', training_data)
    np.savetxt(path + '/train_label', training_label)
    np.savetxt(path + '/test_data', testing_data)
    np.savetxt(path + '/test_label', testing_label)


# 日志函数
def logger(msg):
    print(msg)


# histogram距离函数（卡方距离）（论文中的chi^2）
def chi_square_dist(h1, h2):
    eps = 1e-10
    distance = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(h1, h2)])
    return distance


# KNN分类模型训练
@jit
def train_model_knn(training_data, training_label):
    # 设置K=1，且根据论文，距离衡量函数为卡方距离-chi_square
    _knn = KNeighborsClassifier(n_neighbors=1, metric=chi_square_dist)
    _knn.fit(np.array(training_data, dtype=np.float32), np.array(training_label, dtype=np.int))
    return _knn


# KNN模型测试
@jit
def test_model_knn(_model, testing_data, testing_label):
    hit_cnt = 0
    # 使用训练后的模型预测分类
    for i in range(len(testing_label)):
        feature = np.array([testing_data[i]], dtype=np.float32)
        predict_id = _model.predict(feature)
        if predict_id == int(testing_label[i]):
            hit_cnt = hit_cnt + 1
    rate = hit_cnt / len(testing_label)
    return rate


# svm参数配置
def svm_config():
    _svm = cv2.ml.SVM_create()
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    _svm.setTermCriteria(criteria)
    _svm.setKernel(cv2.ml.SVM_LINEAR)
    _svm.setType(cv2.ml.SVM_C_SVC)
    _svm.setC(10000)
    return _svm


# SVM分类模型训练
@jit
def train_model_svm(training_data, training_label):
    # 创建svm分类器
    svm = svm_config()
    # 开始训练
    print('开始训练...')
    svm.train(np.array(training_data, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(training_label, dtype=np.int))
    return svm


# SVM模型测试
@jit
def test_model_svm(_model, testing_data, testing_label):
    hit_cnt = 0
    # 使用训练后的模型预测分类
    for i in range(len(testing_label)):
        feature = np.array([testing_data[i]], dtype=np.float32)
        _, predict_id = _model.predict(feature)
        if predict_id == int(testing_label[i]):
            hit_cnt = hit_cnt + 1
    rate = hit_cnt / len(testing_label)
    return rate


if __name__ == '__main__':
    print('NOW RUNNING ON [{0}] FILTER BANK'.format(FILTER_BANK_TYPE))
    # 读取类别信息
    CLASS_NUM, classes_info = load_dataset_info('./KTH_TIPS_GRAY')
    # 加载滤波池
    logger('Loading filter bank...')
    FILTER_BANK_SIZE, filter_bank = load_filter_bank()
    # 词典路径
    dictionary_path = './dictionary/{0}_dictionary'.format(FILTER_BANK_TYPE)
    if os.path.exists(dictionary_path):
        logger('Dictionary already exists, loading...')
        dictionary = np.loadtxt(dictionary_path)
    else:
        logger('Creating texton dictionary...')
        dictionary = build_texton_dictionary()
        np.savetxt(dictionary_path, dictionary)
    # 训练集、测试集路径
    dataset_path = './dataset/' + FILTER_BANK_TYPE
    if os.path.exists(dataset_path + '/train_data'):
        logger('Dataset already exists, loading...')
        # 从本地读取数据集
        train_data, train_label, test_data, test_label = read_from_local_dataset(dataset_path)
    else:
        logger('Building dataset...')
        os.makedirs(dataset_path, exist_ok=True)
        # 划分数据集
        train_data, train_label, test_data, test_label = build_dataset()
        logger('Saving dataset...')
        # 写入本地
        write_to_local_dataset(dataset_path, train_data, train_label, test_data, test_label)
    logger('Model training...')
    model = train_model_knn(train_data, train_label)
    # model = train_model_svm(train_data, train_label)
    logger('Model testing...')
    accuracy = test_model_knn(model, test_data, test_label)
    # accuracy = test_model_svm(model, test_data, test_label)
    print('The accuracy is %.2f%%' % (accuracy * 100))
