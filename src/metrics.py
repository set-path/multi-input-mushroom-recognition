from numpy.lib.function_base import average, percentile
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, auc, accuracy_score, precision_score, f1_score, recall_score, jaccard_score
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
from torch.nn import functional
import os
import numpy as np
import cv2
import shutil


def dataloader(path, batch_size):
    def generation():
        data = []
        labels = []
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.name.endswith('.jpg'):
                    img = cv2.imread(entry.path)
                    img = cv2.resize(img, (224, 224))
                    img = np.transpose(img, (2, 0, 1))
                    data.append(img)
                    for key in label_dict.keys():
                        if entry.name.startswith(key):
                            labels.append(label_dict[key])
                if len(data) == batch_size:
                    yield dygraph.to_variable(data), dygraph.to_variable(labels)
                    data = []
                    labels = []
            if len(data) < batch_size:
                yield dygraph.to_variable(data), dygraph.to_variable(labels)
    return generation


def save_confusion_matrix(y_real, y_pred, labels, path):
    cm = confusion_matrix(y_real, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(xticks_rotation='vertical')
    # plt.show()
    plt.savefig(path+'confusion_matrix/confusion_matrix.png',
                dpi=500, bbox_inches='tight')


def save_roc_curve(y_real, y_score, labels, path):
    y_real_one_hot = functional.one_hot(
        torch.LongTensor(y_real), num_classes=len(labels)).numpy()
    y_score = np.array(y_score)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(
            y_real_one_hot[:, i], y_score[:, i])
        roc_auc[i] = auc(tpr[i], fpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_real_one_hot.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(tpr["micro"], fpr["micro"])

    for i, label in enumerate(labels):
        disp = RocCurveDisplay(
            fpr=tpr[i], tpr=fpr[i], roc_auc=roc_auc[i], estimator_name=label)
        disp.plot()
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.savefig(path+'roc/'+label+'_roc.png', dpi=500, bbox_inches='tight')
    disp = RocCurveDisplay(
        fpr=tpr['micro'], tpr=fpr['micro'], roc_auc=roc_auc['micro'], estimator_name='micro_average')
    disp.plot()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.savefig(path+'roc/micro_average_roc.png', dpi=500, bbox_inches='tight')

    _, ax = plt.subplots()
    for i, label in enumerate(labels):
        disp = RocCurveDisplay(
            fpr=tpr[i], tpr=fpr[i], roc_auc=roc_auc[i], estimator_name=label)
        disp.plot(ax=ax)
    disp = RocCurveDisplay(
        fpr=tpr['micro'], tpr=fpr['micro'], roc_auc=roc_auc['micro'], estimator_name='micro_average')
    disp.plot(ax=ax)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc=(1.05, 0))
    # plt.show()
    plt.savefig(path+'roc/total_roc.png', dpi=500, bbox_inches='tight')


def save_accuracy_score(y_real, y_pred, labels, path):
    scores = jaccard_score(y_true=y_real, y_pred=y_pred, average=None)
    for i, label in enumerate(labels):
        with open(path+'metrics/'+label+'_metrics.txt', mode='a') as f:
            f.write('accuracy_score: '+str(scores[i])+'\n')
    score = accuracy_score(y_true=y_real, y_pred=y_pred)
    with open(path+'metrics/total_metrics.txt', mode='a') as f:
        f.write('accuracy_score: '+str(score)+'\n')


def save_precision_score(y_real, y_pred, labels, path):
    for i, label in enumerate(labels):
        scores = precision_score(y_true=y_real, y_pred=y_pred, average=None)
        with open(path+'metrics/'+label+'_metrics.txt', mode='a') as f:
            f.write('precision_score: '+str(scores[i])+'\n')
    score = precision_score(y_true=y_real, y_pred=y_pred, average='micro')
    with open(path+'metrics/total_metrics.txt', mode='a') as f:
        f.write('precision_score: '+str(score)+'\n')


def save_f1_score(y_real, y_pred, labels, path):
    for i, label in enumerate(labels):
        scores = f1_score(y_true=y_real, y_pred=y_pred, average=None)
        with open(path+'metrics/'+label+'_metrics.txt', mode='a') as f:
            f.write('f1_score: '+str(scores[i])+'\n')
    score = f1_score(y_true=y_real, y_pred=y_pred, average='micro')
    with open(path+'metrics/total_metrics.txt', mode='a') as f:
        f.write('f1_score: '+str(score)+'\n')


def save_recall_score(y_real, y_pred, labels, path):
    for i, label in enumerate(labels):
        scores = recall_score(y_true=y_real, y_pred=y_pred, average=None)
        with open(path+'metrics/'+label+'_metrics.txt', mode='a') as f:
            f.write('recall_score: '+str(scores[i])+'\n')
    score = recall_score(y_true=y_real, y_pred=y_pred, average='micro')
    with open(path+'metrics/total_metrics.txt', mode='a') as f:
        f.write('recall_score: '+str(score)+'\n')

# 获取 Grad-CAM 类激活热图


# def get_gradcam(grad_maps, data, label, class_dim=18):
    # conv = model.model_1(data)  # 得到模型最后一个卷积层的特征图
    # predict:(32,16)
    # predict = model.model_2(conv)  # 得到前向计算的结果
    # label:(32)
    # label = np.reshape(label, [-1])
    # predict_one_hot = torch.nn.functional.one_hot(
        # label, class_dim) * predict  # 将模型输出转化为one-hot向量
    # score = torch.mean(predict_one_hot)  # 得到预测结果中概率最高的那个分类的值
    # score.backward()  # 反向传播计算梯度

    # grad_map = conv.grad  # 得到目标类别的loss对最后一个卷积层输出的特征图的梯度
    # grad:(32,1024,1,1)
    # grad = torch.mean(torch.Tensor(grad_map), (2, 3),
        #    keepdim=True)  # 对特征图的梯度进行GAP（全局平局池化）
    # gradcam:(32,7,7)
    # 将最后一个卷积层输出的特征图乘上从梯度求得权重进行各个通道的加和
    # gradcam = torch.sum(grad * conv, axis=1)
    # gradcam = torch.maximum(
    #     gradcam, torch.Tensor(0.))  # 进行ReLU操作，小于0的值设为0
    # for j in range(gradcam.shape[0]):
    #     gradcam[j] = gradcam[j] / torch.max(gradcam[j])  # 分别归一化至[0, 1]
    # return gradcam

# 将 Grad-CAM 叠加在原图片上显示激活热图的效果


def save_gradcam(filename, grad_maps, conv, data, path):
    heat_maps = []
    # gradcams = get_gradcam(model, data, label)
    # (12,1280,1,1)
    grad_maps = np.array(grad_maps)
    conv = np.array(conv)
    data = np.array(data)
    grads = torch.mean(torch.Tensor(grad_maps), (2, 3),
                       keepdim=True)  # 对特征图的梯度进行GAP（全局平局池化
    gradcams = torch.sum(grads * conv, axis=1)
    gradcams = torch.maximum(
        gradcams, torch.tensor(0))  # 进行ReLU操作，小于0的值设为0
    for j in range(gradcams.shape[0]):
        gradcams[j] = gradcams[j] / torch.max(gradcams[j])  # 分别归一化至[0, 1]
    # (12,1280,1,1)
    for i in range(data.shape[0]):
        img = data[i].astype('uint8').transpose([1, 2, 0])
        # heatmap:(224,224)
        heatmap = cv2.resize(gradcams[i].numpy(
        ) * 255., (data.shape[2], data.shape[3])).astype('uint8')  # 调整热图尺寸与图片一致、归一化
        # heatmap:(224,224,3)
        heatmap = cv2.applyColorMap(
            heatmap, cv2.COLORMAP_JET)  # 将热图转化为“伪彩热图”显示模式
        superimposed_img = cv2.addWeighted(
            heatmap, .3, img, .7, 1.)  # 将特图叠加到原图片上
        heat_maps.append(superimposed_img)
    # heat_maps = np.array(heat_maps)

    # heat_maps = heat_maps.reshape([-1, 8, pic_size, pic_size, 3])
    # heat_maps = np.concatenate(tuple(heat_maps), axis=1)
    # heat_maps = np.concatenate(tuple(heat_maps), axis=1)
    for i in range(len(heat_maps)):
        cv2.imwrite(path+'gradcam/'+filename[i]+'.jpg', heat_maps[i])


def save(filename, y_real, y_pred, y_score, labels, gradmaps, conv, data, path):
    if not os.path.exists(path+'metrics'):
        os.mkdir(path+'metrics')
    else:
        shutil.rmtree(path+'metrics')
        os.mkdir(path+'metrics')
    if not os.path.exists(path+'roc'):
        os.mkdir(path+'roc')
    else:
        shutil.rmtree(path+'roc')
        os.mkdir(path+'roc')
    if not os.path.exists(path+'confusion_matrix'):
        os.mkdir(path+'confusion_matrix')
    else:
        shutil.rmtree(path+'confusion_matrix')
        os.mkdir(path+'confusion_matrix')
    # if not os.path.exists(path+'gradcam'):
    #     os.mkdir(path+'gradcam')
    # else:
    #     shutil.rmtree(path+'gradcam')
    #     os.mkdir(path+'gradcam')
    save_confusion_matrix(y_real=y_real, y_pred=y_pred,
                          labels=labels, path=path)
    save_roc_curve(y_real=y_real, y_score=y_score, labels=labels, path=path)
    save_accuracy_score(y_real=y_real, y_pred=y_pred, labels=labels, path=path)
    save_precision_score(y_real=y_real, y_pred=y_pred,
                         labels=labels, path=path)
    save_f1_score(y_real=y_real, y_pred=y_pred, labels=labels, path=path)
    save_recall_score(y_real=y_real, y_pred=y_pred, labels=labels, path=path)
    # save_gradcam(filename, grad_maps=gradmaps, conv=conv, data=data, path=path)
