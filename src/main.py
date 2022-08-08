import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os
from glob import glob
from dataset import branch_dataset, multi_dataset
from model import get_branch_model, get_multi_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def branch_val(model, epoch, best_acc):

    model.eval()
    avg_acc = 0
    avg_loss = 0
    idx = 0
    for batch_id, data in enumerate(valdataloader):
        data, labels = data
        out = model(data.to(device))
        pred = out.argmax(dim=1)
        loss = criterion(out.to(device), labels.to(device))
        avg_loss += loss.cpu().detach().numpy()
        acc = torch.sum(labels.to(device).view(-1) == pred.to(device).view(-1)).item()/len(labels.to(device))
        avg_acc += acc
        idx = batch_id + 1
#     print('epoch: ',epoch,'loss: ', avg_loss/idx, 'acc: ', avg_acc/idx)
    with open(os.path.join('..','result',model_name,'repeat_'+str(repeat),'branch_val_'+str(n)+'.txt'), 'a') as f:
        f.write('epoch '+str(epoch)+' loss '+str(avg_loss/idx)+' acc ' +
                str(avg_acc/idx)+'\n')
    if avg_acc/idx > best_acc:
        best_acc = avg_acc/idx
        torch.save(model, os.path.join('..','checkpoint',model_name,'repeat_'+str(repeat),'branch_'+str(n)+'.pkl'))
    return best_acc

def multi_val(model, epoch, best_acc):
    model.setEval()
    avg_acc = 0
    avg_loss = 0
    idx = 0
    for batch_id, data in enumerate(valdataloader):
        data, labels = data
        out = model(data.to(device))
        pred = out.argmax(dim=1)
        loss = criterion(out.to(device), labels.to(device))
        avg_loss += loss.cpu().detach().numpy()
        acc = torch.sum(labels.to(device).view(-1) == pred.to(device).view(-1)).item()/len(labels.to(device))
        avg_acc += acc
        idx = batch_id + 1
#     print('epoch: ', epoch, 'loss: ', avg_loss /
#           idx, 'acc: ', avg_acc/idx)
    with open(os.path.join('..','result', model_name,'repeat_'+str(repeat), 'val.txt'), 'a') as f:
        f.write('epoch '+str(epoch)+' loss '+str(avg_loss/idx)+' acc ' +
                str(avg_acc/idx)+'\n')
    if avg_acc/idx > best_acc:
        best_acc = avg_acc/idx
        model.saveModel()
    return best_acc

if __name__=='__main__':
    
    model_list = ['squeezenet1_1','shufflenetv2','Alexnet','densenet121','efficientnet_b0','mobilenetv3_large_100','resnet50','resnext50_32x4d','xception']
    repeat_num = 10
    for model_name in model_list:
        for repeat in range(1,repeat_num+1):
            os.mkdir(os.path('..','result',model_name,'repeat_'+str(repeat)))
            os.mkdir(os.path('..','checkpoint',model_name,'repeat_'+str(repeat)))
            for n in range(4):
                if n != 4:
                    # 配置参数
                    branch_mode = True
                    multi_mode = False
                else:
                    branch_mode = False
                    multi_mode = True
                epoch = 30
                lr = 1e-4


                if branch_mode:
                    traindataset = branch_dataset('trainset', n)
                    traindataloader = DataLoader(traindataset, batch_size=6, shuffle=True)
                    valdataset = branch_dataset('valset', n)
                    valdataloader = DataLoader(dataset=valdataset, batch_size=6, shuffle=False)
                    testset = branch_dataset('testset', n)
                    testdataloader = DataLoader(dataset=testset, batch_size=6, shuffle=False)

                    model = get_branch_model(model_name, num_classes=37)
                    model = model.to(device)

                    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
                    criterion = nn.CrossEntropyLoss()

                    best_acc = 0

                    for epoch in range(epoch):
                        model.train()
                        for batch_id, data in enumerate(traindataloader):
                            data, labels = data
                            opt.zero_grad()
                            out = model(data.to(device))
                            pred = out.argmax(dim=1)
                            loss = criterion(out.to(device), labels.to(device))
                            acc = torch.sum(labels.to(device).view(-1) == pred.to(device).view(-1)).item()/len(labels.to(device))
                            loss.backward()
                            opt.step()
#                             print('epoch:', epoch, 'batch_id:', batch_id,
#                                 'loss: ', loss.cpu().detach().numpy(), 'acc: ', acc)
                            with open(os.path.join('..','result',model_name,'repeat_'+str(repeat),'branch_train_'+str(n)+'.txt'), 'a') as f:
                                f.write('epoch '+str(epoch)+' batch_id '+str(batch_id) +
                                        ' loss '+str(loss.cpu().detach().numpy())+' acc '+str(acc)+'\n')
                        best_acc = branch_val(model,epoch, best_acc)

                    model = torch.load(os.path.join('..','checkpoint',model_name,'repeat_'+str(repeat),'branch_'+str(n)+'.pkl'),map_location=device)

                    model.eval()
                    avg_acc = 0
                    avg_loss = 0
                    idx = 0
                    for batch_id, data in enumerate(testdataloader):
                        data, labels = data
                        out = model(data.to(device))
                        pred = out.argmax(dim=1)
                        loss = criterion(out.to(device), labels.to(device))
                        avg_loss += loss.cpu().detach().numpy()
                        acc = torch.sum(labels.to(device).view(-1) == pred.to(device).view(-1)).item()/len(labels.to(device))
                        avg_acc += acc
                        idx = batch_id
#                     print('loss: ', avg_loss/idx, 'acc: ', avg_acc/idx)
                    with open(os.path.join('..','result',model_name,'repeat_'+str(repeat),'branch_test_'+str(n)+'.txt'), 'a') as f:
                            f.write('loss '+str(avg_loss/idx)+' acc '+str(avg_acc/idx)+'\n')

                elif multi_mode:
                    traindataset = multi_dataset('trainset')
                    traindataloader = DataLoader(traindataset, batch_size=6, shuffle=True)
                    valdataset = multi_dataset('valset')
                    valdataloader = DataLoader(dataset=valdataset, batch_size=6, shuffle=False)
                    testdataset = multi_dataset('testset')
                    testdataloader = DataLoader(dataset=testdataset, batch_size=6, shuffle=False)

                    model = get_multi_model(init=True, branch_num=3, model_name=model_name, num_classes=37,repeat=repeat)
                    model = model.to(device)

                    opt_classifier = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

                    criterion = nn.CrossEntropyLoss()

                    best_acc = 0

                    for epoch in range(epoch):
                        model.setTrain()
                        for batch_id, data in enumerate(traindataloader):
                            data, labels = data
                            opt_classifier.zero_grad()
                            out = model(data.to(device))
                            pred = out.argmax(dim=1)
                            loss = criterion(out.to(device), labels.to(device))
                            acc = torch.sum(labels.to(device).view(-1) == pred.to(device).view(-1)).item()/len(labels.to(device))
                            loss.backward()
                            opt_classifier.step()
#                             print('epoch:', epoch, 'batch_id:', batch_id,
#                                 'loss: ', loss.cpu().detach().numpy(), 'acc: ', acc)
                            with open(os.path.join('..','result',model_name,'repeat_'+str(repeat),'train.txt'), 'a') as f:
                                f.write('epoch '+str(epoch)+' batch_id '+str(batch_id) +
                                        ' loss '+str(loss.cpu().detach().numpy())+' acc '+str(acc)+'\n')
                        best_acc = multi_val(model, epoch, best_acc)

                    model = get_multi_model(init=False, branch_num=3, model_name=model_name, num_classes=37,repeat=repeat)
                    model.loadModel()
                    model.to(device)
                    model.setEval()

                    avg_acc = 0
                    avg_loss = 0

                    for batch_id, data in enumerate(testdataloader):
                        data, labels = data
                        out = model(data.to(device))
                        pred = out.argmax(dim=1)
                        loss = criterion(out.to(device), labels.to(device))
                        avg_loss += loss.cpu().detach().numpy()
                        acc = torch.sum(labels.to(device).view(-1) == pred.to(device).view(-1)).item()/len(labels.to(device))
                        avg_acc += acc
                        idx = batch_id + 1
#                     print('loss: ', avg_loss/idx, 'acc: ', avg_acc/idx)
                    with open(os.path.join('..','result',model_name,'repeat_'+str(repeat),'test.txt'), 'a') as f:
                        f.write('loss '+str(avg_loss/idx)+' acc ' +
                                str(avg_acc/idx)+'\n')