import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import timm
import cv2
import numpy as np
import os



# traindataset = DataSet('./data/multiangleMushrooms/trainset/',height=224,width=224)
# traindataloader = DataLoader(traindataset, batch_size=12, shuffle=True)
# valdataset = DataSet(path='./data/multiangleMushrooms/valset/',height=224, width=224)
# valdataloader = DataLoader(dataset=valdataset, batch_size=12, shuffle=False)
testdataset = DataSet(path='./data/multiangleMushrooms/testset/', height=224, width=224)
testdataloader = DataLoader(dataset=testdataset, batch_size=12, shuffle=False)

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


# def val(model, epoch):
#     criterion = nn.CrossEntropyLoss()
#     criterion.to(device)

#     model.eval()
#     avg_acc = 0
#     avg_loss = 0
#     y_real = []
#     y_pred = []
#     y_score = []
#     idx = 0
#     for batch_id, data in enumerate(valdataloader):
#         data, labels = data
#         y_real.extend(labels.numpy())
#         out = model(data.to(device))
#         y_score.extend(out.cpu().detach().numpy())
#         pred = out.argmax(dim=1)
#         y_pred.extend(pred.cpu().numpy())
#         loss = criterion(out.to(device), labels.to(device))
#         avg_loss += loss.cpu().detach().numpy()
#         acc = torch.sum(labels.to(device).view(-1) == pred.to(device).view(-1)).item()/len(labels.to(device))
#         avg_acc += acc
#         idx = batch_id
#         # print('batch_id:', batch_id,
#         # 'loss: ', loss.detach().numpy(), 'acc: ', acc)
#         # with open('./save_models/Mobilenetv3_large_100/valInfo/ .txt', 'a') as f:
#             # f.write('epoch '+str(epoch)+' batch_id '+str(batch_id) +
#                     # ' loss '+str(loss.detach().numpy())+' acc '+str(acc)+'\n')
#     # print('----------------------------------------------')
#     correct = 0
#     error = 0
#     for i in range(len(y_real)):
#         if y_real[i] == y_pred[i]:
#             correct += 1
#         else:
#             error += 1
#     print('epoch: ',epoch,'avg_loss: ', avg_loss/idx, 'acc: ', correct/(correct+error))
#     with open('./save_models/combineMushroomNet/mobilenetv3_large_100/branch/branch_2_val.txt', 'a') as f:
#         f.write('epoch '+str(epoch)+' avg_loss '+str(avg_loss/idx)+' acc ' +
#                 str(correct/(correct+error))+'\n')



# model = timm.create_model(
#     'mobilenetv3_large_100', pretrained=True, num_classes=31)
# model.to(device)

# # print(model)

# for param in model.parameters():
#     param.requires_grad = False


# for param in model.blocks[6].parameters():
#     param.requires_grad = True


# opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
# criterion = nn.CrossEntropyLoss()

# criterion.to(device)


# for epoch in range(30):
#     model.train()
#     for batch_id, data in enumerate(traindataloader):
#         data, labels = data
#         opt.zero_grad()
#         out = model(data.to(device))
#         pred = out.argmax(dim=1)
#         loss = criterion(out.to(device), labels.to(device))
#         acc = torch.sum(labels.to(device).view(-1) == pred.to(device).view(-1)).item()/len(labels.to(device))
#         loss.backward()
#         opt.step()
#         print('epoch:', epoch, 'batch_id:', batch_id,
#               'loss: ', loss.cpu().detach().numpy(), 'acc: ', acc)
#         with open('./save_models/combineMushroomNet/mobilenetv3_large_100/branch/branch_2_train.txt', 'a') as f:
#             f.write('epoch '+str(epoch)+' batch_id '+str(batch_id) +
#                     ' loss '+str(loss.cpu().detach().numpy())+' acc '+str(acc)+'\n')
#     val(model,epoch)

# torch.save(model, './save_models/combineMushroomNet/mobilenetv3_large_100/branch/branch_2.pkl')

# Test

model = torch.load(
'./save_models/combineMushroomNet/mobilenetv3_large_100/branch/branch_0.pkl')
model.to(device)

for param in model.parameters():
    if param.requires_grad == False:
        param.requires_grad = True


criterion = nn.CrossEntropyLoss()
criterion.to(device)

model.eval()
avg_acc = 0
avg_loss = 0
y_real = []
y_pred = []
y_score = []
idx = 0
for batch_id, data in enumerate(testdataloader):
    data, labels = data
    y_real.extend(labels.numpy())
    out = model(data.to(device))
    y_score.extend(out.cpu().detach().numpy())
    pred = out.argmax(dim=1)
    y_pred.extend(pred.cpu().numpy())
    loss = criterion(out.to(device), labels.to(device))
    avg_loss += loss.cpu().detach().numpy()
    acc = torch.sum(labels.to(device).view(-1) == pred.to(device).view(-1)).item()/len(labels.to(device))
    avg_acc += acc
    idx = batch_id
    print('batch_id:', batch_id,
        'loss: ', loss.cpu().detach().numpy(), 'acc: ', acc)
    with open('./save_models/combineMushroomNet/mobilenetv3_large_100/branch/branch_0_test.txt', 'a') as f:
            f.write('batch_id '+str(batch_id) +
                    ' loss '+str(loss.cpu().detach().numpy())+' acc '+str(acc)+'\n')
print('----------------------------------------------')
correct = 0
error = 0
for i in range(len(y_real)):
    if y_real[i] == y_pred[i]:
        correct += 1
    else:
        error += 1
print('avg_loss: ', avg_loss/idx, 'acc: ', correct/(correct+error))
with open('./save_models/combineMushroomNet/mobilenetv3_large_100/branch/branch_0_test.txt', 'a') as f:
            f.write('avg_loss '+str(avg_loss/idx)+' acc '+str(correct/(correct+error))+'\n')