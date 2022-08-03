from torchvision import models
import torch
import torch.nn as nn
import os
from SELayer import SELayer
from ECANet import ECANet
import timm
from shufflenetv2 import shufflenet_v2_x1_0
from squeezenet import squeezenet1_1


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_branch_model(model_name, num_classes=37):
    timm_model_list = ['densenet121','efficientnet_b0','mobilenetv3_large_100','resnet50','resnext50_32x4d','vgg11','xception']
    if model_name == 'Alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096,num_classes,bias=True)
    elif model_name in timm_model_list:
        model = timm.create_model(model_name, pretrained=True, num_classes=37)
    elif model_name == 'shufflenetv2':
        model = shufflenet_v2_x1_0(pretrained=True)
        model.fc = torch.nn.Linear(in_features=1024,out_features=37,bias=True)
    elif model_name == 'squeezenet1_1':
        model = squeezenet1_1(pretrained=True)
        model.classifier[1] = torch.nn.Conv2d(512,37,kernel_size=(1,1),stride=(1,1))
    return model

def get_classifier(model_name, model, num_classes=37):
    if model_name == 'Alexnet':
        fc = nn.Sequential(*list(model.children()))[-2:]
        fc[1] = nn.Linear(in_features=27648,
                        out_features=num_classes, bias=True)
        classifier = nn.Sequential(
            SELayer(in_channels=768, reduction_coefficient=16),
            ECANet(in_channels=768, kernel_size=3),
            fc[0],
            nn.Flatten(1),
            fc[1]
        )
        for param in classifier[2].parameters():
            param.requires_grad = True
    elif model_name == 'densenet121':
        fc = nn.Sequential(*list(model.children()))[-2:]
        fc[1] = nn.Linear(in_features=3072,
                            out_features=num_classes, bias=True)
        classifier = nn.Sequential(
            SELayer(in_channels=3072, reduction_coefficient=16),
            ECANet(in_channels=3072, kernel_size=3),
            fc[0],
            fc[1]
        )
        for param in classifier[1].parameters():
            param.requires_grad = True
    elif model_name == 'efficientnet_b0':
        fc = nn.Sequential(*list(model.children()))[-2:]
        fc[1] = nn.Linear(in_features=3840,
                            out_features=num_classes, bias=True)
        classifier = nn.Sequential(
            SELayer(in_channels=3840, reduction_coefficient=16),
            ECANet(in_channels=3840, kernel_size=3),
            fc[0],
            fc[1]
        )
        for param in classifier[1].parameters():
            param.requires_grad = True
    elif model_name == 'mobilenetv3_large_100':
        fc = nn.Sequential(*list(model.children()))[-5:]
        fc[4] = nn.Linear(in_features=1280,
                            out_features=num_classes, bias=True)
        fc[1] = nn.Conv2d(2880, 1280, kernel_size=(1, 1), stride=(1, 1))
        classifier = nn.Sequential(
            SELayer(in_channels=2880, reduction_coefficient=16),
            ECANet(in_channels=2880, kernel_size=3),
            fc[0],
            fc[1],
            fc[2],
            fc[3],
            fc[4]
        )
        for param in classifier[1].parameters():
            param.requires_grad = True
    elif model_name == 'resnet50':
        fc = nn.Sequential(*list(model.children()))[-2:]
        fc[1] = nn.Linear(in_features=6144,
                            out_features=num_classes, bias=True)
        classifier = nn.Sequential(
            SELayer(in_channels=6144, reduction_coefficient=16),
            ECANet(in_channels=6144, kernel_size=3),
            fc[0],
            fc[1]
        )
        for param in classifier[1].parameters():
            param.requires_grad = False
    elif model_name == 'resnext50_32x4d':
        fc = nn.Sequential(*list(model.children()))[-2:]
        fc[1] = nn.Linear(in_features=6144,
                            out_features=num_classes, bias=True)
        classifier = nn.Sequential(
            SELayer(in_channels=6144, reduction_coefficient=16),
            ECANet(in_channels=6144, kernel_size=3),
            fc[0],
            fc[1]
        )
        for param in classifier[1].parameters():
            param.requires_grad = True
    elif model_name == 'shufflenetv2':
        fc = nn.Sequential(*list(model.children()))[-1]
        fc[1] = nn.Linear(in_features=3072,
                            out_features=num_classes, bias=True)
        classifier = nn.Sequential(
            SELayer(in_channels=3072, reduction_coefficient=16),
            ECANet(in_channels=3072, kernel_size=3),
            fc
        )
        for param in classifier[3].parameters():
            param.requires_grad = True
    elif model_name == 'squeezenet1_1':
        fc = nn.Sequential(*list(model.children()))[-2:]
        fc[1] = nn.Linear(in_features=6144,
                            out_features=num_classes, bias=True)
        classifier = nn.Sequential(
            SELayer(in_channels=6144, reduction_coefficient=16),
            ECANet(in_channels=6144, kernel_size=3),
            nn.Flatten(1),
            fc[1]
        )
        for param in classifier[3].parameters():
            param.requires_grad = True
    elif model_name == 'vgg11':
        fc = nn.Sequential(*list(model.children()))[-1]
        fc.fc = nn.Linear(in_features=12288, out_features=num_classes, bias=True)
        classifier = nn.Sequential(
#             SELayer(in_channels=12288, reduction_coefficient=16),
#             ECANet(in_channels=12288, kernel_size=3),
            fc
        )
        for param in classifier[2].parameters():
            param.requires_grad = True
    elif model_name == 'xception':
        fc = nn.Sequential(*list(model.children()))[-2:]
        fc[1] = nn.Linear(in_features=6144,
                            out_features=num_classes, bias=True)
        classifier = nn.Sequential(
            SELayer(in_channels=6144, reduction_coefficient=16),
            ECANet(in_channels=6144, kernel_size=3),
            fc[0],
            fc[1]
        )
        for param in classifier[1].parameters():
            param.requires_grad = True
    return classifier


def deal_model(model, model_name):
    if model_name == 'Alexnet':
        newModel = nn.Sequential()
        for child in model.named_children():
            if child[0] == 'avgpool':
                break
            newModel.add_module(child[0], child[1])
    elif model_name == 'densenet121':
        newModel = nn.Sequential()
        for child in model.named_children():
            if child[0] == 'global_pool':
                break
            newModel.add_module(child[0], child[1])
    elif model_name == 'efficientnet_b0':
        newModel = nn.Sequential()
        for child in model.named_children():
            if child[0] == 'global_pool':
                break
            newModel.add_module(child[0], child[1])
    elif model_name == 'mobilenetv3_large_100':
        newModel = nn.Sequential()
        for child in model.named_children():
            if child[0] == 'global_pool':
                break
            newModel.add_module(child[0], child[1])
    elif model_name == 'resnet50':
        newModel = nn.Sequential()
        for child in model.named_children():
            if child[0] == 'global_pool':
                break
            newModel.add_module(child[0], child[1])
    elif model_name == 'resnext50_32x4d':
        newModel = nn.Sequential()
        for child in model.named_children():
            if child[0] == 'global_pool':
                break
            newModel.add_module(child[0], child[1])
    elif model_name == 'shufflenetv2':
        newModel = nn.Sequential()
        for child in model.named_children():
            if child[0] == 'fc':
                break
            newModel.add_module(child[0], child[1])
    elif model_name == 'squeezenet1_1':
        newModel = nn.Sequential()
        for child in model.named_children():
            if child[0] == 'global_pool':
                break
            newModel.add_module(child[0], child[1])
    elif model_name == 'vgg11':
        newModel = nn.Sequential()
        for child in model.named_children():
            if child[0] == 'head':
                break
            newModel.add_module(child[0], child[1])
    elif model_name == 'xception':
        newModel = nn.Sequential()
        for child in model.named_children():
            if child[0] == 'global_pool':
                break
            newModel.add_module(child[0], child[1])
    return newModel

class combineMushrooms(nn.Module):
    def __init__(self, init=True, branch_num=None, model_name=None, num_classes=None):
        super(combineMushrooms, self).__init__()
        self.branch_num = branch_num
        self.model_name = model_name
        self.num_classes = num_classes

        if init:
            self.models = []
            for i in range(branch_num):
                model = torch.load(os.path.join('..','checkpoint', model_name,'branch_'+str(i)+'.pkl'),map_location=device)
                model.to(device)
                new_model = deal_model(model, self.model_name)
                new_model = self.freeze(new_model)
                self.models.append(new_model)

                if i == branch_num - 1:
                    self.classifier = get_classifier(model_name, model, num_classes)
                
    def forward(self, inputs):
        outs = None
        inputs = inputs.permute(1, 0, 2, 3, 4)
        for i in range(len(inputs)):
            temp = self.models[i](inputs[i])
            if outs == None:
                outs = torch.zeros([inputs.shape[1],0,temp.shape[2],temp.shape[3]])
            outs = torch.cat(
                [outs.to(device), temp], dim=1)
        outs = self.classifier(outs)
        return outs

    def loadModel(self):
        self.models = []
        for i in range(self.branch_num):
            model = torch.load(os.path.join('..','checkpoint',self.model_name,'model_'+str(i)+'.pkl'), map_location=device)
            self.models.append(model)
        self.classifier = torch.load(os.path.join('..','checkpoint',self.model_name,'classifier.pkl'), map_location=device)

    def saveModel(self):
        for i, model in enumerate(self.models):
            torch.save(model, os.path.join('..','checkpoint',self.model_name,'model_'+str(i)+'.pkl'))
        torch.save(self.classifier, os.path.join('..','checkpoint',self.model_name,'classifier.pkl'))

    def setTrain(self):
        for i in range(self.branch_num):
            self.models[i].train()
        self.classifier.train()

    def setEval(self):
        for i in range(self.branch_num):
            self.models[i].eval()
        self.classifier.eval()
    
    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model
        

def get_multi_model(init=True, branch_num=None, model_name=None, num_classes=None):
    model = combineMushrooms(init=init, branch_num=branch_num, model_name=model_name, num_classes=num_classes)
    return model