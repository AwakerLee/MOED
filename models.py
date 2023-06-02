import torch
import math
import torch.nn as nn
import torchvision
import clip
from pygcn.layers import GraphConvolution
import torch.nn.functional as F

class Teacher1(nn.Module):
    def __init__(self):
        super(Teacher1, self).__init__()
        self.clip_image_encode, _ = clip.load("RN101", device='cuda:0') #ViT-L/14-#768 ViT-B/32#512

    def forward(self, x):
        with torch.no_grad():
            feat_I = self.clip_image_encode.encode_image(x)
            feat_I = feat_I.type(torch.float32)
        return feat_I

class Teacher2(nn.Module):
    def __init__(self):
        super(Teacher2, self).__init__()
        self.clip_image_encode, _ = clip.load("ViT-B/32", device='cuda:0')

    def forward(self, x):
        with torch.no_grad():
            feat_I = self.clip_image_encode.encode_image(x)
            feat_I = feat_I.type(torch.float32)
        return feat_I

class Teacher3(nn.Module):
    def __init__(self):
        super(Teacher3, self).__init__()
        self.clip_image_encode, _ = clip.load("ViT-B/16", device='cuda:0')

    def forward(self, x):
        with torch.no_grad():
            feat_I = self.clip_image_encode.encode_image(x)
            feat_I = feat_I.type(torch.float32)
        return feat_I

class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        # self.clip_image_encode, _ = clip.load("ViT-B/16", device='cuda:0')
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16.classifier = self.vgg16.classifier[:-1]

        self.feature_layer = nn.Sequential(nn.Linear(4096, 4096),
                                           nn.ReLU(),
                                          )
        self.hash_layer = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat_I = self.vgg16(x)
        mid_feat_I = self.feature_layer(feat_I)
        hid = self.hash_layer(mid_feat_I)
        code = torch.tanh(self.alpha * hid)
        return feat_I, mid_feat_I, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        mid_feat_T = torch.relu(self.fc1(x))
        hid = self.fc2(mid_feat_T)
        code = torch.tanh(self.alpha * hid)
        return x, mid_feat_T, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GCNet_IMG(nn.Module):
    def __init__(self, bit, gamma, batch_size):
        super(GCNet_IMG, self).__init__()

        self.gc1 = GraphConvolution(512, 4096)
        self.gc2 = GraphConvolution(4096, 4096)
        self.linear = nn.Linear(4096, bit)
        self.alpha = 1.0
        # self.gamma = gamma
        # self.weight = nn.Parameter(torch.FloatTensor(batch_size, batch_size))
        # nn.init.kaiming_uniform_(self.weight)
        # nn.init.constant_(self.weight, 1e-6)

    def forward(self, x, adj):
        # adj = adj + self.gamma * self.weight
        x = torch.relu(self.gc1(x, adj))
        feat_G_I = torch.relu(self.gc2(x, adj))
        x = self.linear(feat_G_I)
        code = torch.tanh(self.alpha * x)
        return feat_G_I, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GCNet_TXT(nn.Module):
    def __init__(self, txt_feat_len, bit, gamma, batch_size):
        super(GCNet_TXT, self).__init__()

        self.gc1 = GraphConvolution(txt_feat_len, 4096)
        self.gc2 = GraphConvolution(4096, 4096)
        self.linear = nn.Linear(4096, bit)
        self.alpha = 1.0
        # self.gamma = gamma
        # self.weight = nn.Parameter(torch.FloatTensor(batch_size, batch_size))
        # nn.init.kaiming_uniform_(self.weight)
        # nn.init.constant_(self.weight, 1e-6)

    def forward(self, x, adj):
        # adj = adj + self.gamma * self.weight
        x = torch.relu(self.gc1(x, adj))
        feat_G_T = torch.relu(self.gc2(x, adj))
        x = self.linear(feat_G_T)
        code = torch.tanh(self.alpha * x)
        return feat_G_T, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
