import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import datasets
import os.path as osp
from models import ImgNet, TxtNet, GCNet_IMG, GCNet_TXT, Teacher1, Teacher2, Teacher3
from utils import compress, calculate_top_map, compress_wiki, calc_map_k
import numpy as np
import scipy.sparse as sp

class AMTS:
    def __init__(self, log, config):
        self.logger = log
        self.config = config

        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.cuda.set_device(self.config.GPU_ID)

        if self.config.DATASET == "MIRFlickr":
            self.train_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform)
            self.test_dataset = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
            self.database_dataset = datasets.MIRFlickr(train=False, database=True, transform=datasets.mir_test_transform)

        if self.config.DATASET == "NUSWIDE":
            self.train_dataset = datasets.NUSWIDE(train=True, transform=datasets.nus_train_transform)
            self.test_dataset = datasets.NUSWIDE(train=False, database=False, transform=datasets.nus_test_transform)
            self.database_dataset = datasets.NUSWIDE(train=False, database=True, transform=datasets.nus_test_transform)

        if self.config.DATASET == "WIKI":
            self.train_dataset = datasets.WIKI(root=self.config.DATA_DIR, train=True,
                                               transform=datasets.wiki_train_transform)
            self.test_dataset = datasets.WIKI(root=self.config.DATA_DIR, train=False,
                                              transform=datasets.wiki_test_transform)
            self.database_dataset = datasets.WIKI(root=self.config.DATA_DIR, train=True,
                                                  transform=datasets.wiki_test_transform)

        if self.config.DATASET == "MSCOCO":
            self.train_dataset = datasets.MSCOCO(train=True, transform=datasets.coco_train_transform)
            self.test_dataset = datasets.MSCOCO(train=False, database=False, transform=datasets.coco_test_transform)
            self.database_dataset = datasets.MSCOCO(train=False, database=True, transform=datasets.coco_test_transform)

        # Data Loader (Input Pipeline)
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.config.BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=self.config.NUM_WORKERS,
                                       drop_last=True)

        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.config.BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=self.config.NUM_WORKERS)

        self.database_loader = DataLoader(dataset=self.database_dataset,
                                          batch_size=self.config.BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=self.config.NUM_WORKERS)

        self.Teacher1 = Teacher1()
        self.Teacher2 = Teacher2()
        self.Teacher3 = Teacher3()
        self.ImgNet = ImgNet(code_len=self.config.HASH_BIT)

        txt_feat_len = datasets.txt_feat_len

        self.TxtNet = TxtNet(code_len=self.config.HASH_BIT, txt_feat_len=txt_feat_len)

        self.GCNet_IMG = GCNet_IMG(bit=config.HASH_BIT, gamma=self.config.gamma, batch_size=self.config.BATCH_SIZE)
        self.GCNet_TXT = GCNet_TXT(txt_feat_len=txt_feat_len, bit=config.HASH_BIT, gamma=self.config.gamma, batch_size=self.config.BATCH_SIZE)

        self.opt_I = torch.optim.SGD(self.ImgNet.parameters(), lr=self.config.LR_IMG, momentum=self.config.MOMENTUM,
                                      weight_decay=self.config.WEIGHT_DECAY)

        self.opt_T = torch.optim.SGD(self.TxtNet.parameters(), lr=self.config.LR_TXT, momentum=self.config.MOMENTUM,
                                      weight_decay=self.config.WEIGHT_DECAY)

        self.opt_GCN_I = torch.optim.SGD(self.GCNet_IMG.parameters(), lr=self.config.LR_IMG, momentum=self.config.MOMENTUM,
                                     weight_decay=self.config.WEIGHT_DECAY)

        self.opt_GCN_T = torch.optim.SGD(self.GCNet_TXT.parameters(), lr=self.config.LR_TXT, momentum=self.config.MOMENTUM,
                                     weight_decay=self.config.WEIGHT_DECAY)

        # self.opt_GCN_I = torch.optim.Adam(self.GCNet_IMG.parameters(), lr=0.01)
        # self.opt_GCN_T = torch.optim.Adam(self.GCNet_TXT.parameters(), lr=0.01)

        self.best_it = 0
        self.best_ti = 0

    def train(self, epoch):

        self.ImgNet.set_alpha(epoch)
        self.TxtNet.set_alpha(epoch)
        self.GCNet_IMG.set_alpha(epoch)
        self.GCNet_TXT.set_alpha(epoch)

        self.ImgNet.cuda().train()
        self.TxtNet.cuda().train()
        self.GCNet_IMG.cuda().train()
        self.GCNet_TXT.cuda().train()


        for idx, (img, txt, _, index) in enumerate(self.train_loader):

            img = torch.FloatTensor(img).cuda()
            txt = torch.FloatTensor(txt.numpy()).cuda()

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            self.opt_GCN_I.zero_grad()
            self.opt_GCN_T.zero_grad()

            feat_I, mid_feat_I, code_I = self.ImgNet(img)
            feat_T, mid_feat_T, code_T = self.TxtNet(txt)

            # multi-teacher selection
            teacher1_FI = self.Teacher1(img)
            teacher2_FI = self.Teacher2(img)
            teacher3_FI = self.Teacher3(img)

            teacher_FI = self.Multi_Teacher_Selection(mid_feat_I, teacher1_FI, teacher2_FI, teacher3_FI)

            S = self.cal_similarity_matrix(teacher_FI, txt)
        
    
            feat_G_I, code_gcn_I = self.GCNet_IMG(teacher3_FI.detach(), S)
            feat_G_T, code_gcn_T = self.GCNet_TXT(txt.detach(), S)

            loss = self.cal_loss(mid_feat_I, mid_feat_T, feat_G_I, feat_G_T, code_I, code_T, code_gcn_I, code_gcn_T, S)

            loss.backward()

            self.opt_I.step()
            self.opt_T.step()
            self.opt_GCN_I.step()
            self.opt_GCN_T.step()

            if (idx + 1) % (len(self.train_dataset) // self.config.BATCH_SIZE / self.config.EPOCH_INTERVAL) == 0:
                self.logger.info('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                                 % (epoch + 1, self.config.NUM_EPOCH, idx + 1, len(self.train_dataset) // self.config.BATCH_SIZE,
                                     loss.item()))


    def eval(self):

        self.ImgNet.cuda().eval()
        self.TxtNet.cuda().eval()
        if self.config.DATASET == "WIKI":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_wiki(self.database_loader, self.test_loader,
                                                                   self.ImgNet, self.TxtNet,
                                                                   self.database_dataset, self.test_dataset)
        if self.config.DATASET == "MIRFlickr" or self.config.DATASET == "NUSWIDE" or self.config.DATASET == "MSCOCO":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.ImgNet,
                                                          self.TxtNet, self.database_dataset, self.test_dataset)


        # MAP_I2T = calc_map_k(qB=qu_BI, rB=re_BT, query_L=qu_L, retrieval_L=re_L, k=None)
        # MAP_T2I = calc_map_k(qB=qu_BT, rB=re_BI, query_L=qu_L, retrieval_L=re_L, k=None)
        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=self.config.topk)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=self.config.topk)

        if (self.best_it + self.best_ti) < (MAP_I2T + MAP_T2I):
            self.best_it = MAP_I2T
            self.best_ti = MAP_T2I
        self.logger.info('--------------------Evaluation: mAP@%d-------------------' % (self.config.topk))
        self.logger.info('mAP@%d I->T: %.4f, mAP@%d T->I: %.4f' % (self.config.topk, MAP_I2T, self.config.topk, MAP_T2I))
        self.logger.info('Best MAP of I->T: %.4f, Best mAP of T->I: %.4f' % (self.best_it, self.best_ti))
        self.logger.info('--------------------------------------------------------------------')


    def Multi_Teacher_Selection(self, mid_feat_I, teacher1_FI, teacher2_FI, teacher3_FI):
        mid_feat_I = F.normalize(mid_feat_I, dim=1)
        S_I_Feature = mid_feat_I.mm(mid_feat_I.t())

        teacher1_FI = F.normalize(teacher1_FI, dim=1)
        S_I_teacher1 = teacher1_FI.mm(teacher1_FI.t())

        teacher2_FI = F.normalize(teacher2_FI, dim=1)
        S_I_teacher2 = teacher2_FI.mm(teacher2_FI.t())

        teacher3_FI = F.normalize(teacher3_FI, dim=1)
        S_I_teacher3 = teacher3_FI.mm(teacher3_FI.t())

        score1, score2, score3 = F.mse_loss(S_I_Feature, S_I_teacher1), F.mse_loss(S_I_Feature, S_I_teacher2), F.mse_loss(S_I_Feature, S_I_teacher3)
        scores = torch.Tensor([score1, score2, score3])
        selected_index = torch.argmin(scores)
        print(selected_index)
        if selected_index == 0:
            teacher_FI = teacher1_FI
        if selected_index == 1:
            teacher_FI = teacher2_FI
        if selected_index == 2:
            teacher_FI = teacher3_FI

        return teacher_FI

    def cal_similarity_matrix(self, teacher_FI, txt):

        teacher_FI = F.normalize(teacher_FI, dim=1)
        S_I_3 = teacher_FI.mm(teacher_FI.t())
        S_I = S_I_3 * 2 - 1

        F_T = F.normalize(txt, dim=1)
        S_T = F_T.mm(F_T.t())
        S_T = S_T * 2 - 1

        S_ = self.config.beta * S_I + self.config.gamma * S_T + self.config.delta * F.normalize(S_I.mm(S_T))
        S = S_ * self.config.mu
        return S

    def cal_loss(self, mid_feat_I, mid_feat_T, feat_G_I, feat_G_T, code_I, code_T, code_gcn_I, code_gcn_T, S):
        B_I = F.normalize(code_I)
        B_T = F.normalize(code_T)
        B_gI = F.normalize(code_gcn_I)
        B_gT = F.normalize(code_gcn_T)

        BI_BI = B_I.mm(B_I.t())
        BT_BT = B_T.mm(B_T.t())
        BI_BT = B_I.mm(B_T.t())
        BT_BI = B_T.mm(B_I.t())

        GBI_GBI = B_gI.mm(B_gI.t())
        GBT_GBT = B_gT.mm(B_gT.t())

        Hashing_level_loss = F.mse_loss(B_I, B_gI) + F.mse_loss(B_T, B_gT)  # responsed-based KD

        Intra_modal_loss = F.mse_loss(BI_BI, S) + F.mse_loss(BT_BT, S)
        Cross_modal_loss = F.mse_loss(BI_BT, S) - (B_I * B_T).sum(dim=1).mean()         #relation graph-based KD
        Graph_level_loss = F.mse_loss(BI_BI, GBI_GBI) + F.mse_loss(BT_BT, GBT_GBT)

        Feature_level_loss = F.mse_loss(mid_feat_I, feat_G_I) + F.mse_loss(mid_feat_T, feat_G_T)   #feature-based KD

        KD_loss = self.config.epsilon * Hashing_level_loss + self.config.tau * Intra_modal_loss + self.config.eta * Cross_modal_loss + self.config.phi * Graph_level_loss + self.config.lamb * Feature_level_loss

        return KD_loss

    def save_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.ImgNet.state_dict(),
            'TxtNet': self.TxtNet.state_dict(),
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')


    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** Fail to load checkpoint %s!*********' % ckp_path)
            raise IOError

            self.ImgNet.load_state_dict(obj['ImgNet'])
            self.TxtNet.load_state_dict(obj['TxtNet'])


