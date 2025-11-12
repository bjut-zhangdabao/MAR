import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MutanLayer(nn.Module):
    def __init__(self, dim, multi):
        super(MutanLayer, self).__init__()

        self.dim = dim
        self.multi = multi

        modal1 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        # modal3 = []
        # for i in range(self.multi):
        #     do = nn.Dropout(p=0.2)
        #     lin = nn.Linear(dim, dim)
        #     modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        # self.modal3_layers = nn.ModuleList(modal3)

    def forward(self, modal1_emb, modal2_emb):
        bs = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            # print('=======>', i)
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            # x_modal3 = self.modal3_layers[i](modal3_emb)
            x_mm.append(torch.mul(x_modal1, x_modal2))
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(bs, self.dim)
        x_mm = torch.relu(x_mm)
        return x_mm