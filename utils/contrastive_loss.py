import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# cross-entropy loss   [supervised CL paper]
class CrossModal_CL(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(CrossModal_CL, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, anchor_feature, features, label):
        # anchor_feature:[bs,dim]
        # features:[bs,4,dim]
        # label : [bs] long
        anchor_feature = anchor_feature.unsqueeze(1) #[bs.1.dim]
        features = features.transpose(2,1) # [ns,dim,4]
        # b, device = anchor_feature.shape[0], anchor_feature.device
        logits = torch.div(torch.matmul(anchor_feature, features), self.temperature)#[bs.4]
        logits_max = logits.max(dim=-1, keepdim=True)[0]
        logits = logits - logits_max.detach()
        loss = F.cross_entropy(logits.squeeze(1), label)
        return loss

# CL loss on features
class CL_feat(nn.Module):
    def __init__(self, temperature=0.2):
        super(CL_feat, self).__init__()
        self.temperature = temperature
        self.contrastive_loss = CrossModal_CL(temperature=self.temperature)

    def forward(self, anchor, pos, neg):
        # anchor " [bs,4,512] QV
        # pos: [bs,4,512] QV+
        # neg: [bs,x,4,512] QV-
        if neg.dim() == 3:
            neg = neg.unsqueeze(1)
        features = torch.cat([pos.unsqueeze(1), neg], dim=1) #[bs,1+x,4,512]
        label = torch.zeros(anchor.shape[0]).long().cuda() # pos at the first
        loss = []
        for i in range(4):
            loss.append(self.contrastive_loss(anchor[:,i,:], features[:,:,i,:], label))

        mean_loss = sum(loss)/len(loss)

        return mean_loss

if __name__ == '__main__':
    #criterion = CrossModal_CL() #SupConLoss(contrast_mode='one')
    criterion = ww_loss()
    anchor_feature = torch.rand(2,512)
    features = torch.rand(2,4,512)
    label = torch.randint(2,(2,))
    #lable = torch.ones()

    loss = criterion(anchor_feature, features, label)