# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from .net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, hidden_size=512, dropout_r = 0.1, multi_head=8, hidden_size_head=64):
        super(MHAtt, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_r = dropout_r
        self.multi_head = multi_head
        self.hidden_size_head = hidden_size_head
        # self.__C = __C

        self.linear_v = nn.Linear(self.hidden_size,self.hidden_size) #512
        self.linear_k = nn.Linear(self.hidden_size,self.hidden_size)
        self.linear_q = nn.Linear(self.hidden_size,self.hidden_size)
        self.linear_merge = nn.Linear(self.hidden_size,self.hidden_size)

        self.dropout = nn.Dropout(dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2) #[bs,8,seq_len,64]

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2) #[bs,8,seq_len,64]

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2) #[bs,8,seq_len,64]

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k) # [bs, 8, seq_len, seq_len]

        if mask is not None:
            mask = (mask==0).byte()
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map) #[bs,8,seq_len.seq_len]

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, hidden_size=512, dropout_r=0.1):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=4*hidden_size, #__C.FF_SIZE,
            out_size=hidden_size,
            dropout_r=dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, hidden_size=512, dropout_r=0.1):
        super(SA, self).__init__()

        self.mhatt = MHAtt(hidden_size=hidden_size, dropout_r=dropout_r)
        self.ffn = FFN(hidden_size=hidden_size, dropout_r=dropout_r)

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        )) #[bs, seq_len. 512]

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        )) #[bs, seq_len. 512]

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, hidden_size=512, dropout_r=0.1):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt()
        self.mhatt2 = MHAtt()
        self.ffn = FFN()

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(dropout_r)
        self.norm3 = LayerNorm(hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

class CO_SGA(nn.Module):
    def __init__(self, hidden_size=512, dropout_r=0.1):
        super(CO_SGA, self).__init__()

        self.mhatt1 = MHAtt()
        self.mhatt2 = MHAtt()
        self.ffn = FFN()
        self.ffn2 = FFN()

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(dropout_r)
        self.norm3 = LayerNorm(hidden_size)

        self.dropout4 = nn.Dropout(dropout_r)
        self.norm4 = LayerNorm(hidden_size)

        self.dropout5 = nn.Dropout(dropout_r)
        self.norm5 = LayerNorm(hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x2 = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x2 = self.norm3(x2 + self.dropout3(
            self.ffn(x2)
        ))

        y2 = self.norm4(y + self.dropout4(
            self.mhatt2(x, x, y, x_mask)
        ))

        y2 = self.norm5(y2 + self.dropout5(
            self.ffn2(y2)
        ))

        return x2, y2


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, layer=1):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA() for _ in range(layer)])
        self.dec_list = nn.ModuleList([CO_SGA() for _ in range(layer)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask) #SA(x)   x should be question

        for dec in self.dec_list:
            y2, x2 = dec(y, x, y_mask, x_mask) #SGA(Y,X)   Y should be objects

        return x2, y2 # question, objects


# ------------------------------------------------
# ---- Attention Flatten at the fusion stage ----
# ------------------------------------------------

class AttFlat(nn.Module):
    def __init__(self, hidden_size=512, flat_mlp_size=512, flat_out_size=1024, flat_glimpses=1, dropout_r=0.1):
        super(AttFlat, self).__init__()
        self.flat_glimpses = flat_glimpses

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_out_size,
            dropout_r=dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            hidden_size * flat_glimpses,
            flat_out_size
        )

    def forward(self, x, x_mask):
        att = self.mlp(x) #att: [bs,seq_len,1024]
        x_mask = (x_mask == 0).byte() #[bs, seq_len]
        att = att.masked_fill(
            x_mask.unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1) #[bs, obj_num, 1024]

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1) #torch.sum(att[:, :, i: i + 1], dim=1)=1
            )

        x_atted = torch.cat(att_list, dim=1) #[bs, 512]
        x_atted = self.linear_merge(x_atted)

        return x_atted

class AttFlat_nofc(nn.Module):
    def __init__(self, hidden_size=512, flat_mlp_size=512, flat_out_size=1024, flat_glimpses=1, dropout_r=0.1):
        super(AttFlat_nofc, self).__init__()
        self.flat_glimpses = flat_glimpses

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_out_size,
            dropout_r=dropout_r,
            use_relu=True
        )

        # self.linear_merge = nn.Linear(
        #     hidden_size * flat_glimpses,
        #     flat_out_size
        # )

    def forward(self, x, x_mask):
        att = self.mlp(x) #att: [bs,seq_len,1024]
        x_mask = (x_mask == 0).byte() #[bs, seq_len]
        att = att.masked_fill(
            x_mask.unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1) #[bs, obj_num, 1024]

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1) #torch.sum(att[:, :, i: i + 1], dim=1)=1
            )

        x_atted = torch.cat(att_list, dim=1) #[bs, 512]
        # x_atted = self.linear_merge(x_atted)

        return x_atted