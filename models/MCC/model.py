from typing import Dict, List, Any
import torch
import torch.nn as nn
from torchvision.models import resnet
from torch.nn.modules import BatchNorm2d,BatchNorm1d
from utils.pytorch_misc import Flattener
import torch.nn.functional as F
import torch.nn.parallel
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
from allennlp.nn import InitializerApplicator

from utils.mca import AttFlat, LayerNorm, AttFlat_nofc
from utils import  contrastive_loss
import random

# image backbone code from https://github.com/rowanz/r2c/blob/master/utils/detector.py
def _load_resnet_imagenet(pretrained=True):
    # huge thx to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
    backbone = resnet.resnet50(pretrained=pretrained)
    for i in range(2, 4):
        getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
    # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
    backbone.layer4[0].conv2.stride = (1, 1)
    backbone.layer4[0].downsample[0].stride = (1, 1)

    # # Make batchnorm more sensible
    # for submodule in backbone.modules():
    #     if isinstance(submodule, torch.nn.BatchNorm2d):
    #         submodule.momentum = 0.01

    return backbone

@Model.register("LSTMBatchNormBUAGlobalNoFinalImageFull")
class LSTMBatchNormBUAGlobalNoFinalImageFull(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 option_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(LSTMBatchNormBUAGlobalNoFinalImageFull, self).__init__(vocab)
        self.vector_dim = 1024
        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2048, self.vector_dim),
            torch.nn.ReLU(inplace=True),
        )
        self.boxes_fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(self.vector_dim + 525*2, self.vector_dim),
            torch.nn.ReLU(inplace=True),
        )

        self.image_BN = BatchNorm1d(self.vector_dim)

        self.option_encoder = TimeDistributed(option_encoder)
        self.option_BN = torch.nn.Sequential(
            BatchNorm1d(self.vector_dim)
        )
        self.query_BN = torch.nn.Sequential(
            BatchNorm1d(self.vector_dim)
        )
        self.final_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.vector_dim*2, 512),
            torch.nn.ReLU(inplace=True),
        )
        self.final_BN = torch.nn.Sequential(
            BatchNorm1d(512)
        )
        self.final_mlp_linear = torch.nn.Sequential(
            torch.nn.Linear(512, 1)
        )

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        self.attFlat_image = AttFlat(hidden_size=self.vector_dim, flat_mlp_size=self.vector_dim, flat_out_size=self.vector_dim)
        self.attFlat_option = AttFlat_nofc(hidden_size=self.vector_dim, flat_mlp_size=self.vector_dim, flat_out_size=self.vector_dim)
        self.attFlat_query = AttFlat(hidden_size=self.vector_dim, flat_mlp_size=self.vector_dim, flat_out_size=self.vector_dim)

        self.proj_norm = LayerNorm(size = self.vector_dim)

        self.fusion_BN = torch.nn.Sequential(
            BatchNorm1d(self.vector_dim)
        )
        # self.fusion_fc = torch.nn.Sequential(
        #     torch.nn.Linear(512, 512)
        # )

        self.CL_answer_feat = contrastive_loss.CrossModal_CL(temperature=0.1)

        initializer(self)

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]
 
        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
           row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)

        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        # add recurrent dropout here
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)

        return span_rep, retrieved_feats

    def fusion_QV(self, q, v):
        fusion_qv = v + q  # element-wise add
        batch_size = q.shape[0]
        num_options = q.shape[1]
        fusion_qv = self.proj_norm(fusion_qv)
        fusion_qv = fusion_qv.contiguous().view(batch_size * num_options, self.vector_dim)
        fusion_qv = self.fusion_BN(fusion_qv)
        fusion_qv = fusion_qv.contiguous().view(batch_size, num_options, self.vector_dim)
        return fusion_qv

    def Flat_img(self, images, box_mask_counterfactal, q_shape):
        images_features = self.attFlat_image(images, box_mask_counterfactal)  # [bs,512]
        images_features = self.image_BN(images_features)  # ???
        images_features = images_features.unsqueeze(1).expand(q_shape)
        return  images_features

    def forward(self,
                det_features:torch.Tensor,
                boxes: torch.Tensor,
                boxes_feat: torch.Tensor,
                box_mask: torch.LongTensor,
                question: Dict[str, torch.Tensor],
                question_tags: torch.LongTensor,
                question_mask: torch.LongTensor,
                answers: Dict[str, torch.Tensor],
                answer_tags: torch.LongTensor,
                answer_mask: torch.LongTensor,
                v_mask: torch.LongTensor,
                neg: bool,
                neg_img: bool,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param metadata: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """

        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        # objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        boxes_feat = boxes_feat[:, :max_len]
        det_features = det_features[:,:max_len]
        # segms = segms[:, :max_len]

        obj_reps_ = det_features
        obj_reps_ = self.obj_downsample(obj_reps_)

        boxes_feat = boxes_feat.repeat(1, 1, 105*2)  # [bs, obj_num, 525] all positive, ReLU无所谓
        obj_reps = torch.cat([obj_reps_, boxes_feat], dim=-1)
        obj_reps = self.boxes_fc(obj_reps)

        # obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)

        # option part
        batch_size, num_options, padded_seq_len, _ = answers['bert'].shape
        options, option_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps)
        assert (options.shape == (batch_size, num_options, padded_seq_len, 768+self.vector_dim))
        option_rep = self.option_encoder(options, answer_mask) # (batch_size, 4, seq_len, emb_len(512))

        # use soft attention instead of the mean
        option_features = torch.ones([option_rep.shape[1], option_rep.shape[0], option_rep.shape[3]], dtype=torch.float)
        for i in range(4):
            option_features[i] = self.attFlat_option(option_rep[:, i, :, :], answer_mask[:, i, :])
        option_features = option_features.transpose(1, 0).cuda()
        option_features = option_features.contiguous().view(batch_size * num_options, self.vector_dim)

        option_features = self.option_BN(option_features)
        option_features = option_features.contiguous().view(batch_size, num_options, self.vector_dim) # (batch_size, 4, emb_len(512))

        # query part
        batch_size, num_options, padded_seq_len, _ = question['bert'].shape
        query, query_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps)
        assert (query.shape == (batch_size, num_options, padded_seq_len, 768+self.vector_dim))
        query_rep = self.option_encoder(query, question_mask) # (batch_size, 4, seq_len, emb_len(512))

        # use soft attention instead of the mean
        query_features = torch.ones([option_rep.shape[1], option_rep.shape[0], option_rep.shape[3]], dtype=torch.float)
        for i in range(4):
            query_features[i] = self.attFlat_query(query_rep[:, i, :, :], question_mask[:, i, :])
        query_features = query_features.transpose(1, 0).cuda()
        query_features = query_features.contiguous().view(batch_size * num_options, self.vector_dim)
        query_features = self.query_BN(query_features)
        query_features = query_features.contiguous().view(batch_size, num_options, self.vector_dim)  # (batch_size, 4, emb_len(512))

        # image part
        images = obj_reps[:, 1:, :]
        # counterfactual samples
        if neg == False:
            if v_mask is None: # origin samples
                box_mask_counterfactal = box_mask[:, 1:]
            else: # positive object samples or postive img samples
                v_mask = v_mask[:, 1:]
                box_mask_counterfactal = box_mask[:, 1:] * v_mask.long()

            images_features = self.Flat_img(images, box_mask_counterfactal, query_features.shape)
            # fusion QV
            fusion_qv = self.fusion_QV(query_features, images_features)
        else:
            if neg_img == False: # negtive object samples
                fusion_qv = torch.ones([v_mask.shape[0], batch_size, 4, self.vector_dim]).float().cuda()  # [3, bs,4,512]
                for idx, mask in enumerate(v_mask):
                    mask = mask[:, 1:]
                    box_mask_counterfactal = box_mask[:, 1:] * mask.long()  # [bs, topv_neg, obj_num]

                    images_features = self.Flat_img(images, box_mask_counterfactal, query_features.shape)
                    # fusion QV
                    fusion_qv[idx] = self.fusion_QV(query_features, images_features)
                fusion_qv = fusion_qv.transpose(1, 0)  # [bs,x,4,512]
            else:
                # negtive img samples
                img_sample_num = 3
                fusion_qv = torch.zeros([img_sample_num, batch_size, 4, self.vector_dim]).cuda()
                for k in range(img_sample_num):
                    images_neg = torch.zeros(images.shape).float().cuda()
                    box_mask_neg = torch.zeros(box_mask.shape).long().cuda()
                    # random select from the batch
                    for i in range(batch_size):
                        rand_idx = random.randint(0, batch_size - 1)
                        while rand_idx == i:
                            rand_idx = random.randint(0, batch_size - 1)
                        images_neg[i] = images[rand_idx]
                        box_mask_neg[i] = box_mask[rand_idx]

                    images_features = self.Flat_img(images_neg, box_mask_neg[:, 1:], query_features.shape)
                    # fusion QV
                    fusion_qv[k] = self.fusion_QV(query_features, images_features)
                fusion_qv = fusion_qv.transpose(1, 0)  # [bs,x,4,512]

        # ------ answer level -------
        if label is not None and neg == False:
            # feature CL loss
            # L2 Normal
            fusion_qv_norm = F.normalize(fusion_qv, dim=-1)
            option_features_norm = F.normalize(option_features, dim=-1)
            loss_answer_feat = self.CL_answer_feat(fusion_qv_norm.mean(dim=1), option_features_norm, label.long().view(-1))
        else:
            loss_answer_feat = -1

        if fusion_qv.dim() == 4:
            query_option_image_cat = torch.cat((option_features, fusion_qv[:, 0, :, :]), -1)
        else:
            query_option_image_cat = torch.cat((option_features, fusion_qv), -1)
        assert (query_option_image_cat.shape == (batch_size, num_options, self.vector_dim*2))

        query_option_image_cat = self.final_mlp(query_option_image_cat)
        query_option_image_cat = query_option_image_cat.contiguous().view(batch_size*num_options, 512)
        query_option_image_cat = self.final_BN(query_option_image_cat)
        query_option_image_cat = query_option_image_cat.contiguous().view(batch_size,num_options, 512)
        logits = self.final_mlp_linear(query_option_image_cat)
        logits = logits.squeeze(2)
        class_probabilities = F.softmax(logits, dim=-1)
        output_dict = {"label_logits": logits, "label_probs": class_probabilities}
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            self._accuracy(logits, label)
            output_dict["loss"] = loss[None]
            output_dict['loss_answer_feat'] = loss_answer_feat
        output_dict['QV'] = fusion_qv

        # print ('one pass')
        return output_dict
    def get_metrics(self,reset=False):
        return {'accuracy': self._accuracy.get_metric(reset)}




