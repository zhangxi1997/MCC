"""
Training script. Should be pretty adaptable to whatever.
"""
import argparse
import os
import shutil

import random
import multiprocessing
import numpy as np
import pandas as pd
import torch
from allennlp.common.params import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm
from tensorboardX import SummaryWriter

import sys
sys.path.append("..")
from dataloaders.vcr import VCR, VCRLoader
# from dataloaders.vcr_attribute import VCR, VCRLoader
from dataloaders.vcr_attribute_box import VCR, VCRLoader
from utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint, Select_obj_new_topn

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

# This is needed to make the imports work
from allennlp.models import Model
from utils import contrastive_loss
import torch.nn.functional as F
import models
#################################
#################################
######## Data loading stuff
#################################
#################################

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '-params',
    dest='params',
    help='Params location',
    type=str,
)
parser.add_argument(
    '-rationale',
    action="store_true",
    help='use rationale',
)
parser.add_argument(
    '-folder',
    dest='folder',
    help='folder location',
    type=str,
)
parser.add_argument(
    '-no_tqdm',
    dest='no_tqdm',
    action='store_true',
)
parser.add_argument(
    '-plot',
    dest='plot',
    help='plot folder location',
    type=str,
)

args = parser.parse_args()
writer = SummaryWriter('runs/' + args.plot)
params = Params.from_file(args.params)
train, val = VCR.splits(mode='rationale' if args.rationale else 'answer',
                              embs_to_load=params['dataset_reader'].get('embs', 'bert_da'),
                              only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', True),
                                expand2obj36=True)
NUM_THREADS = 2
torch.set_num_threads(NUM_THREADS)
NUM_GPUS = torch.cuda.device_count()
print (NUM_GPUS)
NUM_CPUS = multiprocessing.cpu_count()
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")

def _to_gpu(td):
    if NUM_GPUS > 1:
        return td
    for k in td:
        if k != 'metadata':
            td[k] = {k2: v.cuda(non_blocking=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(
                non_blocking=True)
    return td

num_workers = 8
loader_params = {'batch_size': 96 // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}
train_loader = VCRLoader.from_dataset(train, **loader_params)
val_loader = VCRLoader.from_dataset(val, **loader_params)

ARGS_RESET_EVERY = 50
print("Loading {} for {}".format(params['model'].get('type', 'WTF?'), 'rationales' if args.rationale else 'answer'), flush=True)
model = Model.from_params(vocab=train.vocab, params=params['model'])

if hasattr(model, 'detector'):
    for submodule in model.detector.backbone.modules():
        # if isinstance(submodule, BatchNorm2d):
            # submodule.track_running_stats = False
        for p in submodule.parameters():
            p.requires_grad = False

model = DataParallel(model).cuda() if NUM_GPUS > 1 else model.cuda()
optimizer = Optimizer.from_params([x for x in model.named_parameters() if x[1].requires_grad],
                                  params['trainer']['optimizer'])

lr_scheduler_params = params['trainer'].pop("learning_rate_scheduler", None)
scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params) if lr_scheduler_params else None

if os.path.exists(args.folder):
    print("Found folder! restoring", flush=True)
    start_epoch, val_metric_per_epoch = restore_checkpoint(model, optimizer, serialization_dir=args.folder,
                                                           learning_rate_scheduler=scheduler)
else:
    print("Making directories")
    os.makedirs(args.folder, exist_ok=True)
    start_epoch, val_metric_per_epoch = 0, []
    shutil.copy2(args.params, args.folder)

param_shapes = print_para(model)
num_batches = 0
global_train_loss = []
global_train_acc = []
global_val_loss = []
global_val_acc = []

CL_obj_feat = contrastive_loss.CL_feat(temperature=params['trainer']['obj_feat_temp'])
CL_img_feat = contrastive_loss.CL_feat(temperature=params['trainer']['img_feat_temp'])

for epoch_num in range(start_epoch, params['trainer']['num_epochs'] + start_epoch):
    train_results_1 = []
    train_results_2 = []
    norms = []
    model.train()
    for b, (time_per_batch, batch) in enumerate(time_batch(train_loader if args.no_tqdm else tqdm(train_loader), reset_every=ARGS_RESET_EVERY)):
        batch = _to_gpu(batch)
        batch['det_features'] = batch['det_features'].requires_grad_()
        batch['v_mask'] = None
        batch['neg'] = False
        batch['neg_img'] = False
        optimizer.zero_grad()

        # ------------------------------------------------
        # ---- first train stage ----
        # ------------------------------------------------
        output_dict = model(**batch)

        label = batch['label'].long().view(-1)
        label = torch.zeros(loader_params['batch_size'], 4).cuda().scatter_(1, label.unsqueeze(1), 1)
        visual_grad = torch.autograd.grad((output_dict['label_logits'] * (label>0).float()).sum(), batch['det_features'], create_graph=True)[0]

        loss_origin = output_dict['loss'].mean()
        if 'loss_answer_feat' in output_dict and 'lambda_obj_feat' in params['trainer']:
            loss_answer_feat_origin = output_dict['loss_answer_feat'].mean()

        QV_anchor = output_dict['QV']
        train_results_1.append(pd.Series({'total_loss': loss_origin.item(),
                                        'VCR_loss': output_dict['loss'].mean().item(),
                                        'accuracy': (model.module if NUM_GPUS > 1 else model).get_metrics(
                                            reset=(b % ARGS_RESET_EVERY) == 0)[
                                            'accuracy'],
                                        'sec_per_batch': time_per_batch,
                                        'hr_per_epoch': len(train_loader) * time_per_batch / 3600,
                                        }))

        # ------------------------------------------------
        # ---- second train stage ----
        # ------------------------------------------------
        v_mask = torch.zeros(loader_params['batch_size'],  batch['det_features'].shape[1]).cuda()
        visual_grad_cam = visual_grad.sum(2)
        visual_mask = (batch['box_mask'] == 0).byte() # mask for the padding objects
        visual_grad_cam = visual_grad_cam.masked_fill(visual_mask, -1e9)
        v_grad = visual_grad_cam # [bs, object_num]
        top_num = params['trainer']['sample_num']

        # choose the critical objects
        v_mask_pos, v_mask_neg = Select_obj_new_topn(visual_grad_cam, batch['box_mask'], top_num, loader_params['batch_size'],
                                      batch['det_features'].shape[1])

        # --------- postive  V+  -------
        batch['v_mask'] = v_mask_pos
        batch['neg'] = False
        batch['neg_img'] = False
        output_dict_pos = model(**batch)
        QV_pos = output_dict_pos['QV']

        loss_pos = output_dict_pos['loss'].mean()
        if 'loss_answer_feat' in output_dict_pos:
            loss_answer_feat_pos = output_dict_pos['loss_answer_feat'].mean()

        # --------- negtive V-  -------
        if v_mask_neg.dim() == 3:
            v_mask = v_mask_neg
        else:
            v_mask = v_mask_neg.unsqueeze(0)
        batch['v_mask'] = v_mask
        batch['neg'] = True
        batch['neg_img'] = False
        output_dict_neg = model(**batch)
        QV_neg = output_dict_neg['QV']
        loss_neg = output_dict_neg['loss'].mean()

        # counterfactual loss, feed the counterfactual samples into VCR model
        loss_obj_VCR = (loss_pos + max(0, params['trainer']['margin_obj_VCR'] - loss_neg))
        loss_obj_VCR = loss_obj_VCR * params['trainer']['lambda_obj_VCR']

        # L2-normalization
        QV_anchor = F.normalize(QV_anchor, p=2, dim=-1)
        QV_pos = F.normalize(QV_pos, p=2, dim=-1)
        QV_neg = F.normalize(QV_neg, p=2, dim=-1)

        loss_obj_feat = CL_obj_feat(QV_anchor, QV_pos, QV_neg)
        loss_obj_feat = loss_obj_feat * params['trainer']['lambda_obj_feat']

        # ----- pos whole image -----
        if params['trainer']['img_level'] == True:
            random_mask = torch.cuda.FloatTensor(loader_params['batch_size'], batch['box_mask'].shape[1]).uniform_() > 0.5
            v_mask = batch['box_mask'] * random_mask.long()
            v_mask = v_mask + v_mask_pos.long()  # the most important object should exists
            v_mask = (v_mask > 0).long().cuda()

            batch['v_mask'] = v_mask
            batch['neg'] = False
            batch['neg_img'] = False

            output_dict_pos_img = model(**batch)
            QV_pos_img = output_dict_pos_img['QV']
            loss_pos_img = output_dict_pos_img['loss'].mean()
            if 'loss_answer_feat' in output_dict_pos_img:
                loss_answer_feat_pos_img = output_dict_pos_img['loss_answer_feat'].mean()

            # ------ neg whole image ------
            batch['v_mask'] = None
            batch['neg'] = True
            batch['neg_img'] = True
            output_dict_neg_img = model(**batch)
            QV_neg_img = output_dict_neg_img['QV'] # [bs, 4, 512]

            loss_neg_img = output_dict_neg_img['loss'].mean()

            loss_img_VCR = (loss_pos_img + max(0, params['trainer']['margin_img_VCR'] - loss_neg_img))
            loss_img_VCR = loss_img_VCR * params['trainer']['lambda_img_VCR']

            # L2-normalization
            QV_pos_img = F.normalize(QV_pos_img, p=2, dim=-1)
            QV_neg_img = F.normalize(QV_neg_img, p=2, dim=-1)
            loss_img_feat = CL_img_feat(QV_anchor, QV_pos_img, QV_neg_img)
            loss_img_feat = loss_img_feat * params['trainer']['lambda_img_feat']

            loss_answer_feat = loss_answer_feat_origin + loss_answer_feat_pos + loss_answer_feat_pos_img
            loss_answer_feat = loss_answer_feat * params['trainer']['lambda_answer_feat']

            loss_total = loss_origin + loss_obj_VCR + loss_obj_feat + loss_img_VCR + loss_img_feat + loss_answer_feat

        else:
            loss_answer_feat = loss_answer_feat_origin + loss_answer_feat_pos
            loss_answer_feat = loss_answer_feat * params['trainer']['lambda_answer_feat']

            loss_total = loss_origin + loss_obj_VCR + loss_obj_feat + loss_answer_feat

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        optimizer.zero_grad()

        train_results_2.append(pd.Series({'total_loss': loss_total.item(),
                                          'origin_loss': loss_origin.item(),
                                          'loss_obj_VCR': loss_obj_VCR.item(),
                                          'loss_obj_feat': loss_obj_feat.item(),
                                          'loss_img_VCR': loss_img_VCR.item() if params['trainer']['img_level'] else -1,
                                          'loss_img_feat': loss_img_feat.item() if params['trainer']['img_level'] else -1,
                                          'loss_answer_feat': loss_answer_feat.item(),
                                          'accuracy': (model.module if NUM_GPUS > 1 else model).get_metrics(
                                              reset=(b % ARGS_RESET_EVERY) == 0)[
                                              'accuracy'],
                                          }))

        if b % ARGS_RESET_EVERY == 0 and b > 0:
            print("\ne{:2d}b{:5d}/{:5d}. ---- \nsumm:\n{}\n   ~~~~~~~~~~~~~~~~~~\n".format(
                epoch_num, b, len(train_loader),
                pd.DataFrame(train_results_2[-ARGS_RESET_EVERY:]).mean(),
            ), flush=True)

    epoch_stats = pd.DataFrame(train_results_1).mean()
    train_loss = epoch_stats['total_loss']
    train_acc = epoch_stats['accuracy']
    writer.add_scalar('loss/train', train_loss, epoch_num)
    writer.add_scalar('accuracy/train', train_acc, epoch_num)
    global_train_loss.append(train_loss)
    global_train_acc.append(train_acc)
    print("---\nTRAIN EPOCH {:2d}: -- origin--\n{}\n--------".format(epoch_num, pd.DataFrame(train_results_1).mean()))

    val_probs = []
    val_labels = []
    val_loss_sum = 0.0
    model.eval()
    for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
        with torch.no_grad():
            batch = _to_gpu(batch)
            batch['v_mask'] = None
            batch['neg'] = False
            batch['neg_img'] = False
            output_dict = model(**batch)
            val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
            val_labels.append(batch['label'].detach().cpu().numpy())
            val_loss_sum += output_dict['loss'].mean().item() * batch['label'].shape[0]
    val_labels = np.concatenate(val_labels, 0)
    val_probs = np.concatenate(val_probs, 0)
    val_loss_avg = val_loss_sum / val_labels.shape[0]

    val_metric_per_epoch.append(float(np.mean(val_labels == val_probs.argmax(1))))
    if scheduler:
        scheduler.step(val_metric_per_epoch[-1], epoch_num)

    print("Val epoch {} has acc {:.3f} and loss {:.3f}".format(epoch_num, val_metric_per_epoch[-1], val_loss_avg),
          flush=True)
    writer.add_scalar('loss/validation', val_loss_avg, epoch_num)
    writer.add_scalar('accuracy/validation',val_metric_per_epoch[-1], epoch_num)
    global_val_loss.append(val_loss_avg)
    global_val_acc.append(val_metric_per_epoch[-1])
    # if int(np.argmax(val_metric_per_epoch)) < (len(val_metric_per_epoch) - 1 - params['trainer']['patience']):
    #     print("Stopping at epoch {:2d}".format(epoch_num))
    #     break
    if scheduler:
        save_checkpoint(model, optimizer, args.folder, epoch_num, val_metric_per_epoch,
                        is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1),learning_rate_scheduler=scheduler)
    else:
        save_checkpoint(model, optimizer, args.folder, epoch_num, val_metric_per_epoch,
                        is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1))

writer.close()
print("STOPPING. now running the best model on the validation set", flush=True)
# Load best
restore_best_checkpoint(model, args.folder)
model.eval()
val_probs = []
val_labels = []
for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
    with torch.no_grad():
        batch = _to_gpu(batch)
        batch['v_mask'] = None
        batch['neg'] = False
        batch['neg_img'] = False
        output_dict = model(**batch)
        val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
        val_labels.append(batch['label'].detach().cpu().numpy())
val_labels = np.concatenate(val_labels, 0)
val_probs = np.concatenate(val_probs, 0)
acc = float(np.mean(val_labels == val_probs.argmax(1)))
print("Final val accuracy is {:.3f}".format(acc))
np.save(os.path.join(args.folder, f'valpreds.npy'), val_probs)
np.save(os.path.join(args.folder, f'global_val_loss.npy'), global_val_loss)
np.save(os.path.join(args.folder, f'global_val_acc.npy'), global_val_acc)
np.save(os.path.join(args.folder, f'global_train_loss.npy'),global_train_loss )
np.save(os.path.join(args.folder, f'global_train_acc.npy'), global_train_acc)
