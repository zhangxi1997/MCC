# Data

Obtain the dataset by visiting [visualcommonsense.com/download.html](https://visualcommonsense.com/download.html). 
 - Extract the images somewhere. I put them in a different directory, `/home/vcr1/vcr1images` and added a symlink in this (`data`): `ln -s /home/vcr1/vcr1images`
 - Put `train.jsonl`, `val.jsonl`, and `test.jsonl` in here (`data`).
 
You can also put the dataset somewhere else, you'll just need to update `config.py` (in the main directory) accordingly.
```
unzip vcr1annots.zip
```

# Precomputed representations
1. Running CCL requires computed bert representations in this folder. Warning: these files are quite large. You can download them from :
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_train.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_train.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_val.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_val.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_test.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_test.h5`

2. Pre-trained visual representations are generated using code in [Bottom Up Attention](https://github.com/peteanderson80/bottom-up-attention), released by paper [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering, Peter Anderson et al., 2018](https://arxiv.org/abs/1707.07998).

   Please refer to [tab-vcr](https://github.com/Deanplayerljx/tab-vcr/tree/master/data) to download the visual features.
   Note that we use the attribute features instead of the new_tag features.
