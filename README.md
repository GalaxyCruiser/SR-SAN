# SR-SAN
Implementation for the paper entitled "[Session-based Recommendation with Self-Attention Networks](https://arxiv.org/abs/2102.01922)"

You can download the datasets which used in our paper from the following links. Then put them in the folder `datasets/`:

- YOOCHOOSE: <http://2015.recsyschallenge.com/challenge.html>

- DIGINETICA: <http://cikm2016.cs.iupui.edu/cikm-cup>

After you download the YOOCHOOSE dataset, add headline with `session_id,timestamp,item_id,category` in the yoochoose-clicks.dat. 

## Usage

Run the file  `datasets/preprocess.py` to preprocess the data before train the model.

For example: `cd datasets; python preprocess.py --dataset=yoochoose`

```bash
usage: preprocess.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  dataset name: diginetica/yoochoose
```

Then you can run the file `python main.py --dataset=yoochoose1_64`  to train the model.

You can also change other parameters according to the usage:

```bash
usage: main.py [-h] [--dataset DATASET] [--batchSize BATCHSIZE]
               [--hiddenSize HIDDENSIZE] [--nhead NHEAD] [--layer LAYER]
               [--feedforward FEEDFORWARD] [--epoch EPOCH] [--lr LR]
               [--lr_dc LR_DC] [--lr_dc_step LR_DC_STEP] [--l2 L2]
               [--patience PATIENCE] [--validation] 
               [--valid_portion VALID_PORTION]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset name:
                        diginetica/yoochoose1_64
  --batchSize BATCHSIZE
                        input batch size
  --hiddenSize HIDDENSIZE
                        hidden state size
  --nhead NHEAD
                        the number of heads of multi-head attention
  --layer LAYER
                        number of SAN layers 
  --feedforward FEEDFORWARD
                        the multipler of hidden state size
  --epoch EPOCH         the number of epochs to train for
  --lr LR               learning rate
  --lr_dc LR_DC         learning rate decay rate
  --lr_dc_step LR_DC_STEP
                        the number of epochs after which the learning rate
                        decay
  --l2 L2               l2 penalty
  --patience PATIENCE   the number of epoch to wait before early stop
  --validation          validation
  --valid_portion VALID_PORTION
                        split the portion of training set as validation set
```

## Requirements

- Python 3
- PyTorch 1.2
