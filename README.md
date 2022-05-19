# Ellen: Evolutionary Learning to Learn Feature Interactions for Recommender Systems

This repository is the official implementation of Ellen: Evolutionary Learning to Learn Feature Interactions for
Recommender Systems submitted to NeurIPS2022.

# Requirements

* Python 3
* Pytorch 1.7+
* CUDA 10.0+ (For GPU)


# Usage

## Dataset

We use three public real-world datasets(Avazu, Criteo, Huawei) and one private dataset in our experiments.</br>
**Avazu** The raw dataset can be downloaded from https://www.kaggle.com/c/avazu-ctr-prediction/overview. If you want to
know how to preprocess the data, please refer to `./data/avazuPreprocess.py`</br>
**Criteo** The raw dataset can be downloaded from https://www.kaggle.com/c/criteo-display-ad-challenge/data. If you want
to know how to preprocess the data, please refer to `./data/criteoPreprocess.py`</br>
**Huawei** The raw dataset can be downloaded
from https://www.kaggle.com/louischen7/2020-digix-advertisement-ctr-prediction. If you want to know how to preprocess
the data, please refer to `./data/huaweiPreprocess.py`</br>
**FinTech** FinTech is collected from a high-tech bank in China and ranges from January 1, 2020, to December 1, 2021. It
totally has 2257 feature fields. We filter it by preliminary screening and remain 36 feature fields. The dataset
contains user basic features, e.g., ages and genders, user financial features, e.g, asset conditions and risk appetites,
and user behavioral features, e.g., investments and consumptions. We put part of the processed data in
the `fintech_dataset` file folder. It contains 100,000 pieces of data and 26 feature fields. To protect data privacy, all the user information has been desensitized and the feature fields have been anonymized.
and we will public the complete FinTech dataset after our paper acceptance. If you want to know how to preprocess the data, please refer
to `./data/fintechPreprocess.py`.</br>



## Example

Before you run the preprocess, plese run the `mkdir.sh`
Then you will get the following three folder structures in the same level directory of the project:

```
criteo
├── bucket
├── feature_map
└── processed

avazu
└── processed

huawei
└── processed

fintech
├── bucket
├── feature_map
└── processed
```

The preprocessing methods of Criteo and Fintech are different from other datasets,so the directory structures are also
different.

Then you can put the downloaded file into the corresponding folder and run the corresponding preprocessing python file.
(Eg. Criteo: put the `train.txt` into the `criteo` folder, then run the `criteoPreprocess.py`)

Here's how to run the training.

```
nohup python -u train.py --dataset=criteo --gpu=0 --nas=1 --fis=1 --mlp=1 --retrain_nas=1 --retrain_fis=1 > criteo.log 2>&1 &
```

You can run the `run.sh`

Then you can see the output log in the `result` folder like this

```
INFO:root:--------------------------------------------------
INFO:root:Tue May 11 11:58:17 2021
INFO:root:Namespace(dataset='huawei', fis=1, gpu=0, mlp=1, model='Ellen', nas=1, retrain_fis=1, retrain_nas=1)
INFO:root:{'general': {'batch_size': 2000, 'data': -1, 'epochs': 1, 'validation_split': 0.11, 'net_optim_lr': 0.001, 'embedding_size': 15}}
INFO:root:nas period param:
INFO:root:{'momentum': 0.9, 'net_optim_lr': 0.025, 'net_optim_weight_decay': 0.0003, 'structure_optim_lr': 0.0003, 'structure_optim_weight_decay': 0.001, 'interaction_fc_output_dim': 1, 'validation_split': 0.5}
INFO:root:model search period start
INFO:root:The total number of parameters：1343770
INFO:root:Epoch 1/1
INFO:root:147s - loss:  0.3355 - binary_crossentropy:  0.3323 - auc:  0.5363 - val_binary_crossentropy:  0.1774 - val_auc:  0.6177
INFO:root:model search period end
INFO:root:model search period cost:147
INFO:root:fis period param:
INFO:root:{'c': 0.5, 'mu': 0.8, 'gRDA_optim_lr': 0.001, 'net_optim_lr': 0.001, 'interaction_fc_output_dim': 1, 'validation_split': 0.11}
INFO:root:feature interaction search period start
INFO:root:The total number of parameters：1431400
INFO:root:Epoch 1/1
INFO:root:66s - loss:  0.9290 - binary_crossentropy:  0.9087 - auc:  0.5415 - val_binary_crossentropy:  0.3707 - val_auc:  0.5451
INFO:root:feature interaction search period end
INFO:root:feature interaction search period cost:66
INFO:root:pre period param:
INFO:root:{'interaction_fc_output_dim': 15, 'dnn_hidden_units': [400, 400]}
INFO:root:[0 0 3 0 0 1 0 0 0 1 0 0 0 0 0 0 0 1 1 0 0 0 2 0 0 0 0 3 2 1 0 0 0 1 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 2 0 0 0 0 2 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 1 0 0 0 0 2 0 0 1 0 0 0 3 0 0 0 0 0
 1 1 1 3 0 0 3 3 0 0 0 0 3 3 3 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 3 1 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 1 0
 0 0 0 2 0 0 0 1 0 0 0 1 0 3 3 3 3 0 0 3 3 1 0 0 3 1 3 3 0 0 0 1 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 3 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 3 3 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 0 0 0 0 3 0
 0 0 1 0 1 1 1 3 0 0 1 1 1 0 1 0 2 3 1 0 0 3 3 0 0 0 0 0 0 0 0 0 0 0 1 0 0
 0 0 1 1 3 0 0 0 3 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 3 3 1 0 0 0 3 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 3 1 0 0 0 0 2 0 0 0 1 3 0 0 0 1 0 0 0 0 3 3 3 0 0 0 1 0 0
 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 2 0 3 1 0 0 0 1 0 0 0 0 1 3 1 0 0 0 3 3 1
 0 0 0 1 0 0 0 0 3 2 1 0 0 0 1 1 0 0 1 1 0 0 0 0 1 3 1 0 0 0 1 0 0 1 1 1 0
 1 1 3 1 1 0 0 0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 2 0 0 0 0 0 0 1 1 0 0 0 0 1 1
 0 0 0 0 1 3 1 0 0 0 1 1 0 3 3 3 2 3 0 1 1 1 0 0 0 3 3 3 0 0 0 1 0 0 1 1 0
 0 0 0 2 0 1 1 3 0 0 0 1 3 1 3 0 0 0 1 2 3 3 1 1 1 3 1 1 1 3 0 1 0 2 0 0 2
 0 3 1]
INFO:root:Parameter containing:
tensor([0.5102, 0.7720, 0.7710, 0.8404, 0.8226, 0.8326, 0.8675, 0.8443, 0.8209,
        0.8288, 0.8707, 0.8353, 0.8554, 0.8222, 0.8560, 0.8355, 0.8569, 0.8528,
        0.8503, 0.8367, 0.8429, 0.8267, 0.8533, 0.8378, 0.8503, 0.8388, 0.8545,
        0.8351, 0.8524, 0.8349, 0.8547, 0.8499, 0.8383, 0.8308, 0.8347],
       requires_grad=True)
INFO:root:Parameter containing:
tensor([-0.1864, -0.1833,  0.4993, -0.0406,  0.0146,  0.4988,  0.0996, -0.0192,
         0.0310,  0.4998,  0.0571,  0.1298, -0.1330,  0.1497,  0.0015,  0.1286,
         0.1441,  0.4992,  0.4988,  0.0517,  0.0426,  0.1379,  0.5244,  0.1294,
         0.0921,  0.1413,  0.1089,  0.4982,  0.5631,  0.4980,  0.0757,  0.1115,
         0.0417,  0.4975, -0.2516,  0.0623, -0.1145, -0.0590,  0.1073,  0.0252,
        -0.0936, -0.0415,  0.1038, -0.0141,  0.0557, -0.2045,  0.0759, -0.0699,
         0.0539,  0.0683,  0.1073,  0.1044, -0.0222, -0.0279,  0.0624,  0.1058,
         0.0535,  0.0180,  0.0653,  0.0330,  0.4998,  0.5259,  0.1057,  0.0000,
         0.0382, -0.0308,  0.5462,  0.0636, -0.1128, -0.0573,  0.1071,  0.0265,
        -0.0933, -0.0412,  0.1053, -0.0140,  0.0573, -0.2024,  0.0756, -0.0700,
         0.0542,  0.0689,  0.1094,  0.1058, -0.0213, -0.0274,  0.0634,  0.1071,
         0.0547,  0.0183,  0.0666,  0.0340,  0.4986,  0.4992,  0.1060,  0.0000,
         0.0371, -0.0304,  0.5479,  0.2065,  0.2584,  0.4991,  0.3380,  0.2245,
         0.2745,  0.4987,  0.3009,  0.3660,  0.1151,  0.3832,  0.2479,  0.4989,
         0.4993,  0.4991,  0.4996,  0.2936,  0.2887,  0.4981,  0.4980,  0.3640,
         0.3317,  0.3747,  0.3457,  0.4998,  0.4982,  0.4991,  0.3162,  0.3498,
         0.2849,  0.4986,  0.0870,  0.2490,  0.1698,  0.0497,  0.1023,  0.2468,
         0.1293,  0.1992, -0.0608,  0.2185,  0.0746,  0.1982,  0.2130,  0.2489,
         0.2459,  0.1221,  0.1146,  0.2068,  0.2494,  0.1990,  0.1623,  0.2100,
         0.1787,  0.4987,  0.4991,  0.2471,  0.1459,  0.1828,  0.1145,  0.4995,
         0.3020,  0.2253,  0.1065,  0.1569,  0.2986,  0.1849,  0.2537, -0.0052,
         0.2730,  0.1294,  0.2521,  0.2660,  0.3022,  0.2991,  0.1754,  0.1718,
         0.2617,  0.3024,  0.2510,  0.2183,  0.2638,  0.2312,  0.4986,  0.4992,
         0.2994,  0.1997,  0.2366,  0.1699,  0.5263,  0.3779,  0.2687,  0.3163,
         0.4997,  0.3417,  0.4059,  0.1591,  0.4987,  0.2914,  0.4983,  0.4996,
         0.4983,  0.4984,  0.3335,  0.3289,  0.4993,  0.4984,  0.4978,  0.3729,
         0.4133,  0.4988,  0.4985,  0.4994,  0.4986,  0.3562,  0.3892,  0.3270,
         0.4987,  0.1895,  0.2404,  0.3755,  0.2657,  0.3334,  0.0798,  0.3511,
         0.2152,  0.3331,  0.3444,  0.3811,  0.3769,  0.2602,  0.2535,  0.3392,
         0.3803,  0.3314,  0.2998,  0.3428,  0.3128,  0.4981,  0.4996,  0.3789,
         0.2829,  0.3168,  0.2513,  0.5130,  0.1213,  0.2667,  0.1503,  0.2187,
        -0.0397,  0.2384,  0.0931,  0.2173,  0.2310,  0.2692,  0.2651,  0.1402,
         0.1363,  0.2263,  0.2683,  0.2175,  0.1831,  0.2302,  0.1990,  0.4995,
         0.4989,  0.2674,  0.1655,  0.2011,  0.1332,  0.5284,  0.3126,  0.2004,
         0.2695,  0.0093,  0.2877,  0.1463,  0.2666,  0.2805,  0.3163,  0.3132,
         0.1932,  0.1862,  0.2751,  0.3172,  0.2664,  0.2324,  0.2786,  0.2485,
         0.4995,  0.5083,  0.3158,  0.2148,  0.2519,  0.1847,  0.4994,  0.3399,
         0.4035,  0.1556,  0.4985,  0.2883,  0.4979,  0.4987,  0.4991,  0.4990,
         0.3312,  0.3275,  0.4998,  0.4985,  0.4997,  0.3707,  0.4997,  0.3829,
         0.4955,  0.4997,  0.4988,  0.3542,  0.3879,  0.4992,  0.4989,  0.2944,
         0.0380,  0.3127,  0.1729,  0.2936,  0.3064,  0.3428,  0.3381,  0.2202,
         0.2138,  0.3023,  0.4993,  0.2929,  0.2605,  0.3042,  0.2745,  0.4995,
         0.4995,  0.4991,  0.2429,  0.2782,  0.2113,  0.4979,  0.1087,  0.3795,
         0.2438,  0.3609,  0.3727,  0.4064,  0.4026,  0.2874,  0.2824,  0.3672,
         0.5000,  0.3602,  0.3268,  0.3690,  0.3413,  0.4990,  0.4981,  0.4991,
         0.3102,  0.3449,  0.2792,  0.4987,  0.1284, -0.0170,  0.1077,  0.1224,
         0.1601,  0.1571,  0.0301,  0.0228,  0.1160,  0.1595,  0.1092,  0.0735,
         0.1204,  0.0880,  0.4996,  0.4995,  0.1602,  0.0539,  0.0914,  0.0216,
         0.5504,  0.2626,  0.3784,  0.3905,  0.4980,  0.4993,  0.3063,  0.3011,
         0.3848,  0.4986,  0.3766,  0.3447,  0.3876,  0.3602,  0.4987,  0.4992,
         0.4994,  0.3278,  0.3622,  0.2997,  0.4989,  0.2417,  0.2548,  0.2909,
         0.2886,  0.1662,  0.1595,  0.2490,  0.2904,  0.2406,  0.2075,  0.2514,
         0.2218,  0.4998,  0.4990,  0.2899,  0.1876,  0.2260,  0.1570,  0.5268,
         0.3701,  0.4990,  0.4993,  0.2868,  0.2802,  0.3666,  0.4983,  0.3595,
         0.3266,  0.3679,  0.3407,  0.4997,  0.4984,  0.4982,  0.3090,  0.3439,
         0.2791,  0.4978,  0.4987,  0.4979,  0.2994,  0.2940,  0.3773,  0.4979,
         0.3708,  0.3400,  0.3819,  0.3516,  0.4998,  0.5042,  0.4986,  0.3228,
         0.3561,  0.2924,  0.4986,  0.4980,  0.3354,  0.3297,  0.4982,  0.4987,
         0.4046,  0.3747,  0.4160,  0.3877,  0.4985,  0.4982,  0.4994,  0.3576,
         0.3907,  0.3278,  0.4986,  0.3333,  0.3259,  0.4983,  0.4985,  0.4982,
         0.3692,  0.4994,  0.4984,  0.4993,  0.4991,  0.4994,  0.3544,  0.3881,
         0.3256,  0.4992,  0.2067,  0.2936,  0.3353,  0.2875,  0.2540,  0.2976,
         0.2675,  0.4991,  0.4983,  0.3348,  0.2354,  0.2715,  0.2033,  0.5222,
         0.2889,  0.3282,  0.2796,  0.2451,  0.2919,  0.2599,  0.4985,  0.4983,
         0.3288,  0.2290,  0.2642,  0.1993,  0.4982,  0.4993,  0.3651,  0.3336,
         0.3754,  0.3473,  0.4993,  0.4982,  0.4981,  0.3160,  0.3504,  0.2879,
         0.4980,  0.4992,  0.3728,  0.4983,  0.4989,  0.4984,  0.5003,  0.4983,
         0.3567,  0.4991,  0.4981,  0.4981,  0.3257,  0.3693,  0.3404,  0.4979,
         0.4980,  0.4979,  0.3093,  0.3428,  0.2782,  0.4990,  0.3358,  0.3066,
         0.4992,  0.4991,  0.3718,  0.2755,  0.3113,  0.2442,  0.5138,  0.3508,
         0.4993,  0.4994,  0.4991,  0.3203,  0.3528,  0.2887,  0.4996,  0.4980,
         0.4990,  0.4980,  0.2901,  0.3232,  0.2606,  0.4982,  0.4951,  0.4984,
         0.4996,  0.4985,  0.4983,  0.4992,  0.4992,  0.4978,  0.4978,  0.4989,
         0.4989,  0.3554,  0.4990,  0.3263,  0.5037,  0.2924,  0.2278,  0.5189,
         0.2626,  0.4995,  0.4987], requires_grad=True)
INFO:root:predict period start
INFO:root:The total number of parameters：5162585
INFO:root:Epoch 1/1
INFO:root:105s - loss:  0.1965 - binary_crossentropy:  0.1951 - auc:  0.6559 - val_binary_crossentropy:  0.4943 - val_auc:  0.7054
INFO:root:test LogLoss:0.4952
INFO:root:test AUC:0.6925
INFO:root:predict period end
INFO:root:predict period cost:108
``` 





