# Classification of Breast Cancer from Mammograms: Do Vision Transformers and MLP-Mixers Outperform CNNs?

## 1. Abstract
Recently, various computer-aid detection/diagnosis systems have been studied to support the
examination of breast cancer from mammograms. Several studies have proposed deep learningbased methods for breast cancer classification; however, the advantages and limitations of
different types of architectures on this task are still not clear. Even though a few studies attempted
to compare the performance of breast cancer classification models based on mammograms,
they suffered from the following limitations which hamper fair performance evaluation: (1)
each study applied its own data augmentation methods, (2) dataset for training and testing was
not consistent, and (3) the objective of a classification task was not identical. Therefore, in
this paper, we conducted a comparative study of up-to-date deep learning-based breast cancer
classification models, including CNNs, vision transformers, and MLP-Mixers, using CBISDDSM mammogram dataset under the same experimental setting. Through the experiments,
we evaluated the performance of the models in terms of effectiveness and efficiency. The
experimental results revealed that lightweight models performed better than complex networks
in the mammogram domain in particular. Specifically, the MobileViT architecture achieved the
highest accuracy of 0.7477 in a binary classification task and 0.7122 in a 4-class classification
task. In addition, MobileViT showed the best efficiency in terms of accuracy/parameter trade-off.
Traditional CNNs and vision transformers with a large number of parameters and layers tended to
show unstable performances. We expect that the performance analysis in this study will provide
various design considerations of deep learning networks for the diagnostics of breast cancers
using mammograms.

## 2. Requirement
Here's the environment we used for our experiments :
```
Python == 3.8.18
Pytorch == 1.11.0
timm == 0.6.5
```
## 3. Training settings

|  Hyperparameter | Value |
| ------------- | ------------- |
| Optimizer | AdamW  |
| Epoch  | 300 |
| Learning rate  | 0.0001 |
| Warm up (epoch) | 30 |
| Batch size  | 64 |

## 4. Dataset and implementation example
Placed your dataset at ../dataset folder

Training on CBIS-DDSM, with ImageNet-1k models with Avanced augmentation option using cosine scheduler 
```
python train.py --model 'ResNet101' --aug 'Advanced' --LR 'w_sch' --iter '1' --num_cls '2' --model_weight '1k'
```

## 5. MobileViT models
We used ml-cvent code for MobileViT-S, MobileViT-XS, and MobileViT-XXS explements  -->
[ml-cvnet](https://github.com/apple/ml-cvnets/blob/7be93d3debd45c240a058e3f34a9e88d33c07a7d/docs/source/en/models/classification/README-classification-tutorial.md)

## 6. CvT models\
CvT pre-trained model weight can be download from ~~~

