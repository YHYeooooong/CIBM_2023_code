import timm
import torch
import torch.nn as nn
import torchvision

def define_model(model_name, num_cls, device) :

    num_classes = num_cls

    model = ''
    if model_name == 'CvT-21' :
        model = torch.load('../ref_model/whole_CvT-21-384x384-IN-1k_2class.pt')
        clipping = 0.1

    elif model_name == 'MLP-Mixer-b16' : # image 1k
        model = timm.create_model('mixer_b16_224', pretrained=True, num_classes=num_classes)
        clipping = 0.1

    elif model_name == 'Beit-base-patch16' : # image 1k
        model = timm.create_model('beit_base_patch16_224', pretrained=True, num_classes=num_classes)
        clipping = 0.1

    elif model_name == 'ViT-base-16' : # imagenet 1k (2012)
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        clipping = 0.1

    elif model_name == 'ResNet101' :
        model = timm.create_model('resnet101', pretrained=True, num_classes=num_classes)
        clipping = 0.1

    elif model_name == 'MobileNetV2' :
        model = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=num_classes)
        clipping = 0.1

    elif model_name == 'DenseNet121' :
        model = timm.create_model('densenet121', pretrained=True, num_classes=num_classes)
        clipping = 0.1

    elif model_name == 'EfficientNetB0' :
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        clipping = 0.1

    elif model_name == 'ShuffleNetV2' :
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        num_f = model.fc.in_features
        model.fc = nn.Linear(num_f, num_classes) 
        clipping = 0.1

    elif model_name == 'gmlp_s16' :
        model = timm.create_model('gmlp_s16_224', pretrained=True, num_classes=num_classes)
        clipping = 0.1

    elif model_name == 'resmlp_24' :
        model = timm.create_model('resmlp_24_224', pretrained=True, num_classes=num_classes)
        clipping = 0.1


    model = model.to(device) 
    model = nn.DataParallel(model)

    return model, clipping