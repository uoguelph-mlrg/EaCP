"""
Script for returning pre-trained neural network models
"""
import torchvision
import torch
import timm


def get_model(data_name: str, model_name: str) -> torch.nn.Module:
    if 'imagenet' in data_name:
        print(f'model: {model_name}')
        if model_name == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True, progress=True)
        elif model_name == 'resnet152':
            model = torchvision.models.resnet152(pretrained=True, progress=True)
        elif model_name == 'deit3B':
            model = timm.create_model(
                'deit3_base_patch16_224', pretrained=True)
            print('Loaded Deit3-B')
        elif model_name == 'deit3S':
            model = timm.create_model(
                'deit3_small_patch16_224', pretrained=True)
            print('Loaded Deit3-S')
        elif model_name == 'vitB':
            model = timm.create_model(
                'vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
            print('Loaded ViT-B')
        elif model_name == 'vitS':
            model = timm.create_model(
                'vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
            print('Loaded ViT-S')
        else:
            raise ValueError(
                'Neural Net Model not supported. Choose from: resnet50, resnet152, deit3B, deit3S, vitB, vitS')

    elif data_name == 'rxrx1':
        weights = torch.load(r'/scratch/ssd004/scratch/kkasa/trained_models/rxrx1/rxrx1_seed:0_epoch:best_model.pth')[
            'algorithm']

        for key in list(weights.keys()):
            weights[key.removeprefix('model.')] = weights.pop(key)

        model = torchvision.models.resnet50(pretrained=True, progress=True)

        model.fc = torch.nn.Linear(model.fc.in_features, 1139)

        model.load_state_dict(weights, strict=True)
    elif data_name == 'iwildcam':
        weights = torch.load(r'/scratch/ssd004/scratch/kkasa/trained_models/iwildcam/best_model.pth')['algorithm']

        for key in list(weights.keys()):
            weights[key.removeprefix('model.')] = weights.pop(key)

        model = torchvision.models.resnet50(pretrained=True, progress=True)

        model.fc = torch.nn.Linear(model.fc.in_features, 182)

        model.load_state_dict(weights, strict=True)
    elif data_name == 'fmow':
        weights = torch.load(r'/scratch/ssd004/scratch/kkasa/trained_models/fmow/fmow_seed:0_epoch:best_model.pth')[
            'algorithm']
        for key in list(weights.keys()):
            weights[key.removeprefix('model.')] = weights.pop(key)

        model = torchvision.models.densenet121(pretrained=False)

        model.classifier = torch.nn.Linear(model.classifier.in_features, 62)

        model.load_state_dict(weights, strict=True)
    else:
        raise ValueError('Dataset not supported')

    return model
