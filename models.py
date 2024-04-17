import torchvision
import torch


def get_model(data_name, model_name):
    if 'imagenet' in data_name:
        print(f'model: {model_name}')
        if model_name == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True, progress=True)
        elif model_name == 'resnet152':
            model = torchvision.models.resnet152(pretrained=True, progress=True)
        else:
            raise ValueError('Neural Net Model not supported')

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
