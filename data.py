import torch
import torchvision
import torchvision.transforms as transforms
import wilds
from wilds.common.data_loaders import get_eval_loader

import datasets
from datasets import imagenet_r_mask, indices_in_1k


def initializeRxrx1Transform():
    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return torchvision.transforms.functional.normalize(x, mean, std)

    t_standardize = transforms.Lambda(lambda x: standardize(x))

    transforms_ls = [
        transforms.ToTensor(),
        t_standardize,
    ]
    return transforms.Compose(transforms_ls)


def get_data(data_name, args):
    if data_name == 'imagenet-r':
        dataset = datasets.INr(args.model)
        mask = imagenet_r_mask
    elif data_name == 'imagenet-a':
        dataset = datasets.INa(args.model)
        mask = indices_in_1k
    elif data_name == 'imagenet-v2':
        dataset = datasets.INv2(args.model)
        mask = None
    elif data_name == 'imagenet-c':
        dataset = datasets.INc(args.corruption, args.severity, args.model)
        mask = None
    elif data_name == 'rxrx1':
        transform = initializeRxrx1Transform()
        base_dataset = wilds.get_dataset('rxrx1', download=False,
                                         root_dir=r'/scratch/ssd004/scratch/kkasa/code/OnlineCP/data/', )
        dataset = base_dataset.get_subset(
            "test",
            transform=transform
        )
        mask = None
    elif data_name == 'iwildcam':
        base_dataset = wilds.get_dataset('iwildcam', download=False, root_dir='/datasets/iWildCam/', version='2.0')
        dataset = base_dataset.get_subset(
            "val",
            transform=transforms.Compose(
                [transforms.Resize((448, 448)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            ),
        )
        mask = None
    elif data_name == 'fmow':
        # Get the datasets
        base_dataset = wilds.get_dataset('fmow', download=False, root_dir='/datasets/domainbed/')
        dataset = base_dataset.get_subset(
            "test",
            transform=transforms.Compose(
                [transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            ),
        )
        mask = None
    else:
        raise ValueError('Dataset not supported')

    if 'imagenet' in data_name:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                 pin_memory=True)
    elif data_name in ['rxrx1', 'iwildcam', 'fmow']:
        # dataloader = get_eval_loader("standard", dataset, batch_size=args.batch_size, num_workers=4)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                 pin_memory=True)

    else:
        raise ValueError('Dataset not supported')

    return dataloader, mask
