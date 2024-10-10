"""
Returns pytorch dataloaders for selected dataset
"""
import torch
import torchvision
import torchvision.transforms as transforms
import wilds

import datasets
from datasets import imagenet_r_mask, indices_in_1k


def initializeRxrx1Transform() -> transforms.Compose:
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


def get_data(data_name: str, args) -> tuple[torch.utils.data.DataLoader, list]:
    """
    Retrieve the specified dataset and its associated dataloader.

    :param data_name: Name of the dataset to load (e.g., 'imagenet-r', 'imagenet-a', etc.).
    :param args: Arguments containing model specifications and batch size.
    :return: A tuple containing the dataloader for the dataset and an optional mask.
    :raises ValueError: If the specified dataset is not supported.
    """

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
                                         root_dir=r'/data/', )
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

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                             pin_memory=True)

    return dataloader, mask
