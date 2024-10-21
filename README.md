# Adapting Prediction Sets to Distribution Shifts Without Labels

This is the code repository for our paper [Adapting Prediction Sets to Distribution Shifts Without Labels](https://arxiv.org/pdf/2406.01416). **EACP** improves the accuracy of prediction sets under distribution shift by adaptively increasing set-sizes under greater uncertainty and simultaneously updating the base model.  

## Getting started
For computational efficiency, we assume that the inference results on calibration datasets (for example, the ImageNet validation set) have been executed and saved to disk. One way this can be done is as follows:
```python
def save_results(save_path: str, model torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device, file_name: str, n_classes: int):
    """
    Save softmax results and corresponding labels for a given dataset and model.

    save_path: Directory for saving results.
    model: Pre-trained model of interest, e.g. torchvision.models.resnet50
    data_loader: Pytorch dataloader with the desired dataset, e.g. ImageNet Val set.
    device: gpu or cpu device.
    file_name: Name of save file.
    n_classes: Number of classes in this dataset. 
    """
    scores = np.ones((len(data_loader.dataset), n_classes))
    labels = np.ones((len(data_loader.dataset),))
    counter = 0
    # do inference
    with torch.no_grad():
        for batch in tqdm(data_loader):
            scores[counter:counter + batch[0].shape[0], :] = model(batch[0].to(device)).softmax(dim=1).cpu().numpy()
            labels[counter:counter + batch[1].shape[0]] = batch[1].numpy().astype(int)
            counter += batch[0].shape[0]

    print("saving the scores and labels")
    os.makedirs(save_path, exist_ok=True)
    np.savez(save_path + file_name + '.npz', smx=scores, labels=labels)

    acc = (np.argmax(scores, axis=1) == labels).mean() * 100
    print('Validation accuracy: {} %'.format(acc))
```

Run main.py to run adaptation techniques on a desired dataset. 

**Example:** Replicating results on ImageNet-v2. 

```
python main.py --dataset imagenet-v2 --model resnet50 --save-name resnet50_results --lr 0.00025 --cal-path /scratch/ssd004/scratch/kkasa/inference_results/IN1k/imagenet-resnet50.npz --scaling-factor 2 --alpha 0.1 --updates none tta ecp eacl
```

## Acknowledgment

This repository borrows from the [EATA](https://github.com/mr-eggplant/EATA?tab=readme-ov-file), [Tent](https://github.com/DequanWang/tent), and [conformal training](https://github.com/google-deepmind/conformal_training) repos. 
