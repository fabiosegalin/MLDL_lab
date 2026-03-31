# Lab - Setup a project from scratch

## Current structure

- `dataset/`: dataset download, transforms, and dataloaders
- `models/`: model definitions
- `utils/`: helper functions
- `train.py`: training entrypoint
- `eval.py`: evaluation entrypoint
- `checkpoints/`: saved model weights

## Tiny ImageNet dataloader and training

The recovered lab content is now split across the template:

- `dataset/tiny_imagenet.py`: dataset download, transforms, train and validation loaders
- `models/custom_net.py`: simple convolutional neural network
- `train.py`: training + validation loop, checkpoint saving
- `eval.py`: load the saved checkpoint and evaluate on validation data

Run:

```bash
python train.py
```

This will:

- download Tiny ImageNet into `data/`
- build train and validation dataloaders
- print dataset stats
- show a few sample images
- train `CustomNet`
- save the best model into `checkpoints/best_custom_net.pt`

To evaluate the saved model later:

```bash
python eval.py
```

Note: your original notebook reorganized the `val/` folder so that `ImageFolder` could read it. In this project, that folder rewrite is not needed because `dataset/tiny_imagenet.py` reads `val_annotations.txt` directly.
