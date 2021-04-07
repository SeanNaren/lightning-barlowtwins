# Lightning Barlow Twins

<p align="center">
  <img src="https://github.com/SeanNaren/lightning-barlowtwins/blob/master/diagram.png?raw=true" width="700">
</p>

This is a [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) port of the [Barlow Twins implementation](https://github.com/facebookresearch/barlowtwins) release by Facebook Research.

Hyper-parameters have been set up based on the README found in the PyTorch Barlow Twins implementation.


### Usage
```
pip install -r requirements.txt
```

### Training
Train your own backbone on the ImageNet dataset (look inside [train.py](./train.py) to use CIFAR10 dataset instead). Requires you to have ImageNet downloaded, instructions [here](https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/datamodules/imagenet_datamodule.py#L132).

Have a look at the [PyTorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html) documentation for more flags to enable.

*Note that the Facebook released ResNet pre-trained weights is trained with a total effective batch size of `2048`. Modify the batch size and the number of GPUs based on compute, with a helpful table [here](https://github.com/facebookresearch/barlowtwins/issues/7#issuecomment-806449220) which can be used to estimate training times.* 

```
python train.py --gpus 8 --batch_size 256
```


### Linear Evaluation

Run linear evaluation on the [pre-trained ResNet weights](https://github.com/facebookresearch/barlowtwins#pretrained-model) provided by Facebook Research. Requires you to have ImageNet downloaded, instructions [here](https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/datamodules/imagenet_datamodule.py#L132). Look inside [evaluate.py](./evaluate.py) to swap to CIFAR10.

```
python evaluate.py --gpus 8 --batch_size 256
```

Run linear evaluation on a trained BarlowTwins `LightningModule`:
```
python evaluate.py --gpus 8 --batch_size 256 --model_path /path/to/lightning_logs/model.ckpt
```

To customize parameters, look at [train.py](./train.py) and [evaluate.py](./evaluate.py) respectively.

### License

Unfortunately I've had to duplicate the licence from the original Barlow Twins Pytorch Implementation, which is a restrictive non-commercial licence :(

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### Citations

Thanks to [Phoeby](https://www.phoebynaren.com) for the illustration :)

```
@article{zbontar2021barlow,
  title={Barlow Twins: Self-Supervised Learning via Redundancy Reduction},
  author={Zbontar, Jure and Jing, Li and Misra, Ishan and LeCun, Yann and Deny, St{\'e}phane},
  journal={arXiv preprint arXiv:2103.03230},
  year={2021}
}
```