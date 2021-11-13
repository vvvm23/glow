# Glow: Generative Flow with Invertible 1x1 Convolutions [Work in Progress]
> Unofficial PyTorch implementation of "Glow: Generative Flow with Invertible 1x1 Convolutions"

The original paper can be found [here](https://arxiv.org/abs/1807.03039).

The code is based off another implementation found [here](https://github.com/rosinality/glow-pytorch).

This repository contain the Glow model code and associated training scripts.

> This repository is a work in progress. Glow model is complete, but training
> is unstable currently.

## Usage

### Glow Training

Run Glow training using config file `cfg.toml`. Defaults to `config/cifar10.toml`

```
python main.py --cfg-path cfg.toml
```

Other useful flags:
```
--nb-samples        # number of samples to generate when evaluating
--resume            # resume training from specified checkpoint
--seed              # set RNG seed 
--no-save           # disable saving of checkpoints
--no-cuda           # disable the use of CUDA device
--no-amp            # disable the use of automatic mixed precision
--nb-workers        # set number of dataloader workers.
```

## Samples

`TODO: add (nice) sample outputs`

## Checkpoints

`TODO: add pretrained checkpoints`

### TODO:

- [X] Glow Model
- [X] Training script
- [ ] Sampling script
- [ ] PyPi library
- [ ] Add pretrained models / nice samples
- [ ] Gradient checkpoints

### Citations:

**Glow: Generative Flow with Invertible 1x1 Convolutions**
> Diederik P. Kingma, Prafulla Dhariwal
```
@misc{kingma2018glow,
      title={Glow: Generative Flow with Invertible 1x1 Convolutions}, 
      author={Diederik P. Kingma and Prafulla Dhariwal},
      year={2018},
      eprint={1807.03039},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
