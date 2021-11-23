# Glow: Generative Flow with Invertible 1x1 Convolutions [Work in Progress]
> Unofficial PyTorch implementation of "Glow: Generative Flow with Invertible 1x1 Convolutions"

The original paper can be found [here](https://arxiv.org/abs/1807.03039).

The code is based off another implementation found [here](https://github.com/rosinality/glow-pytorch).

This repository contain the Glow model code and associated training / sampling scripts.

> This repository is a work in progress. Default parameters may not be optimal!

## Usage

### Glow Training

Run Glow training using config file `cfg.toml`. Defaults to `config/cifar10.toml`

```
python main.py --cfg-path cfg.toml --no-amp
```
> currently recommend NOT using automatic mixed precision (AMP)

Other useful flags:
```
--nb-samples            # number of samples to generate when evaluating [16]
--resume                # resume training from specified checkpoint 
--seed                  # set RNG seed 
--no-save               # disable saving of checkpoints [False]
--no-cuda               # disable the use of CUDA device [False]
--no-amp                # disable the use of automatic mixed precision [False]
--nb-workers            # set number of dataloader workers. [4]
--no-grad-checkpoint    # don't checkpoint gradients [False]
--temperature           # set temperature when sampling at evaluation [0.7]
```

### Glow Sampling
Run Glow sampling using config file `cfg.toml` from checkpoint `checkpoint.pt` using sample mode `mode`:
```
python main.py --sample --sample-mode mode --resume checkpoint.pt --cfg-path cfg.toml --no-amp
```
Other flags from training will also work during sampling.

The sampling modes are:
- `normal`: samples random latent and displays corresponding samples, saving to `sample.jpg`
- `vtemp`: samples random latent and varies temperature, dumping samples
  `samples-vtemp/`
- `interpolate`: computes latent of dataset items, then linearly interpolates
  between them, dumping samples in `samples-interpolate/` 

## Samples

`TODO: add (nice) sample outputs`

## Checkpoints

`TODO: add pretrained checkpoints`

### TODO:

- [X] Glow Model
- [X] Training script
- [X] Sampling script
- [X] Gradient checkpoints
- [ ] PyPi library
- [ ] Add pretrained models / nice samples

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
