# beta-vae
Implementation of beta-VAE with TensorFlow.

https://arxiv.org/pdf/1606.05579.pdf

## dependencies
- Python3
- tensorflow
- opencv-python

## train
```
$ python train.py
```

## test trained models
```
$ python test.py --model saved_models/path/models.ckpt --config saved_models/path/constants.json
```

## test each element of the latent variable
```
$ python test_latent.py --model saved_models/path/models.ckpt --config saved_models/path/constants.json
```
