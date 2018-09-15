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

### beta = 1.0
![image](https://user-images.githubusercontent.com/5235131/45580499-acfe8280-b8cc-11e8-94bf-be33f11e475d.png)

### beta = 4.0
![image](https://user-images.githubusercontent.com/5235131/45580489-84768880-b8cc-11e8-81d0-af7142e7ff2f.png)
