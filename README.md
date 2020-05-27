# Play with MNIST

## Prepare `Pytorch`

I created this MNIST training scripts.
I checked the scripts with Miniconda3.

Download Miniconda3 installer (Python 3.7 64-bit) from [URL](https://docs.conda.io/en/latest/miniconda.html).

I prepared `requirements.yaml`.
You can make the same environment as I did.

```shell
$ conda env create -f requirements.yaml
$ conda activate pytorch
```

## MNIST

```shell
$ python mnist.py
```


## Fashion MNIST

```shell
$ python fashon_mnist.py
```