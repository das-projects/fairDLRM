### Deep learning project seed
Use this seed to start new deep learning / ML projects.

- Built in setup.py
- Built in requirements
- Examples with MNIST
- Badges
- Bibtex

#### Goals  
The goal of this seed is to structure ML paper-code the same so that work can easily be extended and replicated.   

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT  
 
---

<div align="center">    
 
# Your Project Name     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```
# Fairness in Machine Learning

This project demonstrates how make fair machine learning models.

![Fair training](https://github.com/equialgo/fairness-in-ml/raw/master/images/training.gif)


## Notebooks

- `fairness-in-ml.ipynb`: keras & TensorFlow implementation of [Towards fairness in ML with adversarial networks](https://blog.godatadriven.com/fairness-in-ml).
- `fairness-in-torch.ipynb`: PyTorch implementation of [Fairness in Machine Learning with PyTorch](http://blog.godatadriven.com/fairness-in-pytorch).
- `playground/*`: Various experiments.


## Getting started

This repo uses conda's virtual environment for __Python 3__.

Install (mini)conda if not yet installed.

For MacOS:
```sh
$ wget http://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh -O miniconda.sh
$ chmod +x miniconda.sh
$ ./miniconda.sh -b
```

`cd` into this directory and create the  conda virtual environment for Python 3 from `environment.yml`:
```sh
$ conda env create -f environment.yml
```

Activate the virtual environment:
```sh
$ source activate fairness-in-ml
```

Install the `fairness` library:

```sh
$ python setup.py develop
```


## Contributing

If you have applied these models to a different dataset or implemented any other fair models, consider submitting a Pull Request!


### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
