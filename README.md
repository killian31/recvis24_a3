## Object recognition and computer vision 2024/2025

### Assignment 3: Sketch image classification

#### Requirements

1. Install PyTorch from <http://pytorch.org>

2. Create a new virtual environment

```bash
pyenv virtualenv 3.11.9 a3
pyenv activate a3
```

3. Run the following command to install additional dependencies

```bash
pip install poetry
poetry install
```

#### Dataset

Download the training/validation/test images from [here](https://www.kaggle.com/competitions/mva-recvis-2024/data). The test image labels are not provided.

#### Training and validating a model

Run the script `main.py` to train your model.

#### Evaluating your model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_file] --model_name [model_name]
```

That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.

#### Logger

We recommend you use an online logger like [Weights and Biases](https://wandb.ai/site/experiment-tracking) to track your experiments. This allows to visualise and compare every experiment you run. In particular, it could come in handy if you use google colab as you might easily loose track of your experiments when your sessions ends.

Note that currently, the code does not support such a logger. It should be pretty straightforward to set it up.

#### Acknowledgments

Adapted from Rob Fergus and Soumith Chintala <https://github.com/soumith/traffic-sign-detection-homework>.<br/>
Origial adaptation done by Gul Varol: <https://github.com/gulvarol><br/>
New Sketch dataset and code adaptation done by Ricardo Garcia and Charles Raude: <https://github.com/rjgpinel>, <http://imagine.enpc.fr/~raudec/>
