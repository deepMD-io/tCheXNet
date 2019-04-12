# tCheXNet: Detecting Pneumothorax on Chest X-Ray Images using Deep Transfer Learning

## Abstract
Pneumothorax (collapsed lung or dropped lung) is an urgent situation and can be life-threatening. It is mostly diagnosed by chest X-ray images. Detecting Pneumothorax on chest X-ray images is challenging, as it requires the expertise of radiologists. Such expertise is time-consuming and expensive to obtain. The recent release of big medical image datasets with labels enabled the Deep Neural Network to be trained to detect diseases autonomously. As the trend moves on, it is expected to foresee more and more medical image big dataset will appear. However, the major limitation is that these datasets have different labels and settings. The know-how to transfer the knowledge learnt from one Deep Neural Network to another, i.e. Deep Transfer Learning, is becoming more and more important. In this study, we explored the use of Deep Transfer Learning to detect Pneumothorax from chest X-ray images. We proposed a model architecture tCheXNet, a Deep Neural Network with 122 layers. Other than training from scratch, we used a training strategy to transfer knowledge learnt in CheXNet to tCheXNet. Experimental results demonstrated that tCheXNet achieved 10% better  in ROC comparing to CheXNet on a testing set verified by three board-certified radiologists using only a training time of 10 epochs.

## Dataset
Please download the CheXpert dataset in the following link
```
https://stanfordmlgroup.github.io/competitions/chexpert/
```
After downloading the dataset, please put it under the directory of chexpert, as follows
```
chexpert/CheXpert-v1.0-small/
```

## Environment
* [Python 3.6](https://www.python.org/downloads/)
* [Keras 2.2.4](https://keras.io)
* [scikit-learn 0.20.3](https://scikit-learn.org/stable/index.html)
* [scikit-image 0.14.2](https://scikit-image.org/)
* [pandas 0.24.2](https://pandas.pydata.org/)
* [Pillow 5.4.1](https://pillow.readthedocs.io/en/stable/)


To setup the environment of tCheXNet, run the following code to prepare a new anaconda environment

```
conda create -n tCheXNet python=3.6 keras scikit-learn scikit-image pandas pillow
```

To activate the environment, use

```
conda activate tCheXNet
```

## Execution
To run tCheXNet, run the hello-world version which detects if there is Pneumothorax on an input image

```
python hello_tCheXNet
```

The expected output is as follows

```
hello_world_images/patient64541_view1_frontal.jpg
Pneumothorax 0.12068107
```
