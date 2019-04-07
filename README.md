# tCheXNet: Detecting Pneumothorax on Chest X-Ray Images using Deep Transfer Learning

## Abstract
Pneumothorax (collapsed lung or dropped lung) is an urgent situation and can be life-threatening. It is mostly diagnosed by chest X-ray images. Detecting Pneumothorax on chest X-ray images is challenging, as it requires the expertise of radiologists. Such expertise is time-consuming and expensive to obtain. The recent release of big medical image datasets with labels enabled the Deep Neural Network to be trained to detect diseases autonomously. As the trend moves on, it is expected to foresee more and more medical image big dataset will appear. However, the major limitation is that these datasets have different labels and settings. The know-how to transfer the knowledge learnt from one Deep Neural Network to another, i.e. Deep Transfer Learning, is becoming more and more important. In this study, we explored the use of Deep Transfer Learning to detect Pneumothorax from chest X-ray images. We proposed a model architecture tCheXNet, a Deep Neural Network with 122 layers. Other than training from scratch, we used a training strategy to transfer knowledge learnt in CheXNet to tCheXNet. Experimental results demonstrated that tCheXNet achieved 10% better  in ROC comparing to CheXNet on a testing set verified by three board-certified radiologists using only a training time of 10 epochs.

## Dataset
Please download the CheXpert dataset in the following link
```
https://stanfordmlgroup.github.io/competitions/chexpert/
```

## Environment
* [Keras 2.2.4](https://keras.io)
* [scikit-learn 0.20.3](http://scikit-learn.org/stable/index.html)

To run tCheXNet, run the following code to prepare a new anaconda environment

```
conda create -n tCheXNet python=2.7 keras scikit-learn pandas PIL
```

To activate the environment, use

```
conda activate tCheXNet
```
