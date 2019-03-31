This repository contains the implementation of the RBFolutional layer described in the paper 

Matěj Kocián, Martin Pilát: "Using Local Convolutional Units to Defend Against
Adversarial Examples". In: _2019 International Joint Conference on Neural Networks (IJCNN 2019)_. IEEE, 2019.

The `custom_layers.py` file contains the definition of the RBFolutional layer, the `simplenet.py` and `simplenet_rbf.py` show the definition and traning of the SimpleNet and SimleNet with RBFolutional first layer models respectively on the CIFAR10 dataset. The `models_mnist.py` contains the models used in the experiments on the smaller datasets.

The file `misc.py` contains code snippets that were used to generate some of the plots presented in the paper.

Some of the expeximents on the MNIST dataset were described and performed in the Master Thesis of the first author and the sources for them are available as the [attachment of that thesis](http://hdl.handle.net/20.500.11956/99233).



