# A Computationally Efficient Neural Network Invariant to the Action of Symmetry Subgroups

This repository contains the code associated with the paper "A Computationally Efficient Neural Network Invariant to the Action of Symmetry Subgroups",
submitted to ICML'20.

### Short description
This repository contains implementation of the G-invariant neural networks.
Those networks are able to approximate functions invariant to the action of a given subgroup G
of the symmetric group on the input data.
The key element of the proposed network architecture is a new G-invariant transformation module,
which produces a G-invariant latent representation of the input data.
This latent representation is then processed with a multi-layer perceptron in the network.

### Repository structure

#### Main structure
* [dataset/](dataset/README.md) - contains files associated with preparation and loading the dataset of convex quadrangles
* [experiments/](experiments/README.md) - contains the code to perform neural networks training and evaluation of the models
* [models/](models/README.md) - contains model of the proposed G-invariant neural network and other models used for the comparison
* [utils/](utils/README.md) - contains a bunch of utilities, such as: polynomials definitions, predefined permutation groups, etc.


### Dependencies
* Tensorflow 1.14
* Keras 2.2.5
* NumPy 1.16.4
* cudatoolkit 10.1.168
* Matplotlib 3.1.1


