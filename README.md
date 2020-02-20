# A Computationally Efficient Neural Network Invariant to the Action of Symmetry Subgroups

This repository contains the code associated with the paper ["A Computationally Efficient Neural Network Invariant to the Action of Symmetry Subgroups"](https://arxiv.org/abs/2002.07528).

### Short description
This repository contains implementation of the G-invariant neural networks.
Those networks are able to approximate functions invariant to the action of a given subgroup G
of the symmetric group on the input data.
The key element of the proposed network architecture is a new G-invariant transformation module,
which produces a G-invariant latent representation of the input data.
This latent representation is then processed with a multi-layer perceptron in the network.

### Repository structure

#### Main structure
* [dataset/](dataset/) - contains files associated with preparation and loading the dataset of convex quadrangles
* [experiments/](experiments/) - contains the code to perform neural networks training and evaluation of the models
* [models/](models/) - contains model of the proposed G-invariant neural network and other models used for the comparison
* [utils/](utils/) - contains a bunch of utilities, such as: polynomials definitions, predefined permutation groups, etc.
* [data_inv/](data_inv/) - contains a dataset used in the experiments (convex quadrangle estimation only)


### Dependencies
* Tensorflow 1.14
* Keras 2.2.5
* NumPy 1.16.4
* cudatoolkit 10.1.168
* Matplotlib 3.1.1


### Citation
```
@misc{kicki2020invariant,
    title={A Computationally Efficient Neural Network Invariant to the Action of Symmetry Subgroups},
    author={Piotr Kicki and Mete Ozay and Piotr Skrzypczyński},
    year={2020},
    eprint={2002.07528},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


