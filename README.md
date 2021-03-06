# Using machine learning to characterize 2d Ising magnetization states

The purpose of this project is to independently validate the claim made by Machine Learning Phases of Matter [(1605.01735)](https://arxiv.org/pdf/1703.05334.pdf).
It is a neural net built with Tensorflow to identify magnetization states of 2d Ising lattices at low temperatures around the critical temperature.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. Build passing with python 3.5.2.

### Prerequisites

Dependencies will be handled by pip, make sure you have python3 installed at least version 3.5

```bash
pip install -r requirements.txt
```

Ensure you have enough memory before running the code, the code is not yet optimized for memory allocation, could take up 1-2gb of memory.

### Running the code
Run the code with python

```
python tf.py
```

Your data should begin generating, or if you have the pre-generated data.p file, tensorflow will build the model and begin training.
The model is expected to have ~98% accuracy on the validation set, and similar accuracy in the test set if you have it loaded.

## Built With

* [Tensorflow](https://www.tensorflow.org/) - To build the neural net

## Authors

* **Danny Kong** - [Github](https://github.com/dannykong12)


## Acknowledgments

* This code was inspired by the paper Machine Learning Phases of Matter (Carrasquilla, J; Melko, R)
* Metropolis algorithm was implemented by Rajesh Rinet [Github](https://github.com/rajeshrinet/compPhy/tree/master/ising), our data was generated by running his algorithm, equilibrating, and sampling at discrete time points
