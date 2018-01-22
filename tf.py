#-------------------------------------------------------------------------------
#    Using machine learning to characterize 2d Ising magnetization states
#
#-------------------------------------------------------------------------------
# author:         Xiangzhou Kong
# date:           Jan 21, 2018
# email:          x32kong@uwaterloo.ca
#
# collaboration:
#   Monte-Carlo code was written by rajeshrinet taken fron his blog
#   Tensorflow code was based off the template from tensorflow examples
#       neural_net.py, adapted to this problem
#-------------------------------------------------------------------------------

from __future__ import division
import tensorflow as tf
import numpy as np
import pickle
import sklearn.model_selection as sk
from numpy.random import rand
import sys
import os
from tqdm import tqdm

#-------------------------------------------------------------------------------
# Code for monte carlo simulations and generating the dataset
#-------------------------------------------------------------------------------

# this code was run prior to training tensorflow code, generating the dataset
# could take very long

def initialstate(N):
    ''' generates a random spin configuration for initial condition'''
    state = 2*np.random.randint(2, size=(N,N))-1
    return state


def mcmove(config, beta):
    '''Monte Carlo move using Metropolis algorithm '''
    for i in range(N):
        for j in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                s =  config[a, b]
                nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
                cost = 2*s*nb
                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost*beta):
                    s *= -1
                config[a, b] = s
    return config


def calcEnergy(config):
    '''Energy of a given configuration'''
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            energy += -nb*S
    return energy/4.


def calcMag(config):
    '''Magnetization of a given configuration'''
    mag = np.sum(config)
    return mag

# These functions were taken from user rajeshrinet on GitHub

#-------------------------------------------------------------------------------
# Main function Block
#-------------------------------------------------------------------------------

# the following functions were written by the author, Xiangzhou Kong

def generate_data(temp, num_instances, sampling_rate=1, steps=1):
    # generates num_instances of samples at the given temp, with sampling_rate
    # mc iterations between each sample after equilibrium, and eqSteps/steps
    # to reach equilibrium
    config = initialstate(N)
    iT=1.0/temp; iT2=iT*iT;

    print("Equilibrating at T = %1.1f...\n"% (temp))

    for i in tqdm(range(np.floor_divide(eqSteps, steps))):         # equilibrate
        mcmove(config, iT)           # Monte Carlo moves


    print("Sampling...\n")

    for i in tqdm(range(num_instances)):
        for ii in range(sampling_rate):
            mcmove(config, iT)
        data.append(np.copy(config).flatten())
        labels.append(np.array([1, 0])) if (temp<2.269) else labels.append(np.array([0, 1]))
    print("Done!\n")


def maybe_generate(force=false):
    # generates data if data.p is not in the same folder
    # produces 4000 data points at each temp at 1.2, 1.3 ... 3.8 K
    if os.path.exists("data.p") and not force:
        print("Dataset already generated, proceeding")
    else:
        for temp in range(26):
            for i in tqdm(range(16)):
                generate_data(1.2+temp/10, 250, sampling_rate=5, steps=1)
        with open("data.p", "wb") as a:
            pickle.dump(data, a)
            a.close()
        print("Dataset ready, proceeding")



#----------------------------------------------------------------------
# Data generation script
#----------------------------------------------------------------------

nt      = 2**8        # number of temperature points
N       = 30        # size of the lattice, N x N
eqSteps = 2**10       # number of MC sweeps for equilibration
mcSteps = 2**10       # number of MC sweeps for calculation

n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N)
tm = 2.269;    T=np.random.normal(tm, .64, nt)
T  = T[(T>1.2) & (T<3.8)];    nt = np.size(T)

data = []

# Generate data if necessary
maybe_generate()

# ------------------------------------------------------------------------------
# Tensorflow code start
# ------------------------------------------------------------------------------

# Initialize tensorflow params
seed = 42
tf.set_random_seed(seed)
tf.logging.set_verbosity(tf.logging.INFO)

# Load data from monte carlo sims
datafile = open("data.p", "rb")
data = pickle.load(datafile)

# Format data for training
features, label = data['data'], data['labels']
labels = []
for i in label:
    labels.append(1 if i[0]==1 else 0)
features, labels = np.asarray(features, dtype="float32"), np.asarray(labels, dtype="float32")

# Use sklearn to split and shuffle data
X_train, X_test, y_train, y_test = sk.train_test_split(features,labels,test_size=0.3, random_state = 42)

# Parameters for training
learning_rate = 0.001
num_steps = 1000
batch_size = 200
display_step = 100

# Network Parameters
n_hidden_1 = 600 # 1st layer number of neurons
num_input = 900 # Ising data input (img shape: 30*30)
num_classes = 2 # total classes, 1 for FM and 0 for PM


# Define the neural net:
# 2-Layer feed forward neural net with 100 hidden units with rectified linear
# activation, l2 regularization at 0.01 beta
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    regularizer = tf.contrib.layers.l2_regularizer(0.01)
    layer_1 = tf.layers.dense(x, n_hidden_1, kernel_regularizer=regularizer, activation=tf.nn.relu)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_1, num_classes)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    logits = neural_net(features)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # Define a logging hook
    logging_hook = tf.train.LoggingTensorHook({"loss": loss_op}, every_n_iter=10)
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op},
        training_hooks = [logging_hook])
    return estim_specs

# Build the Estimator
print("Building Model")
model = tf.estimator.Estimator(model_fn)


# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': X_train}, y=y_train,
    batch_size=batch_size, num_epochs=None, shuffle=True)


# Train the Model
print("Beginning Training")
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': X_test}, y=y_test,
    batch_size=batch_size, shuffle=False)
in_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': X_train[0:1000]}, y=y_train[0:1000],
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
print("Evaluating Model")
e = model.evaluate(input_fn)
f = model.evaluate(in_fn)

print("Validation Accuracy:", e['accuracy'])
print("Training Accuracy:", f['accuracy'])

# Load Test Set
# Assumes you have a separately generated test set in the same folder
# this code will not generate the test set for you, but you can tweak the code
# used to generate the test set to generate your own test set
try:
    test_file = open("test.p", "rb")
except Exception as e:
    print("No test set found. Do you have test.p in your current directory?\n A test set can be generated with the generate_data function.")
test_data = pickle.load(test_file)
test_features, test_label = data['data'], data['labels']
test_labels = []
for i in label:
    test_labels.append(1 if i[0]==1 else 0)
test_features, test_labels = np.asarray(features, dtype="float32"), np.asarray(labels, dtype="float32")
test_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': test_features}, y=test_labels,
    batch_size=batch_size, shuffle=False)
g = model.evaluate(test_fn)
print("Test Accuracy:", g['accuracy'])
