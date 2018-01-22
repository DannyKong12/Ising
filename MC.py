from __future__ import division
import numpy as np
from numpy.random import rand
from tqdm import tqdm
import sys
import pickle

#----------------------------------------------------------------------
##  BLOCK OF FUNCTIONS USED IN THE MAIN CODE
#----------------------------------------------------------------------

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

#----------------------------------------------------------------------
# These functions were taken from user rajeshrinet on GitHub
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# Function Block
#----------------------------------------------------------------------

def generate_data(temp, num_instances, sampling_rate=1, steps=1):
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









#----------------------------------------------------------------------
# Function Block End
#----------------------------------------------------------------------


## change the parameter below if you want to simulate a smaller system
nt      = 2**8        # number of temperature points
N       = 30        # size of the lattice, N x N
eqSteps = 2**10       # number of MC sweeps for equilibration
mcSteps = 2**10       # number of MC sweeps for calculation

print("Starting up...")

n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N)
tm = 2.269;    T=np.random.normal(tm, .64, nt)
T  = T[(T>1.2) & (T<3.8)];    nt = np.size(T)

print("Allocating space...")
print("Generating data...\n")
data = []
labels = []

for temp in range(26):
    for i in range(16):
        generate_data(1.2+temp/10, 250, sampling_rate=5, steps=1)

with open("data.p", "wb") as a:
    pickle.dump({'data': data, 'labels': labels}, a)
    a.close()
print("Size in ram: ", sys.getsizeof(current_data))
