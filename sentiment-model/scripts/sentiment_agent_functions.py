import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.integrate import nquad
from functools import reduce
import operator

# Defining the sum of sigmoid functions.
def sigmoid_sum(var, params):
    output = 0
    for a, b, c in params:
        output += a / (1 + np.exp(-b * (var - c)))
    return np.clip(output, 0, 1)

# Defining the age function.
def age_function(var, params):
    return np.exp(-(params * var.days))

# Defining the probability function to be included in the subset S.
def g(p, params_sp):
    return age_function(p['age'], params_sp['age']) * reduce(operator.mul, [sigmoid_sum(p[i], params_sp[i]) for i in ['s', 'v', 'a', 'd']])

# Defining the normalized probability function and mapping to be included in the subset S.
def create_subset(df_P, params, m):
    # Determining the normalization sum of g for P.
    normal_sum = sum(g(row.to_dict(), params['sp']) for _, row in df_P[['age', 's', 'v', 'a', 'd']].iterrows())

    # Determining the probability for each entry in P.
    if normal_sum > 0:
        rho = [g(row.to_dict(), params['sp']) for _, row in df_P[['age', 's', 'v', 'a', 'd']].iterrows()] / normal_sum
    else: rho = None
  
    # Creating the subset.
    return df_P.loc[np.random.choice(df_P.index, size=m, replace=False, p=rho)]

##########################################

# Defining the post distribution initial state function.
def post_distribution_initial_state(df_P):
    kde_models = {}

    # Defining the bandwidth of the kernel density function.
    bandwidth = 0.1

    # Fitting a kde for each column.
    for i in ['s', 'v', 'a', 'd']:
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(df_P[i].values.reshape(-1,1))
        kde_models[i] = kde

    # Generating a new random entry for each column.
    new_entries = {}
    for i, kde_model in kde_models.items():
        new_entry = np.clip(kde_model.sample(), -1, 1)
        new_entries[i] = new_entry.item()
    
    return new_entries

# Defining the totally random initial state function.
def random_initial_state():
     return {key: np.random.uniform(-1, 1) for key in ['s', 'v', 'a', 'd']}

# Defining the initial state mapping.
def create_initial_state(df_P):
    state = post_distribution_initial_state(df_P)
    return state

##########################################

# Defining the simple influence function.
def simple_influence_function(u, p):
    return (1/2) * (np.array(list(u.values())) + np.array(list(p.values())))

# Defining the contagion influence function.
def contagion_influence_function(u, p, params_cif):
    # Determining the influence/contagion factor.
    epsilon = sigmoid_sum(reduce(operator.add, [abs(u[i] - p[i]) for i in ['v', 'a', 'd']]), params_cif)
    
    # Returning the influenced state.
    return (1 - epsilon) * np.array(list(u.values())) + epsilon * np.array(list(p.values()))

# Defining the total mapping from state u to v of the agent.
def influence_mapping(u, df_S, params):
    columns = ['s', 'v', 'a', 'd']
    vector = (1/len(df_S)) * sum(contagion_influence_function(u, row.to_dict(), params['cif']) for _, row in df_S[columns].iterrows())
    return {key: value for key, value in zip(columns, vector)}

##########################################

# Defining the post probability function.
def h(v, params_pp):
    return reduce(operator.mul, [sigmoid_sum(v[i], params_pp[i]) for i in ['s', 'v', 'a', 'd']])

# Defining the post boolean return function based on h.
def post_boolean(v, params):
    probability = h(v, params['pp'])
    return np.random.choice([True, False], p=[probability, 1 - probability])