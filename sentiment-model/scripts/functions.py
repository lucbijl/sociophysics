import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.integrate import nquad

# Defining the various probability functions to be included in the subset S.
def g_age(p_age):
        a = 0.1
        return a * np.exp(-(a*p_age.days))

def g_s(p_s):
      a_1 = 0.4
      a_2 = 0.6
      b_1 = 2
      b_2 = 2
      c_1 = 1
      c_2 = -1
      return a_1 * np.exp(-b_1 * (p_s - c_1)**2) + a_2 * np.exp(-b_2 * (p_s - c_2)**2)

def g_v(p_v):
      a_1 = 0.4
      a_2 = 0.6
      b_1 = 1
      b_2 = 1
      c_1 = 1
      c_2 = -1
      return a_1 * np.exp(-b_1 * (p_v - c_1)**2) + a_2 * np.exp(-b_2 * (p_v - c_2)**2)

def g_a(p_a):
      a = 0.5
      return np.exp(a * p_a) * a / (np.exp(a) - np.exp(-a))

def g_d(p_d):
      return 1

# Defining the probability function to be included in the subset S.
def g(p):
      return g_age(p['age']) * g_s(p['s']) * g_v(p['v']) * g_a(p['a']) * g_d(p['d'])

# Defining the normalized probability function and mapping to be included in the subset S.
def create_subset(df_P):
    # Determining the normalization sum of g for P.
    normal_sum = sum(g(row.to_dict()) for _, row in df_P[['age', 's', 'v', 'a', 'd']].iterrows())

    # Determining the probability for each entry in P.
    rho = [g(row.to_dict()) for _, row in df_P[['age', 's', 'v', 'a', 'd']].iterrows()] / normal_sum

    # Defining the cardinality m of the subset S.
    m = 3
    
    # Creating the subset
    return df_P.loc[np.random.choice(df_P.index, size=m, replace=False, p=rho)]

# Defining the initial state function.
def create_initial_state(df_P):
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

# Defining the various probability influence functions
def I_s(u, p, epsilon):
    mu_1, mu_2 = 1, 10
    sigma_1, sigma_2 = 1, 2
    mu = mu_1 * mu_2**(u['a'] + p['a'] - u['d'] + p['d'] - 4)
    sigma = sigma_1 * sigma_2**(-(u['v'] - p['v'])**2 - (u['s'] - p['s'])**2)
    return np.exp((-1 / (2 * sigma**2)) * (epsilon - mu)**2) * 1 / (sigma * np.sqrt(2 * np.pi))

def I_v(u, p, epsilon):
    mu = 0.1
    return np.exp(mu * epsilon) * mu / (np.exp(mu) - np.exp(-mu))

def I_a(u, p, epsilon):
    mu = 0.1
    return np.exp(mu * epsilon) * mu / (np.exp(mu) - np.exp(-mu))

def I_d(u, p, epsilon):
    mu = 0.1
    return np.exp(mu * epsilon) * mu / (np.exp(mu) - np.exp(-mu))

# Defining the inverse transform function.
def inverse_transform(u, p, pdf):
    values = np.linspace(-1, 1, 1000)

    cdf_values = np.cumsum(pdf(u, p, values))
    cdf_values /= cdf_values[-1]

    sample = np.random.rand(1)

    inverse_sample = np.interp(sample, cdf_values, values)

    return inverse_sample

# Defining the probability influence function.
def probability_influence_function(u, p):

    # Determing the influence factors by taking the inverse transforms of the various probability influence functions.
    epsilon_s = inverse_transform(u, p, I_s)
    epsilon_v = inverse_transform(u, p, I_v)
    epsilon_a = inverse_transform(u, p, I_a)
    epsilon_d = inverse_transform(u, p, I_d)

    # Creating the diagonal matrix E, with the influence factors.
    E = np.diag([epsilon_s, epsilon_v, epsilon_a, epsilon_d])
    return (1/2) * (np.array(list(u.values())) + np.dot(E, np.array(list(p.values())).reshape(1,4)))

# Defining the simple influence function.
def simple_influence_function(u, p):
    return (1/2) * (np.array(list(u.values())) + np.array(list(p.values())))

# Defining the coupled influence function.
def coupled_influence_function(u, p):
    v_1 = 2 * u['s'] + (1 - u['d']) * (1 + u['a']) * (1 + p['a']) * p['s']
    v_2 = 7 * u['v'] + 3 * p['v']
    v_3 = 6 * u['a'] + 4 * p['a']
    v_4 = 6 * u['d'] + 4 * u['d']
    return (1/10) * np.array([v_1, v_2, v_3, v_4])

# Defining the total mapping from state u to v of the agent.
def influence_mapping(u, df_S):
    columns = ['s', 'v', 'a', 'd']
    vector = (1/len(df_S)) * sum(probability_influence_function(u, row.to_dict()) for _, row in df_S[['s', 'v', 'a', 'd']].iterrows())
    return {key: value for key, value in zip(columns, vector)}
    
# Defining the various post probability functions.
def h_s(v_s):
    mu_1 = 1
    mu_2 = 1
    nu_1 = 1
    nu_2 = 1
    rho_1 = 1
    rho_2 = -1
    return mu_1 * np.exp(-nu_1 * (v_s - rho_1)**2) + mu_2 * np.exp(-nu_2 * (v_s - rho_2)**2)
    
def h_v(v_p):
    mu_1 = 1
    mu_2 = 1
    nu_1 = 1
    nu_2 = 1
    rho_1 = 1
    rho_2 = -1
    return mu_1 * np.exp(-nu_1 * (v_p - rho_1)**2) + mu_2 * np.exp(-nu_2 * (v_p - rho_2)**2)
    
def h_a(v_a):
    mu = 0.2
    return np.exp(mu * v_a) * (mu + 0.1) / (np.exp(mu) - np.exp(-mu))

def h_d(v_d):
    mu = 0.2
    return np.exp(mu * v_d) * (mu + 0.1) / (np.exp(mu) - np.exp(-mu))

# Defining the post probability function.
def h(v):
    return h_s(v['s']) * h_v(v['v']) * h_a(v['a']) * h_d(v['d'])

# Defining the post probability function wrapper.
def wrapper_h(v_s, v_v, v_a, v_d):
    v = {'s': v_s, 'v': v_v, 'a': v_a, 'd': v_d}
    return h(v)

# Defining the post boolean return function based on h.
def post_boolean(v):
    # interval = [[-1,1], [-1,1], [-1,1], [-1,1]]

    # Determining the normal integral of the post probability function.
    # normal_sum, error = nquad(wrapper_h, interval)

    # Determining the probabality by normalizing the image of h with the normal sum.
    probability = h(v)

    return np.random.choice([True, False], p=[probability, 1 - probability])