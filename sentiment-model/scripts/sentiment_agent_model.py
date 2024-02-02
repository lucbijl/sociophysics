import pandas as pd
import numpy as np
import scripts.sentiment_agent_functions as fct
from sklearn.neighbors import KernelDensity
from scipy.integrate import nquad
from tqdm import tqdm

def sentiment_agent_model(import_posts, cardinality_S, string_start, string_end, string_threshold, post_period, params):

    ### Begin of important definitions.
   
    # Stating the start and end time point.
    time_start = pd.to_datetime(string_start)
    time_end = pd.to_datetime(string_end)

    # Stating the age threshold
    age_threshold = pd.Timedelta(string_threshold)

    # Stating the post density.
    post_frequency = pd.Timedelta(seconds=post_period)
    
    ### End of important definitions.

    # Importing the csv dataset.
    df_posts = pd.read_csv(import_posts)

    # Defining the list P of all posts.
    P = []

    # Determining the posts from before the time point and appending the necessary information to P.
    for _, row in df_posts.iterrows():
        t_row = pd.to_datetime(row['date'])
        if t_row < time_start and (time_start - t_row) < age_threshold:
            P.append([time_start - t_row] + list(row[['s', 'v', 'a', 'd']]))

    df_P = pd.DataFrame(P, columns=['age', 's', 'v', 'a', 'd'])
    print(len(df_P))

    # Defining the agent loop.
    for time in tqdm(pd.date_range(start=time_start, end=time_end, freq=post_frequency)):
        # Updating the age of the posts.
        if time > time_start:
            df_P['age'] += post_frequency

        # Creating the subset S.
        df_S = fct.create_subset(df_P, params, cardinality_S)

        # Creating the initial state of the agent.
        u = fct.create_initial_state(df_P)

        # Determining the influenced state of the agent.
        v = fct.influence_mapping(u, df_S, params)

        # Adding the age of the influenced state.
        v['age'] = pd.Timedelta(seconds=0)
        
        # Determining if the state will be added to P with the post probability.
        if fct.post_boolean(v, params):
            df_P = pd.concat([df_P, pd.DataFrame([v])], ignore_index=True)

    # Redefining the list P of all posts.
    P = []

    # Converting the age in the posts dataframe back to datetime.
    for _, row in df_P.iterrows():
        age_row = row['age']
        P.append([(time_end - age_row)] + list(row[['s', 'v', 'a', 'd']]))

    df_modeled_posts = pd.DataFrame(P, columns=['date', 's', 'v', 'a', 'd'])

    return df_modeled_posts 