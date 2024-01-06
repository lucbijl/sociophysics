import pandas as pd
import numpy as np
import functions as fct
from sklearn.neighbors import KernelDensity
from scipy.integrate import nquad
from tqdm import tqdm

def main():
    # Importing the csv dataset.
    import_posts = '../test-posts-scores.csv'
    df_posts = pd.read_csv(import_posts)

    # Stating the initial (start) time point.
    time_start = pd.to_datetime('2024-01-03 00:00:00')

    # Stating the end time point.
    time_end = pd.to_datetime('2024-01-6 00:00:00')

    # Stating the post density.
    post_frequency = pd.Timedelta(seconds=60)

    # Defining the list P of all posts.
    P = []

    # Determining the posts from before the time point and appending the necessary information to P.
    for _, row in df_posts.iterrows():
        t_row = pd.to_datetime(row['Date'])
        if t_row < time_start:
            P.append([time_start - t_row] + list(row[['S', 'V', 'A', 'D']]))

    df_P = pd.DataFrame(P, columns=['age', 's', 'v', 'a', 'd'])

    # Defining the agent loop.
    for time in tqdm(pd.date_range(start=time_start, end=time_end, freq=post_frequency)):
        # Updating the age of the posts.
        if time > time_start:
            df_P['age'] += post_frequency

        # Creating the subset S.
        df_S = fct.create_subset(df_P)

        # Creating the initial state of the agent.
        u = fct.create_initial_state(df_P)

        # Determining the influenced state of the agent.
        v = fct.influence_mapping(u, df_S)

        # Adding the age of the influenced state.
        v['age'] = pd.Timedelta(seconds=0)
        
        # Determining if the state will be added to P with the post probability.
        if fct.post_boolean(v):
            df_P = pd.concat([df_P, pd.DataFrame([v])], ignore_index=True)

    # Redefining the list P of all posts.
    P = []

    # Converting the age in the posts dataframe back to datetime.
    for _, row in df_P.iterrows():
        age_row = row['age']
        P.append([(time_end - age_row)] + list(row[['s', 'v', 'a', 'd']]))

    df_modeled_posts = pd.DataFrame(P, columns=['date', 's', 'v', 'a', 'd'])

    # Exporting the posts dataframe to a csv dataset.
    export_posts = '../test-posts-scores-predicted.csv'
    df_modeled_posts.to_csv(export_posts, index=False)

if __name__ == '__main__':
    main()