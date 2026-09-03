import json
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
import os


def complete_individuals(individuals:pd.DataFrame, column_names:list[str]) -> pd.DataFrame:
    '''
    **completes the missing columns of an individual. The new values are 0.0**
    
    - parameters:
        `individual`: the individual to complete
        `columns_names`: names of all the columns

    - returns:
        dataframe with the completed individuals
    '''

    individuals = individuals.copy()
    for col in column_names:
        if col not in individuals.columns:
            individuals[col] = 0.0
    return individuals

def add_other_columns(df):
    """
    **Assigns similar_grade, TrialType, containsMB columns to a given dataframe**

    - Parameters:
        `df`: DataFrame to modify (Pandas DataFrame)

    - Returns:
        DataFrame with the columns assigned
    """
    return df.assign(**{"similar_grade": "TS200F6", "Trialtype": "LT", "containsMB": False})


def normalize_individual(individual, max_values, offset=0.2):
    """
    **Normalizes an individual or individuals based on the max value it has in the dataset + offset**

    - Params:
        `individual`: dataframe containing 1 or more individuals (pandas DataFrame)
        `max_values`: maximum values for each concentration from the dataframe (pandas Series)
        `offset`: offset in % to add on top of the max values
    
    - Returns:
        Dataframe containing the normalized individual(s)
    """

    if not (0.0 <= offset <= 1.0):
        raise ValueError('offset must be between 0.0 and 1.0')
    
    normalized_individual = individual*(max_values.values*(1+offset))

    return normalized_individual

def recipe_to_lab(recipes_df, model_):
    """
    **Transforms color recipes (pigment concentrations) to L a b color format, using the predictive model**

    - parameters:
        `recipes_df` (pandas.DataFrame): dataframe containing the recipes
        `model_`: model used for the transformation
    
    - returns:
        pandas.DataFrame containing the corresponding L a b values
    """

    recipes_df = recipes_df.pipe(add_other_columns)
    return pd.DataFrame(model_.predict(recipes_df)[0], columns=[f'L', 'a', 'b'])


def read_summary_metrics(summary_metrics_path):
    """
    **Transforms output.txt file to structured data**

    - parameters:
        `summary_metrics_path`: path to the output.txt file

    - returns:
        dict containing the structured data
    """

    # open and read the file
    with open(summary_metrics_path, 'r') as file:
        lines = file.readlines()

    # extract values into a dictionary
    results = {}
    c = 0
    for line in lines:
        try:
            if c==7:
                break
            if c == 5:
                key, value = line.split('              ', 1) 
            else:
                key, value = line.split(':', 1)
            results[key.strip()] = float(value) if '.' in value or 'e' in value.lower() else int(value)
        except ValueError:
            pass
        c+=1

    return results


def summarize_results(dir_path, modulo=1):
    """
    **Summarizes and saves the results of the saved GA results**

    - parameters:
        `dir_path` (str): path to the directory containing the GA results
        `modulo` (int): modulo the recipe number to take into account. (default: 1)
    
    - returns:
        Dataframe containing the summary (pd.DataFrame)

    """

    summary = []

    base_dir = Path(dir_path)

    # get and sort folders by creation time
    folders = sorted([f for f in base_dir.iterdir() if f.is_dir()],
                    key=lambda x: x.stat().st_ctime)
    
    pbar = tqdm(range(len(folders)), desc=f'Summarizing results for {dir_path}')
    for folder in folders:
        # path to the folder
        folder_path = f'{dir_path}/{folder.name}'

        

        if int(folder.name.split('_')[-1]) % modulo != 0:
            continue

        # skip if the recipe is not fully evaluated yet
        if not os.path.exists(f'{folder_path}/.done'):
            continue


        sub_dir = Path(folder_path)
        
        best_fitness = -1
        best_fitness_num = -1

        recipe_trials = [] # contains the dicts for the trials of the recipe

        c=0
        for subfolder in sub_dir.iterdir():

            if not subfolder.is_dir():
                continue  # skip non-folder entries

            res_path = f'{folder_path}/{subfolder.name}'

            #print(res_path)

            # transform the output to structured data
            res_summary = read_summary_metrics(f'{res_path}/output.txt')

            res_summary['Name'] = subfolder.name
            res_summary['Recipe'] = folder.name
            res_summary['Best_fitness'] = False

            # count the mathced pigments
            expected_pigments = list(pd.read_csv(f'{res_path}/expected_recipe.csv').columns)
            try:
                predicted_pigments = list(pd.read_csv(f'{res_path}/best_individual.csv').columns)
            except Exception:
                predicted_pigments = []
            matched_pigments = list(set(expected_pigments) & set(predicted_pigments))

            res_summary['Expected_pigments'] = expected_pigments
            res_summary['Predicted_pigments'] = predicted_pigments
            res_summary['Matched_pigments'] = matched_pigments
            if len(matched_pigments) == 0:
                res_summary['Matched_pigments_ratio'] = 0
            else:
                res_summary['Matched_pigments_ratio'] = len(matched_pigments) / len(expected_pigments)


            recipe_trials.append(res_summary)

            if res_summary['Fitness'] > best_fitness:
                best_fitness = res_summary['Fitness']
                best_fitness_num = c

            c+=1
        
        # identify the trial with best fitness for the recipe
        recipe_trials[best_fitness_num]['Best_fitness'] = True

        # add the results to the summary list
        summary += recipe_trials

        pbar.update(1)

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(f'{dir_path}/summary.csv', index=False)

    return summary_df




def metrics_counter(metrics_csv_path, type=None):
    """
    **Counts and saves the metrics to create overviews. Saves to the same folder as the summary file**

    - parameters:
        `metrics_csv_path` (str): path to the summary.csv file
        `type` (str): type for the gann results. set to None to ignore it (default: None)

    - returns:
        Dictionnary containing the metrics overview
    """
    
    df = pd.read_csv(metrics_csv_path)

    if type is not None:
        df = df[(df['Best_fitness'] == True) & (df['Type'] == type)]
    else:
        df = df[(df['Best_fitness'] == True)]


    metrics = {}

    metrics['dE94_0-1'] = df[(df['Delta E94'] >= 0) & (df['Delta E94'] <= 1)].shape[0]
    metrics['dE94_1-2'] = df[(df['Delta E94'] > 1) & (df['Delta E94'] <= 2)].shape[0]
    metrics['dE94_2-4'] = df[(df['Delta E94'] > 2) & (df['Delta E94'] <= 4)].shape[0]
    metrics['dE94_4+'] = df[(df['Delta E94'] > 4)].shape[0]

    metrics['dE94_0-1_pigments_mean'] = df[(df['Delta E94'] >= 0) & (df['Delta E94'] <= 1)]['Non zero pigments'].mean()
    metrics['dE94_1-2_pigments_mean'] = df[(df['Delta E94'] > 1) & (df['Delta E94'] <= 2)]['Non zero pigments'].mean()
    metrics['dE94_2-4_pigments_mean'] = df[(df['Delta E94'] > 2) & (df['Delta E94'] <= 4)]['Non zero pigments'].mean()
    metrics['dE94_4+_pigments_mean'] = df[(df['Delta E94'] > 4)]['Non zero pigments'].mean()

    metrics['dE94_0-1_pigments_median'] = df[(df['Delta E94'] >= 0) & (df['Delta E94'] <= 1)]['Non zero pigments'].median()
    metrics['dE94_1-2_pigments_median'] = df[(df['Delta E94'] > 1) & (df['Delta E94'] <= 2)]['Non zero pigments'].median()
    metrics['dE94_2-4_pigments_median'] = df[(df['Delta E94'] > 2) & (df['Delta E94'] <= 4)]['Non zero pigments'].median()
    metrics['dE94_4+_pigments_median'] = df[(df['Delta E94'] > 4)]['Non zero pigments'].median()

    metrics['dE94_0-1_pigments_std'] = df[(df['Delta E94'] >= 0) & (df['Delta E94'] <= 1)]['Non zero pigments'].std()
    metrics['dE94_1-2_pigments_std'] = df[(df['Delta E94'] > 1) & (df['Delta E94'] <= 2)]['Non zero pigments'].std()
    metrics['dE94_2-4_pigments_std'] = df[(df['Delta E94'] > 2) & (df['Delta E94'] <= 4)]['Non zero pigments'].std()
    metrics['dE94_4+_pigments_std'] = df[(df['Delta E94'] > 4)]['Non zero pigments'].std()

    metrics['dE94_0-1_pigments_quantile'] = df[(df['Delta E94'] >= 0) & (df['Delta E94'] <= 1)]['Non zero pigments'].quantile([0.25, 0.5, 0.75]).to_dict()
    metrics['dE94_1-2_pigments_quantile'] = df[(df['Delta E94'] > 1) & (df['Delta E94'] <= 2)]['Non zero pigments'].quantile([0.25, 0.5, 0.75]).to_dict()
    metrics['dE94_2-4_pigments_quantile'] = df[(df['Delta E94'] > 2) & (df['Delta E94'] <= 4)]['Non zero pigments'].quantile([0.25, 0.5, 0.75]).to_dict()
    metrics['dE94_4+_pigments_quantile'] = df[(df['Delta E94'] > 4)]['Non zero pigments'].quantile([0.25, 0.5, 0.75]).to_dict()

    metrics['dE94_0-1_fitness_mean'] = df[(df['Delta E94'] >= 0) & (df['Delta E94'] <= 1)]['Fitness'].mean()
    metrics['dE94_1-2_fitness_mean'] = df[(df['Delta E94'] > 1) & (df['Delta E94'] <= 2)]['Fitness'].mean()
    metrics['dE94_2-4_fitness_mean'] = df[(df['Delta E94'] > 2) & (df['Delta E94'] <= 4)]['Fitness'].mean()
    metrics['dE94_4+_fitness_mean'] = df[(df['Delta E94'] > 4)]['Fitness'].mean()

    metrics['dE94_0-1_fitness_median'] = df[(df['Delta E94'] >= 0) & (df['Delta E94'] <= 1)]['Fitness'].median()
    metrics['dE94_1-2_fitness_median'] = df[(df['Delta E94'] > 1) & (df['Delta E94'] <= 2)]['Fitness'].median()
    metrics['dE94_2-4_fitness_median'] = df[(df['Delta E94'] > 2) & (df['Delta E94'] <= 4)]['Fitness'].median()
    metrics['dE94_4+_fitness_median'] = df[(df['Delta E94'] > 4)]['Fitness'].median()

    metrics['dE94_0-1_fitness_std'] = df[(df['Delta E94'] >= 0) & (df['Delta E94'] <= 1)]['Fitness'].std()
    metrics['dE94_1-2_fitness_std'] = df[(df['Delta E94'] > 1) & (df['Delta E94'] <= 2)]['Fitness'].std()
    metrics['dE94_2-4_fitness_std'] = df[(df['Delta E94'] > 2) & (df['Delta E94'] <= 4)]['Fitness'].std()
    metrics['dE94_4+_fitness_std'] = df[(df['Delta E94'] > 4)]['Fitness'].std()

    metrics['dE94_0-1_fitness_quantile'] = df[(df['Delta E94'] >= 0) & (df['Delta E94'] <= 1)]['Fitness'].quantile([0.25, 0.5, 0.75]).to_dict()
    metrics['dE94_1-2_fitness_quantile'] = df[(df['Delta E94'] > 1) & (df['Delta E94'] <= 2)]['Fitness'].quantile([0.25, 0.5, 0.75]).to_dict()
    metrics['dE94_2-4_fitness_quantile'] = df[(df['Delta E94'] > 2) & (df['Delta E94'] <= 4)]['Fitness'].quantile([0.25, 0.5, 0.75]).to_dict()
    metrics['dE94_4+_fitness_quantile'] = df[(df['Delta E94'] > 4)]['Fitness'].quantile([0.25, 0.5, 0.75]).to_dict()

    try:
        metrics['dE94_0-1_dEm_mean'] = df[(df['Delta E94'] >= 0) & (df['Delta E94'] <= 1)]['Delta E metamerism'].mean()
        metrics['dE94_1-2_dEm_mean'] = df[(df['Delta E94'] > 1) & (df['Delta E94'] <= 2)]['Delta E metamerism'].mean()
        metrics['dE94_2-4_dEm_mean'] = df[(df['Delta E94'] > 2) & (df['Delta E94'] <= 4)]['Delta E metamerism'].mean()
        metrics['dE94_4+_dEm_mean'] = df[(df['Delta E94'] > 4)]['Delta E metamerism'].mean()

        metrics['dE94_0-1_dEm_std'] = df[(df['Delta E94'] >= 0) & (df['Delta E94'] <= 1)]['Delta E metamerism'].std()
        metrics['dE94_1-2_dEm_std'] = df[(df['Delta E94'] > 1) & (df['Delta E94'] <= 2)]['Delta E metamerism'].std()
        metrics['dE94_2-4_dEm_std'] = df[(df['Delta E94'] > 2) & (df['Delta E94'] <= 4)]['Delta E metamerism'].std()
        metrics['dE94_4+_dEm_std'] = df[(df['Delta E94'] > 4)]['Delta E metamerism'].std()
    except KeyError:
        pass


    metrics['fitness09-10'] = df[(df['Fitness'] > 0.9) & (df['Fitness'] <= 1)].shape[0]
    metrics['fitness08-09'] = df[(df['Fitness'] > 0.8) & (df['Fitness'] <= 0.9)].shape[0]
    metrics['fitness07-08'] = df[(df['Fitness'] > 0.7) & (df['Fitness'] <= 0.8)].shape[0]
    metrics['fitness06-07'] = df[(df['Fitness'] > 0.6) & (df['Fitness'] <= 0.7)].shape[0]
    metrics['fitness05-06'] = df[(df['Fitness'] > 0.5) & (df['Fitness'] <= 0.6)].shape[0]
    metrics['fitness04-05'] = df[(df['Fitness'] > 0.4) & (df['Fitness'] <= 0.5)].shape[0]
    metrics['fitness03-04'] = df[(df['Fitness'] > 0.3) & (df['Fitness'] <= 0.4)].shape[0]
    metrics['fitness02-03'] = df[(df['Fitness'] > 0.2) & (df['Fitness'] <= 0.3)].shape[0]
    metrics['fitness01-02'] = df[(df['Fitness'] > 0.1) & (df['Fitness'] <= 0.2)].shape[0]
    metrics['fitness00-01'] = df[(df['Fitness'] <= 0.1)].shape[0]

    metrics['fitness_mean'] = df['Fitness'].mean()

    metrics['total_values'] = len(df)



    save_path = metrics_csv_path.replace(metrics_csv_path.split('/')[-1], 'metrics.json')

    with open(save_path, 'w') as f:
        json.dump(metrics, f)
    


    return metrics