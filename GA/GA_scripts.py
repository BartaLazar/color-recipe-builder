import inspect
import json
from matplotlib import pyplot as plt
import os
import pickle
import sys
import pandas as pd
import colour
from dotenv import load_dotenv
from Code.Utils.util_methods import UtilMethods
from Code.GANN.utils.utils import complete_individuals
from Code.Arnold_models import PAR_ModellingFunctions_pipelines
from Code.GA.utils import add_other_columns, normalize_individual
#from tqdm.notebook import tqdm
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from Code.GA.utils import recipe_to_lab
from skimage.color import lab2rgb
import matplotlib.pyplot as plt
import random
import numpy as np

sys.modules['PAR_ModellingFunctions_pipelines'] = PAR_ModellingFunctions_pipelines

def get_tqdm(use_notebook):
    if use_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    return tqdm


base = UtilMethods.find_project_root(os.getcwd())
print(f"Project root found: {base}")

if load_dotenv(f'{base}/.env'):
    print(".env found")
else:
    print("ERROR .env not found")

def fitness(individuals, expectation, model, all_available_pigments, reflectance=True, internal_light_source='D65', uncertanity_bias=0.05, save_results=None, debug=False):
    """
    **Computes the fitness function of an individual by comparing the expected Lab values with the obtained Lab values, using the DELTAe94 colour difference**

    - Parameters:
        `individuals`: df containing the ingredients of the candidate recipes (pd.Dataframe)
        `expectation`: df of 1 row containing the expected reflectance or Lab values (pd.DataFrame)
        `model`: Model to use for the Lab or reflectance prediction
        `all_available_pigments`: list of all the available pigments
        `reflectance`: the expectation are reflectance values. If false, then it means Lab. (bool) (Default: True)
        `internal_light_source`: !!NOT USED ANYMORE!! light source to use inside of the function for the color similarity when reflectance is used (default: D65) (str)
        `uncertanity_bias`: weight of the uncertanity of the model predicting Lab/reflection from a recipe
        `save_results`: path to save the fitness results as txt. Provide the absolut path (str) (Default: None)
        `debug`: Flag indicating wether some internal values should be printed (bool) (Default: False)
    
    - Returns:
        Fitness values (np.ndarray), DeltaE94 (np.ndarray), Number of used pigments (np.ndarray), Incertitudes of Lab prediction (np.ndarray), dE distance of the metamerism (np.ndarray)
    """

    #if individual.shape[0] != 1:
    #    raise Exception('individual parameter should have one and only one row')

    individuals = complete_individuals(individuals, all_available_pigments)
    
    individual_added_columns = individuals.pipe(add_other_columns)

    # transform the individual to Lab values
    pred = model.predict(individual_added_columns)
    incertitude = pd.DataFrame(pred[1]).iloc[:,0].values
    if reflectance:
        reflectance_df = pd.DataFrame(pred[0], columns=[f'{w}nm' for w in range(400,741,10)])
        individual_lab = {'D65': None, 'FL2': None, 'Studio LED': None}
        expectation_lab = {'D65': None, 'FL2': None, 'Studio LED': None}
        delta_e_dict = {'D65': None, 'FL2': None, 'Studio LED': None}
        # delta: expected Lab - individual Lab for every light source
        delta = {'D65': None, 'FL2': None, 'Studio LED': None}

        for light_source in individual_lab.keys():
            individual_lab[light_source] = reflectance_df.pipe(UtilMethods.addLabcols, light_source)[[f'L_{light_source}', f'a_{light_source}', f'b_{light_source}']].values
            expectation_lab[light_source] =  expectation.pipe(UtilMethods.addLabcols, light_source)[[f'L_{light_source}', f'a_{light_source}', f'b_{light_source}']].iloc[0].values
            
            # compute delta E94 distances
            repeated_expectation = np.tile(expectation_lab[light_source], (individual_lab[light_source].shape[0], 1))
            delta_e_dict[light_source] = colour.difference.delta_E_CIE1994(repeated_expectation, individual_lab[light_source])

            # compute delta
            delta[light_source] = expectation_lab[light_source] - individual_lab[light_source]
    else:
        #individual_lab = pred[0]
        #expectation_lab = expectation.iloc[0].values
        raise Exception('Lab expectation values are no longer supported')

    # sum of each individual's concentrations
    concentration_sums = individuals.sum(axis=1)

    # count of non-zero pigments
    non_zero_counts = (individuals != 0.0).sum(axis=1)

    # compute weighterd average dE94 (color experts want more weight on D65)
    delta_e = 0.4*delta_e_dict['D65'] + 0.3*delta_e_dict['FL2'] + 0.3*delta_e_dict['Studio LED']


    # spread: maximum - minimum of the deltas max(L_l1, L_l2, L_l3) - min(L_l1, L_l2, L_l3) for L a and b -> 3 values: metamersm in L, a and b
    # stack arrays into shape (num_lights, num_samples, 3), transposed matrix
    stacked_delta = np.stack(list(delta.values()), axis=0)

    # compute max and min across lighting conditions (axis=0)
    max_values = np.max(stacked_delta, axis=0)
    min_values = np.min(stacked_delta, axis=0)

    # compute difference: max - min for each sample and each channel
    spread = max_values - min_values

    # compute the the carthesian distance of the spread vector -> dE of the metamerism (dEm) -> not add to the fitness function but report on it and show on graphs
    dEm = np.linalg.norm(spread, axis=1) # dE of metamerism

    
    # print('Delta E94:')
    # print(delta_e)
    # print('-------')
    # print('Incertitude')
    # print(incertitude)

    # apply fitness function components
    delta_fitness = np.exp(-0.5 * delta_e)
    ingredients_fitness = 1 - 0.008 * non_zero_counts**2
    ingredients_fitness[non_zero_counts > 10] = 0.0
    incertitude_fitness = 1 - 0.05 * incertitude**2
    incertitude_fitness[incertitude > 4.5] = 0.0
    fitness_vals = (1-uncertanity_bias)*(delta_fitness * ingredients_fitness) + uncertanity_bias*incertitude_fitness
    #fitness_vals[fitness_vals < 0] = 0.0

    # print('------')
    # print('Incertitude fitness')
    # print(incertitude_fitness)

    # apply penalty for concentration sum > 1
    fitness_vals[concentration_sums > 1] = 0.0


    # only if there is one individual
    if debug and len(individuals) == 1:

        non_zero_concentrations = individuals.loc[:, (individuals != 0).any(axis=0)]
        non_zero_concentrations_dict = non_zero_concentrations.iloc[0].to_dict()

        print(f'Fitness:                        {fitness_vals.values[0]}')
        print(f'Incertitude:                    {incertitude[0]}')
        print(f'Non zero pigments:              {non_zero_counts.iloc[0]}')
        print(f'Delta E94:                      {delta_e[0]}')
        print(f'Delta E metamerism              {dEm[0]}')
        print(f'Sum of concentrations:          {concentration_sums.iloc[0]}')
        print("Non-zero pigments and their concentrations:")
        for pigment, value in non_zero_concentrations_dict.items():
            print(f"   {pigment}: {value}")

    if save_results is not None and len(individuals) == 1:
        os.makedirs(save_results, exist_ok=True)

        non_zero_concentrations = individuals.loc[:, (individuals != 0).any(axis=0)]
        non_zero_concentrations_dict = non_zero_concentrations.iloc[0].to_dict()

        with open(f'{save_results}/output.txt', 'a') as f:
            f.write(f'Fitness:                        {fitness_vals.values[0]}\n')
            f.write(f'Uncertanity:                    {incertitude[0]}\n')
            f.write(f'Non zero pigments:              {non_zero_counts.iloc[0]}\n')
            f.write(f'Delta E94:                      {delta_e[0]}\n')
            f.write(f'Delta E metamerism              {dEm[0]}\n')
            f.write(f'Sum of concentrations:          {concentration_sums.iloc[0]}\n')
            f.write("Non-zero pigments and their concentrations:\n")
            for pigment, value in non_zero_concentrations_dict.items():
                f.write(f"   {pigment}: {value}\n")




    return fitness_vals.values, delta_e, non_zero_counts, incertitude, dEm



def generate_individual(size_individual):
    '''
    Not used
    '''

    individual = np.zeros(size_individual)
    # choose a random number of nonzero entries between size-10 and size
    num_nonzero = np.random.randint(1,11)
    # select random positions to fill
    nonzero_indices = np.random.choice(size_individual, num_nonzero, replace=False)
    # assign random values between 0 and 1 to selected positions
    individual[nonzero_indices] = np.random.rand(num_nonzero)

    normalized_individual = UtilMethods.normalize_recipes(individual)

    return normalized_individual


def create_population(size_population:int, max_pigment_values:pd.Series, occurences:pd.Series, forced_columns:list=None, unused_columns:list=None):
    """
    **Create the initial population**

    - Parameters:
        `size`: size of the population (number of individuals)
        `max_pigment_values`: maximum pigment values for each pigment
        `occurences`: occurences in % for the given pigment in the historic dataset
        `forced_columns`: columns that must be included no matter what
        `unused_columns`: columns that should not be used no matter what

    - Returns:
        pd.DataFrame containing the normalized initial population
    """


    all_columns = max_pigment_values.index

    size_individual = len(max_pigment_values)

    population = np.random.rand(size_population, size_individual)
    random_occurence_value = np.random.rand(size_population, size_individual)

    # normalize each idnividual of the population and remove if the random occurence value for that pigment is less than the occurence rate in the original dataset
    normalized_population = normalize_individual(population, max_pigment_values, offset=0.2)#*(random_occurence_value < occurences.values)

    # create the initial mask based on occurrence
    mask = (random_occurence_value < occurences.values)

    # force mask to True for the forced columns
    if forced_columns is not None:
        forced_indices = [max_pigment_values.index.get_loc(col) for col in forced_columns]
        mask[:, forced_indices] = True  # force these columns to be kept

    # apply mask
    normalized_population = normalized_population * mask

    # transform to df
    normalized_population = pd.DataFrame(normalized_population, columns=max_pigment_values.index)

    # set values at unused columns to 0
    if unused_columns is not None:
        normalized_population[unused_columns] = 0.0

    # todo: try
    
    return normalized_population


def tournament_selection(population_df, expectation, model, all_available_pigments, reflectance, tournament_size=5, internal_light_source='D65', uncertanity_bias=0.05, debug=False):
    """
    Selects parents from the population using tournament selection.

    In each tournament, a subset of the population is chosen (size = tournament_size) and 
    the individual with the best fitness is added to the parent pool.

    - parameters:
        `population_df` (pd.DataFrame): each row represents an individual.
        `expectation` (pd.DataFrame): 1-row dataframe with expected Lab or reflectance values.
        `model`: Model used to predict reflectance or Lab values from recipes
        `all_available_pigments`: list of all the available pigments
        `reflectance` (bool): Indicates wether it is working with reflectance or Lab
        `tournament_size` (int): number of individuals per tournament. (default: 5)
        `internal_light_source` (bool): internal light source to use in the fitenss function (default: D65)
        `uncertanity_bias`: weight of the uncertanity of the model predicting Lab/reflection from a recipe
        `debug` (bool): flag to print internal information. (default: False)

    - returns:
        (pd.DataFrame, pd.DataFrame): tuple of selected parents, best individual overall.
    """
    parent_idx = []
    population_size = len(population_df)
    num_parents = int(np.ceil(population_size / tournament_size))

    best_individual = None
    best_fitness = -np.inf

    for i in range(num_parents):
        start = i * tournament_size
        end = min((i + 1) * tournament_size, population_size)
        participants = population_df.iloc[start:end]

        # compute fitnesses for all participants in this group
        fitnesses = fitness(participants, expectation, model, all_available_pigments, reflectance=reflectance, internal_light_source=internal_light_source, uncertanity_bias=uncertanity_bias, debug=debug)[0]

        # get the index of the winner (relative to population)
        local_best = np.argmax(fitnesses)
        winner_id = start + local_best
        parent_idx.append(winner_id)

        # check for global best
        if fitnesses[local_best] >= best_fitness:
            best_fitness = fitnesses[local_best]
            best_individual = population_df.iloc[[winner_id]]

        if debug:
            print(f'Tournament {i+1} fitnesses: {fitnesses}')
            print(f'Max fitness in tournament: {fitnesses[local_best]}')

    parents_df = population_df.iloc[parent_idx].reset_index(drop=True)
    return parents_df, best_individual.reset_index(drop=True)


def enforce_sparse_limit(individual, max_nonzeros=15):
    # maybe use this as mutation
    """
    keeps only a random subset of non-zero values if they exceed max_nonzeros.
    all other non-zero values are set to zero.
    
    - parameters:
        `individual` (np.array): 1D array representing an individual
        `max_nonzeros` (int): maximum allowed number of non-zero values

    - returns:
        np.array: modified individual with limited number of non-zero values
    """
    nonzero_indices = np.where(individual > 0)[0]
    
    if len(nonzero_indices) > max_nonzeros:
        # randomly select indices to keep
        selected_indices = np.random.choice(nonzero_indices, size=max_nonzeros, replace=False)
        keep_mask = np.zeros_like(individual, dtype=bool)
        keep_mask[selected_indices] = True
        individual[~keep_mask] = 0.0

    return individual


def crossover(parents_df, offspring_size, forbidden_pigments=None):
    """
    **Generates offspring using arithmetic crossover between randomly selected pairs of parents.**

    - parameters:
        `parents_df` (pd.DataFrame): dataframe of selected parents (each row = 1 parent)
        `offspring_size` (int): number of offspring to produce
        `forbidden_pigments` (list of str): pigments that should not be used (default: None)

    - returns:
        pd.DataFrame: dataframe of offspring individuals
    """

    offspring = []
    columns = parents_df.columns
    for _ in range(offspring_size):
        p1, p2 = parents_df.sample(2, replace=False).values
        alpha = np.random.rand()

        # define a mask: only blend positions where at least one parent is non-zero
        nonzero_mask = (p1 != 0) | (p2 != 0)

        child = np.zeros_like(p1)
        child[nonzero_mask] = alpha * p1[nonzero_mask] + (1 - alpha) * p2[nonzero_mask]

        child = np.clip(child, 0, 1)
        child = enforce_sparse_limit(child, max_nonzeros=10)
        offspring.append(child)
    
    offspring_df = pd.DataFrame(offspring, columns=columns)

    if forbidden_pigments is not None:
        offspring_df[forbidden_pigments] = 0.0

    return offspring_df


def mutate(individual, nb_generation, total_generations, all_pigments, max_pigment_values, mandatory_pigments=None, forbidden_pigments=None, mutation_rate=0.1, min_mutation=0.7, max_mutation=1.3, zero_prob=0.05, new_pigment_prob=0.05):
    """
    **Mutates an individual by randomly adjusting up to half of its non-zero values.**

    - parameters:
        `individual` (np.array): 1D array of pigment concentrations
        `nb_generation` (int): The number of generation (>=1)
        `total_generations` (int): Total generations in the GA
        `all_pigments` (list of str): List of all the pigments in order
        `max_pigment_values` (pd.Series): Series containing the max values encountered in the historic dataset for each pigment
        `mandatory_pigments` (list of str): list of pigments that must be contained in the mutated individual
        `forbidden_pigments` (list of str): list of pigments that can't be contained in the mutated individual
        `mutation_rate` (float): probability that the individual undergoes mutation (default: 0.1)
        `min_mutation` (float): minimum multiplier for the mutation (default: 0.7)
        `max_mutation` (float): maximum multiplier for the mutation (default: 1.3)
        `zero_prob` (float): probability of randomly setting a gene to zero (default: 0.05)
        `new_pigment_prob` (float): probability of randomly setting a zero gene to non zero (default: 0.05)


    - returns:
        pd.Series: mutated individual
    """

    if nb_generation < 1:
        raise ValueError(f'nb_generation shoud be greater or equal to 1. Got {nb_generation} instead')
    if total_generations < nb_generation:
        raise ValueError(f'total_generations should not be inferior to nb_generation. Got {total_generations} < {nb_generation}')

    if np.random.rand() > mutation_rate:
        return individual  # no mutation

    mutated = individual.copy()
    nonzero_indices = np.where(mutated > 0)[0]
    zero_indices = np.where(mutated == 0)[0]

    nonzero_pigments = [all_pigments[i] for i in nonzero_indices]

    if len(nonzero_indices) == 0:
        return mutated  # nothing to mutate  

    num_to_mutate = max(1, len(nonzero_indices) // 2)
    #selected_indices = np.random.choice(nonzero_indices, size=num_to_mutate, replace=False)
    selected_indices = random.sample(nonzero_pigments, num_to_mutate)


    # progress from 0 (start) to 1 (final generation)
    progress = nb_generation / total_generations

    # linearly interpolate min/max toward 1.0 as generations increase
    adj_min = (1 - progress) * min_mutation + progress * 0.95
    adj_max = (1 - progress) * max_mutation + progress * 1.05

    for pigment in selected_indices:
        factor = np.random.uniform(adj_min, adj_max)
        mutated[pigment] *= factor

        # randomly zero out some mutated values
        if np.random.rand() < zero_prob:
            if mandatory_pigments is None or pigment not in mandatory_pigments:
                mutated[pigment] = 0.0
    
    # randomly activate a new pigment
    if len(zero_indices) > 0 and np.random.rand() < new_pigment_prob:
        new_idx = np.random.choice(zero_indices)
        pigment = all_pigments[new_idx]
        max_value = max_pigment_values[pigment] * 1.2
        new_value = (np.random.beta(1, 50, 1) * 6).clip(0, 1).item() # beta function
        mutated.iloc[new_idx] = new_value * max_value
    
    # set forbidden pigments to 0
    if forbidden_pigments is not None:
        mutated[forbidden_pigments] = 0.0


    # clip the if the maximum concentration of a pigment is above the limit
    # clip values between 0 and 20% of the corresponding max pigment value
    max_values_scaled = max_pigment_values.values * 0.2
    mutated = np.clip(mutated, 0, max_values_scaled)

    
    return mutated




def run_ga(expectation:pd.DataFrame, model, max_pigment_values:pd.Series, occurences:pd.Series, pipeline_dict_file:dict, all_available_pigments:list[str], generations:int=50, population:pd.DataFrame=None, population_size:int=30,
                          tournament_size:int=5, mutation_rate:float=0.2, min_mutation:float=0.7, max_mutation:float=1.3, zero_prob:float=0.05, uncertanity_bias:float = 0.05,
                          reflectance:bool=True, internal_light_source:str='D65', early_stopping:int=None, early_stopping_start:int=3, 
                          early_stopping_tolerance:float=0.01, mandatory_pigments:list=None, forbidden_pigments:list=None, expected_recipe:pd.DataFrame=None,
                          plot_fitness:bool=True, plot_best_colors:bool=True, visualization_offset:int=None, save_results:bool=True, save_final_result:bool=False, save_folder:str=None, progressbar_text:str='Running genetic algroithm', run_from_notebook:bool=True, debug:bool=False)->pd.Series:
    """
    **Runs the genetic algorithm to evolve pigment mixtures that approximate the target color. After each generation, it creates a new population from the mutated crossovers of the parents and the parents themselves.**

    - parameters:
        `expectation` (pd.DataFrame): target reflectance or Lab values (1 row)
        `model`: predictive model for reflectance or Lab
        `max_pigment_values` (pd.Series): Max value of each pigment in the historic dataset
        `occurences`: (pd.Series): Ratio of occurence of each pigment in the historic dataset
        `pipeline_dict_file`: Contains the models to use for the internal light source
        `all_available_pigments`: Contains all the pigments available
        `generations` (int): number of generations (default: 50)
        `population` (pd.DataFrame): DO NOT USE! Provide an initial population instead of a random one (default: None)
        `population_size` (int): number of individuals in the population (default: 30)
        `tournament_size` (int): tournament size for selection (default: 5)
        `mutation_rate` (float): mutation rate applied to each offspring (default: 0.2)
        `min_mutation` (float): lower mutation bound. The mutated value will be between value*min_mutation and value*max_mutation. min_mutation gradually gets closer to 1 with the generations (default: 0.7)
        `max_mutation` (float): upper mutation bound. The mutated value will be between value*min_mutation and value*max_mutation. max_mutation gradually gets closer to 1 with the generations (default: 1.3)
        `zero_prob` (float): probability to set a pigment to zero during mutation (default: 0.05)
        `uncertanity_bias` (float): weight of the uncertanity of the prediction model from recipe to Lab/reflectance (default: 0.05)
        `reflectance` (bool): whether the target values are reflectance (default: True)
        `internal_light_source` (str): light source for color conversion (default: 'D65')
        `early_stopping` (int): stop the GA if no improvements in early_stopping generations (default: None)
        `early_stopping_start` (int): start generation of the early stopping. No early stopping before this generation (default: 0)
        `early_stopping_tolearance` (float): the early stopping counter is not reset if the new fitness is in the tolerance range. Between 0 and 1 (default: 0.05)
        `pigments` (list): NOT IN USE, HAS NO EFFECT!! list of the pigments to use ['all'] means that all the available pigments are used (default: ['all'])
        `mandatory_pigments` (list): Pigments that must be included. Note that pigments that are not used in the historic recipes will have 0 concentration (default: None)
        `forbidden_pigments` (list): Pigments that are not allowed to be used (default: None)
        `expected_recipe` (pd.DataFrame): For experiments only. dataframe of 1 row containing the expected recipe. (default: None)
        `plot_fitness` (bool): plot the fitness for each generation (default: False)
        `plot_best_color` (bool): plot the color on the L a b space corresponding to the best fitness for each generation
        `visualization_offset` (int): better visualization of the best colors by multiplying the distance to the expected values of each L a b value respectively (default: None)
        `save_results` (bool): save the results and the plots of the GA (default: False) 
        `save_final_result` (bool): save the final output only (default: False)
        `save_folder` (str): parent folder where the results will be saved. If None, saves to /Code/GA/Results/. (default: None)
        `progressbar_text` (str): text to show on the progressbar description (default: 'Running genetic algroithm')
        `debug` (bool): whether to print debug information (default: False)

    - returns:
        pd.Series: best individual found
    """

    if population is not None:
        raise Exception('POPULATION IS NOT NONE! IT MUST BE SET TO NONE!!!!')
    
    if len(expectation) != 1:
        raise Exception("expectation_lab must be a pandas.DataFrame of 1 row!")
    
    all_pigments = list(max_pigment_values.index) #list(pandas.core.indexes.base.Index)


    # initialize population
    if population is not None:
        population_df = population
        population_size = len(population_df)
    else:
        population_df = create_population(population_size, max_pigment_values, occurences, forced_columns=mandatory_pigments, unused_columns=forbidden_pigments)

    best_fitnesses = []
    best_individuals_delta_e94 = []
    best_individuals_dEm = []
    best_individuals_pigment_number = []
    best_individuals = []

    best_fitness_early_stopping = 0
    best_individual_early_stopping = None
    early_stopping_counter = 0

    timestamp = datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y-%m-%dT%H:%M:%SZ')

    if save_results or save_final_result:
        if save_folder is None:
            save_path = f'{base}/Code/GA/Results/{timestamp}-{generations}gen-{population_size}pop-{tournament_size}tour-{mutation_rate}smartmut-metamerism'
        else:
            save_path = f'{save_folder}/{timestamp}-{generations}gen-{population_size}pop-{tournament_size}tour-{mutation_rate}smartmut'
        os.makedirs(save_path, exist_ok=True)

        with open(f'{save_path}/.notdone', 'w') as f:
            f.write(f"{datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
            f.write('INTERNAL FLAG TO INDICATE THAT THE GA HAS NOT COMPLETED FOR THIS TIMESTAMP!\nDO NOT DELETE OR MODIFY!!!!')


        expected_recipe.loc[:, ~(expected_recipe == 0.0).all()].to_csv(f'{save_path}/expected_recipe.csv', index=False)

        best_recipes_columns = ['generation', 'fitness', 'dE94', 'number_of_pigmnets', 'incertitude'] + all_pigments
        best_recipes_df = pd.DataFrame(columns=best_recipes_columns	)

    else:
        save_path = None

    
    if expected_recipe is not None:
        expected_recipe_fitness, expected_recipe_delta_e94s, expected_recipe_used_pigments, incertitudes, dE_metamerisms = fitness(expected_recipe, expectation, model, all_available_pigments, reflectance, internal_light_source, uncertanity_bias=uncertanity_bias)


    # same as in tournament_selection function
    num_parents = int(np.ceil(population_size / tournament_size))

    tqdm_module = get_tqdm(run_from_notebook)

    pbar = tqdm_module(range(generations), desc=progressbar_text, leave=False)

    for gen in range(generations): #tqdm(range(generations), desc=progressbar_text, leave=False):
        
        if debug:
            print(f'{datetime.now()} beginning of generation {gen+1}')
            print(f"\nGeneration {gen + 1}")

        # select parents
        parents_df, best_individual = tournament_selection(population_df, expectation, model, all_available_pigments, reflectance,
                                          tournament_size=tournament_size, internal_light_source=internal_light_source, uncertanity_bias=uncertanity_bias, debug=debug)
        
        if debug: print(f'{datetime.now()} done tournament selection')

        # generate offspring via crossover
        offspring_df = crossover(parents_df, offspring_size=population_size - num_parents, forbidden_pigments=forbidden_pigments)

        if debug: print(f'{datetime.now()} done crossover')

        # apply mutation to each offspring
        mutated_offspring = offspring_df.apply(
            lambda row: mutate(row, gen+1, generations, all_pigments, max_pigment_values, mandatory_pigments=mandatory_pigments, forbidden_pigments=forbidden_pigments, mutation_rate=mutation_rate, min_mutation=min_mutation, max_mutation=max_mutation, zero_prob=zero_prob), axis=1, result_type='expand'
        )
        #mutated_offspring.columns = offspring_df.columns

        if debug: print(f'{datetime.now()} done mutation')

        # new population: elitism (keep parents) + mutated offspring
        population_df = pd.concat([parents_df, mutated_offspring], ignore_index=True)

        if plot_fitness or save_results or (early_stopping is not None and gen >= early_stopping_start):

            # compute the best fitness of this population
            gen_fitnesses, gen_delta_e94s, gen_used_pigments, incertitudes, dE_metamerisms = fitness(population_df, expectation, model, all_available_pigments, reflectance, internal_light_source, uncertanity_bias=uncertanity_bias)
            mean_fitness = np.mean(gen_fitnesses)
            best_idx = np.argmax(gen_fitnesses)
            best_fitnesses.append(gen_fitnesses[best_idx])
            best_individuals_delta_e94.append(gen_delta_e94s[best_idx])
            best_individuals_dEm.append(dE_metamerisms[best_idx])
            best_individuals_pigment_number.append(gen_used_pigments[best_idx])
            best_individuals.append(population_df.iloc[best_idx])

            if save_results:
                best_individual_gen = [gen+1, gen_fitnesses[best_idx], gen_delta_e94s[best_idx], gen_used_pigments[best_idx], incertitudes[best_idx]] + population_df.iloc[best_idx].tolist()
                best_recipes_df.loc[gen] = best_individual_gen

            # create the main figure and axis
            fig, ax1 = plt.subplots()

            ax1.plot(gen_fitnesses, label='Fitness values')
            ax1.axhline(mean_fitness, color='orange', linestyle='--', label=f'Mean fitness {mean_fitness:.4f}')
            if expected_recipe is not None:
                ax1.axhline(expected_recipe_fitness[0], color='green', linestyle='--', label=f'Expected recipe fitness {expected_recipe_fitness[0]:.4f}')
            ax1.set_xlabel('Individual ID')
            ax1.set_ylabel('Fitness value')
            ax1.set_title(f'Fitness values for generation {gen+1} of timestamp {timestamp}')

            # # create a secondary y-axis sharing the same x-axis
            # ax2 = ax1.twinx()

            # # plot the mean fitness as a horizontal line
            # ax2.plot(gen_delta_e94s, color='green', label='Delta E94', linestyle='--')
            # ax2.plot(gen_used_pigments, color='purple', label='Number of used pigments', linestyle='--')
            # ax2.set_ylabel('Number of used pigments and Delta E94')



            ax1.legend()
            #ax2.legend()

            plt.title(f'Fitness values for generation {gen+1} of timestamp {timestamp}')
            plt.tight_layout()
            if save_results:
                plt.savefig(f'{save_path}/{timestamp}_gen{gen+1}.png', bbox_inches='tight')
            if plot_fitness:
                plt.show()
            plt.close()

            
            
            if early_stopping is not None and gen >= early_stopping_start:
                if gen_fitnesses[best_idx] > best_fitness_early_stopping*(1+early_stopping_tolerance):
                    best_fitness_early_stopping = gen_fitnesses[best_idx]
                    best_individual_early_stopping = population_df.iloc[[best_idx]]
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                #print(f'Best fitness:                    {gen_fitnesses[best_idx]}')
                #print(f'Best fitenss for early stopping: {best_fitness_early_stopping}')

                if early_stopping_counter >= early_stopping:
                    if debug: print(f'Early stopping at generation {gen+1}, returning the best candidate from generation {(gen+1)-early_stopping_counter}')

                    if save_results:
                        with open(f'{save_path}/output.txt', 'a') as f:
                            f.write(f'Early stop at generation:       {gen-early_stopping_counter}\n')

                    break
        
        if debug: print(f'{datetime.now()} done generation {gen+1}')

        pbar.update(1)
            

    # final evaluation

    if early_stopping is None or early_stopping_counter < early_stopping:
        fitnesses = population_df.apply(
            lambda row: fitness(row.to_frame().T, expectation, model, all_available_pigments, reflectance, internal_light_source, uncertanity_bias=uncertanity_bias)[0], axis=1
        )
        best_idx = fitnesses.idxmax()
        best_individual = population_df.iloc[[best_idx]]
        best_fitness = fitnesses[best_idx]
    
    else:
        best_individual = best_individual_early_stopping

    # plot fitnesses
    if plot_fitness or save_results:

       # create the main figure and axis
        fig, ax1 = plt.subplots()

        x_vals = range(1, len(best_fitnesses) + 1)

        # plot on primary y-axis
        scatter1 = ax1.scatter(x_vals, best_fitnesses, color='blue', label='Best fitnesses')
        line1, = ax1.plot(x_vals, best_fitnesses, color='blue', linewidth=0.8)  # thin line

        # add labels and title
        ax1.set_xlabel('Generations')
        ax1.set_ylabel('Max fitness value')
        ax1.set_title('Maximum fitness value for each generation')

        # create a secondary y-axis
        ax2 = ax1.twinx()

        # plot on secondary y-axis
        scatter2 = ax2.scatter(x_vals, best_individuals_delta_e94, color='green', label='Delta E94')
        line2, = ax2.plot(x_vals, best_individuals_delta_e94, color='green', linewidth=0.8)

        scatter3 = ax2.scatter(x_vals, best_individuals_pigment_number, color='purple', label='Pigment number')
        line3, = ax2.plot(x_vals, best_individuals_pigment_number, color='purple', linewidth=0.8)

        scatter4 = ax2.scatter(x_vals, best_individuals_dEm, color='gray', label='Delta Metamerism')
        line4, = ax2.plot(x_vals, best_individuals_dEm, color='gray', linewidth=0.8)

        ax2.set_ylabel('Delta E94, Pigment number and Delta Metamerism')

        # combine all handles and labels into one legend
        handles = [scatter1, scatter2, scatter3, scatter4]
        labels = [h.get_label() for h in handles]

        # place legend outside the plot on the right
        ax1.legend(handles, labels, loc='center left', bbox_to_anchor=(1.1, 0.5))

        # set x-ticks at every integer unit
        ax1.set_xticks(x_vals)

        # add grid with vertical lines at each x-tick
        ax1.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)

        if save_results or save_final_result:
            fig.savefig(f'{save_path}/{timestamp}_best-fitnesses.png', bbox_inches='tight')


    if save_results or save_final_result:

        expectation.to_csv(f'{save_path}/expectation.csv', index=False)

        hyperparameters = {
            'model': str(type(model).__name__),
            'generations': generations, 
            'population_size': population_size,
            'tournament_size': tournament_size, 
            'mutation_rate': mutation_rate,
            'min_mutation': min_mutation, 
            'max_mutation': max_mutation,
            'zero_prob': zero_prob,
            'reflectance': reflectance,
            'early_stopping': early_stopping,
            'early_stopping_tolerance': early_stopping_tolerance,
            'visualization_offset': visualization_offset,
            'uncertanity_bias': uncertanity_bias
        }

        with open(f'{save_path}/hyperparameters.json', 'w') as f:
            json.dump(hyperparameters, f, indent=4)

        best_individual_save = best_individual.loc[:, ~(best_individual == 0.0).all()]
        best_individual_save.to_csv(f'{save_path}/best_individual.csv', index=False)

        if save_results:
            best_recipes_df = best_recipes_df.loc[:, ~(best_recipes_df == 0.0).all()]
            best_recipes_df.to_csv(f'{save_path}/best_recipes.csv', index=False)

    # show the plot
    if plot_fitness:
        plt.show()
    plt.close()


    if save_results or plot_best_colors:

        if save_results:
            spath = f'{save_path}/best_individuals_representation.png'
        else:
            spath = None

        if internal_light_source == 'F2':
            internal_model = pipeline_dict_file['model']['Lab']
        elif internal_light_source == 'D65':
            internal_model = pipeline_dict_file['model']['LabD65']
        elif internal_light_source == 'StudioLED':
            internal_model = pipeline_dict_file['model']['LabStudioLED']
        else:
            raise ValueError('The internal light source is not valid. Should be F2, D65 or StudioLED')
        
        best_individuals_extended_df = complete_individuals(pd.DataFrame(best_individuals).reset_index(drop=True), all_available_pigments)
            
        best_individuals_lab = recipe_to_lab(best_individuals_extended_df, internal_model)
        #expectation_lab = expectation.pipe(UtilMethods.addLabcols, internal_light_source)[[f'L_{internal_light_source}', f'a_{internal_light_source}', f'b_{internal_light_source}']].iloc[[0]].rename(columns={f'L_{internal_light_source}': 'L', f'a_{internal_light_source}': 'a', f'b_{internal_light_source}': 'b'}).reset_index(drop=True)
        expectation_lab = expectation.pipe(UtilMethods.addLabcols, internal_light_source)[[f'L_{internal_light_source}', f'a_{internal_light_source}', f'b_{internal_light_source}']].iloc[[0]].rename(columns={f'L_{internal_light_source}': 'L', f'a_{internal_light_source}': 'a', f'b_{internal_light_source}': 'b'}).reset_index(drop=True)
        UtilMethods.visualize_lab(best_individuals_lab, expectation_lab, show_plot=plot_best_colors, save_path=spath, title=f'Best individuals for each generation for timestamp {timestamp}', numbering=True)
        UtilMethods.visualize_lab(best_individuals_lab, expectation_lab, show_plot=plot_best_colors, save_path=spath.replace('.png', '_interactive.html'), title=f'Best individuals for each generation for timestamp {timestamp}', numbering=True, interactive=True)
        UtilMethods.visualize_lab(best_individuals_lab, expectation_lab, show_plot=plot_best_colors, save_path=spath.replace('.png', '_3d.html'), title=f'Best individuals for each generation for timestamp {timestamp}', numbering=True, threed_plot=True)


        
        if save_results:
            best_individuals_lab.to_csv(f'{save_path}/best_Lab.csv', index=False)
            expectation_lab.to_csv(f'{save_path}/expectation_Lab.csv', index=False)


        # multiply the L a b distances with an offset (visualization_offset), so the differnces are more pronounced
        if visualization_offset is not None:
            offset_best_individuals_lab = best_individuals_lab.copy(deep=True)
            offset_best_individuals_lab = expectation_lab.iloc[0] + ((best_individuals_lab - expectation_lab.iloc[0]) * visualization_offset)
            #offset_best_individuals_lab = offset_best_individuals_lab*offset_best_individuals_lab

            if save_results: offset_best_individuals_lab.to_csv(f'{save_path}/best_Lab_with_offset_unclipped.csv', index=False)


            # clip the values to stay within CIELAB bounds
            offset_best_individuals_lab['L'] = offset_best_individuals_lab['L'].clip(lower=0, upper=100)
            offset_best_individuals_lab['a'] = offset_best_individuals_lab['a'].clip(lower=-128, upper=127)
            offset_best_individuals_lab['b'] = offset_best_individuals_lab['b'].clip(lower=-128, upper=127)

            if save_results:
                spath = f'{save_path}/best_individuals_representation_with_offset.png'

                offset_best_individuals_lab.to_csv(f'{save_path}/best_Lab_with_offset.csv', index=False)

            UtilMethods.visualize_lab(offset_best_individuals_lab, expectation_lab, show_plot=plot_best_colors, save_path=spath, title=f'Best individuals for each generation for timestamp {timestamp}\n*{visualization_offset} offset on the L a b distance compared to the target', numbering=True)
            UtilMethods.visualize_lab(offset_best_individuals_lab, expectation_lab, show_plot=plot_best_colors, save_path=spath.replace('.png', '_interactive.html'), title=f'Best individuals for each generation for timestamp {timestamp}<br>*{visualization_offset} offset on the L a b distance compared to the target', numbering=True, interactive=True)
            UtilMethods.visualize_lab(offset_best_individuals_lab, expectation_lab, show_plot=plot_best_colors, save_path=spath.replace('.png', '_3d.html'), title=f'Best individuals for each generation for timestamp {timestamp}<br>*{visualization_offset} offset on the L a b distance compared to the target', numbering=True, threed_plot=True)




    # Evaluate the best individual
    fitness(best_individual.iloc[[0]], expectation, model, all_available_pigments,
        reflectance=reflectance,
        internal_light_source=internal_light_source, save_results=save_path,
        uncertanity_bias=uncertanity_bias, debug=debug)
    
    if save_results or save_final_result:
        with open(f'{save_path}/output.txt', 'a') as f:
            f.write(f'Mandatory pigments:             {mandatory_pigments}\n')
            f.write(f'Forbidden pigments:             {forbidden_pigments}\n')
            f.write(f'The metamersim is addressed by taking the weighted average of the dE94 color distances of D65, FL2 and Studio LED when comparing the predicted Lab representation of a recipe to the Lab representation of the expected color\n')
            f.write(f'0.4*D65+0.3*FL2+0.3*Studio LED\n')
        
        with open(f'{save_path}/.done', 'w') as f:
            f.write(f"{datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
            f.write('INTERNAL FLAG TO INDICATE THAT THE GA HAS COMPLETED FOR THIS TIMESTAMP!\nTHIS FLAG OVERWRITES THE .notdone FLAG IF NOT ERASED. WITH THIS FLAG PRESENT, THE .notdone FLAG IS MEANINGLESS.\nDO NOT DELETE OR MODIFY!!!!')

        if os.path.exists(f'{save_path}/.notdone'):
            os.remove(f'{save_path}/.notdone')

    
    pbar.clear()
    pbar.close()

    
    return best_individual