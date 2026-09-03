


# # Incorporation of GA and NN (reflectance only)


import os
import sys
import pandas as pd
from dotenv import load_dotenv
from Code.Arnold_models import PAR_ModellingFunctions_pipelines
from Code.Utils.util_methods import UtilMethods
from keras.models import load_model
from Code.NN.utils.keras_functions import binary_focal_loss
import pickle
import shutil
from tqdm import tqdm
from Code.GA.GA_scripts import run_ga
from datetime import datetime
from zoneinfo import ZoneInfo
from Code.GANN.utils.utils import summarize_results_GANN

sys.modules['PAR_ModellingFunctions_pipelines'] = PAR_ModellingFunctions_pipelines

base = UtilMethods.find_project_root(os.getcwd())
print(f"Project root found: {base}")

if load_dotenv(f'{base}/.env'):
    print(".env found")
else:
    print("ERROR .env not found")


# ## Variables


NN_MODEL_DIR = 'prc-split-val-small-threshold-04/tuned_model_435'
NN_MODEL_PATH = f'{base}/Code/NN/Results/Tuning/{NN_MODEL_DIR}'
THRESHOLD = 0.4
RESULT_FOLDER = f'{base}/Code/GANN/Results/{NN_MODEL_DIR}'
CONFIDENCE_OCCURENCE = True # used the confidence of the predictions instead of the occurences
STRICT_PIGMENT_USE = True # only uses the predicted pigments (confidence above threshold)


# ## Import the data


X = pd.read_csv(f'{base}/Dataset/traintest/X.csv')
y = pd.read_csv(f'{base}/Dataset/traintest/y.csv')
all_pigments = list(y.columns)


# ## Load NN model


NN_model = load_model(f"{NN_MODEL_PATH}/model.keras", custom_objects={"loss": binary_focal_loss(gamma=2.0, alpha=0.8)})


# ## Load Lab prediction model


with open(f'{base}/Dataset/Arnold/GP_pipeline_models_dict_16April2025.pkl', 'rb') as f:
    pipeline_dict_file = pickle.load(f)


LAB_model =  pipeline_dict_file['model']['curve']


# ## Run the GA with pigment prediction


max_pigment_values = y.max()
max_pigment_values


# predict the pigments to use:
NN_result = NN_model.predict(X, verbose=0)
NN_result_bin = (NN_result >= 0.4).astype(int)
NN_result_bin_df = pd.DataFrame(NN_result_bin, columns=all_pigments)
row_df = NN_result_bin_df.loc[[0]]
columns_to_use = list(row_df.loc[:, row_df.iloc[0] != 0].columns)
columns_to_use


max_pigment_values[columns_to_use]


# ratio of appearance of each pigment 
occurences = (y>0).sum()/len(y)
occurences



NN_result = NN_model.predict(X, verbose=1)
NN_result_bin = (NN_result >= THRESHOLD).astype(int)
NN_result_bin_df = pd.DataFrame(NN_result_bin, columns=all_pigments)
NN_result_df = pd.DataFrame(NN_result, columns=all_pigments)

step = 10

for i in tqdm(range(0, len(X)), desc=f'Iterating through all recipes by steps of {step}'):

    if i % step != 0:
        continue

    save_main_dir = f'{RESULT_FOLDER}/recipe_{i}'

    if os.path.exists(save_main_dir):
        if os.path.exists(f'{save_main_dir}/.done'):
            continue
        else:
            # delte the folder and recreate later
            shutil.rmtree(save_main_dir)

    row = y.iloc[[i]]

    count_per_row = (row > 0.0).sum(axis=1).iloc[0]
    if count_per_row <= 1:
        continue
    
    row_df = NN_result_bin_df.loc[[i]]
    columns_to_use = list(row_df.loc[:, row_df.iloc[0] != 0].columns)

    possibilities = [True, False]

    os.makedirs(save_main_dir, exist_ok=True)

    # save the model predictions
    NN_result_df.iloc[[i]].to_csv(f'{save_main_dir}/model_prediction.csv', index=False)
    NN_result_df[columns_to_use].iloc[[i]].to_csv(f'{save_main_dir}/predicted_pigments.csv', index=False)

    c=-1
    additions = ['sp-co','sp','co','-']
    for a in possibilities:
        for b in possibilities:

            c+=1

            if c in [0,1,3]:
                continue
            
            strict_pigment_use = a
            confidence_occurence = b

            save_folder = f'{save_main_dir}/{additions[c]}'

            if strict_pigment_use:
                max_pigment_values_row = max_pigment_values[columns_to_use]
                if confidence_occurence:
                    occurences_row = NN_result_df.loc[i, columns_to_use]
                else:
                    occurences_row = occurences[columns_to_use]
            else:
                max_pigment_values_row = max_pigment_values
                if confidence_occurence:
                    occurences_row = NN_result_df.iloc[i]
                else:
                    continue
        

            for j in range(3):
                run_ga(X.iloc[[i]], LAB_model, max_pigment_values_row, occurences_row, pipeline_dict_file, all_pigments, population=None, generations=200, population_size=300,
                                            tournament_size=30, mutation_rate=0.5, min_mutation=0.4, max_mutation=1.6, zero_prob=0.05,
                                            reflectance=True, early_stopping=5, early_stopping_start=3, early_stopping_tolerance=0.0005,
                                            mandatory_pigments=None, forbidden_pigments=None, expected_recipe=row, uncertanity_bias=0.05,
                                            plot_fitness=False, save_results=True, save_final_result=True, debug=False, plot_best_colors=False, 
                                            save_folder=save_folder, visualization_offset=50, progressbar_text=f'Running genetic algroithm with NN ({additions[c]}) for recipe {i} - {j+1}', run_from_notebook=False)
            

    
    with open(f'{save_main_dir}/.done', 'w') as f:
        f.write(f"{datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        f.write('INTERNAL FLAG TO INDICATE THAT THE GA HAS COMPLETED FOR THIS RECIPE!\nDO NOT DELETE OR MODIFY!!!!')







summary_df = summarize_results_GANN(RESULT_FOLDER, modulo=1)


# (
#     summary_df
#     .groupby('Type').agg(Fitavg= ('Fitness','mean'),Fitstd=('Fitness','std'),N=('Fitness','count'))
# )


# summary_df.groupby('Recipe').agg(N=('Fitness','count'))


# summary_df.loc[lambda df:df['Recipe']=='recipe_20']





