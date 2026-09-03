from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import Precision, Recall, CategoricalAccuracy, BinaryAccuracy
from Code.NN.utils.keras_functions import binary_focal_loss
from Code.NN.utils.random_combinations import create_hyperparameter_combinations
from Code.NN.utils.utils import default_serializer, results_summarizer
from Code.Utils.util_methods import UtilMethods
import keras.backend as K
from keras.utils import set_random_seed


base = UtilMethods.find_project_root(os.getcwd())
print(f"Project root found: {base}")

if load_dotenv(f'{base}/.env'):
    print(".env found")
else:
    print("ERROR .env not found")

def get_tqdm(use_notebook=False):
    if use_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    return tqdm

def load_data(val:int=True)->tuple:
    """
    **Loads the train and test datasets from Dataset/testrain folder**

    - parameters:
        `val` (bool): flac indicating if validation set should also be retourned

    - returns:
        6 or 4 * Pandas.DataFrame: X_train, y_train, X_test, y_test(, X_val, y_val)
    """
    X_train = pd.read_csv(f'{base}/Dataset/traintest/X_train.csv')
    X_test = pd.read_csv(f'{base}/Dataset/traintest/X_test.csv')
    y_train = pd.read_csv(f'{base}/Dataset/traintest/y_train.csv')
    y_test = pd.read_csv(f'{base}/Dataset/traintest/y_test.csv')
    if val:
        X_val = pd.read_csv(f'{base}/Dataset/traintest/X_val.csv')
        y_val = pd.read_csv(f'{base}/Dataset/traintest/y_val.csv')

    # Discretize the y data
    y_train = (y_train > 0).astype(int)
    y_test = (y_test > 0).astype(int)
    if val: 
        y_val = (y_val > 0).astype(int)

        return X_train, y_train, X_test, y_test, X_val, y_val

    return X_train, y_train, X_test, y_test

def compare_binary_dataframes(expected_df: pd.DataFrame, predicted_df: pd.DataFrame) -> pd.DataFrame:
    '''
    **Compares the expected pigments to the predicted pigments**

    - parameters:
        `expected_df` (pd.DataFrame): Contains the expected recipes (likely the y_test dataframe)
        `predicted_df` (pd.DataFrame): Contains the predicted recipes, binarized.

    - returns:
        pd.Dataframe containing comparison informations about the given dataframes. Columns: ['Expected pigment #', 'Predicted pigment #', 'Pigment match #', 'Match ratio', 'Overprediction ratio', 'Expected pigments', 'Predicted pigments', 'Pigment match']
        dict containing a summary. Keys: ['complete_match_total', 'complete_match_ratio', 'match_total', 'match_ratio', 'exact_predictions', 'matched_pigments', 'average_prediction_ratio', 'average_predicted_pigment_number', 'average_expected_pigment_number']

    '''

    if not expected_df.columns.equals(predicted_df.columns) or len(expected_df) != len(predicted_df):
        raise ValueError("dataframes must have the same shape and columns")
    
    def get_indices(row):
        return expected_df.columns[row.astype(bool)].tolist()
    
    # find positions of 1s
    cols_expected_df = expected_df.apply(get_indices, axis=1)
    cols_predicted_df = predicted_df.apply(lambda row: predicted_df.columns[row.astype(bool)].tolist(), axis=1)
    cols_overlap = (expected_df & predicted_df).apply(lambda row: expected_df.columns[row.astype(bool)].tolist(), axis=1)

    # count 1s
    count_expected_df = expected_df.sum(axis=1)
    count_predicted_df = predicted_df.sum(axis=1)
    overlap_count = (expected_df & predicted_df).sum(axis=1)

    # compute ratio of overlap relative to expected_df
    overlap_ratio_expected_df = np.where(count_expected_df != 0, overlap_count / count_expected_df, 0)

    # compute the ratio of overprediction
    prediction_ratio = np.where(count_expected_df != 0, count_predicted_df / count_expected_df, 0)

    # combine into result
    result = pd.DataFrame({
        'Expected pigment #': count_expected_df,
        'Predicted pigment #': count_predicted_df,
        'Pigment match #': overlap_count,
        'Match ratio': overlap_ratio_expected_df,
        'Prediction ratio': prediction_ratio,
        'Expected pigments': cols_expected_df,
        'Predicted pigments': cols_predicted_df,
        'Pigment match': cols_overlap
    })

    # compute the summary of the comparison
    complete_match_total = ((result['Match ratio'] == 1.0) & (result['Prediction ratio'] == 1.0)).sum()
    complete_match_ratio = complete_match_total / len(result)

    match_total = (result['Match ratio'] == 1.0).sum()
    match_ratio = match_total / len(result)

    average_prediction_ratio = result['Prediction ratio'].mean()

    average_predicted_pigment_number = result['Predicted pigment #'].mean()
    average_expected_pigment_number = result['Expected pigment #'].mean()

    exact_predictions = complete_match_total / len(result)
    matched_pigments = match_total / len(result)

    summary_dict ={
        'complete_match_total': complete_match_total,
        'complete_match_ratio': complete_match_ratio,
        'match_total': match_total,
        'match_ratio': match_ratio,
        'exact_predictions': exact_predictions,
        'matched_pigments': matched_pigments,
        'average_prediction_ratio': average_prediction_ratio,
        'average_predicted_pigment_number': average_predicted_pigment_number,
        'average_expected_pigment_number': average_expected_pigment_number
    }

    return result, summary_dict

def compute_confidence(pred_df, threshold=0.5):
    '''
    **Computes the confidence of the predictions for being True and False. It takes the threshold and computes how confident the model is to predict a class for above the threshold (True) and below the Threshold (False)**

    - parameters:
        `pred_df` (pd.dataframe): dataframe of model probabilities
        `threshold` (float): cutoff for splitting values

    - returns:
        pd.dataframe: per-column stats above and below the threshold
    '''
    # compute the confidence for each pigment

    # create empty dictionaries to store results
    results = {
        'mean_below': {},
        'std_below': {},
        'mean_above': {},
        'std_above': {}
    }

    # iterate over each column
    for col in pred_df.columns:
        values = pred_df[col]

        # split the values based on the threshold
        below = values[values < threshold]
        above = values[values >= threshold]

        # compute mean and std
        results['mean_below'][col] = below.mean() if not below.empty else np.nan
        results['std_below'][col] = below.std() if not below.empty else np.nan
        results['mean_above'][col] = above.mean() if not above.empty else np.nan
        results['std_above'][col] = above.std() if not above.empty else np.nan

    # convert results to dataframe
    stats_df = pd.DataFrame(results)
    
    return stats_df

def train_model(X_train, y_train, X_val, y_val, hyperparameters, save_directory, early_stopping_patience=10, debug=False):
    '''
    **Trains the neural network model**

    - parameters:
        `X_train` (pd.dataframe): training features 
        `y_train` (pd.dataframe): training labels
        `X_val` (pd.dataframe): validation features
        `y_val` (pd.dataframe): validation labels
        `hyperparameters` (dict): model configuration
        `save_directory` (str): output folder
        `early_stopping_patience` (int): patience for early stopping
        `debug` (bool): verbose output flag

    - returns:
        tuple[tf.keras.model, tf.keras.callbacks.history]: the trained model and its training history
    '''
    
    set_random_seed(42)

    verbose = 2 if debug else 0

    model_name = save_directory.split('/')[-1]

    hyperparameters['early_stopping_patience'] = early_stopping_patience

    # save the train and val sets for in case
    X_train.to_csv(f'{save_directory}/X_train.csv', index=False)
    y_train.to_csv(f'{save_directory}/y_train.csv', index=False)
    X_val.to_csv(f'{save_directory}/X_val.csv', index=False)
    y_val.to_csv(f'{save_directory}/y_val.csv', index=False)

    # define the model
    nb_hidden_layers = len(hyperparameters['layer_sizes'])
    hidden_layers = hyperparameters['layer_sizes']
    learning_rate = hyperparameters['learning_rate']
    batch_size = hyperparameters['batch_size']
    dropout = hyperparameters['dropout']

    with open(f'{save_directory}/model_summary.txt', "w") as f:
        f.write(f'{model_name}\n')
        f.write(f'Training start time: {datetime.now()}\n')
        f.write('------------------------------------------\n')
        f.write(f'Hyperparameters:\n')
        f.write(f'  Hidden layer:  ------------ {hidden_layers}\n')
        f.write(f'  Learning rate: ------------ {learning_rate}\n')
        f.write(f'  Batch size: --------------- {batch_size}\n')
        f.write(f'  Dropout: ------------------ {dropout}\n')
        f.write(f'  Early stopping patience:--- {early_stopping_patience}\n')
        f.write('------------------------------------------\n')

    with open(f"{save_directory}/hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=4) 


    model = Sequential()

    for i in range(nb_hidden_layers):
        if i == 0:
            model.add(Dense(hidden_layers[i], input_shape=(X_train.shape[1],), activation='relu'))
            model.add(BatchNormalization())
        else:
            model.add(Dense(hidden_layers[i], activation='relu'))
        model.add(Dropout(dropout))
    
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    # compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss = binary_focal_loss(gamma=2.0, alpha=0.8),
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), BinaryAccuracy(name='binary_accuracy')]
    )

    # define early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)

    # train the model
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=batch_size,
        #validation_split=0.2,
        validation_data = (X_val, y_val),
        callbacks=[early_stop],
        verbose=verbose
    )
    train_stop_timestamp = datetime.now()

    model.save(f'{save_directory}/model.keras')

    nb_train_epochs = len(history.history['loss'])

    with open(f'{save_directory}/history.json', 'w') as f:
        json.dump(history.history, f, indent=4)

    with open(f'{save_directory}/model_summary.txt', "a") as f:
        f.write(f'Training stopped at {train_stop_timestamp}\n')
        f.write(f'Trained over {nb_train_epochs} epochs with early stopping\n')
        f.write(f'Model saved at: {save_directory}\n')
        f.write('------------------------------------------\n')
        model.summary(print_fn=lambda x: f.write(x + "\n"))
        f.write('------------------------------------------\n')

    if debug:
        print(model.summary())

    return model, history


def evaluate_model(X_test, y_test, save_directory, threshold, predict=True, debug=False):
    '''
    **Evaluates a trained neural network model**

    - parameters:
        `x_test` (pd.dataframe): test features
        `y_test` (pd.dataframe): test labels
        `save_directory` (str): folder with saved model
        `threshold` (float): binarization cutoff for predictions
        `predict` (bool): predict on the trained model or load the existing preictions if any (default: True)
        `debug` (bool): flag for debug output (default: False)

    - returns:
        none
    '''

    verbose = 2 if debug else 0

    with open(f'{save_directory}/model_summary.txt', "a") as f:
        f.write('------------------------------------------\n')
        f.write(f'Evaluation:\n\n')


    # load the model
    model = load_model(f"{save_directory}/model.keras", custom_objects={"loss": binary_focal_loss(gamma=2.0, alpha=0.8)})

    if not predict and os.path.exists(f'{save_directory}/.predictions.csv'):
        #read the existing preictions
        y_pred_df = pd.read_csv(f'{save_directory}/.predictions.csv')
    else:
        # predict on the test set
        y_pred = model.predict(X_test, verbose=verbose)
        y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns)
    
    y_pred_bin_df = (y_pred_df > threshold).astype(int)
    
    # save the predictions
    y_pred_df.to_csv(f'{save_directory}/predictions.csv', index=False)
    y_pred_bin_df.to_csv(f'{save_directory}/thresholded_preictions.csv', index=False)

    # save the test set for in case
    X_test.to_csv(f'{save_directory}/X_test.csv', index=False)
    y_test.to_csv(f'{save_directory}/y_test.csv', index=False)

    # evaluate the model
    eval_dict = model.evaluate(X_test, y_test, verbose=verbose, return_dict=True)
    with open(f'{save_directory}/test_performance.json', 'w') as f:
        json.dump(eval_dict, f, indent=4)

    with open(f'{save_directory}/model_summary.txt', "a") as f:
         f.write(f'Loss: ------------ {eval_dict["loss"]}\n')
         f.write(f'Precision: ------- {eval_dict["precision"]}\n')
         f.write(f'Recall: ---------- {eval_dict["recall"]}\n')
         f.write(f'Binary accuracy: - {eval_dict["binary_accuracy"]}\n')
         f.write(f'Threshold: ------- {threshold}\n\n')

    # compare the prediction to the expectations
    comparison_df, comparison_summary_dict = compare_binary_dataframes(y_test, y_pred_bin_df)
    comparison_df.to_csv(f'{save_directory}/comparison.csv', index=False)

    with open(f'{save_directory}/comparison_summary.json', 'w') as f:
        json.dump(comparison_summary_dict, f, default=default_serializer, indent=4)

    with open(f'{save_directory}/model_summary.txt', "a") as f:
        f.write(f'Exact predictions:  {comparison_summary_dict["exact_predictions"]}  ({(comparison_summary_dict["complete_match_ratio"]*100):.2f}%)\n')
        f.write(f'Matched pigments:   {comparison_summary_dict["matched_pigments"]}  ({(comparison_summary_dict["match_ratio"]*100):.2f}%)\n')
        f.write(f'Average # of predicted pigments ratio compared to the expected # of pigments:   {comparison_summary_dict["average_prediction_ratio"]*100:.2f}%\n')
        f.write(f'Average # of expected pigments:     {comparison_summary_dict["average_expected_pigment_number"]:.2f}\n')
        f.write(f'Average # of predicted pigments:    {comparison_summary_dict["average_predicted_pigment_number"]:.2f}\n')

    confidence_df = compute_confidence(y_pred_df, threshold=threshold)

    confidence_df.to_csv(f'{save_directory}/confidence.csv', index=False)


def re_evaluate_models(directory:str, threshold:float=0.5, predict:bool=False, debug:bool=False)->None:
    '''
    **Re evaluates the models in a given directory**

    - parameters:
        `directory` (str): Path to the directory to evaluate in
        `threshold` (float): Threshold to evaluate on (default: 0.5)
        `predict` (bool): Re-predict or use the existing preictions (default: False)
        `debug` (bool): flag that indicates if debug prints are shown (default: False)
    '''

    base_dir = Path(directory)

    # get and sort folders by creation time
    folders = sorted([f for f in base_dir.iterdir() if f.is_dir()],
                    key=lambda x: x.stat().st_ctime)
    
    tqdm_func = get_tqdm(use_notebook=True)
    
    pbar = tqdm_func(range(len(folders)), desc=f'Re-evaluating models in directory {directory}')
    for folder in folders:
        # path to the folder
        folder_path = f'{directory}/{folder.name}'

        # skip if the model is not fully trained or evaluated yet
        if not os.path.exists(f'{folder_path}/.done'):
            continue

        X_test = pd.read_csv(f'{folder_path}/X_test.csv')
        y_test = pd.read_csv(f'{folder_path}/y_test.csv')

        evaluate_model(X_test, y_test, folder_path, threshold=threshold, predict=predict, debug=debug)

        pbar.update(1)



         



def run_NN_tuning(nb_tirals:int, save_path:str, hyperparameter_ranges:dict, start:int=0, fixed_hyperparameters:dict=None, manual_combinations:list[dict]=None, early_stopping_patience:int=10, run_from_main:bool=False, debug:bool=False)->None:
    '''
    **Runs the tuning for the neural network**

    - parameters:
        `nb_tirals` (int): number of models to train. It will first start with the manual combinations and if there are still trials left it will generate random combinations
        `save_path` (str): base output directory
        `hyperparameter_ranges` (dict): random sampling ranges for the hyperparameters. The keys should be: ['hidden_layer_size_choices', 'nb_hidden_layer_choices', 'learning_rate_choices', 'batch_size_choices', 'dropout_choices']
        `start` (int): start of the iterations. Once an iteration is done, it won't be repeated for the same base folder (default: 0)
        `fixed_hyperparameters` (dict or none): fixed parameter values. The should be: ['nb_hidden_layers', 'hidden_layers', 'learning_rate', 'batch_size', 'dropout']
        `manual_combinations` (list[dict] or none): predefined parameter sets
        `early_stopping_patience` (int): patience for early stopping
        `run_from_main` (bool): flag indicating wether the code is run from the main class or notebook. Useful to decide which tqdm to use (default: False)
        `debug` (bool): debug output flag

    - returns:
        none
    '''

    # import the data
    X_train, y_train, X_test, y_test, X_val, y_val = load_data()

    # generate random hyperparameters:
    hyperparameters = create_hyperparameter_combinations(
        nb_tirals, 
        hyperparameter_ranges['hidden_layer_size_choices'], 
        hyperparameter_ranges['nb_hidden_layer_choices'], 
        hyperparameter_ranges['learning_rate_choices'], 
        hyperparameter_ranges['batch_size_choices'], 
        hyperparameter_ranges['dropout_choices'], 
        manual_combinations=manual_combinations, 
        nb_hidden_layers_= fixed_hyperparameters['nb_hidden_layers'] if fixed_hyperparameters is not None else -1,
        hidden_layers_= fixed_hyperparameters['hidden_layers'] if fixed_hyperparameters is not None else -1,
        learning_rate_= fixed_hyperparameters ['learning_rate'] if fixed_hyperparameters is not None else -1.0,
        batch_size_= fixed_hyperparameters['batch_size'] if fixed_hyperparameters is not None else -1,
        dropout_= fixed_hyperparameters['dropout'] if fixed_hyperparameters is not None else -1.0,
        randomize=False, 
        directory=save_path,
        progressbar=True
    )

    tqdm_func = get_tqdm(use_notebook=not run_from_main)

    # run the tuning
    for h in tqdm_func(range(start, len(hyperparameters)), desc='Tuning NN models', unit='model'):

        model_folder = f'{save_path}/tuned_model_{h}'

        if os.path.exists(model_folder):
            if os.path.exists(f'{model_folder}/.done'):
                continue
            else:
                # delte the folder and recreate later
                shutil.rmtree(model_folder)

        os.makedirs(model_folder, exist_ok=True)

        train_model(X_train, y_train, X_val, y_val, hyperparameters[h], model_folder, early_stopping_patience=early_stopping_patience, debug=debug)

        evaluate_model(X_test, y_test, model_folder, threshold=0.4, debug=False)

        with open(f'{model_folder}/.done', 'w') as f:
            #f.write(f"{datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
            f.write('INTERNAL FLAG TO INDICATE THAT THE MODEL HAS COMPLETED TRAINING AND EVALUATION!\nDO NOT DELETE OR MODIFY!!!!')




if __name__ == "__main__":

    run_from_main = True

    hyperparameter_ranges = {
        'hidden_layer_size_choices': [32 * 2**i for i in range(5)],
        #'hidden_layer_size_choices': [30, 50, 70, 90, 110, 130],
        'nb_hidden_layer_choices': [i for i in range(2, 5)],
        'learning_rate_choices': [0.0001, 0.001, 0.01],
        'batch_size_choices': [16 * 2**i for i in range(5)],
        'dropout_choices': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

    hyperparameter_ranges_small = {
        'hidden_layer_size_choices': [30, 50, 70, 90, 110, 130],
        'nb_hidden_layer_choices': [2,2,3,3,4],
        'learning_rate_choices': [0.0001, 0.001, 0.01],
        'batch_size_choices': [16 * 2**i for i in range(5)],
        'dropout_choices': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

    fixed_hyperparameters = {
        'nb_hidden_layers': -1,
        'hidden_layers': -1,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'dropout': -0.1
    }

    manual_combinations_small = [
        {
            "layer_sizes": [30, 70],
            "learning_rate": 0.0001,
            "batch_size": 16,
            "dropout": 0.1
        },
        {
            "layer_sizes": [30, 70],
            "learning_rate": 0.01,
            "batch_size": 16,
            "dropout": 0.1
        },
        {
            "layer_sizes": [30, 70],
            "learning_rate": 0.001,
            "batch_size": 32,
            "dropout": 0.1
        },
        {
            "layer_sizes": [30, 70],
            "learning_rate": 0.001,
            "batch_size": 16,
            "dropout": 0.2
        },
        {
            "layer_sizes": [30, 70],
            "learning_rate": 0.001,
            "batch_size": 16,
            "dropout": 0.0
        },
        {
            "layer_sizes": [30, 70],
            "learning_rate": 0.001,
            "batch_size": 16,
            "dropout": 0.3
        }
    ]

    manual_combinations = [
        {
            "layer_sizes": [128, 32, 128],
            "learning_rate": 0.001,
            "batch_size": 64,
            "dropout": 0.2
        },
        {
            "layer_sizes": [128, 32, 128],
            "learning_rate": 0.01,
            "batch_size": 32,
            "dropout": 0.2
        },
        {
            "layer_sizes": [128, 32, 128],
            "learning_rate": 0.01,
            "batch_size": 64,
            "dropout": 0.1
        },
        {
            "layer_sizes": [128, 32, 128],
            "learning_rate": 0.01,
            "batch_size": 64,
            "dropout": 0.3
        },
        {
            "layer_sizes": [128, 32, 128, 32],
            "learning_rate": 0.01,
            "batch_size": 64,
            "dropout": 0.2
        },
        {
            "layer_sizes": [128, 32, 64],
            "learning_rate": 0.01,
            "batch_size": 64,
            "dropout": 0.2
        },
        {
            "layer_sizes": [128, 32, 256],
            "learning_rate": 0.01,
            "batch_size": 64,
            "dropout": 0.2
        },
        {
            "layer_sizes": [128, 64, 128],
            "learning_rate": 0.01,
            "batch_size": 64,
            "dropout": 0.2
        },
        {
            "layer_sizes": [64, 32, 128],
            "learning_rate": 0.01,
            "batch_size": 64,
            "dropout": 0.2
        },
        {
            "layer_sizes": [64, 64, 128],
            "learning_rate": 0.01,
            "batch_size": 64,
            "dropout": 0.2
        },
        {
            "layer_sizes": [32, 32, 128],
            "learning_rate": 0.01,
            "batch_size": 64,
            "dropout": 0.2
        },

    ]

    manual_combinations_small_2 = [

        {
            "layer_sizes": [30, 70],
            "learning_rate": 0.01,
            "batch_size": 16,
            "dropout": 0.1
        },

    ]

    
    #run_NN_tuning(1170, f'{base}/Code/NN/Results/Tuning/prc-split-val-small-reduced-pigments-threshold-04', hyperparameter_ranges_small, fixed_hyperparameters=None, run_from_main=True, manual_combinations=manual_combinations_small_2)

    results_summarizer(f'{base}/Code/NN/Results/Tuning/prc-split-val-small-threshold-04')