import os

import colour
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
from skimage.color import lab2rgb
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
#from Code.Utils.util_methods import UtilMethods

# get the directory of this file
script_dir = os.path.dirname(os.path.abspath(__file__))

# change cwd to script directory
os.chdir(script_dir)

print(os.getcwd())
base = '../..' # UtilMethods.find_project_root(os.getcwd())
#print(f"Project root found: {base}")

if not load_dotenv(f'{base}/.env'):
    print(f"ERROR .env not found @ {os.getcwd}")

# Add the Apple Studio Led to the colour package
StudioLed=pd.read_csv(f'{base}/Dataset/Studio_Led.csv')
colour.SDS_ILLUMINANTS['Studio Led']=colour.SpectralDistribution({row['wavelength']:row['energy'] for name,row in StudioLed.iterrows()})


class UtilMethods():

    @staticmethod
    def find_project_root(current_file, marker=".rootfolder"):
        """
        Find the project root directory containing a specific marker file.

        - Parameters:
            `current_file`: Path to the current file. Use os.getcwd() if unsure. (str)
            `marker`: File or directory name in the root folder that indicates the project root. Default is ".rootfolder" (str)
        
        - Returns:
            The path to the project root directory. (str)
        """
        # start with the directory containing the current file
        current_dir = current_file #os.path.abspath(os.path.dirname(current_file))

        while current_dir != os.path.dirname(current_dir):  # stop when at filesystem root
            if marker in os.listdir(current_dir):
                return current_dir
            current_dir = os.path.dirname(current_dir)  # move one directory up

        raise FileNotFoundError(f"Project root with marker '{marker}' not found.")

    @staticmethod
    def select_prescriptive_x_y(df, lightsource='FL2', custom_pigments=None, pigment_count=False):
        '''
        Creates the X and y dataframes

        - Parameters:
            `df`: The entier dataframe (Pandas DataFrame)
            `lightsource`: The light source 
            `custom_pigments`: The custom pigments. Default is None, in this case the default pigments are used. (list)
            `pigment_count`: The number of used pigments. Default is False. (int)

        - Retuns:
            The X and y dataframes
        '''

        X = df[[f'L_{lightsource}', f'a_{lightsource}', f'b_{lightsource}']]
        X.rename(columns={f'L_{lightsource}': 'L', f'a_{lightsource}': 'a', f'b_{lightsource}': 'b'}, inplace=True)
        y = df[['Amaplast Orange YXL',
                'Amaplast Red RP',
                'Amaplast Orange GXP',
                'Bayferrox 110M',
                'Bayferrox 140M',
                'Bayferrox 180M',
                'Black N774',
                'Black Pearls 800',
                'Black Pearls 120',
                'Black Pearls 717',
                'Black Pearls 1300',
                'Blue 299',
                'Cinquasia Magenta K 4535',
                'Cinquasia Pink K 4430 FP',
                'COLORSOL Orange HT2R',
                'COLORSOL Orange HT6G',
                'Cromophtal Yellow K 1310',
                'GreenTop Orange S',
                'GreenTop Red S',
                'GreenTop Light Orange S',
                'Heliogen Blue K 7097',
                'Heliogen Blue K 7104 LW',
                'Heucodur Black 953-1',
                'Hififast Yellow HF7R',
                'Hostasol Red GG',
                'Keyplast Resist Orange 9185',
                'Keyplast Resist Yellow 9187',
                'Macrolex Green 5B',
                'Macrolex Red 5B',
                'Macrolex Red EG',
                'Oracet Red 350 FA',
                'PV Fast Orange 6RL',
                'PolySynthren Yellow RL',
                'R-105',
                'Sachtolith HD-S',
                'Sachtolith L',
                'Shepherd Green 10G655',
                'Shepherd Violet 92',
                'Sicotan Yellow K2001FG',
                'Ultramarine Blue 26',
                'Ultramarine Blue 32',
                'Vanadur Plus Yellow 9010',
                'Yellow 2GTI',
                'Chemikos SG28',
                'Chemikos SR135',
                'Chemikos SR52',
                'Chemikos SV49',
                'Chemikos SY104',
                'Chemikos SY154',
                'Hostasol Yellow 3G',
                'Keyplast Resist Blue 9778',
                'Keyplast Resist Red 9179',
                'Keyplast Resist Yellow 9785',
                'Keyplast Resist Yellow 9882',
                'Macrolex Blue RR',
                'Macrolex Orange 3G',
                'Oracet Blue 700 FA',
                'Oracet Orange 220',
                'S720-SY21',
                'S723-DY54',
                'S725-SY116',
                'Shepherd Green 10G603',
                'Shepherd Orange 10P320',
                'Shepherd Orange 10P340',
                'Shepherd Yellow 10G148E',
                'Shepherd Yellow 10G155',
                'Sicopal Orange K 2430',
                'Heliogen Green K 8730',
                'Paliogen Blue K 6500 FK',
                'Paliogen Red K3911HD',
                'Ultramarine Violet',
                'PV Fast Yellow HGR',
                'Eupolen Blue 69-2001',
                'Macrolex Orange HT',
                'Macrolex Red E2G',
                'Neolor Light Orange H',
                'Neolor Orange H',
                'Neolor Red H',
                'PV Fast Red',
                'S-20',
                'Shepherd Orange 10C341']] if custom_pigments is None else df[custom_pigments]
        if pigment_count:
                y.loc[:, 'count_pigments'] = (y > 0.0).sum(axis=1)

        return X, y

    @staticmethod
    def divide_train_and_test_data(X, y, random_state=42, cluster_data='y', cluster_nb=1, test_size=0.2, plot_clusters=False):
        '''
        Divides the X and y datasets into train and test sets. For the test set it selects n% from each cluster. A side effect of this method is that count_pigments column is removed from y if present.
        
        - Parameters:
            `X`: Input dataset (L, a, b values) (Pandas DataFrame)
            `y`: Output dataset (Pandas DataFrame)
            `random_state`: Random state for shuffling. Default = 42 (int)
            `cluster_data`: Cluster the X dataset according to the given dataset (X or y). Default = 'y' (str)
            `cluster_nb`: Number of clusters to split. Default = 1 (int)
            `test_size`: Size of the test set. Default = 0.2 (float)
            `plot_clusters`: If true, plots the clusters. Default = False (bool)

        - Returns:
            X_train, X_test, y_train, y_test (Pandas DataFrame)
        '''

        if 'count_pigments' in y.columns:
            y.drop('count_pigments', axis=1, inplace=True)

        # clustering using KMeans
        kmeans = KMeans(n_clusters=cluster_nb, random_state=random_state)
        if cluster_data == 'X':
            X.loc[:,'cluster_nb'] = kmeans.fit_predict(X[['L', 'a', 'b']])
        elif cluster_data == 'y':
            X.loc[:, 'cluster_nb'] = kmeans.fit_predict(y)
        else:
            raise ValueError('cluster_data must be "X" or "y"')
        centroids = kmeans.cluster_centers_

        if plot_clusters:
            # count the sizes of the clusters
            cluster_counts = X['cluster_nb'].value_counts()
            print(cluster_counts)

            # interactive scatter plot with clusters
            fig = px.scatter(X,
                             x='a',
                             y='b',
                             color='cluster_nb',
                             # symbol='cluster',
                             labels={'a': 'a value', 'b': 'b value', 'cluster': 'cluster_nb'},
                             title='interactive k-means clustering of a-b values',
                             width=800,
                             height=600)

            # add centroids to plot
            # fig.add_scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers',
            #                marker=dict(size=15, color='black', symbol='x'),
            #                name='centroids')

            # enforce square aspect ratio
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

            fig.show()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=X['cluster_nb'],
            random_state=random_state
        )

        X.drop('cluster_nb', axis=1, inplace=True)
        X_train.drop('cluster_nb', axis=1, inplace=True)
        X_test.drop('cluster_nb', axis=1, inplace=True)

        return X_train, X_test, y_train, y_test


    @staticmethod
    def binarize_dataset(y):
        '''
        Indicates the presence or absence of a pigment for each recipe from the y dataset.

        - Parameters:
            `y`: Dataset containing pigments and their concentrations (Pandas DataFrame)
        
        - Returns:
            Dataset containing information about the presence or absence of a pigment (Pandas DataFrame)
        '''

        binary_df = y.copy()

        # convert ingredient quantities to binary (1 if quantity > 0, else 0)
        binary_df.iloc[:, :] = (binary_df.iloc[:, 0:] > 0).astype(int)

        return binary_df


    @staticmethod
    def normalize_recipes(recipe_df, offset=0.2):
        '''
        Normalizes the recipes using the min and max values for each pigment.

        - Parameters:
            `recipe_df`: df containing pigments and their concentrations (Pandas DataFrame)
            `offset`: indication of the offset to use above or below the max and min values for each pigment. It allows to have higher or lower concentrations than the max and min values for each pigment currently in the dataset. Default = 0.2 (float between 0 and 1)
        
        - Returns:
            normalized recipe dataframe (Pandas DataFrame)
        '''

        if not (0.0 <= offset <= 1.0):
            raise ValueError('offset must be between 0.0 and 1.0')

        # calculate min and max with the offset
        min_offset = 0 #recipe_df.min() - offset * (recipe_df.max() - recipe_df.min())
        max_offset = recipe_df.max() + offset * (recipe_df.max() - recipe_df.min())

        # apply min-max normalization with offset
        recipe_df_nomralized = (recipe_df - min_offset) / (max_offset - min_offset)

        return recipe_df_nomralized
    

    @staticmethod
    def CalculateLab(spectrum: np.array, illuminant: str):
        '''
        Calculates the actual Lab values from wavelengths (refelctance)

        - Returns:
            XYZ color and Lab color (tuple with 3 values)
        '''

        sd = colour.SpectralDistribution(spectrum)
        cmfs = colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer']
        lightsource = colour.SDS_ILLUMINANTS[illuminant]
        XYZ = colour.colorimetry.sd_to_XYZ_ASTME308(sd, cmfs, lightsource) / 100
        #RGB = colour.XYZ_to_RGB(XYZ)
        Lab = colour.XYZ_to_Lab(XYZ, colour.XYZ_to_xy(colour.colorimetry.sd_to_XYZ_ASTME308(lightsource, cmfs)))
        return (XYZ, Lab)
    
    @staticmethod
    def getLab(row,illuminant='D65', **kwargs):
        '''
        **Prepares a row in the dataframe to be suitable to the CalculateLab fucntion**

        - Params:
            `row`: row to be prepared, contains reflectance values
            `illuminant`: light source
        
        - Returns:
            Lab value
        '''
        wavecols = [f'{w}nm' for w in np.arange(400,741,10)]
        wavs=[int(f[:-2]) for f in wavecols]
        spectrum = dict(zip(wavs, row[wavecols].values/100))
        Lab=UtilMethods.CalculateLab(spectrum,illuminant)[1]
        return Lab

    @staticmethod
    def addLabcols(df,illuminant='D65'):
        '''
        **Add Lab values to a reflectance dataframe**

        - Params:
            `df`: Reflectance df
            `illuminant`: Light source
        
        - Returns:
            The original dataframe + Lab columns
        '''
        Lab=df.apply(UtilMethods.getLab,illuminant=illuminant,axis=1).apply(pd.Series)
        kwargs={'L_'+illuminant :Lab[0],'a_'+illuminant :Lab[1],'b_'+illuminant :Lab[2]}
        return df.assign(**kwargs)


    @staticmethod
    def visualize_lab(lab_df, target_df=None, lab_columns=['L', 'a', 'b'], title='a b values with their actual L a b colors', numbering=False, save_path=None, show_plot=True, interactive=False, threed_plot=False):
        """
        **Visualizes Lab color data on an a vs b scatterplot, coloring points with their actual Lab colors.**

        - parameters:
            `lab_df` (pd.DataFrame): dataframe containing L, a, b columns for the main points
            `target_df` (pd.DataFrame): optional dataframe for target points to highlight (default: None)
            `lab_columns` (list of str): list of column names for L, a, b values (default: ['L', 'a', 'b'])
            `title` (str): title of the plot (default: 'a b values with their actual L a b colors')
            `numbering` (bool): whether to show index numbers next to each point (default: False)
            `save_path` (str): if not none, the plot will be saved to this location (default: None)
            `show_plot` (bool): indicates if the plot should be shown or not. Only set to False if the plot is saved, else calling this method makes no sense (default: True)
            `interactive` (bool): if true, creates an interactive plot using plotly.express (default: False)
            `threed_plot` (bool): if true, creates an interactive 3D plot (default: False)


        """

        if save_path is None and not show_plot:
            return
        

        lab_array = lab_df[lab_columns].to_numpy().reshape(-1, 1, 3)
        #lab_array = np.clip(lab_array, [0, -128, -128], [100, 127, 127])  # clip values to valid lab range
        rgb_array = lab2rgb(lab_array).reshape(-1, 3)
        hex_colors = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' for r, g, b in rgb_array]


        ## compute eauclidan distance to target
        lab_values = lab_df[lab_columns].to_numpy()
        target_values = target_df[lab_columns].to_numpy()

        # compute all pairwise distances
        dist_matrix = cdist(lab_values, target_values)  # shape (n_lab, n_target)

        # get the minimum distance to any target for each lab point
        min_distances = dist_matrix.min(axis=1)



        if target_df is not None:
            target_array = target_df[lab_columns].to_numpy().reshape(-1, 1, 3)
            #target_array = np.clip(lab_array, [0, -128, -128], [100, 127, 127])  # clip values to valid lab range
            rgb_target_array = lab2rgb(target_array).reshape(-1, 3)
            hex_target_colors = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' for r, g, b in rgb_target_array]

        L_name, a_name, b_name = lab_columns
        
        if threed_plot:
            # create 3d scatter plot
            fig = go.Figure()

            cd = min_distances
            

            fig.add_trace(go.Scatter3d(
                x=lab_df[a_name],
                y=lab_df[b_name],
                z=lab_df[L_name],
                mode='markers+text' if numbering else 'markers',
                marker=dict(
                    size=6,
                    color=hex_colors,
                    line=dict(width=1, color='black')
                ),
                customdata=cd,
                text=[str(i+1) if numbering else '' for i in range(len(lab_df))],
                hovertemplate='L: %{z}<br>a: %{x}<br>b: %{y}<br>Distance to target: %{customdata}<br>Label: %{text}<extra></extra>',
                name='best individuals'
            ))

            if target_df is not None:
                fig.add_trace(go.Scatter3d(
                    x=target_df[a_name],
                    y=target_df[b_name],
                    z=target_df[L_name],
                    mode='markers+text' if numbering else 'markers',
                    marker=dict(
                        size=8,
                        color=hex_target_colors,
                        line=dict(width=2, color='red')
                    ),
                    text=['TARGET'] * len(target_df) if numbering else None,
                    hovertemplate='a: %{x}<br>b: %{y}<br>L: %{z}<br>label: TARGET<extra></extra>',
                    name='target'
                ))

            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='a',
                    yaxis_title='b',
                    zaxis_title='L'
                ),
                width=800,
                height=700,
                showlegend=True
            )

            if show_plot:
                fig.show()
            if save_path:
                fig.write_html(save_path)

        
        elif interactive:

            fig = go.Figure()

            cd = lab_df[L_name].to_list()

            # add main points
            fig.add_trace(go.Scatter(
                x=lab_df[a_name],
                y=lab_df[b_name],
                mode='markers+text' if numbering else 'markers',
                marker=dict(color=hex_colors, size=8, line=dict(width=1, color='black')),
                text=[str(i+1) if numbering else '' for i in range(len(lab_df))],
                textposition='top center',
                name='best individuals',
                customdata=cd,
                hovertemplate='L: %{customdata}<br>a: %{x}<br>b: %{y}<br>label: %{text}<extra></extra>'
            ))

            # handle target points
            if target_df is not None:

                cd = target_df[L_name].to_list()
               
                fig.add_trace(go.Scatter(
                    x=target_df[a_name],
                    y=target_df[b_name],
                    mode='markers+text' if numbering else 'markers',
                    marker=dict(color=hex_target_colors, size=10, line=dict(width=2, color='red')),
                    text=['TARGET' for _ in range(len(target_df))] if numbering else None,
                    textposition='top center',
                    name='target',
                    customdata=cd,
                    hovertemplate='L: %{customdata}<br>a: %{x}<br>b: %{y}<br>label: TARGET<extra></extra>'
                ))

            fig.update_layout(
                title=title,
                xaxis_title='a',
                yaxis_title='b',
                width=800,
                height=600,
                yaxis_scaleanchor='x',
                yaxis_scaleratio=1,
                showlegend=True
            )

            if show_plot:
                fig.show()
            if save_path:
                fig.write_html(save_path)

        else:

            # create scatter plot
            #plt.figure(figsize=(16,12))
            #plt.xlim(-100, 100)
            #plt.ylim(-50, 100)
            plt.scatter(lab_df[a_name], lab_df[b_name], color=rgb_array, label='best individuals')

            if numbering:
                for i, (x, y) in enumerate(zip(lab_df['a'], lab_df['b'])):
                    plt.text(x, y, f'  {str(i+1)}', fontsize=8)  # small offset to avoid overlap

            if target_df is not None:
                plt.scatter(target_df[a_name], target_df[b_name], color=rgb_target_array, edgecolor='red', label='target')
                for i, (x, y) in enumerate(zip(target_df[a_name], target_df[b_name])):
                    plt.text(x, y, 'TARGET', fontsize=8)  # small offset to avoid overlap


            plt.xlabel('a')
            plt.ylabel('b')
            plt.title(title)
            plt.legend()
            #plt.grid(True)
            if save_path:
                plt.savefig(save_path)
            if show_plot:
                plt.show()
            else:
                plt.close()