import warnings
import timeit
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from clean_explore import data_restructuring
from data_loader import load_data
from plot import plot_pred
warnings.filterwarnings("ignore")

"""
This file compare various ML and statistical methods, try to fit in different way
    -------------------------------------------------------------------------
    Eight models were tested as follows, but only three of them shown in the report.
    becasue of the limitation of paper length.

    SVR, LinearSVR, NuSVR
    KNeighborsRegressor
    GaussianProcessRegressor
    DecisionTreeRegressor
    GradientBoostingRegressor
    MLPRegressor

    -------------------------------------------------------------------------

There are four main parts in this project: run, data loader, clean and explore and plot.

run.py:
  - main file, help us train the model and test the performance.

data_loader.py:
  - load the flights, airlines, airports information from csv files.

clean_explore.py:
  - data reconstruction file, composite the dataset with useful information.

plot.py:
  - plot the prediction and mean delayed time.

NOTE: SVM and  Gaussian Process will takes longer time (around 80 seconds).

"""

##########################################################################
# TODO: Load and restructure the data for building up model          #
##########################################################################

df, airport, airlines_names = load_data()
df = data_restructuring(df)
abbr_companies = airlines_names.set_index('IATA_CODE')['AIRLINE'].to_dict()

##########################################################################
# TODO: Display the information of delay by different airlines          #
##########################################################################

def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}


global_stats = df['DEPARTURE_DELAY'].groupby(
    df['AIRLINE']).apply(get_stats).unstack()
global_stats = global_stats.sort_values('count')
print(global_stats)

##########################################################################
# TODO: Display the information of delay by different airports          #
##########################################################################

carrier = 'WN'
check_airports = df[(df['AIRLINE'] == carrier)]['DEPARTURE_DELAY'].groupby(
    df['ORIGIN_AIRPORT']).apply(get_stats).unstack()
check_airports.sort_values('count', ascending=False, inplace=True)
print(check_airports)

##########################################################################
# TODO: Extract the departure delay and convert to second               #
##########################################################################

def get_flight_delays(df, carrier, id_airport, extrem_values=False):
    df2 = df[(df['AIRLINE'] == carrier) & (df['ORIGIN_AIRPORT'] == id_airport)]
    #_______________________________________
    # remove extreme values before fitting
    if extrem_values:
        df2['DEPARTURE_DELAY'] = df2['DEPARTURE_DELAY'].apply(
            lambda x: x if x < 60 else np.nan)
        df2.dropna(how='any')
    #__________________________________
    # Conversion: date + heure -> heure
    df2.sort_values('SCHEDULED_DEPARTURE', inplace=True)
    df2['heure_depart'] = df2['SCHEDULED_DEPARTURE'].apply(lambda x: x.time())
    #___________________________________________________________________
    # regroupement des vols par heure de départ et calcul de la moyenne
    test2 = df2['DEPARTURE_DELAY'].groupby(
        df2['heure_depart']).apply(get_stats).unstack()
    test2.reset_index(inplace=True)
    #___________________________________
    # conversion de l'heure en secondes

    def fct(x): return x.hour * 3600 + x.minute * 60 + x.second
    test2.reset_index(inplace=True)
    test2['heure_depart_min'] = test2['heure_depart'].apply(fct)
    return test2

##########################################################################
# TODO: Extract the departure delay with different airlines              #
##########################################################################

def get_merged_delays(df, carrier):
    liste_airports = df[df['AIRLINE'] == carrier]['ORIGIN_AIRPORT'].unique()
    i = 0
    liste_columns = ['AIRPORT_ID', 'heure_depart_min', 'mean']
    for id_airport in liste_airports:
        test2 = get_flight_delays(df, carrier, id_airport, True)
        test2.loc[:, 'AIRPORT_ID'] = id_airport
        test2 = test2[liste_columns]
        test2.dropna(how='any', inplace=True)
        if i == 0:
            merged_df = test2.copy()
        else:
            merged_df = pd.concat([merged_df, test2], ignore_index=True)
        i += 1
    return merged_df


carrier = 'WN'
merged_df = get_merged_delays(df, carrier)


##########################################################################
# TODO: One hot encoding to present the tags of the airports            #
##########################################################################

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(merged_df['AIRPORT_ID'])
zipped = zip(integer_encoded, merged_df['AIRPORT_ID'])
label_airports = list(set(list(zipped)))
label_airports.sort(key=lambda x: x[0])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

##########################################################################
# TODO: Set up data and label                                           #
##########################################################################

b = np.array(merged_df['heure_depart_min'])
b = b.reshape(len(b), 1)
X = np.hstack((onehot_encoded, b))
Y = np.array(merged_df['mean'])
Y = Y.reshape(len(Y), 1)

##########################################################################
# TODO: Load model                                                      #
##########################################################################

from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


start = timeit.default_timer()

##########################################################################
# TODO: Cross validation                                                #
##########################################################################

MSE_list = []
NLL_list = []
Min_list = []
N = 10
for i in range(N):
    # Split data
    ratio = 0.1  # Sample usage
    idx_test = np.random.choice(
        len(X), int(
            ratio * len(X)), replace=False)
    idx_train = np.setxor1d(np.arange(len(X)), idx_test)

    train_data, train_target = X[idx_train], Y[idx_train]
    test_data, test_target = X[idx_test], Y[idx_test]
    # uncomment to check the shape of data.
    # print(train_data.shape,train_target.shape)
    # Generate random targets as baseline

    # uncomment to check other regressors.
    # lm = SVR()
    # lm = LinearSVR()
    # lm = NuSVR()
    # lm = GaussianProcessRegressor()
    # lm = GradientBoostingRegressor()
    # uncomment to check other parameter.
    # lm = KNeighborsRegressor(n_neighbors=10, weights='distance')
    # lm = MLPRegressor(
    #        solver='adam', hidden_layer_sizes=(
    #            100,100,100), random_state=1, activation='relu')
    lm = SVR(C=1.4, epsilon=0.1)
    # lm = DecisionTreeRegressor()
    model = lm.fit(train_data, train_target)
    predictions = lm.predict(test_data)
    MSE = metrics.mean_squared_error(predictions, test_target)
    print('========Cross Validation Iteration %d Start.========' % i)
    print("MSE = %.5f" % MSE)
    pi = 3.1415926
    NLL = 0.5 * np.log(2 * pi * np.var(predictions))
    NLL1 = np.square(test_target - np.mean(predictions))
    NLL2 = 2 * pi * np.var(predictions)
    final_NLL = np.mean(NLL + NLL1 / NLL2)
    Min = np.around(np.sqrt(MSE), decimals=2)
    print("NLL = %.5f" % final_NLL)
    print('Ecart = {:.2f} min'.format(np.sqrt(MSE)))
    MSE_list.append(MSE)
    NLL_list.append(final_NLL)
    Min_list.append(Min)

end = timeit.default_timer()

##########################################################################
# TODO: Display the mean MSE and NLL as well as the efficiency          #
##########################################################################

print('========The average reuslt========')
print('MSE: ', np.mean(MSE_list))
print('NLL: ', np.mean(NLL_list))
print('Prediction Delay: %.2f min' % np.mean(Min_list))
print('Run time: %f s' % (end - start))

plot_pred(predictions, Y)
