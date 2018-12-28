import pandas as pd
import numpy as np
import datetime

##########################################################################
# TODO: Convert the string to datetime format                            #
##########################################################################


def format_heure(chaine):
    if pd.isnull(chaine):
        return np.nan
    else:
        if chaine == 2400:
            chaine = 0
        chaine = "{0:04d}".format(int(chaine))
        heure = datetime.time(int(chaine[0:2]), int(chaine[2:4]))
        return heure
#_____________________________________________________________________
# Function that combines a date and time to produce a datetime.datetime


def combine_date_heure(x):
    if pd.isnull(x[0]) or pd.isnull(x[1]):
        return np.nan
    else:
        return datetime.datetime.combine(x[0], x[1])
#_________________________________________________________________________
# Function that combine two columns of the dataframe to create a datetime
# format


def create_flight_time(df, col):
    liste = []
    for index, cols in df[['DATE', col]].iterrows():
        if pd.isnull(cols[1]):
            liste.append(np.nan)
        elif float(cols[1]) == 2400:
            cols[0] += datetime.timedelta(days=1)
            cols[1] = datetime.time(0, 0)
            liste.append(combine_date_heure(cols))
        else:
            cols[1] = format_heure(cols[1])
            liste.append(combine_date_heure(cols))
    return pd.Series(liste)

##########################################################################
# TODO: Extract the useful data for original dataset                     #
##########################################################################


def data_restructuring(df, month=1):
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
    df = df[df['MONTH'] == month]
    df['SCHEDULED_DEPARTURE'] = create_flight_time(df, 'SCHEDULED_DEPARTURE')
    df['DEPARTURE_TIME'] = df['DEPARTURE_TIME'].apply(format_heure)
    df['SCHEDULED_ARRIVAL'] = df['SCHEDULED_ARRIVAL'].apply(format_heure)
    df['ARRIVAL_TIME'] = df['ARRIVAL_TIME'].apply(format_heure)
    df = df[['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
             'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY',
             'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',
             'SCHEDULED_TIME', 'ELAPSED_TIME']]
    return df

##########################################################################
# TODO: Display the missing ratio and drop nan data                     #
##########################################################################

# def drop_missing_data(df):
#     missing_df = df.isnull().sum(axis=0).reset_index()
#     missing_df.columns = ['variable', 'missing values']
#     missing_df['filling factor (%)']=(df.shape[0]-missing_df['missing values'])/df.shape[0]*100
#     missing_df.sort_values('filling factor (%)').reset_index(drop = True)
#     df.dropna(inplace = True)
#     return(df)
