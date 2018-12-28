import pandas as pd

##########################################################################
# TODO: Load the flight, airports, airlines                             #
##########################################################################


def load_data(print_info=False):

    df = pd.read_csv('Data/flights.csv', low_memory=False)
    print('Successfully Load Flights information')
    airports = pd.read_csv("Data/airports.csv")
    print('Successfully Load airports information')
    airlines = pd.read_csv('Data/airlines.csv')
    print('Successfully Load airlines information')
    #____________________________________________________________
    # gives some infos on columns types and number of null values
    tab_info = pd.DataFrame(df.dtypes).T.rename(index={0: 'column type'})
    tab_info = tab_info.append(
        pd.DataFrame(
            df.isnull().sum()).T.rename(
            index={
                0: 'null values (nb)'}))
    tab_info = tab_info.append(
        pd.DataFrame(
            df.isnull().sum() /
            df.shape[0] *
            100) .T.rename(
            index={
                0: 'null values (%)'}))
    if print_info:
        print(tab_info)

    return df, airports, airlines
