import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
##########################################################################
# TODO: Display relationship between prediction and mean delays          #
##########################################################################


def plot_pred(predictions, Y):
    tips = pd.DataFrame()
    tips["prediction"] = pd.Series([float(s) for s in predictions])
    tips["original_data"] = pd.Series([float(s) for s in Y])
    sns.jointplot(
        x="original_data",
        y="prediction",
        data=tips,
        size=6,
        ratio=7,
        joint_kws={
            'line_kws': {
                'color': 'limegreen'}},
        kind='reg')
    plt.xlabel('Mean delays (min)', fontsize=15)
    plt.ylabel('Predictions (min)', fontsize=15)
    plt.plot(list(range(-10, 25)), list(range(-10, 25)),
             linestyle=':', color='r')
    plt.show()
