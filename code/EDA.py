# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, RocCurveDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 
import seaborn as sns


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU, LSTM
from keras.layers import Dropout
from keras import backend as K
import os,sys

# %%
src_file_dir = os.path.dirname(os.path.realpath(__file__))  # obtain path to directory holding this file
os.chdir(src_file_dir)                                      # change working directory to this directory
new_df = pd.read_csv("../data/new_df2.csv")
new_df.drop(columns = 'Unnamed: 0', inplace=True)


# %% [markdown]
# ## Density diagarams of the different variables per class

# %%
new_df['pi_cat'] = np.where(new_df.pi_24>15,1,0)

for f in ['wd', 'ws', 'wsf', 'bp', 'tt', 'ah']:
    plt.figure()
    sns.kdeplot(data=new_df,x=f, hue ="pi_cat").set(title='Distribution of {}'.format(f))
    plt.show
# plt.savefig('pics.png', dpi=300)

# %%
split_index = round(new_df.shape[0] * 0.8)

test = new_df[split_index:]
train = new_df[:split_index]

# %%
#normalizing
scaler = StandardScaler()
new_df.iloc[:, 6:496] = scaler.fit_transform(new_df.iloc[:, 6:496])




