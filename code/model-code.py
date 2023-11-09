# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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
src_file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(src_file_dir) 
new_df = pd.read_csv("../data/new_df.csv")
new_df.drop(columns = 'Unnamed: 0', inplace=True)

# %%
#normalizing
scaler = StandardScaler()
new_df.iloc[:, 6:496] = scaler.fit_transform(new_df.iloc[:, 6:496])

# %% [markdown]
# ## Density diagarams of the different variables per class

# %%
new_df['pi_cat'] = np.where(new_df.pi24>15,1,0)

sns.kdeplot(data=new_df,x="ah", hue ="pi_cat").set(title='Distribution of absolute humidity')
plt.savefig('pics.png', dpi=300)

# %%
split_index = round(new_df.shape[0] * 0.8)

test = new_df[split_index:]
new_df = new_df[:split_index]

# %%
# KNN
KNN = KNeighborsClassifier(n_neighbors=1, metric='cosine', n_jobs=-1)

KNN.fit(X = new_df.iloc[:, [6,7,8,9,12,14,15,
                            36,37,38,39,42,44,45,
                            66,67,68,69,72,74,75,
                            96,97,98,99,102,104,105]], y = new_df.loc[:, 'pi_cat'])

# %%
preds = KNN.predict(test.iloc[:, [6,7,8,9,12,14,15,
                            36,37,38,39,42,44,45,
                            66,67,68,69,72,74,75,
                            96,97,98,99,102,104,105]])

# %%
confusion_matrix(test.loc[:, 'pi_cat'], preds)

# %%
RocCurveDisplay.from_predictions(test.loc[:, 'pi_cat'], KNN_proba[:,1])
plt.show()

# %%
f1_score(test.loc[:, 'pi_cat'], preds)

# %%
accuracy_score(test.loc[:, 'pi_cat'], preds)

# %% [markdown]
# # Random Forest

# %%
RF = RandomForestClassifier(n_jobs= -1, class_weight=None)

RF.fit(X = new_df.iloc[:, [6,7,8,9,12,14,15,
                            36,37,38,39,42,44,45,
                            66,67,68,69,72,74,75,
                            96,97,98,99,102,104,105]], y = new_df.loc[:, 'pi_cat'])

# %%
RF.feature_importances_

# %%
RF_preds = RF.predict(test.iloc[:, [6,7,8,9,12,14,15,
                            36,37,38,39,42,44,45,
                            66,67,68,69,72,74,75,
                            96,97,98,99,102,104,105]])
RF_proba = RF.predict_proba(test.iloc[:, [6,7,8,9,12,14,15,
                            36,37,38,39,42,44,45,
                            66,67,68,69,72,74,75,
                            96,97,98,99,102,104,105]])


# %%
RocCurveDisplay.from_predictions(test.loc[:, 'pi_cat'], RF_proba[:,1])
plt.savefig('rf_roc.png', dpi=300)

# %%
confusion_matrix(test.loc[:, 'pi_cat'], RF_preds)

# %%
f1_score(test.loc[:, 'pi_cat'], RF_preds)
# accuracy_score(test.loc[:, 'pi_cat'], RF_preds)

# %% [markdown]
# # LightGBM

# %%
parameters = {'boosting_type':['dart'], 
              'learning_rate':[.3,.4,.5, .6],
              'n_jobs': [-1],
              'class_weight': ['balanced']}

lgbmc = LGBMClassifier()

clf = GridSearchCV(lgbmc, 
                   parameters, 
                   scoring= 'f1',
                   n_jobs=-1)

lgbm_fit = clf.fit(X = new_df.iloc[:, 6:106], y = new_df.loc[:, 'pi_cat'])

# %%
pd.DataFrame(lgbm_fit.cv_results_)[['params','mean_test_score','rank_test_score']].sort_values('rank_test_score').head(10)

# %%
lgbmc = LGBMClassifier(boosting_type='dart', learning_rate=.4, n_jobs=-1, class_weight='balanced')

lgbmc.fit(X = new_df.iloc[:, [6,7,8,9,12,14,15,
                            36,37,38,39,42,44,45,
                            66,67,68,69,72,74,75,
                            96,97,98,99,102,104,105]], y = new_df.loc[:, 'pi_cat'])

# %%
LG_preds = lgbmc.predict(test.iloc[:, [6,7,8,9,12,14,15,
                            36,37,38,39,42,44,45,
                            66,67,68,69,72,74,75,
                            96,97,98,99,102,104,105]])

# %%
LG_proba = lgbmc.predict_proba(test.iloc[:, [6,7,8,9,12,14,15,
                            36,37,38,39,42,44,45,
                            66,67,68,69,72,74,75,
                            96,97,98,99,102,104,105]])

# %%
confusion_matrix(test.loc[:, 'pi_cat'], LG_preds)

# %%
f1_score(test.loc[:, 'pi_cat'], LG_preds)

# %%
RocCurveDisplay.from_predictions(test.loc[:, 'pi_cat'], LG_proba[:,1])
plt.savefig('lgbm_roc.png', dpi=300)

# %% [markdown]
# # GRU

# %%
X = np.array(new_df.iloc[:, [6,7,8,9,12,14,15,
                            36,37,38,39,42,44,45,
                            66,67,68,69,72,74,75,
                            96,97,98,99,102,104,105]]).reshape(new_df.shape[0],-1, 7)
y = np.array(new_df['pi_cat'])

# %%
X_test = np.array(test.iloc[:, [6,7,8,9,12,14,15,
                            36,37,38,39,42,44,45,
                            66,67,68,69,72,74,75,
                            96,97,98,99,102,104,105]]).reshape(test.shape[0],-1,7)
y_test = np.array(test['pi_cat'])

# %%
class_weights = {0: 1., 1: 70.}
grumod = Sequential()

grumod.add(LSTM(units = 10, return_sequences = True, input_shape = (4, 7)))
grumod.add(Dropout(0.2))

grumod.add(LSTM(units = 3, return_sequences = False))
grumod.add(Dropout(0.2))

grumod.add(Dense(units = 1, activation='sigmoid'))

grumod.compile(optimizer = 'adam', loss = 'binary_crossentropy')

history = grumod.fit(X, new_df.pi_cat, validation_data=(X_test,test.pi_cat), epochs = 50, batch_size = 512, class_weight = class_weights)

# %%
gru_preds = grumod.predict(X_test[:,:10,:])

# %%
confusion_matrix(test.pi_cat,np.round(gru_preds))

# %%
f1_score(test.pi_cat,np.round(gru_preds))

# %%
RocCurveDisplay.from_predictions(test.pi_cat,gru_preds)
plt.savefig('gru_roc.png', dpi=300)


