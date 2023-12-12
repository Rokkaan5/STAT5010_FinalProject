# %%
# libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, RocCurveDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
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
new_df = pd.read_csv("../data/new_df2.csv")
new_df.drop(columns = 'Unnamed: 0', inplace=True)
new_df.head()


# %% [markdown]
# Where                  
#                                        
# - `wd` = 'wind direction'
# - `ws` = 'wind speed'
# - `wsf` = "wind steadiness factor"
# - `bp` = "barometric pressure" 
# - `t2` = "temp at 2m" (degrees Celsius)
# - `t10` = "temp at 10m" (degrees Celsius)
# - `tt` = "temp at top of measuring device" (degrees Celsius)
# - `rh` = "relative humidity" (percent)
# - `ah` = "absolute humidity"
# - `pi` = "precipitation intensity"
#     - `pi_24` = rolling sum of the previous 24hr precipitation intensity
# 
# Time history (lag data): 48 (hrs)
# 
# - i.e. `wd6` is wind direction at 6hrs prior to time step of that row 

# %%
new_df.columns

# %%
new_df[["pi_24","wd","ws","wsf","bp","t2","t10","tt","rh","ah","pi"]].describe()

# %%
new_df['pi_cat'] = np.where(new_df.pi_24>15,1,0)

# %% [markdown]
# # Training, testing, features, targets

# %%
selective_features = []
for i in ['wd','ws','wsf','bp','tt','ah','pi']:
    for j in range(25,49):
        selective_features.append(i+str(j))


# selective_features = ['wd3', 'ws3', 'wsf3', 'bp3',
#                      'tt3', 'ah3', 'pi3', 'wd6', 'ws6', 'wsf6', 'bp6', 'tt6', 'ah6', 'pi6',
#                      'wd9', 'ws9', 'wsf9', 'bp9', 'tt9', 'ah9', 'pi9']
cat_target = ['pi_cat']
reg_target = ['pi_24']

# %% [markdown]
# ## Sequential split

# %%
#Ujas' code
split_index = round(new_df.shape[0] * 0.8)

test = new_df[split_index:]
train = new_df[:split_index]

# %%
X_train_seq = train[selective_features]
y_train_seq = train[['pi_cat']]
X_test_seq = test[selective_features]
y_test_seq = test[['pi_cat']]


# %%
scaler=StandardScaler()
X_train_seq = scaler.fit_transform(X_train_seq)
X_test_seq = scaler.transform(X_test_seq)

# %% [markdown]
# ## Random split

# %%
X = new_df[selective_features]
cat_y = new_df[cat_target]
reg_y = new_df[reg_target]

# %%
X_train_rand,X_test_rand,y_train_rand,y_test_rand = train_test_split(X,cat_y,test_size= 0.2,random_state=123)

# %%
scaler=StandardScaler()
X_train_rand = scaler.fit_transform(X_train_rand)
X_test_rand = scaler.transform(X_test_rand)

# %% [markdown]
# # KNN

# %%
KNN = KNeighborsClassifier(n_neighbors=1, metric='cosine', n_jobs=-1)

KNN.fit(X = X_train_rand, y = y_train_rand)

# %%
%%time
preds = KNN.predict(X_test_rand)

# %%
cm = confusion_matrix(y_test_rand, preds)
cm

# %%
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues')  
#annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(["Not Extreme", "Extreme"])
ax.yaxis.set_ticklabels(["Not Extreme", "Extreme"])

# %%
%%time
KNN_proba = KNN.predict_proba(X_test_rand)

RocCurveDisplay.from_predictions(y_test_rand, KNN_proba[:,1])
plt.show()

# %%
f1_score(y_test_rand, preds)

# %%
accuracy_score(y_test_rand, preds)

# %% [markdown]
# # Random Forest

# %%
%%time
RF = RandomForestClassifier(n_jobs= -1, class_weight=None)

RF.fit(X = X_train_rand, y = y_train_rand)

# %%
RF.feature_importances_

# %%
%%time
RF_preds = RF.predict(X_test_rand)

# %%
%%time
RF_proba = RF.predict_proba(X_test_rand)

# %%
RocCurveDisplay.from_predictions(y_test_rand, RF_proba[:,1])

# %%
rf_cm = confusion_matrix(y_test_rand, RF_preds)
rf_cm

# %%
ax= plt.subplot()
sns.heatmap(rf_cm, annot=True, fmt='g', ax=ax, cmap='Blues')  
#annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(["Not Extreme", "Extreme"])
ax.yaxis.set_ticklabels(["Not Extreme", "Extreme"])

# %%
f1_score(y_test_rand, RF_preds)

# %%
accuracy_score(y_test_rand, RF_preds)

# %% [markdown]
# # RandomForest Regressor

# %%
X_train_rand,X_test_rand,reg_y_train_rand,reg_y_test_rand = train_test_split(X,reg_y,test_size= 0.2,random_state=123)

# %%
scaler=StandardScaler()
X_train_rand = scaler.fit_transform(X_train_rand)
X_test_rand = scaler.transform(X_test_rand)

# %%
%%time
RF_reg = RandomForestRegressor(n_estimators=100,n_jobs=-1,random_state = 123)

RF_reg.fit(X = X_train_rand, y = reg_y_train_rand)

# %%
%%time
RF_reg_preds = RF_reg.predict(X_test_rand)

# %%
from sklearn.metrics import r2_score
r2_score(reg_y_test_rand, RF_reg_preds)

# %%
RF_reg_preds

# %%
cat_pred = np.where(RF_reg_preds>15,1,0)

# %%
rf_reg_cm = confusion_matrix(y_test_rand,cat_pred)
rf_reg_cm

# %%
ax= plt.subplot()
sns.heatmap(rf_reg_cm, annot=True, fmt='g', ax=ax, cmap='Blues')  
#annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(["Not Extreme", "Extreme"])
ax.yaxis.set_ticklabels(["Not Extreme", "Extreme"])

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

lgbm_fit = clf.fit(X = X_train_rand, y = y_train_rand)

# %%
pd.DataFrame(lgbm_fit.cv_results_)[['params','mean_test_score','rank_test_score']].sort_values('rank_test_score').head(10)

# %%
lgbmc = LGBMClassifier(boosting_type='dart', learning_rate=.4, n_jobs=-1, class_weight='balanced')

lgbmc.fit(X =X_train_rand, y = y_train_rand)

# %%
LG_preds = lgbmc.predict(X_test_rand)

# %%
LG_proba = lgbmc.predict_proba(X_test_rand)

# %%
lg_cm = confusion_matrix(y_test_rand, LG_preds)
lg_cm

# %%
ax= plt.subplot()
sns.heatmap(lg_cm, annot=True, fmt='g', ax=ax, cmap='Blues')  
#annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(["Not Extreme", "Extreme"])
ax.yaxis.set_ticklabels(["Not Extreme", "Extreme"])

# %%
f1_score(y_test_rand, LG_preds)

# %%
RocCurveDisplay.from_predictions(y_test_rand, LG_proba[:,1])

# %% [markdown]
# # GRU

# %%
X_train = np.array(X_train_seq).reshape(X_train_seq.shape[0],-1, 7)
X_test = np.array(X_test_seq).reshape(X_test_seq.shape[0],-1, 7)

# %%
X_train.shape[1]

# %%
%%time
class_weights = {0: 1., 1: 70.}
grumod = Sequential()

grumod.add(LSTM(units = 10, return_sequences = True, input_shape = (X_train.shape[1], 7)))
grumod.add(Dropout(0.2))

grumod.add(LSTM(units = 3, return_sequences = False))
grumod.add(Dropout(0.2))

grumod.add(Dense(units = 1, activation='sigmoid'))

grumod.compile(optimizer = 'adam', loss = 'binary_crossentropy')

history = grumod.fit(X_train, y_train_seq, validation_data=(X_test,y_test_seq), epochs = 50, batch_size = 512, class_weight = class_weights)

# %%
gru_preds = grumod.predict(X_test[:,:24,:])

# %%
gru_cm = confusion_matrix(y_test_seq,np.round(gru_preds))
gru_cm

# %%
ax= plt.subplot()
sns.heatmap(gru_cm, annot=True, fmt='g', ax=ax, cmap='Blues')  
#annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(["Not Extreme", "Extreme"])
ax.yaxis.set_ticklabels(["Not Extreme", "Extreme"])

# %%
f1_score(y_test_seq,np.round(gru_preds))

# %%
RocCurveDisplay.from_predictions(y_test_seq,gru_preds)


