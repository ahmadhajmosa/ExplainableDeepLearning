import keras

import lime

import numpy as np
import lime
import lime.lime_tabular
import pandas as pd
import sklearn
import sklearn.datasets
import sklearn.ensemble
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.random.seed(1)
pd.options.mode.chained_assignment = None  # hide some warnings

data = pd.DataFrame(np.random.randint(0, 2, (200, 4)))
data_target = (data[0] & ~data[1]) | (~data[0] & data[1])
data_target
data[[0, 1]]

plt.scatter(data[0], data[1], c=data_target)

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier


# Keras models can be used by scikit-learn but require a build function for GridsearchCV
def create_model(layer_one_nodes=8, layer_two_nodes=4, optimizer='sgd'):
    keras_model = Sequential()
    keras_model.add(Dense(units=layer_one_nodes, activation='relu', input_dim=4))
    keras_model.add(Dense(units=layer_one_nodes, activation='relu'))

    # keras_model.add(Dense(units=layer_one_nodes, activation='relu'))

    optim = adam(lr=0.001)
    keras_model.add(Dense(units=1, activation='sigmoid'))

    keras_model.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])

    return keras_model


keras_model = KerasClassifier(build_fn=create_model, verbose=1, epochs=300)

keras_param_grid = {'layer_one_nodes': [20], 'layer_two_nodes': [2], 'optimizer': ['adam']}

data.columns = ['X0', 'X1', 'X2', 'X3']

class_names = list(data_target.astype('category').cat.categories)
cat_names = {0: [0, 1], 1: [0, 1]}

gs = GridSearchCV(keras_model, keras_param_grid, n_jobs=1, cv=2)
gs.fit(data.values, data_target.values)

score = gs.score(data.values, data_target.values.ravel())

print("accuracy: ", "%12.2f%%" % (score * 100.0))

gs.best_params_


#create the explainer
explainer = lime.lime_tabular.LimeTabularExplainer(data.values, feature_names = list(data.columns),
                                                   categorical_features = list(np.arange(len(data.columns))),
                                                   categorical_names = cat_names,
                                                   class_names=class_names,
                                                   discretize_continuous=True,verbose=1,kernel_width=1)

X=data.iloc[5,:]
predict_fn = lambda x: gs.predict_proba(x) # include one hot encoding in predict function

exp = explainer.explain_instance(X, predict_fn, num_samples=2000,
                                     num_features=len(data.columns))
exp.show_in_notebook(show_table=True, show_all=True)