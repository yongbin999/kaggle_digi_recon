import numpy as np
import pandas as pd
import scikit_models

# create the training & test sets, skipping the header row with [1:]
dataset = pd.read_csv("./input/train.csv")
labels_train = dataset[[0]].values.ravel()
features_train = dataset.iloc[:,1:].values
features_test = pd.read_csv("./input/test.csv").values

#run through 3 base models and 2 ensembles

#prediction = scikit_models.scikit_ran_forest(features_train, labels_train,features_test,alpha=500,beta =50)


prediction = scikit_models.scikit_ada_boost(features_train, labels_train,features_test,alpha=50)

"""
best_model = scikit_models.select_best_model(features_train,labels_train, data_title="digit recon")
prediction = scikit_models.run_model(best_model[0],features_train,labels_train,features_test,alpha=best_model[1],beta=best_model[2])
"""




np.savetxt('./output/submission_rand_forest.csv', np.c_[range(1,len(features_test)+1),prediction], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
 

