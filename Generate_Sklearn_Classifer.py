import numpy as np
from sklearn import svm
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import random
from sklearn.neural_network import MLPClassifier
import config


X = []
Y = []
i = 0
neg_count = 0
pos_count = 0
with open("cleaned_train", "r") as ins:
    for line in ins:
        line = line.strip()
        line1 = line.split(',')
        if (i == 0):
            i += 1
            continue
        L = list(map(int, line1[:-1]))
        # L[sens_arg-1]=-1
        X.append(L)

        if (int(line1[-1]) == 0):
            Y.append(-1)
            neg_count = neg_count + 1
        else:
            Y.append(1)
            pos_count = pos_count + 1


X = np.array(X)
Y = np.array(Y)
print(neg_count, pos_count)

model_svm = svm.SVC(gamma=0.0025)
model_svm.fit(X, Y)
print('SVM:', cross_val_score(model_svm, X, Y, scoring='accuracy'))
joblib.dump(model_svm, 'SVM_standard_unfair.pkl', compress=3)

model_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(7, 5), random_state=1, max_iter=20000)
model_mlp.fit(X, Y)
print('MLPC:', cross_val_score(model_mlp, X, Y, scoring='accuracy'))
joblib.dump(model_mlp, 'MLPC_standard_unfair.pkl', compress=3)

model_dt = DecisionTreeClassifier()
model_dt.fit(X, Y)
print('DT:', cross_val_score(model_dt, X, Y, scoring='accuracy'))
joblib.dump(model_dt, 'Decision_tree_standard_unfair.pkl', compress=3)

model_rf = RandomForestClassifier()
model_rf.fit(X, Y)
print('RF:', cross_val_score(model_rf, X, Y, scoring='accuracy'))
joblib.dump(model_rf, 'Random_Forest_standard_unfair.pkl', compress=3)
