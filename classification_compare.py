import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score

def nearest_neighbors(train_data, test_row, k):
    list1=[]
    for x in train_data:
        list1.append(np.linalg.norm(x-test_row, ord=2))
        
    num_list1 = np.array(list1)
    bottomlist =list(num_list1.argsort()[:5][::1])
    return bottomlist

data = pd.read_csv('Dataset.csv',delimiter=',',header=0)
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=45931)

std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = std_scale.transform(X_train)
X_test_scaled = std_scale.transform(X_test)
Y_train_array = np.array(Y_train)
Y_test_array = np.array(Y_test)

Y_actual=[]
for x in Y_test_array:
    Y_actual.append(x)

Y_prediction =[]
num_neighbors = int(input("Please enter the number of nearest neighbors: "))
                    
for k in X_test_scaled:
    list2 = nearest_neighbors(X_train_scaled,k,num_neighbors)
    list4 =[]
    for p in list2 :
        list4.append(Y_train_array[p])
    if sum(list4)>=1:
        Y_prediction.append(1)
    else:
        Y_prediction.append(-1)
        
conf_matrix = confusion_matrix(Y_actual, Y_prediction)
accuracy = accuracy_score(Y_actual, Y_prediction)
Accuracy_KNN = accuracy

print("KNN_Accuracy:", accuracy)

# Naive Bayes Implementation

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, Y_train)
y_pred = gnb.predict(X_test)
Accuracy_NaiveBayes = accuracy_score(Y_test, y_pred)
print("Naive_Accuracy:", accuracy_score(Y_test, y_pred))

# Decision Tree

from sklearn.tree import DecisionTreeClassifier 

clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
Accuracy_DT = accuracy_score(Y_test, y_pred)
print("Tree_Accuracy:", accuracy_score(Y_test, y_pred))

# SVM

from sklearn.svm import SVC

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train,Y_train)
y_pred = svclassifier.predict(X_test)
Accuracy_SVM = accuracy_score(Y_test, y_pred)
print("SVM_Accuracy:", accuracy_score(Y_test, y_pred))

# Graphs plot

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
Methods = ['KNN', 'Naive Bayes', 'Decision Tree','SVM']
accuracy = [Accuracy_KNN, Accuracy_NaiveBayes, Accuracy_DT, Accuracy_SVM]
ax.bar(Methods,accuracy)
plt.show()
