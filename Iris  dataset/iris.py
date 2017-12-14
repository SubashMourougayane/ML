import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
iris = load_iris()
test_idx = [0,50,100]

#trainind data
train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx,axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

print(train_target)
print("****************************************")
print(train_data)
print("****************************************")
print(test_target)
print("****************************************")
print(test_data)
print("****************************************")
print("****************************************")

#classifier

clf = tree.DecisionTreeClassifier().fit(train_data,train_target)

for i in range(len(test_target)):
    print(test_data[i])
    print(test_target[i])
