from sklearn import tree
features = [[140,1],[130,1],[150,0],[170,0]]
lables = [0,0,1,1]
clf = tree.DecisionTreeClassifier().fit(features,lables)
clf = clf.fit(features,lables)
print(clf.predict([[150,0]]))
