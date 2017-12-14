from sklearn import tree
from sklearn.externals.six import StringIO
import pydot

features = [[300,2],[450,2],[200,8],[150,9]]
lables = [1,1,0,0]
clf = tree.DecisionTreeClassifier().fit(features,lables)
print(clf.predict([[250,4]]))
# small program for prediction
