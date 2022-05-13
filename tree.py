from sklearn import tree

x = [[0, 0], [1, 1]]
y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf.fit(x, y)
print(clf.predict([[2, 2]]))
