import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

x = np.array([[-1,-1],[-2,-1],[2,1],[1,1]])
y = np.array([1,1,2,2])

clf = SVC()
clf.fit(x,y)

print(clf.predict([[-0.8,-1]]))
# print(accuracy_score(y,))