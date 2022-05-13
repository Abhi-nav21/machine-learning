x = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x, y)

print(neigh.predict([[0.9], [1.1], [1.5], [3.3]]))
