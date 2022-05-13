# import all library

import numpy
import random
import matplotlib.pyplot as plt
import seaborn as sns


# LinearRegression
def studentReg(age_train, net_worth_train):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(age_train, net_worth_train)
    return reg


numpy.random.seed(42)
ages = []
for i in range(250):
    ages.append(random.randint(18, 75))

net_worth = [i * 6.25 + numpy.random.normal(scale=40) for i in ages]

ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
net_worth = numpy.reshape(numpy.array(net_worth), (len(net_worth), 1))

from sklearn.model_selection import train_test_split

ages_train, ages_test, net_worth_train, net_worth_test = train_test_split(ages, net_worth)

reg1 = studentReg(ages_train, net_worth_train)

print("co-eff", reg1.coef_)
print("intercept", reg1.intercept_)
print("training data", reg1.score(ages_train, net_worth_train))
print("testing data", reg1.score(ages_test, net_worth_test))


# plot graph

plt.figure(figsize=(5, 4))
# sns.regplot(x=ages_train,y=net_worth_train,scatter=True,color="b",marker="*")
plt.scatter(ages_train, net_worth_train, color="b", marker="*")
plt.scatter(ages_test, net_worth_test, color="r", marker="o")
plt.plot(ages_test, reg1.predict(ages_test), color="black")
plt.xlabel("Ages")
plt.ylabel("Net Worth")
plt.show()
