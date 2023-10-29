import math
import numpy as np
import matplotlib
import sklearn
from collections import Counter

import tqdm
import xgboost as xgboost
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import r2_score, accuracy_score
import seaborn as sns
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt

matplotlib.use('TKagg')
import seaborn as sns

# df = pd.read_excel('Вариант 2.xlsx', skiprows=2, skipfooter=1)
# print(df.head())
# print(df.describe())
# mean = df.Балл.mean()
# df_under = df[df['Балл'] < mean]
# print(df_under)
# percentage = df_under.shape[0] / df.shape[0] * 100
# print(percentage)
# nesdali = df[df['Балл'] < df['Минимальный балл']]
# n_percentage = nesdali.shape[0] / df.shape[0] * 100
# print(n_percentage)
# data=[n_percentage,100-n_percentage]
# label=['fial','pass']
# plt.pie(data,labels=label)
#
# sns.kdeplot(df.Балл)
# plt.show()


# plt.show()
# def func(x):
#     if x < 63:
#         return 2
#     elif x > 62 and x < 77:
#         return 3
#     elif x > 76 and x < 93:
#         return 4
#     else:
#         return 5
#
#
# df1 = df.loc[:, 'Балл']
# df['Баллы'] = df['Балл'].apply(func)
# print(df.groupby('Баллы').agg({'Баллы': 'count'}) / df.shape[0] * 100)
# print(df.groupby('Пол').agg({"Пол": 'count'}) / df.shape[0] * 100)
# print(len(df['№ школы'].unique().tolist()))
# long = Counter(df['Задания с кратким ответом'].tolist())
# print(len(long.values())/4)
# short = df.loc[0, 'Задания с развёрнутым ответом']
# for i in range(int(len(short) / 4)):
#     df.insert(len(df.columns), f'B{i + 1}', 0)
# for i, _ in df.iterrows():
#     short = df.loc[i, 'Задания с кратким ответом']
#     for j in range(len(short)): df.loc[i, f"B{j + 1}"] = short[j]
# print(df)
# for i in range(int(len(long) / 4)):
#     df.insert(len(df.columns), f'C{i + 1}', 0)
# for i, _ in df.iterrows():
#     short = df.loc[i, 'Задания с развёрнутым ответом']
#     for j in range(len(long)): df.loc[i, f"C{j + 1}"] = long[j]
# print(df)
# df.drop('Устная часть', axis=1, inplace=True)

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# x = pd.read_csv(r'x.csv', sep=',', index_col=0)['0']
# y = pd.read_csv(r'y.csv', sep=',', index_col=0)['0']
#
#
# class Model(object):
#     def __init__(self):
#         self.b0=0
#         self.b1=0
#     def predict(self,x):
#         return self.b0+self.b1*x
#
#     def error(self, X, Y):
#         return sum((self.predict(X) - Y) ** 2) / (2 * len(X))
#
#     def fit(self, X, Y):
#         alpha = 0.1
#         dJ0 = sum(self.predict(X) - Y) /len(X)
#         dJ1 = sum((self.predict(X) - Y) * X) /len(X)
#         self.b0 -= alpha * dJ0
#         self.b1 -= alpha * dJ1
# hyp = Model()
# print(hyp.predict(0))
# print(hyp.predict(100))
# J = hyp.error(x, y)
# print("initial error:", J)
# class Model(object):
#     """Модель парной линейной регрессии"""
#
#     def __init__(self):
#         self.b0 = 0
#         self.b1 = 0
#
#     def predict(self, X):
#         if isinstance(X, pd.Series):
#             return self.b0 + self.b1 * X
#         elif isinstance(X, pd.DataFrame):
#             assert len(X.columns) > 1,"DataFrame should contain only one column"
#             return self.b0 + self.b1 * X.iloc[:, 0]
#         # return self.b0 + self.b1 * X
#
#     def error(self, X, Y):
#         return sum(((self.predict(X) - Y) ** 2) / (2 * len(X)))
#
#     def fit(self, X, Y, alpha=0.001, accuracy=0.01, max_steps=5000, value_of_mistake=1 / 10 ** 6):
#         steps, errors = [], []
#         step = 0
#         for _ in range(max_steps):
#             dJ0 = sum(self.predict(X) - Y) / len(X)
#             dJ1 = sum((self.predict(X) - Y) * X) / len(X)
#             self.b0 -= alpha * dJ0
#             self.b1 -= alpha * dJ1
#             new_err = hyp.error(X, Y)
#             step += 1
#             steps.append(step)
#             errors.append(new_err)
#             if not all((errors[i - 1] > errors[i]) for i in range(1, len(errors))):
#                 alpha1 = alpha / 2
#                 max_steps1 = int(max_steps / 100)
#                 self.fit(X, Y, alpha=alpha1, max_steps=max_steps1)
#                 # print(errors)
#
#             if all((errors[i] - errors[i - 1]) < value_of_mistake for i in range(1, len(errors))):
#                 pass
#             else:
#                 print('end')
#                 break
#         return steps, errors
#     def fit_normalized (self, X, Y, alpha=0.001, accuracy=0.01, max_steps=5000, value_of_mistake=1 / 10 ** 6):
#         steps, errors = [], []
#         step = 0
#         min_x, max_x = min(X), max(X)
#         x_normalized = [(x - min_x) / (max_x - min_x) for x in X]
#         min_y, max_y = min(Y), max(Y)
#         y_normalized = [(y - min_y) / (max_y - min_y) for y in Y]
#         for _ in range(max_steps):
#             dJ0 = sum(self.predict(X) - Y) / len(X)
#             dJ1 = sum((self.predict(X) - Y) * X) / len(X)
#             self.b0 -= alpha * dJ0
#             self.b1 -= alpha * dJ1
#             new_err = hyp.error(X, Y)
#             step += 1
#             steps.append(step)
#             errors.append(new_err)
#             if not all((errors[i - 1] > errors[i]) for i in range(1, len(errors))):
#                 alpha1 = alpha / 2
#                 max_steps1 = int(max_steps / 100)
#                 self.fit(X, Y, alpha=alpha1, max_steps=max_steps1)
#                 # print(errors)
#
#             if all((errors[i] - errors[i - 1]) < value_of_mistake for i in range(1, len(errors))):
#                 pass
#             else:
#                 print('end')
#                 break
#         return steps, errors
#
#     def plots(self, X, Y,X0,Y0):
#         min_x, max_x = min(self.predict(X)), max(self.predict(X))
#         x_normalized = [(x - min_x) / (max_x - min_x) for x in self.predict(X)]
#         # print(len(x_normalized))
#         min_y, max_y = min(Y), max(Y)
#         y_normalized = [(y - min_y) / (max_y - min_y) for y in Y]
#         # print(len(y_normalized))
#         min_x0, max_x0 = min(X0), max(X0)
#         x0_normalized = [(x0 - min_x0) / (max_x0 - min_x0) for x0 in X0]
#         # print(len(x0_normalized))
#         min_y0, max_y0 = min(Y0), max(Y0)
#         y0_normalized = [(y0 - min_y0) / (max_y0 - min_y0) for y0 in Y0]
#         # print(len(y0_normalized))
#         fig, ax = plt.subplots(2, 1)
#         ax[0].plot(x_normalized,y_normalized, color='green')
#         ax[0].scatter(x_normalized,y_normalized,color='black')
#         ax[1].plot(x0_normalized,y0_normalized,color='orange')
#         ax[1].scatter(x0_normalized, y0_normalized, color='red')
#
#         plt.show()
#     def score(self,Y_test,Y_predicted):
#         u= (Y_test - Y_predicted ** 2).sum()
#         v=((Y_test - Y_test.mean()) ** 2).sum()
#         R2=1-(u/v)
#         return R2
#
#
# hyp = Model()
# steps, errors = hyp.fit(x, y, alpha=2, max_steps=100000)
# # ПРи 1.6 спуск разошелся
# J = hyp.error(x, y)
# print("error after gradient descent:", J)
# # hyp.plots(h, y)
# z=hyp.predict(x).to_numpy().reshape(20,1)
# lr=LinearRegression()
# X0=x.to_numpy().reshape(20,1)
# Y0=y.to_numpy().reshape(20,1)
# X_train , X_test,  y_train, y_test = train_test_split(X0,Y0)
# lr.fit(X_train, y_train)
# y_prediction = lr.predict(X_test)
# print(r2_score(z,y))
# print(r2_score(y_test,y_prediction))
# hyp.plots(hyp.predict(x),y,y_test,y_prediction)
# # hyp.fit_normalized(x,y)
# print(hyp.score(y,hyp.predict(x)))


# 2 лаба
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# x = pd.read_csv("https://raw.githubusercontent.com/koroteevmv/ML_course/main/ML1.2_regression/data/0_x.csv",
#                 header=None)
# y = pd.read_csv("https://raw.githubusercontent.com/koroteevmv/ML_course/main/ML1.2_regression/data/0_y.csv",
#                 header=None)
# regression=LinearRegression()
# regression.fit(x[1].to_numpy()[:,np.newaxis],y)
# print(regression.coef_)
# res=regression.predict(x[[1]])
# xx=np.linspace(x[1].max(),x[1].min(),100).reshape((-1, 1))

# plt.scatter(x[1], y)
# plt.plot(xx, regression.predict(xx), c='r')
# plt.show()
# print(r2_score(y,res))
# multiple = LinearRegression()
# multiple.fit(x, y)
# print(multiple.score(x, y))
# yy = multiple.predict(x)
# plt.scatter(yy, y)
# plt.plot(yy, yy, c='r')
# plt.show()
# from sklearn.model_selection import train_test_split
# def atr_targets(X,Y):
#     X=pd.DataFrame(X)
#     Y=pd.DataFrame(Y)
#     for ind in range(X.shape[1]):
#         el=X[ind].to_numpy()[:,np.newaxis]
#         regressions=LinearRegression()
#         X_train,X_test,Y_train,Y_test=train_test_split(el,Y,test_size=0.2,random_state=42)
#         regressions.fit(X_train,Y_train)
#         res=regressions.predict(X_test)
#         yield r2_score(Y_test,res)


# for el1 in atr_targets(x,y):
#     print(el1)
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn. metrics import mean_absolute_error as mae
# import time
#
# def atr_targets1(X,Y):
#     dict1=dict()
#     res1 = []
#     X=pd.DataFrame(X)
#     Y=pd.DataFrame(Y)
#     for i in range(X.shape[1]):
#         el = X[i].to_numpy()[:, np.newaxis]
#         for j in range(2, 11):
#             X_train,X_test,Y_train,Y_test=train_test_split(el,Y,test_size=0.2,random_state=42)
#             start_time = time.time()
#             poly = PolynomialFeatures(degree=j)
#             X_train_transformed,X_test_transformed = poly.fit_transform(X_train),poly.fit_transform(X_test)
#             regressor = LinearRegression()
#             regressor.fit(X_train_transformed, Y_train)
#             y_pred = regressor.predict(X_test_transformed)
#             end_time=time.time()
#             r2 = r2_score(Y_test, y_pred)
#             me=mae(Y_test,y_pred)
#             elapsed_time = end_time - start_time
#             print(f'R^2 for {i+1} feature, degree {j}: {r2}')
#             print(f'MAE for {i+1} feature, degree {j}: {me}')
#             print(f'TIME for {i+1} feature, degree {j}: {elapsed_time}')
#             res1.append((r2,me,elapsed_time))
#         s = f'Model{i + 1}'
#         dict1.update({s:res1})
#     return dict1
# table=pd.DataFrame(atr_targets1(x,y))
# print(table.head())


# 3 лаба
# from sklearn.datasets import make_classification
#
# X, y = make_classification(n_samples=1000,
#                            n_features=2,
#                            n_informative=2,
#                            n_redundant=0,
#                            n_classes=2,
#                            class_sep=2,
#                            random_state=1)


# print(X)
# plt.scatter(X[:, 0][y==0], X[:, 1][y==0], marker="o", c='r', s=100)
# plt.scatter(X[:, 0][y==1], X[:, 1][y==1], marker="x", c='b', s=100)
# plt.show()
# class SGD(object):
#     def __init__(self, alpha=0.5, n_iters=1000):
#         self.theta = None
#         self._alpha = alpha
#         self._n_iters = n_iters
#
#     def gradient_step(self, theta, theta_grad):
#         return theta - self._alpha * theta_grad
#
#     def optimize(self, X, y, start_theta, n_iters):
#         theta = start_theta.copy()
#         for i in range(n_iters):
#             theta_grad = self.grad_func(X, y, theta)
#             theta = self.gradient_step(theta, theta_grad)
#         return theta
#
#     def fit(self, X, y):
#         m = X.shape[1]
#         start_theta = np.ones(m)
#         self.theta = self.optimize(X, y, start_theta, self._n_iters)
#
#
# class LogReg(SGD):
#     def sigmoid(self, X, theta):
#         return 1. / (1. + np.exp(-X.dot(theta)))
#
#     def add_intercept_column(self, X):
#         intercept_column = np.ones((X.shape[0], 1))
#         return np.concatenate((intercept_column, X), axis=1)
#
#     def grad_func(self, X, y, theta):
#         n = X.shape[0]
#         grad = 1. / n * X.transpose().dot(self.sigmoid(X, theta) - y)
#         return grad
#
#     def predict_proba(self, X):
#         return self.sigmoid(X, self.theta)
#
#     def predict(self, X):
#         y_pred = self.predict_proba(X) > 0.5
#         return y_pred


# logreg = LogReg()
# X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])
# print(X)
# logreg.fit(X, y)
# y_pred = logreg.predict(X)
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.linear_model import LogisticRegression
#
# ac = accuracy_score(y, y_pred)
# f1 = f1_score(y, y_pred)
# print(f'accuracy = {ac:.2f} F1-score = {f1:.2f}')
# model = LogisticRegression()

# X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42, test_size=0.2)
# model.fit(X_train, y_train)
# x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
# y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
#
# h = 0.02
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
# np.arange(y_min, y_max, h))
# Z = model.predict(np.c_[np.ones(len(xx.ravel())), xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.figure(figsize=(10, 6))
# plt.contourf(xx, yy, Z, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.title('Результаты логистической регрессии')
# plt.show()
#
# logreg = LogReg(alpha=0.0001)
# X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])
# print(X)
# logreg.fit(X, y)
# y_pred = logreg.predict(X)
# print(r2_score(y,y_pred))
class_seps = [0.2, 0.5, 0.8, 1, 2, 5, 12]

# def different_parametrs(class_seps):
#     subtitles = ['class_sep {}'.format(value) for value in class_seps]
#     fig, axs = plt.subplots(1, len(class_seps), figsize=(15, 5))
#     for i, class_sep in enumerate(class_seps):
#         X, y = make_classification(
#             n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=0,
#             class_sep=class_sep
#         )
#         axs[i].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
#         axs[i].set_title(subtitles[i])
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
#         logreg = LogReg()
#         logreg.fit(X_train, y_train)
#
#         y_pred = logreg.predict(X_test)
#
#         accuracy = accuracy_score(y_test, y_pred)
#         print(accuracy)
#     plt.show()


# different_parametrs(class_seps)

# X, y = make_classification(
#             n_samples=1000, n_features=20, n_informative=2, n_redundant=0, random_state=0,
#             class_sep=5
#         )
# X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42, test_size=0.2)
# logreg.fit(X_train,y_train)
# print(r2_score(y_test,logreg.predict(X_test)))
# def onevsall():
#     X, y = make_classification(
#                 n_samples=1000, n_features=200, n_informative=2, n_redundant=0, random_state=0,
#                 class_sep=5
#             )
#     X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42, test_size=0.2)
#     from sklearn.multiclass import OneVsRestClassifier
#     from sklearn.model_selection import cross_val_score
#     classifier = OneVsRestClassifier(LogisticRegression())
#     classifier.fit(X_train,y_train)
#     scores = cross_val_score(classifier, X_test,y_test,cv=6)
#     print("Cross-validation scores:", scores)
#     print("Mean score:", scores.mean())
#     print("Standard deviation of scores:", scores.std())
# onevsall()
# def  rocs():
#     from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
#     import matplotlib.pyplot as plt
#     X, y = make_classification(
#                         n_samples=1000, n_features=20, n_informative=2, n_redundant=0, random_state=0,
#                         class_sep=5,n_classes=2
#                     )
#     X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42, test_size=0.2)
#     model=LogisticRegression()
#     model.fit(X_train,y_train)
#     y_pred_prob = model.predict_proba(X_test)[:,1]
#     fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#     plt.plot(fpr, tpr)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.show()
#     auc=roc_auc_score(y_test, y_pred_prob)
#     print('AUC:', auc)
# rocs()
# class SGDLIn(object):
#     def __init__(self, alpha=0.5, n_iters=1000):
#         self.theta = None
#         self._alpha = alpha
#         self._n_iters = n_iters
#
#     def gradient_step(self, theta, theta_grad):
#         return theta - self._alpha * theta_grad
#
#     def optimize(self, X, y, start_theta, n_iters):
#         theta = start_theta.copy()
#         for i in range(n_iters):
#             theta_grad = self.grad_func(X, y, theta)
#             theta = self.gradient_step(theta, theta_grad)
#         return theta
#
#     def fit(self, X, y):
#         m = X.shape[1]
#         start_theta = np.ones(m)
#         self.theta = self.optimize(X, y, start_theta, self._n_iters)
# class LinearRegressions(SGDLIn):
#
#
#     def add_intercept_column(self, X):
#         intercept_column = np.ones((X.shape[0], 1))
#         return np.concatenate((intercept_column, X), axis=1)
#
#     def grad_func(self, X, y, theta):
#         n = X.shape[0]
#         grad = 1. / n * X.transpose().dot(self.sigmoid(X, theta) - y)
#         return grad
#
#     def predict_proba(self, X):
#         return self.sigmoid(X, self.theta)
#
#     def predict(self, X):
#         y_pred = self.predict_proba(X) > 0.5
#         return y_pred

# 4 лаба

# col_names = ['pregnant', 'glucose', 'bp', 'skin',
#              'insulin', 'bmi', 'pedigree', 'age', 'label']
# pima = pd.read_csv('diabetes.csv', header=None, names=col_names)
# pima = pima[1:]
# Y = pima.label
# X = pima.drop(['label'], axis=1)
# x_train, x_test, y_train, y_test = train_test_split(X, Y,
#                                                     test_size=0.2,
#                                                     random_state=42)
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,StackingClassifier
# from sklearn.linear_model import PassiveAggressiveClassifier
# from sklearn.linear_model import Ridge,ElasticNet,Lasso
# from sklearn.metrics import confusion_matrix
#
# def linears():
#     l=Lasso()
#     r=Ridge()
#     net=ElasticNet()
#     l.fit(x_train, y_train)
#     print('Lasso->' + str(r2_score(y_test, l.predict(x_test))))
#     print('Lasso->' + str((y_test, l.predict(x_test))))
#     r.fit(x_train, y_train)
#     print('Ridge->' + str(r2_score(y_test, r.predict(x_test))))
#     net.fit(x_train, y_train)
#     print('Net->' + str(r2_score(y_test, net.predict(x_test))))
#
#
# def different_kernels():
#     for el in ('rbf', 'poly', 'linear', 'sigmoid'):
#         machine = SVC(kernel=el)
#         machine.fit(x_train, y_train)
#         y_pred = machine.predict(x_test)
#         print(f'{el}:' + str(accuracy_score(y_test, y_pred)))
#         print(f'{el}:' + str(confusion_matrix(y_test, y_pred)))
#     aggressor = PassiveAggressiveClassifier()
#     aggressor.fit(x_train,y_train)
#     print('Aggressor->' + str(accuracy_score(y_test, aggressor.predict(x_test))))
#     print('Aggressor->' + str(confusion_matrix(y_test, aggressor.predict(x_test))))
#
#
#
# def neighbours():
#     clf = KNeighborsClassifier()
#     clf.fit(x_train, y_train)
#     print('Kneighbours->' + str(accuracy_score(y_test, clf.predict(x_test))))
#     print('Kneighbours->' + str(confusion_matrix(y_test, clf.predict(x_test))))
#
#
# def trees():
#     tree = DecisionTreeClassifier()
#     tree.fit(x_train, y_train)
#     print('DecesionTree->' + str(accuracy_score(y_test, tree.predict(x_test))))
#     print('DecesionTree->' + str(confusion_matrix(y_test, tree.predict(x_test))))
#     forest = RandomForestClassifier()
#     forest.fit(x_train, y_train)
#     print('Forest->' + str(accuracy_score(y_test, forest.predict(x_test))))
#     print('Forest->' + str(confusion_matrix(y_test, forest.predict(x_test))))
#
#
# def gauss():
#     gausss = DecisionTreeClassifier()
#     gausss.fit(x_train, y_train)
#     print('GAusss->' + str(accuracy_score(y_test, gausss.predict(x_test))))
#     print('GAusss->' + str(confusion_matrix(y_test, gausss.predict(x_test))))
# def baggings():
#     bag = BaggingClassifier(n_estimators=13,estimator=DecisionTreeClassifier())
#     bag.fit(x_train, y_train)
#     print('Bag->' + str(confusion_matrix(y_test, bag.predict(x_test))))
#     estimators = [('DecisionTree', DecisionTreeClassifier()),
#                   ('KNeighbors', KNeighborsClassifier()),
#                   ('ElasticNet', ElasticNet()),
#                   ('Ridge',Ridge(alpha=0.5))]
#     stacks=StackingClassifier(estimators=estimators,final_estimator=GaussianNB())
#     stacks.fit(x_train, y_train)
#     print('Stack->' + str(accuracy_score(y_test, stacks.predict(x_test))))
#     print('Stack->' + str(confusion_matrix(y_test, stacks.predict(x_test))))
#
#
#
#
#
#
#
# def different_models():
#     print('SVM->')
#     different_kernels()
#     neighbours()
#     trees()
#     gauss()
#     linears()
#     baggings()
#
#
# different_models()
# cls = LogisticRegression()
# cls.fit(x_train, y_train)
# y_pred = cls.predict(x_test)
# from sklearn import metrics
#
# metrics.confusion_matrix(y_test, y_pred)
import numpy as np
import matplotlib.pyplot as plt

# class_names = [0, 1]
# fig, ax = plt.subplots()
# ticks = np.arange(len(class_names))
# plt.xticks(ticks, class_names)
# plt.yticks(ticks, class_names)
# sns.heatmap(pd.DataFrame(
#     metrics.confusion_matrix(y_test, y_pred)),
#     annot=True)
# plt.ylabel('Действительные значения')
# plt.xlabel('Предсказанные значения')
# plt.show()
# print(metrics.accuracy_score(y_test, y_pred))
# print(metrics.precision_score(y_test, y_pred))\


# clf=KNeighborsClassifier()
# params ={"n_neighbors":[1,2,3,4, 5,6,7,8,9,10],
# "metric":['manhattan', 'euclidean'],
# "weights":['uniform', 'distance']}
# clf_grid=GridSearchCV(clf,params,scoring='accuracy',n_jobs=-1,cv=5)
# clf_grid.fit(x_train,y_train)
# clf_grid.predict(x_test)
# print(clf_grid.best_params_)
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# params_neighbors ={"n_neighbors":[1,2,3,4,5,6,7,8,9,10],
# "metric":['manhattan', 'euclidean'],
# "weights":['uniform', 'distance']}
# logistic_regression_params = {
# 'penalty':['l2','l1','elasticnet',None],
# 'C': [1.0,2.0,3.0,4.0,5.0,6.0,7.0],
# 'fit_intercept': [True,False],
# 'solver': ['lbfg', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
# }
# clf= KNeighborsClassifier()
# log=LogisticRegression()
# scale=StandardScaler()
#
#
# iris=load_iris(as_frame=True)
# iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
#                      columns= iris['feature_names'] + ['target']).reset_index(drop=True)
# col_name=list(iris.columns)
# iris=scale.fit_transform(iris)
# iris=pd.DataFrame(iris,columns=col_name)
# X=iris.iloc[:,:-1]
# Y=iris.iloc[:,-1].astype(int)
#
# x_train, x_test, y_train, y_test = train_test_split(X, Y,
#                                                     test_size=0.2,
#                                                     random_state=42)
#
# grid=GridSearchCV(clf,params,scoring='accuracy',cv=6)
# grid1=GridSearchCV(log,logistic_regression_params,scoring='accuracy',cv=6)
# grid.fit(x_train,y_train)
# grid1.fit(x_train,y_train)
# print(accuracy_score(y_test,grid.predict(x_test)))
# print(grid.best_params_)
# print(r2_score(y_test,grid1.predict(x_test)))
# print(grid1.best_params_)
#
# 5 лаба
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_california_housing

# california = fetch_california_housing()
# print(type(california))
# print(california.data.shape, california.target.shape)
# data = pd.DataFrame(california.data, columns=california.feature_names)
# data['Price'] = california.target
# print(data.info())
# y = data['Price'].astype(int)
# X = data.drop('Price', axis=1).astype(int)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#
# poly = PolynomialFeatures(5).fit_transform(X)
# polynomial = LinearRegression(fit_intercept=True)
# polynomial.fit(poly, y)
# y_pred_poly = polynomial.predict(poly)
# print(polynomial.score(poly, y))

# def linears():
#     l = Lasso()
#     r = Ridge()
#     net = ElasticNet()
#     l.fit(x_train, y_train)
#     y_pred = l.predict(x_test)
#     print('Lasso->' + str(r2_score(y_test, y_pred)))
#     plt.scatter(y_pred, y_test)
#     plt.plot(y_test, y_test, c='r')
#     r.fit(x_train, y_train)
#     y_pred = r.predict(x_test)
#     print('Ridge->' + str(r2_score(y_test, y_pred)))
#     plt.scatter(y_pred, y_test)
#     plt.plot(y_test, y_test, c='r')
#     net.fit(x_train, y_train)
#     y_pred = net.predict(x_test)
#     print('Net->' + str(r2_score(y_test, y_pred)))
#
#
# def different_kernels():
#     for el in ('rbf', 'poly', 'linear', 'sigmoid'):
#         machine = SVC(kernel=el)
#         machine.fit(x_train, y_train)
#         y_pred = machine.predict(x_test)
#         print(f'{el}:' + str(accuracy_score(y_test, y_pred)))
#         plt.scatter(y_pred, y_test)
#         plt.plot(y_test, y_test, c='r')
#
#     aggressor = PassiveAggressiveClassifier()
#     aggressor.fit(x_train, y_train)
#     y_pred = aggressor.predict(x_test)
#     print('Aggressor->' + str(accuracy_score(y_test, y_pred)))
#     plt.scatter(y_pred, y_test)
#     plt.plot(y_test, y_test, c='r')
#
#
# def neighbours():
#     clf = KNeighborsClassifier()
#     clf.fit(x_train, y_train)
#     print('Kneighbours->' + str(accuracy_score(y_test, clf.predict(x_test))))
#
#
# def trees():
#     tree = DecisionTreeClassifier()
#     tree.fit(x_train, y_train)
#     print('DecesionTree->' + str(accuracy_score(y_test, tree.predict(x_test))))
#     forest = RandomForestClassifier()
#     forest.fit(x_train, y_train)
#     print('Forest->' + str(accuracy_score(y_test, forest.predict(x_test))))
#
#
# def gauss():
#     gausss = DecisionTreeClassifier()
#     gausss.fit(x_train, y_train)
#     print('GAusss->' + str(accuracy_score(y_test, gausss.predict(x_test))))
#
#
# def baggings():
#     bag = BaggingClassifier(n_estimators=13, estimator=DecisionTreeClassifier())
#     bag.fit(x_train, y_train)
#     print('Bag->' + str(accuracy_score(y_test, bag.predict(x_test))))
#     estimators = [('DecisionTree', DecisionTreeClassifier()),
#                   ('KNeighbors', KNeighborsClassifier()),
#                   ('ElasticNet', ElasticNet()),
#                   ('Ridge', Ridge(alpha=0.5))]
#     stacks = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
#     stacks.fit(x_train, y_train)
#     print('Stack->' + str(accuracy_score(y_test, stacks.predict(x_test))))
#
#
# def different_models():
#     print('SVM->')
#     different_kernels()
#     plt.show()
#     neighbours()
#     trees()
#     gauss()
#     linears()
#     baggings()


# different_models()
# from sklearn.datasets import load_diabetes
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import StackingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import ElasticNet
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import cross_val_predict
# import pandas as pd
#
# diabetes = load_diabetes()
#
# data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# data['Target'] = diabetes.target
#
# probs = GaussianNB()
# estimators = [
#     ('DecisionTree', DecisionTreeClassifier()),
#     ('KNeighbors', KNeighborsClassifier()),
#     ('SVC', SVC()),
#     ('ElasticNet', ElasticNet()),
#     ('GaussianNB', GaussianNB())
# ]
#
# # Create the stacking classifier
# stack = StackingClassifier(estimators=estimators, final_estimator=KNeighborsClassifier())
#
# # Split the data into train and test sets
# X, Y = data.iloc[:, :-1], data.iloc[:, -1]
# threshold = 150
# Y = (diabetes.target > threshold).astype(int)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# stack.fit(X_train, Y_train)
# probs.fit(X_train, Y_train)
#
# print("Accuracy on test set (GaussianNB):", accuracy_score(Y_test, probs.predict(X_test)))
# print("Accuracy on test set (StackingClassifier):", accuracy_score(Y_test, stack.predict(X_test)))
# import pandas as pd
# from sklearn.datasets import load_iris
#
# iris = load_iris()
# X = iris.data
# y = iris.target
# iris_data = pd.DataFrame(iris['data'], columns=iris['feature_names'])
# name_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
# iris_data['class'] = [name_map[k] for k in iris['target']]
# # print(iris_data.head(10))
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=5)
# import sklearn.metrics
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_score, LeavePOut
# 6 лаба недообкчение и переобучение
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=10000, n_features=500,
                           n_informative=50, n_repeated=0,
                           class_sep=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=3)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(X_train, y_train)

print(f"Training score: {lr.score(X_train, y_train):.4f}")
print(f"Test score: {lr.score(X_test, y_test):.4f}")
from yellowbrick.model_selection import LearningCurve

# visualizer = LearningCurve(
#     LogisticRegression(), train_sizes=np.linspace(0.5, 1.0, 10)
# ).fit(X, y).show()
from sklearn.linear_model import RidgeClassifier
lr = RidgeClassifier(alpha=1000000).fit(X_train, y_train)

print(f"Training score: {lr.score(X_train, y_train):.4f}")
print(f"Test score: {lr.score(X_test, y_test):.4f}")
visualizer = LearningCurve(
    RidgeClassifier(alpha=1000000), train_sizes=np.linspace(0.1, 1.0, 10)
).fit(X, y) .show()
# 7 лаба с валидацией
# model = LogisticRegression(solver='liblinear')
# model.fit(X_train, y_train) #Обучение трейновой выборке
# y_pred = model.predict(X_test) #Предсказание для тестовой выборки
# print(accuracy_score(y_test, y_pred))
# print(sklearn.metrics.f1_score(y_test, y_pred, average='macro'))
# sns.heatmap(sklearn.metrics.confusion_matrix(y_test, y_pred), annot=True)
# kf = KFold(n_splits=3, shuffle=True, random_state=15)
# metrics_accuracy = []
# metrics_f1 = []
# model = LogisticRegression(solver='liblinear')
# for i, (train_index, test_index) in enumerate(kf.split(y)):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     metrics_accuracy.append(accuracy_score(y_test, y_pred))
#     metrics_f1.append(sklearn.metrics.f1_score(y_test, y_pred, average='macro'))
# print('Значения метрики accuracy: {} \nЗначения метрики f1: {}'.format(metrics_accuracy, metrics_f1))
# print("Среднее по кросс-валидации: ", np.array(metrics_f1).mean())
# cv_results = cross_val_score(model,  # модель
#                              X,  # матрица признаков
#                              y,  # вектор цели
#                              cv=kf,  # тип разбиения (можно указать просто число фолдов cv = 3)
#                              scoring='accuracy',  # метрика
#                              n_jobs=-1)  # используются все ядра CPU
#
# print("Кросс-валидация: ", cv_results)
# print("Среднее по кросс-валидации: ", cv_results.mean())
# print("Дисперсия по кросс-валидации: ", cv_results.std())
# skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=15)
# skf.get_n_splits(X, y)
# for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#     print(f"Fold {i + 1}:")
#     print('Train: index={}\n Test:  index={}'.format(train_index, test_index))
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
# cv_results = cross_val_score(model,  # модель
#                              X,  # матрица признаков
#                              y,  # вектор цели
#                              cv=skf,  # тип разбиения
#                              scoring='f1_macro',  # метрика
#                              n_jobs=-1)  # используются все ядра CPU
#
# print("Кросс-валидация: ", cv_results)
# print("Среднее по кросс-валидации: ", cv_results.mean())
# loo = LeaveOneOut()
# for i, (train_index, test_index) in enumerate(loo.split(X)):
#     print(f"Fold {i + 1}:")
#     print('Train: index={}\n Test:  index={}'.format(train_index, test_index))
#
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
# cv_results = cross_val_score(model,  # модель
#                              X,  # матрица признаков
#                              y,  # вектор цели
#                              cv=loo,  # тип разбиения
#                              scoring='f1_macro',  # метрика
#                              n_jobs=-1)  # используются все ядра CPU
#
# print("Кросс-валидация: ", cv_results)
# print("Среднее по кросс-валидации: ", cv_results.mean())

# lpp = LeavePOut(p=3)
# lpp.get_n_splits(X,y)
# for i, (train_index, test_index) in enumerate(lpp.split(X)):
#     # print(f"Fold {i+1}:")
#     # print('Train: index={}\n Test:  index={}'.format(train_index, test_index))
#
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
# cv_results = cross_val_score(model,                  # модель
#                              X,                      # матрица признаков
#                              y,                      # вектор цели
#                              cv = lpp,           # тип разбиения
#                              scoring = 'f1_macro',   # метрика
#                              n_jobs=-1)              # используются все ядра CPU
#
# print("Кросс-валидация: ", cv_results)
# print("Среднее по кросс-валидации: ", cv_results.mean())
# import tqdm
# cv_results=sklearn.model_selection.cross_validate(X,y,cv=6,scoring=['f1_macro','accuracy'],estimator=model)
# print(tqdm.tqdm(cv_results))
# from sklearn.model_selection import cross_validate
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
#
# estimators_stck = [('logreg', LogisticRegression(solver='liblinear')),
#                    ('tree', DecisionTreeClassifier()),
#                    ('randomForest', sklearn.ensemble.RandomForestClassifier())]
#
#
# def evaluate_classifiers(X, y):
#     models = [
#         ('logreg', LogisticRegression(solver='liblinear')),
#         ('tree', DecisionTreeClassifier()),
#         ('randomForest', sklearn.ensemble.RandomForestClassifier()),
#         ('Stacks', sklearn.ensemble.StackingClassifier(estimators=estimators_stck,
#                                                        final_estimator=
#                                                        sklearn.neighbors.KNeighborsClassifier(n_neighbors=2)))
#     ]
#     scores_list = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
#     results = {}
#     for score in tqdm.tqdm(scores_list):
#         results[score] = {}
#         for name, model in models:
#             scores = cross_validate(model, X, y, cv=5, scoring=score)
#             results[score][name] = scores['test_score'].mean()
#
#     return results
#
#
# print(evaluate_classifiers(X, y))
# from sklearn.datasets import load_diabetes
# import xgboost
#
# diabets = load_diabetes()
# X = diabets.data
# Y = diabets.target
# xgb_model = xgboost.XGBRegressor()
# kf = KFold(n_splits=10, shuffle=True)
# cv_results = cross_validate(xgb_model,  # модель
#                             X,  # матрица признаков
#                             Y,  # вектор цели
#                             cv=kf,  # тип разбиения (можно указать просто число фолдов cv = 10)
#                             scoring='neg_mean_squared_error',  # метрика
#                             n_jobs=-1)
# print(cv_results['test_score'].mean())
# print('Бустинг с кроссвалидацией\n')
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
#
# pipes = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2'))
# kf = KFold(n_splits=10, shuffle=True)
# cv_results = cross_val_score(pipes, X, Y, cv=kf, n_jobs=-1, scoring='neg_mean_squared_log_error')
# print(str(cv_results.mean()) + 'с pipe')
