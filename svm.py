import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from useful_functions import plot_decision_regions

# Конфигурация исходных данных для задачи классификации
X, y = make_moons(n_samples=512, random_state=123, noise=0.18)

# Отображение исходных данных
plt.figure(1)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red',
            marker='^', alpha=0.5, label='0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue',
            marker='o', alpha=0.5, label='1')
plt.legend()
plt.title("Исходные данные (Луны)")
plt.show()

# Pазделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Подбор наилучших гиперпараметров
params_for_search = {
    'degree': [2, 3],
    'C': np.logspace(-2, 2, num=20),
    'gamma': ['auto'],
    'coef0': np.arange(0, 2, 0.5),
}

svm = SVC(kernel='poly')
search = GridSearchCV(svm, params_for_search, n_jobs=-1, scoring='roc_auc')
search.fit(X_train, y_train)

cvres = search.cv_results_
best_params = search.best_params_

# Вывод лучших параметров
print(f"CV best score = {search.best_score_}")
print(f"CV error = {1 - search.best_score_}")
print(f"best C = {search.best_estimator_.C}")
print(f"best degree = {search.best_estimator_.degree}")
print(f"coef0 = {search.best_estimator_.coef0}")

# Обучение оптимизированной модели
svm_best = search.best_estimator_

print("Модель bestSVM:",
      "\n   kernel=", svm_best.kernel,
      "\n   C=",      svm_best.C,
      "\n   gamma=",  svm_best.gamma,
      "\n   degree=", svm_best.degree)

svm_best.fit(X_train, y_train)

# Ошибки обучения на обучающей и тестовой выборках
err_train = np.mean(y_train != svm_best.predict(X_train))
err_test = np.mean(y_test != svm_best.predict(X_test))
print(f"{err_train=: };\n{err_test=: }")

# Постороение графика области решений
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plt.figure(figsize=(12, 8))
plot_decision_regions(X_combined, y_combined,
                      classifier=svm_best)
plt.show()
