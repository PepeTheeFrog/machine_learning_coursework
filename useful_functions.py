#    ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ
# Построение графика поверхности решения


# Для работы с массивами элементов одного типа
import numpy as np
import matplotlib.pyplot as plt              # Для построения научных графиков

from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02, test_idx=None):

    # Настройка генератора маркеров и палитры
    markers = ('s', 'v', 'o', '^', 'x')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Построение графика поверхности решения
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Отобразить образцы классов

    # Пометить тестовые образцы точками (при необходимости)
    if test_idx:                # если заданы номера тестовых образцов
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='w',
                    alpha=0.8,  edgecolor='black', linewidths=1, marker='o',
                    s=120, label='test set')

    for idx, cl in enumerate(np.unique(y)):

        if markers[idx] != 'x':
            edgecolor = 'black'
        else:
            edgecolor = None

        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    edgecolor=edgecolor,
                    marker=markers[idx],
                    label=cl)
