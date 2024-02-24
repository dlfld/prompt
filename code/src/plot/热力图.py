import itertools

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, classes, normalize=False, title='State transition matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")

    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")

    plt.ylabel('Self patt')
    plt.xlabel('Transition patt')

    plt.tight_layout()
    # plt.savefig('res/method_2.png', transparent=True, dpi=800)

    plt.show()


trans_mat = np.array([[410, 1, 2, 0, 0, 0, 0, 0, 0, 0, 7, 0, 18, 0, 3, 0, 117, 1],
                      [3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 8, 0],
                      [11, 0, 31, 0, 0, 0, 0, 0, 0, 0, 5, 0, 3, 0, 3, 0, 99, 7],
                      [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 4, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 2, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0],
                      [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 7, 0, 2, 0, 0, 0, 36, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0],
                      [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 212, 0, 0, 0, 0, 0, 31, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 18, 1],
                      [24, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 28, 0, 1, 0, 117, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0],
                      [7, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 207, 0, 19, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                      [50, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 872, 0],
                      [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 0, 0, 14, 14]], dtype=int)

"""method 2"""
if True:
    # label = [f'{i}' for i in range(1, trans_mat.shape[0] + 1)]
    label = [ "NR", "NN", "AD", "PN", "OD", "CC", "DEG",
                      "SP", "VV", "M", "PU", "CD", "BP", "JJ", "LC", "VC",
                      "VA", "VE"]
    plot_confusion_matrix(trans_mat, label)
