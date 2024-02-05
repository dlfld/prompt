# Given label names and confusion matrix data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

confusion_matrix = np.array([
    [498, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 16, 8, 1, 0, 32, 2],
    [14, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 111, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 39, 3],
    [0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 1, 0, 1, 0, 4, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
    [7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 4, 27, 0],
    [0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 243, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0],
    [119, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 1, 0, 0, 31, 0],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [13, 0, 16, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 0, 26, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [15, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 924, 0],
    [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 7, 24]])
print(confusion_matrix.shape)
categories = ['NR', 'NN', 'AD', 'PN', 'OD', 'CC', 'DEG', 'SP', 'VV', 'M', 'PU', 'CD', 'BP', 'JJ', 'LC', 'VC', 'VA',
              'VE']
print(len(categories))
df_cm = pd.DataFrame(confusion_matrix, index=categories, columns=categories)
fig = plt.figure(figsize=(10, 10))
heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
