'''
Visualize embeddings via TSNE
'''


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


def visualize_distribution(
        embeddings,
        num_class,
        sample_per_class,
        class_names=None):
    '''
    Inputs:
        embeddings: numpy array with shape
            [num_class * sample_per_class, emb_dim]
    '''
    emb_2d = TSNE(n_components=2).fit_transform(embeddings)
    emb_2d = emb_2d.reshape([num_class, sample_per_class, 2])
    fig, axes = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    if class_names is None:
        class_names = [' ' for i in range(num_class)]
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', \
        'orange', 'purple', 'pink']
    for i, class_name in enumerate(class_names):
        x = emb_2d[i, :, 0]
        y = emb_2d[i, :, 1]
        axes.scatter(x=x, y=y, c=color_list[i], label=class_name)

    axes.set_xlabel("Component 1")
    axes.set_ylabel("Component 2")
    axes.legend(loc='upper left')
    axes.set_xlim(-300, 300)
    axes.set_ylim(-300, 300)
    plt.show()
