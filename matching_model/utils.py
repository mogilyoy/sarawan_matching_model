import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon

def gradient_fill(x, y, fill_color=None, ax=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    
    if ax is None:
        ax = plt.gca()

    line, = ax.plot(x, y, **kwargs)
    
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return line, im


def plot_train_process(train_loss, train_acc, val_loss = None, val_acc = None):
    
    fig, axes = plt.subplots(2, 5, figsize=(30, 10)) #create one row and two columns with graphs
    fig.set_facecolor('#2a2d31')      #border color
    
    axes[0, 0].set_facecolor('#2a2d31')
    axes[0, 1].set_facecolor('#2a2d31')
    axes[0, 2].set_facecolor('#2a2d31')
    axes[0, 3].set_facecolor('#2a2d31')
    axes[0, 4].set_facecolor('#2a2d31') 
    
    axes[0, 0].set_title('Loss_1', color='#787878')
    axes[0, 1].set_title('Loss_2', color='#787878')
    axes[0, 2].set_title('Loss_3', color='#787878')
    axes[0, 3].set_title('Loss_arc', color='#787878')
    axes[0, 4].set_title('Loss_main', color='#787878')
    
    axes[1, 0].set_facecolor('#2a2d31')
    axes[1, 1].set_facecolor('#2a2d31')
    axes[1, 2].set_facecolor('#2a2d31')
    axes[1, 3].set_facecolor('#2a2d31')
    axes[1, 4].set_facecolor('#2a2d31') 
    
    axes[1, 0].set_title('Accuracy_1', color='#787878')
    axes[1, 1].set_title('Accuracy_2', color='#787878')
    axes[1, 2].set_title('Accuracy_3', color='#787878')
    axes[1, 3].set_title('Accuracy_arc', color='#787878')
    axes[1, 4].set_title('Accuracy_main', color='#787878')
    
    
    x1 = [i for i in range(len(train_loss[0]))]
    x2 = [i for i in range(len(train_loss[1]))]
    x3 = [i for i in range(len(train_loss[2]))]
    x_arc = [i for i in range(len(train_loss[3]))]
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    x_arc = np.array(x_arc)
    if np.isnan(train_loss[4]).any() != True:
        x_main = [i for i in range(len(train_loss[4]))]
        x_main = np.array(x_main)
        line, im  = gradient_fill(x_main, train_loss[4], fill_color='#8c5096', ax=axes[0, 4], color="#8c5096", label='train')
        
    x1 = [i for i in range(len(train_acc[0]))]
    x2 = [i for i in range(len(train_acc[1]))]
    x3 = [i for i in range(len(train_acc[2]))]
    x_arc = [i for i in range(len(train_acc[3]))]
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    x_arc = np.array(x_arc)
    if np.isnan(train_acc[4]).any() != True:
        x_main = [i for i in range(len(train_acc[4]))]
        x_main = np.array(x_main)
        line, im  = gradient_fill(x_main, train_acc[4], fill_color='#8c5096', ax=axes[1, 4], color="#8c5096", label='train')
    
    
    line, im  = gradient_fill(x1, train_loss[0], fill_color='#8c5096', ax=axes[0, 0], color="#8c5096", label='train')
    line, im  = gradient_fill(x2, train_loss[1], fill_color='#8c5096', ax=axes[0, 1], color="#8c5096", label='train')
    line, im  = gradient_fill(x3, train_loss[2], fill_color='#8c5096', ax=axes[0, 2], color="#8c5096", label='train')
    line, im  = gradient_fill(x_arc, train_loss[3], fill_color='#8c5096', ax=axes[0, 3], color="#8c5096", label='train')
    
    line, im  = gradient_fill(x1, train_acc[0], fill_color='#8c5096', ax=axes[1, 0], color="#8c5096", label='train')
    line, im  = gradient_fill(x2, train_acc[1], fill_color='#8c5096', ax=axes[1, 1], color="#8c5096", label='train')
    line, im  = gradient_fill(x3, train_acc[2], fill_color='#8c5096', ax=axes[1, 2], color="#8c5096", label='train')
    line, im  = gradient_fill(x_arc, train_acc[3], fill_color='#8c5096', ax=axes[1, 3], color="#8c5096", label='train')
    
    if val_loss[0] is not None:
        line, im  = gradient_fill(x1, val_loss[0], fill_color='#cd5f82', ax=axes[0, 0], color="#cd5f82", label='validation')
        axes[0, 0].legend(facecolor='#363a40',labelcolor='linecolor')
    if val_loss[1] is not None:
        line, im  = gradient_fill(x2, val_loss[1], fill_color='#cd5f82', ax=axes[0, 1], color="#cd5f82", label='validation')
        axes[0, 1].legend(facecolor='#363a40',labelcolor='linecolor')
    if val_loss[2] is not None:
        line, im  = gradient_fill(x3, val_loss[2], fill_color='#cd5f82', ax=axes[0, 2], color="#cd5f82", label='validation')
        axes[0, 2].legend(facecolor='#363a40',labelcolor='linecolor')
    if val_loss[3] is not None:
        line, im  = gradient_fill(x_arc, val_loss[3], fill_color='#cd5f82', ax=axes[0, 3], color="#cd5f82", label='validation')
        axes[0, 3].legend(facecolor='#363a40',labelcolor='linecolor')
    if val_loss[4] is not None and np.isnan(val_loss[4]).any() != True:
        line, im  = gradient_fill(x_main, val_loss[4], fill_color='#cd5f82', ax=axes[0, 4], color="#cd5f82", label='validation')
        axes[0, 4].legend(facecolor='#363a40',labelcolor='linecolor')
    
    if val_acc[0] is not None:
        line, im  = gradient_fill(x1, val_acc[0], fill_color='#cd5f82', ax=axes[1, 0], color="#cd5f82", label='validation')
        axes[1, 0].legend(facecolor='#363a40',labelcolor='linecolor')
    if val_acc[1] is not None:
        line, im  = gradient_fill(x2, val_acc[1], fill_color='#cd5f82', ax=axes[1, 1], color="#cd5f82", label='validation')
        axes[1, 1].legend(facecolor='#363a40',labelcolor='linecolor')
    if val_acc[2] is not None:
        line, im  = gradient_fill(x3, val_acc[2], fill_color='#cd5f82', ax=axes[1, 2], color="#cd5f82", label='validation')
        axes[1, 2].legend(facecolor='#363a40',labelcolor='linecolor')
    if val_acc[3] is not None:
        line, im  = gradient_fill(x_arc, val_acc[3], fill_color='#cd5f82', ax=axes[1, 3], color="#cd5f82", label='validation')
        axes[1, 3].legend(facecolor='#363a40',labelcolor='linecolor')
    if val_acc[4] is not None and np.isnan(val_acc[4]).any() != True:
        line, im  = gradient_fill(x_main, val_acc[4], fill_color='#cd5f82', ax=axes[1, 4], color="#cd5f82", label='validation')
        axes[1, 4].legend(facecolor='#363a40',labelcolor='linecolor')

    axes[0, 0].tick_params(axis='both', which='both', labelsize=14, labelcolor='#787878')
    axes[0, 1].tick_params(axis='both', which='both', labelsize=14, labelcolor='#787878')
    axes[0, 2].tick_params(axis='both', which='both', labelsize=14, labelcolor='#787878')
    axes[0, 3].tick_params(axis='both', which='both', labelsize=14, labelcolor='#787878')
    axes[0, 4].tick_params(axis='both', which='both', labelsize=14, labelcolor='#787878')
    
    axes[1, 0].tick_params(axis='both', which='both', labelsize=14, labelcolor='#787878')
    axes[1, 1].tick_params(axis='both', which='both', labelsize=14, labelcolor='#787878')
    axes[1, 2].tick_params(axis='both', which='both', labelsize=14, labelcolor='#787878')
    axes[1, 3].tick_params(axis='both', which='both', labelsize=14, labelcolor='#787878')
    axes[1, 4].tick_params(axis='both', which='both', labelsize=14, labelcolor='#787878')

    
    for spine in axes[0, 0].spines.values():
        spine.set_edgecolor('#787878')
        
    for spine in axes[0, 1].spines.values():
        spine.set_edgecolor('#787878')
    
    for spine in axes[0, 2].spines.values():
        spine.set_edgecolor('#787878')
    
    for spine in axes[0, 3].spines.values():
        spine.set_edgecolor('#787878')
    
    for spine in axes[0, 4].spines.values():
        spine.set_edgecolor('#787878')
        
    for spine in axes[0, 0].spines.values():
        spine.set_edgecolor('#787878')
        
    for spine in axes[1, 1].spines.values():
        spine.set_edgecolor('#787878')
    
    for spine in axes[1, 2].spines.values():
        spine.set_edgecolor('#787878')
    
    for spine in axes[1, 3].spines.values():
        spine.set_edgecolor('#787878')
    
    for spine in axes[1, 4].spines.values():
        spine.set_edgecolor('#787878')
        
    plt.show()

