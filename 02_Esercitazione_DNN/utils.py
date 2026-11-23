import numpy as np
import matplotlib.pyplot as plt

def hist_plot(history_dict, title):
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, loss, 'bo', label='Training loss')
    ax_loss.plot(epochs, val_loss, 'b', label='Validation loss')
    ax_loss.set_title('Loss')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend(loc='upper right')

    ax_acc.plot(epochs, acc, 'bo', label='Training acc')
    ax_acc.plot(epochs, val_acc, 'b', label='Validation acc')
    eps = 3e-3
    n = 7
    ym = min(np.min(acc), np.min(val_acc)) - eps
    yM = max(np.max(acc), np.max(val_acc)) + eps
    ax_acc.set_yticks(np.linspace(ym, yM, n + 1))   # n intervalli uguali tra minimo e massimo
    ax_acc.set_title('Accuracy')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(loc='lower right')

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
