import os

os.environ["QT_QPA_PLATFORM"] = "wayland"

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# plot training losses in one plot
def multiple_plot(epochs, train_losses, file_name, labels, val_losses=None):
    # plt.plot(epochs, train_losses, 'bo', label='Training loss')
    for idx, label in enumerate(labels):
        plt.plot(epochs[idx], train_losses[idx], label=f'{label} Train')
        if val_losses:
            plt.plot(epochs[idx], val_losses[idx], label=f'{label} Val', linestyle='--')

    # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
    # plt.title()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('off')
    # print(
    #     f'Using SGD Optimizer: Final Training Loss was {round(train_losses[-1], 2)}')
    plt.savefig(file_name)
    plt.close()


def single_plot(epochs, train_losses, val_losses, train_accuracies, val_accuracies):
    plt.subplot(211)
    plt.plot(epochs, train_losses, 'bo', label='Training loss')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.title('Training Loss and Validation Loss, using SGD Optimizer')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('off')
    print(
        f'Using SGD Optimizer: Final Training Loss was {round(train_losses[-1], 2)} and Final Validation Loss was {round(val_losses[-1], 2)}')
    plt.show()
    plt.subplot(212)
    plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.title('Training Accuracy and Validation Accuracy, using SGD Optimizer')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
    plt.legend()
    plt.grid('off')
    print(
        f'\n Using SGD Optimizer: Final Training Accuracy was {round(train_accuracies[-1], 2)} and Final Validation Accuracy was {round(val_accuracies[-1], 2)}')
    plt.show()
