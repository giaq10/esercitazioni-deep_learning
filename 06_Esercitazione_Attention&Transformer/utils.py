import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

def hist_plot(history_dict):

    epochs = range(1, len(history_dict['loss']) + 1)
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict['loss'], 'bo', label='Training loss')
    plt.plot(epochs, history_dict['val_loss'], 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-entropy')
    plt.legend(loc='upper right')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict['accuracy'], 'bo', label='Training accuracy')
    plt.plot(epochs, history_dict['val_accuracy'], 'b', label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (token, con teacher forcing)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def train(model, optimizer, criterion, train_loader, test_loader, epochs, device):

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)

    for epoch in tqdm(range(epochs)):

        model.train()
        train_loss = torch.zeros((), device=device)
        train_correct = torch.zeros((), device=device)

        for train_inputs, train_targets in train_loader:
            optimizer.zero_grad()

            train_inputs = train_inputs.to(device)
            train_targets = train_targets.to(device)

            outputs = model(train_inputs)
            loss = criterion(outputs, train_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach() * len(train_targets)
            train_correct += (outputs.argmax(dim=1) == train_targets).sum()

        model.eval()
        with torch.no_grad():
            val_loss = torch.zeros((), device=device)
            val_correct = torch.zeros((), device=device)

            for val_inputs, val_targets in test_loader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)

                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets) * len(val_targets)
                val_correct += (val_outputs.argmax(dim=1) == val_targets).sum()

        history["loss"].append(train_loss.item() / n_train)
        history["val_loss"].append(val_loss.item() / n_test)
        history["accuracy"].append(train_correct.item() / n_train)
        history["val_accuracy"].append(val_correct.item() / n_test)

    return history