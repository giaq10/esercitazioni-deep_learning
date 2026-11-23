import torch
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

def hist_plot(history_dict, title):
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(loss) + 1) # 2 3
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    plt.show()

def train_autoencoder(model, optimizer, criterion, train_loader, test_loader, epochs, device):

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)

    for epoch in tqdm(range(epochs)):

        model.train()
        train_loss = torch.zeros((), device=device)
        train_correct = torch.zeros((), device=device)

        for train_inputs, _ in train_loader:

            optimizer.zero_grad()

            train_inputs = train_inputs.to(device)
            train_targets = train_inputs.to(device)

            outputs = model(train_inputs)
            loss = criterion(outputs, train_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach()*len(train_targets)

        model.eval()
        with torch.no_grad():
            val_loss = torch.zeros((), device=device)
            val_correct = torch.zeros((), device=device)

            for val_inputs, _ in test_loader:

                val_inputs = val_inputs.to(device)
                val_targets = val_inputs.to(device)

                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets)*len(val_targets)

        history["loss"].append(train_loss.item() / n_train)
        history["val_loss"].append(val_loss.item() / n_test)

    return history



def train_denoising_autoencoder(model, optimizer, criterion, train_loader, test_loader, epochs, device):

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
            train_loss += loss.detach()*len(train_targets)

        model.eval()
        with torch.no_grad():
            val_loss = torch.zeros((), device=device)
            val_correct = torch.zeros((), device=device)

            for val_inputs, val_targets in test_loader:

                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)

                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets)*len(val_targets)

        history["loss"].append(train_loss.item() / n_train)
        history["val_loss"].append(val_loss.item() / n_test)

    return history

def summary(model, input_size):

    rows, hooks = [], []
    for name, layer in model.named_children():
        hooks.append(layer.register_forward_hook(
            lambda m, i, o, name=name: rows.append((f'{name} ({m.__class__.__name__})', tuple(o.shape), sum(p.numel() for p in m.parameters())))))
    model.eval()
    with torch.no_grad():
        model(torch.zeros((1, *input_size), device=next(model.parameters()).device))
    for h in hooks:
        h.remove()
    print(f"{'Layer (type)':<28}{'Output Shape':<24}{'Param #':>10}")
    print('=' * 62)
    for n, s, p in rows:
        print(f"{n:<28}{str((None, *s[1:])):<24}{p:>10,}")
    print('=' * 62)
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {tot:,}\nTrainable params: {trn:,}\nNon-trainable params: {tot - trn:,}')