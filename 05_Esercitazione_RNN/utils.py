# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import os
# import pickle
#
# import torch
#
# from tqdm import tqdm

# def summary(model, input_size):
#
#     rows, hooks = [], []
#     for name, layer in model.named_children():
#         hooks.append(layer.register_forward_hook(
#             lambda m, i, o, name=name: rows.append((f'{name} ({m.__class__.__name__})', tuple(o.shape), sum(p.numel() for p in m.parameters())))))
#     model.eval()
#     with torch.no_grad():
#         model(torch.zeros((1, *input_size), device=next(model.parameters()).device))
#     for h in hooks:
#         h.remove()
#     print(f"{'Layer (type)':<28}{'Output Shape':<24}{'Param #':>10}")
#     print('=' * 62)
#     for n, s, p in rows:
#         print(f"{n:<28}{str((None, *s[1:])):<24}{p:>10,}")
#     print('=' * 62)
#     tot = sum(p.numel() for p in model.parameters())
#     trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f'Total params: {tot:,}\nTrainable params: {trn:,}\nNon-trainable params: {tot - trn:,}')
#
#
# def evaluate(model, x, y, loss_fn, batch_size=32, device = 'cuda'):
#     model.eval()
#     loss_sum, correct = 0.0, 0
#     with torch.no_grad():
#         for i in range(0, len(x), batch_size):
#             xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
#             logits = model(xb)
#             loss_sum += loss_fn(logits, yb).item() * len(xb)
#             correct += (logits.argmax(1) == yb).sum().item()
#     return loss_sum / len(x), correct / len(x)
#
#
# def fit(model, optimizer, loss_fn, x, y, epochs, verbose=0, validation_data=None, batch_size=32, device='cuda'):
#
#     x = torch.as_tensor(x, dtype=torch.float32)
#     y = torch.as_tensor(y, dtype=torch.long)
#     if validation_data is not None:
#         x_val = torch.as_tensor(validation_data[0], dtype=torch.float32)
#         y_val = torch.as_tensor(validation_data[1], dtype=torch.long)
#     history = {'loss': [], 'accuracy': []}
#     if validation_data is not None:
#         history.update({'val_loss': [], 'val_accuracy': []})
#     for epoch in tqdm(range(epochs)):
#         model.train()
#         perm = torch.randperm(len(x))
#         loss_sum, correct = 0.0, 0
#         for i in range(0, len(x), batch_size):
#             idx = perm[i:i + batch_size]
#             xb, yb = x[idx].to(device), y[idx].to(device)
#             optimizer.zero_grad()
#             logits = model(xb)
#             loss = loss_fn(logits, yb)
#             loss.backward()
#             optimizer.step()
#             loss_sum += loss.item() * len(xb)
#             correct += (logits.argmax(1) == yb).sum().item()
#         history['loss'].append(loss_sum / len(x))
#         history['accuracy'].append(correct / len(x))
#         msg = f"Epoch {epoch + 1}/{epochs} - loss: {history['loss'][-1]:.4f} - accuracy: {history['accuracy'][-1]:.4f}"
#         if validation_data is not None:
#             val_loss, val_acc = evaluate(model, x_val, y_val, loss_fn, batch_size)
#             history['val_loss'].append(val_loss)
#             history['val_accuracy'].append(val_acc)
#             msg += f' - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}'
#         if verbose:
#             print(msg)
#     return history
#
# def hist_plot(history_dict, title):
#     acc = history_dict['accuracy']
#     val_acc = history_dict['val_accuracy']
#     loss = history_dict['loss']
#     epochs = range(1, len(acc) + 1) # 2 3
#     plt.figure()
#     plt.plot(epochs, acc, 'bo', label='Training acc')
#     plt.plot(epochs, val_acc, 'b', label='Validation acc')
#     plt.title(title)
#     eps = 3e-3
#     n = 7
#     ym = np.min([np.min(acc), np.min(val_acc)])-eps
#     yM = np.max([np.max(acc), np.max(val_acc)])+eps
#     yt = np.array([*range(n+1)])*(yM-ym)/n+ym
#     plt.yticks(yt)
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend(loc='lower right')
#     plt.ion()
#     plt.show()
#     plt.pause(3.0001)

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

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
    ax_acc.set_ylim(ym, yM)
    ax_acc.set_yticks(np.linspace(ym, yM, n + 1))   # n intervalli uguali tra minimo e massimo
    ax_acc.yaxis.set_major_formatter('{x:.3f}')
    ax_acc.set_title('Accuracy')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(loc='lower right')

    fig.suptitle(title)
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

def summary(model, input_size, dtype=None):
    # input_size NON include la dimensione del batch (viene aggiunta qui sotto).
    # Se il primo layer è un Embedding, l'input deve essere fatto di indici interi
    # (torch.long); negli altri casi si usa float32. Si può forzare con dtype=...
    if dtype is None:
        first = next(model.children(), None)
        dtype = torch.long if isinstance(first, torch.nn.Embedding) else torch.float32

    rows, hooks = [], []
    for name, layer in model.named_children():
        def hook(m, i, o, name=name):
            if isinstance(o, tuple):  # es. nn.RNN/nn.LSTM restituiscono (output, h_n)
                o = o[0]
            rows.append((f'{name} ({m.__class__.__name__})', tuple(o.shape), sum(p.numel() for p in m.parameters())))
        hooks.append(layer.register_forward_hook(hook))
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros((1, *input_size), dtype=dtype, device=device))
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
