import torch
from tqdm.auto import tqdm
def train_vae(model, optimizer, criterion, train_loader, test_loader, epochs, device):
    """Come utils.train_autoencoder, ma per un VAE: il modello restituisce (x_hat, mean, log_var)
    e la loss (ricostruzione + KL) è una somma sul batch. La history riporta la loss media per immagine."""
    history = {'loss': [], 'val_loss': []}
    for epoch in tqdm(range(epochs)):
        model.train()
        total = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = criterion(x, x_hat, mean, log_var)
            loss.backward()
            optimizer.step()
            total += loss.item()
        history['loss'].append(total / len(train_loader.dataset))

        model.eval()
        total = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                x_hat, mean, log_var = model(x)
                total += criterion(x, x_hat, mean, log_var).item()
        history['val_loss'].append(total / len(test_loader.dataset))
    return history