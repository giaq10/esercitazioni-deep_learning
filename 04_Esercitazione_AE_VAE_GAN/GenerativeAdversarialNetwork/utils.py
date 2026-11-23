import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import Subset


def hist_plot(history_dict, title):
    """Grafico di training loss e validation loss a partire da una history con chiavi 'loss' e 'val_loss'."""
    epochs = range(1, len(history_dict['loss']) + 1)
    plt.figure()
    plt.plot(epochs, history_dict['loss'], 'bo', label='Training loss')
    plt.plot(epochs, history_dict['val_loss'], 'b', label='Validation loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()


def _to_image(x):
    """Da tensore (C, H, W) in [-1, 1] ad array (H, W, C) in [0, 1] per plt.imshow."""
    return ((x.detach().cpu() + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()


def _show_gan_samples(generator, noise, real_images, title):
    """Mostra le immagini generate da `noise` (riga sopra) e alcune immagini reali (riga sotto)."""
    generator.eval()
    with torch.no_grad():
        generated = generator(noise)
    generator.train()

    n = len(noise)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.3))
    for i in range(n):
        axes[0, i].imshow(_to_image(generated[i]))
        axes[1, i].imshow(_to_image(real_images[i]))
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    axes[0, n // 2].set_title('Immagini generate')
    axes[1, n // 2].set_title('Immagini reali')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def train_gan(generator, discriminator, g_optimizer, d_optimizer, criterion, train_loader,
              latent_size, device, display_interval=100, clip_value=1.0):
    """Addestra una GAN alternando, per ogni batch di immagini reali, un passo del discriminatore
    e uno del generatore. Ogni batch di train_loader è un'iterazione; le immagini devono stare in [-1, 1].

    Ogni `display_interval` iterazioni mostra immagini generate (sempre dallo stesso rumore,
    così se ne vede l'evoluzione) e immagini reali. I gradienti sono limitati in [-clip_value, clip_value].

    Returns:
        history: dict con 'd_loss', 'g_loss', 'd_acc' (accuratezza del discriminatore su reali e generate)
                 e 'g_acc' (frazione di immagini generate che il discriminatore scambia per reali)
    """
    history = {'d_loss': [], 'g_loss': [], 'd_acc': [], 'g_acc': []}
    fixed_noise = torch.randn(3, latent_size, device=device)
    generator.train()
    discriminator.train()

    bar = tqdm(train_loader, desc='Addestramento GAN', unit='iter')
    for iteration, (real_images,) in enumerate(bar, start=1):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        sigma = 0.1 * max(0.0, 1 - iteration / len(train_loader))   # rumore

        # PASSO DEL DISCRIMINATORE: immagini generate con etichetta 0, immagini reali con etichetta 1
        z = torch.randn(batch_size, latent_size, device=device)
        generated_images = generator(z).detach()          # detach: questo passo non aggiorna il generatore
        combined_images = torch.cat([generated_images, real_images])
        labels = torch.cat([torch.zeros(batch_size, 1),
                            torch.full((batch_size, 1), 0.9)]).to(device)
        hard_labels = (labels > 0.5).float()
        d_optimizer.zero_grad()
        d_pred = discriminator(combined_images + sigma * torch.randn_like(combined_images))
        d_loss = criterion(d_pred, labels)
        d_loss.backward()
        nn.utils.clip_grad_value_(discriminator.parameters(), clip_value)
        d_optimizer.step()
        d_acc = ((d_pred > 0).float() == hard_labels).float().mean()

        # PASSO DEL GENERATORE: discriminatore congelato, immagini generate con etichetta 1 ("reali")
        discriminator.requires_grad_(False)
        z = torch.randn(batch_size, latent_size, device=device)
        fake = generator(z)
        g_pred = discriminator(fake + sigma * torch.randn_like(fake))
        misleading_targets = torch.ones(batch_size, 1, device=device)
        g_loss = criterion(g_pred, misleading_targets)

        g_optimizer.zero_grad()
        g_loss.backward()
        nn.utils.clip_grad_value_(generator.parameters(), clip_value)
        g_optimizer.step()
        discriminator.requires_grad_(True)
        g_acc = (g_pred > 0).float().mean()             # immagini generate scambiate per reali

        history['d_loss'].append(d_loss.item())
        history['g_loss'].append(g_loss.item())
        history['d_acc'].append(d_acc.item())
        history['g_acc'].append(g_acc.item())
        bar.set_postfix({'D_loss': f"{history['d_loss'][-1]:.4f}", 'D_acc': f"{history['d_acc'][-1]:.0%}",
                         'G_loss': f"{history['g_loss'][-1]:.4f}", 'G_acc': f"{history['g_acc'][-1]:.0%}"})

        if iteration % display_interval == 0:
            _show_gan_samples(generator, fixed_noise, real_images[:3],
                              f"Iterazione {iteration}/{len(train_loader)}  |  "
                              f"D loss {history['d_loss'][-1]:.3f}, G loss {history['g_loss'][-1]:.3f}")
    return history
