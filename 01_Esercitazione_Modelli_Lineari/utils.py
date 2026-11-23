import numpy as np
import matplotlib.pyplot as plt

def get_random_poly2D_ds(n, poly, test_size=0.5, x_min=-1, x_max=1, to_plot=False):

    # Generazione dei punti: x uniforme, y = polinomio + rumore
    x = np.random.uniform(x_min, x_max, n)
    y = poly(x) + np.random.uniform(-1, 1, n) * np.random.exponential(1, n)

    # Split train/test: le x sono già estratte a caso, quindi basta tagliare
    n_test = int(round(n * test_size))
    x_train, y_train = x[n_test:], y[n_test:]
    x_test,  y_test  = x[:n_test], y[:n_test]

    # Grafico opzionale (dati originali, prima della normalizzazione)
    if to_plot:
        x_plot = np.linspace(x_min, x_max, 200)
        plt.plot(x_plot, poly(x_plot), label="Polinomio")
        plt.plot(x_train, y_train, '.', label="Train set")
        plt.plot(x_test,  y_test,  '.', label="Test set")
        plt.xlim(x_min, x_max)
        plt.legend()
        plt.title(f"Train: {len(x_train)} punti, Test: {len(x_test)} punti")
        plt.show()

    return x_train, y_train, x_test, y_test