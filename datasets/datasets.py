import os
import re
import csv
import time
import zipfile
import warnings
import numpy as np
import pandas as pd
import urllib.request
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Cartella che contiene questo file: i dataset vengono salvati qui,
# indipendentemente dalla cartella da cui viene eseguito il notebook
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_MNIST(root=DATA_DIR, batch_size=256, verbose=True):
    """Scarica (se necessario) MNIST e restituisce dataset e DataLoader.

    Le immagini sono tensori float in [0, 1] di forma (1, 28, 28).
    Il train loader mescola i dati a ogni epoca, il test loader no.

    Returns:
        train_dataset, test_dataset, train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if verbose:
        print(f"TRAIN SET shape: {train_dataset.data.shape}")
        print(f"TEST SET shape: {test_dataset.data.shape}")

    return train_dataset, test_dataset, train_loader, test_loader



def load_FMNIST(root=DATA_DIR, batch_size=256, verbose=True):
    """Scarica (se necessario) FMNIST e restituisce dataset e DataLoader.

    Le immagini sono tensori float in [0, 1] di forma (1, 28, 28).
    Il train loader mescola i dati a ogni epoca, il test loader no.

    Returns:
        train_dataset, test_dataset, train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if verbose:
        print(f"TRAIN SET shape: {train_dataset.data.shape}")
        print(f"TEST SET shape: {test_dataset.data.shape}")

    return train_dataset, test_dataset, train_loader, test_loader


def load_CIFAR10(root=DATA_DIR, batch_size=256, verbose=True):
    """Scarica (se necessario) CIFAR-10 e restituisce dataset e DataLoader.

    Le immagini sono tensori float in [0, 1] di forma (3, 32, 32).
    Il train loader mescola i dati a ogni epoca, il test loader no.

    Nota: a differenza di MNIST, `dataset.data` è un array numpy di forma
    (N, 32, 32, 3) e `dataset.targets` è una lista Python, non un tensore.
    I nomi delle classi sono in `dataset.classes`.

    Returns:
        train_dataset, test_dataset, train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=root, train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if verbose:
        print(f"TRAIN SET shape: {train_dataset.data.shape}")
        print(f"TEST SET shape: {test_dataset.data.shape}")

    return train_dataset, test_dataset, train_loader, test_loader


def load_AUTOMPG(root = DATA_DIR):
    """Scarica (se necessario) AUTO-MPG e restituisce un dataframe pandas.

        Returns:
            dataset
    """
    dataset_path = f"{root}/auto-mpg.data"
    if not os.path.exists(dataset_path):
        urllib.request.urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                                   dataset_path)
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']  # MPG = Miles per gallons
    raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=" ",skipinitialspace=True)
    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    dataset.pop("Model Year")
    dataset.pop("Origin")

    return dataset


def load_HAR(root = DATA_DIR):
    """
        Scarica (se necessario) AUTO-MPG e restituisce un dataframe pandas.

        Returns:
            (x_train_view_1, x_train_view_2, y_train), (x_test_view_1, x_test_view_2, y_test)
    """
    train_set = pd.read_csv(f"{root}/HAR/train.csv")
    x_train_view_1 = train_set.filter(like="Acc")
    x_train_view_2 = train_set.filter(like="Gyro")
    y_train = train_set["Activity"]


    test_set = pd.read_csv(f"{root}/HAR/test.csv")
    x_test_view_1 = test_set.filter(like="Acc")
    x_test_view_2 = test_set.filter(like="Gyro")
    y_test = test_set["Activity"]

    return (x_train_view_1, x_train_view_2, y_train), (x_test_view_1, x_test_view_2, y_test)

def load_AIRLINE(root = DATA_DIR):
    """Scarica (se necessario) AIRLINE PASSENGERS e restituisce un dataframe pandas.

    Numero mensile di passeggeri delle linee aeree internazionali (in migliaia),
    da gennaio 1949 a dicembre 1960: 144 righe con le colonne 'Month' (datetime)
    e 'Passengers' (int).

        Returns:
            dataset
    """
    dataset_path = f"{root}/airline-passengers.csv"
    if not os.path.exists(dataset_path):
        urllib.request.urlretrieve("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
                                   dataset_path)
    dataset = pd.read_csv(dataset_path, parse_dates=['Month'])

    return dataset

def load_BASICMOTIONS(root = DATA_DIR):
    """Scarica (se necessario) BASICMOTIONS (archivio UEA) e restituisce array numpy.

    Classificazione di serie temporali multivariate: uno smartwatch registra accelerometro
    e giroscopio (3 assi ciascuno, 6 canali) per 10 secondi a 10 Hz (100 passi) mentre si
    svolge una di 4 attivita': Standing, Running, Walking, Badminton.
    40 serie di training e 40 di test, 10 per classe.

        Returns:
            (x_train, y_train), (x_test, y_test), classes
            x: array float32 di forma (serie, 100, 6) = (serie, passi temporali, canali)
            y: array int64 con l'indice della classe nella lista classes
    """

    def _read_ts(path):
        """Legge un file .ts dell'archivio UCR/UEA con serie di lunghezza fissa e senza valori mancanti.

        Ogni riga dopo @data contiene i canali separati da ':' (valori separati da ',')
        e come ultimo campo l'etichetta.

            Returns:
                x (array float32 di forma (serie, passi temporali, canali)), y (indici delle classi), classes
        """
        classes, x, labels, in_data = None, [], [], False
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if not in_data:
                    if line.lower().startswith('@classlabel'):
                        classes = line.split()[2:]
                    in_data = line.lower() == '@data'
                    continue
                *dims, label = line.split(':')
                x.append([[float(v) for v in d.split(',')] for d in dims])
                labels.append(label)
        x = np.array(x, dtype=np.float32).transpose(0, 2, 1)  # (serie, canali, passi) -> (serie, passi, canali)
        y = np.array([classes.index(label) for label in labels], dtype=np.int64)
        return x, y, classes

    folder = f"{root}/BasicMotions"
    os.makedirs(folder, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/aeon-toolkit/aeon/v1.0.0/aeon/datasets/data/BasicMotions"

    splits = []
    for split in ["TRAIN", "TEST"]:
        path = f"{folder}/BasicMotions_{split}.ts"
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"{base_url}/BasicMotions_{split}.ts", path)
        x, y, classes = _read_ts(path)
        splits.append((x, y))

    return splits[0], splits[1], classes


def load_IMDB(root = DATA_DIR, num_words=10000, maxlen=500):
    """Scarica (se necessario) IMDB e restituisce array numpy, come keras.datasets.imdb.load_data
    seguito da keras.utils.pad_sequences.

    50.000 recensioni di film (25.000 di training e 25.000 di test, meta' positive e meta' negative),
    gia' convertite in sequenze di indici di parole. Gli indici seguono la convenzione di Keras:
    0 = padding, 1 = inizio della recensione, 2 = parola fuori vocabolario, da 3 in poi le parole
    ordinate per frequenza. Si tengono solo gli indici minori di num_words (gli altri diventano 2).
    Ogni recensione viene portata a maxlen parole: a quelle piu' lunghe si tolgono le parole iniziali,
    a quelle piu' corte si aggiungono zeri all'inizio, cosi' l'ultimo passo e' sempre una parola vera.

        Returns:
            (x_train, y_train), (x_test, y_test)
            x: array int64 di forma (25000, maxlen) con gli indici delle parole
            y: array int64 con 0 = negativa, 1 = positiva
    """
    dataset_path = f"{root}/imdb.npz"
    if not os.path.exists(dataset_path):
        urllib.request.urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz",
                                   dataset_path)

    # Le recensioni hanno lunghezze diverse e sono salvate come liste Python: serve allow_pickle
    with np.load(dataset_path, allow_pickle=True) as f:
        splits = [(f["x_train"], f["y_train"]), (f["x_test"], f["y_test"])]

    rng = np.random.RandomState(113)    # stesso mescolamento di Keras
    result = []
    for x, y in splits:
        indices = np.arange(len(x))
        rng.shuffle(indices)
        x, y = x[indices], y[indices]
        padded = np.zeros((len(x), maxlen), dtype=np.int64)
        for i, review in enumerate(x):
            review = [1] + [w + 3 if w + 3 < num_words else 2 for w in review]   # inizio, indici spostati di 3, parole rare -> 2
            review = review[-maxlen:]                                            # tiene le ultime maxlen parole
            padded[i, maxlen - len(review):] = review                            # zeri all'inizio
        result.append((padded, y.astype(np.int64)))

    return result[0], result[1]



def load_PROMESSI_SPOSI(root = DATA_DIR, filename = "da_verona_i_promessi_sposi.txt"):
    """Legge I PROMESSI SPOSI di Guido da Verona da un file di testo e restituisce il testo del romanzo.

    E' la parodia del romanzo di Manzoni pubblicata da Guido da Verona nel 1930 (edizione
    elettronica Liber Liber, progetto Manuzio): segue la trama e molte frasi dell'originale,
    riscritte in chiave comica e piene di anacronismi. Il file va messo nella cartella di questo
    modulo. Si tiene solo il romanzo, dal titolo del primo capitolo all'ultima frase: si tolgono
    le note di Liber Liber, la prefazione dell'autore e i titoli dei capitoli.
    Il testo viene restituito su un'unica riga, con gli spazi normalizzati.

        Returns:
            testo (stringa)
    """
    dataset_path = os.path.join(root, filename)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"File non trovato: copia {filename} nella cartella {os.path.abspath(root)}")
    with open(dataset_path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")        # i testi di Liber Liber sono spesso nella codifica di Windows
    text = text.replace("\r\n", "\n").replace("\u2019", "'")      # apostrofo tipografico -> '

    # Solo il romanzo: dal titolo del primo capitolo all'ultima frase
    # (non da "Quel ramo del lago di Como", che compare gia' nella prefazione)
    start = re.search(r"^\s*CAPITOLO\s+(I|PRIMO)\.?\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    end = re.search(r"non\s+s'è\s+fatto\s+apposta\.", text)
    if start and end:
        text = text[start.end():end.end()]
    else:
        warnings.warn("Inizio o fine del romanzo non trovati: si usa tutto il file")

    lines = [line for line in text.split("\n")
             if re.search(r"[a-zàèéìòù]", line)]   # via i titoli tutti maiuscoli (CAPITOLO II, CAP. IV)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()

def _download(url, path, manual_hint, attempts=3):
    """Scarica url e lo salva in path.

    Alcuni siti rifiutano le richieste con lo User-Agent predefinito di Python, quindi se ne
    manda uno da browser; gli errori temporanei (come 503) si riprovano con una pausa crescente.
    Il file viene scritto solo a download completato, cosi' non resta un file a meta'.
    """
    errors = []
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            with open(path, "wb") as f:
                f.write(data)
            return
        except Exception as e:
            errors.append(str(e))
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Download di {url} non riuscito ({'; '.join(errors)}).\n"
                       f"In alternativa {manual_hint} e salvalo come {os.path.abspath(path)}")

def load_ITA_ENG(root = DATA_DIR):
    """Scarica (se necessario) le coppie di frasi inglese-italiano e restituisce un dataframe pandas.

    Frasi del progetto Tatoeba raccolte da www.manythings.org/anki: ogni riga e' una frase
    inglese con una sua traduzione italiana. La stessa frase inglese puo' comparire piu' volte,
    con traduzioni diverse (es. "I'm tired." -> "Sono stanco." e "Sono stanca.").
    Licenza CC-BY 2.0 (France): attribuzione a tatoeba.org e www.manythings.org/anki.

        Returns:
            dataset con le colonne 'english' e 'italian'
    """
    txt_path = f"{root}/ita.txt"
    zip_path = f"{root}/ita-eng.zip"
    if not os.path.exists(txt_path):
        if not os.path.exists(zip_path):
            _download("https://www.manythings.org/anki/ita-eng.zip", zip_path,
                      "scarica a mano ita-eng.zip da https://www.manythings.org/anki/")
        with zipfile.ZipFile(zip_path) as z:
            z.extract("ita.txt", root)

    # Colonne separate da tabulazioni: inglese, italiano, attribuzione. Le virgolette fanno parte delle
    # frasi (QUOTE_NONE) e frasi come "NA" non vanno lette come valori mancanti (na_filter=False)
    dataset = pd.read_csv(txt_path, sep="\t", header=None, names=["english", "italian", "attribution"],
                          quoting=csv.QUOTE_NONE, na_filter=False, encoding="utf-8")
    return dataset[["english", "italian"]]