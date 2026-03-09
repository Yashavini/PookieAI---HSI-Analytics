import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64

def load_indian_pines(data_path, gt_path):
    data = loadmat(data_path)
    gt = loadmat(gt_path)
    # Common keys: 'indian_pines_corrected' and 'indian_pines_gt'
    X = None
    y = None
    for key in data:
        if not key.startswith('__'):
            X = data[key]
            break
    for key in gt:
        if not key.startswith('__'):
            y = gt[key]
            break
    return X, y

def preprocess(X, y, test_size=0.2, n_components=30, window_size=5):
    h, w, bands = X.shape
    scaler = MinMaxScaler()
    X_flat = X.reshape(-1, bands)
    X_scaled = scaler.fit_transform(X_flat).reshape(h, w, bands)

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled.reshape(-1, bands)).reshape(h, w, n_components)

    # Pad for patches
    pad = window_size // 2
    X_padded = np.pad(X_pca, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    patches = []
    labels = []
    for i in range(h):
        for j in range(w):
            if y[i, j] > 0:  # skip background
                patch = X_padded[i:i + window_size, j:j + window_size, :]
                patches.append(patch)
                labels.append(y[i, j] - 1)  # 0-indexed

    X_out = np.array(patches)
    y_out = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X_out, y_out, test_size=test_size, random_state=42, stratify=y_out
    )
    return X_train, X_test, y_train, y_test, pca, scaler

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

def make_confusion_matrix_img(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    return fig_to_base64(fig)

def make_classification_map(model, X_full, y_full, pca, scaler, window_size=5):
    h, w, bands = X_full.shape
    X_flat = X_full.reshape(-1, bands)
    X_scaled = scaler.transform(X_flat).reshape(h, w, bands)
    X_pca = pca.transform(X_scaled.reshape(-1, bands)).reshape(h, w, pca.n_components_)

    pad = window_size // 2
    X_padded = np.pad(X_pca, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    pred_map = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if y_full[i, j] > 0:
                patch = X_padded[i:i + window_size, j:j + window_size, :]
                patch = patch.reshape(1, window_size, window_size, pca.n_components_, 1)
                pred = model.predict(patch, verbose=0)
                pred_map[i, j] = np.argmax(pred) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(y_full, cmap='jet')
    axes[0].set_title('Ground Truth')
    axes[1].imshow(pred_map, cmap='jet')
    axes[1].set_title('Predicted')
    return fig_to_base64(fig)

def make_pca_visualization(X, n_components=3):
    h, w, bands = X.shape
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X.reshape(-1, bands)).reshape(h, w, n_components)
    # Normalize to 0-1 for RGB display
    for i in range(n_components):
        band = X_pca[:, :, i]
        X_pca[:, :, i] = (band - band.min()) / (band.max() - band.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(X_pca[:, :, :3])
    ax.set_title('PCA (First 3 Components as RGB)')
    return fig_to_base64(fig)

def get_spectral_signature(X, y, pixel_row, pixel_col):
    spectrum = X[pixel_row, pixel_col, :]
    label = y[pixel_row, pixel_col]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(spectrum)
    ax.set_xlabel('Band')
    ax.set_ylabel('Reflectance')
    ax.set_title(f'Spectral Signature at ({pixel_row},{pixel_col}) - Class {label}')
    return fig_to_base64(fig), int(label)
