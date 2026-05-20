import numpy as np

# Dados de Treinamento (Apendice)
# x1, x2, d
train_data_raw = [
    [0.2563, 0.9503, -1],
    [0.2405, 0.9018, -1],
    [0.1157, 0.3676, 1],
    [0.5147, 0.0167, 1],
    [0.4127, 0.3275, 1],
    [0.2809, 0.5830, 1],
    [0.8263, 0.9301, -1],
    [0.9359, 0.8724, -1],
    [0.1096, 0.9165, -1],
    [0.5158, 0.8545, -1],
    [0.1334, 0.1362, 1],
    [0.6371, 0.1439, 1],
    [0.7052, 0.6277, -1],
    [0.8703, 0.8666, -1],
    [0.2612, 0.6109, 1],
    [0.0244, 0.5279, 1],
    [0.9588, 0.3672, -1],
    [0.9332, 0.5499, -1],
    [0.9623, 0.2961, -1],
    [0.7297, 0.5776, -1],
    [0.4560, 0.1871, 1],
    [0.1715, 0.7713, 1],
    [0.5571, 0.5485, -1],
    [0.3344, 0.0259, 1],
    [0.4803, 0.7635, -1],
    [0.9721, 0.4850, -1],
    [0.8318, 0.7844, -1],
    [0.1373, 0.0292, 1],
    [0.3660, 0.8581, -1],
    [0.3626, 0.7302, -1],
    [0.6474, 0.3324, 1],
    [0.3461, 0.2398, 1],
    [0.1353, 0.8120, 1],
    [0.3463, 0.1017, 1],
    [0.9086, 0.1947, -1],
    [0.5227, 0.2321, 1],
    [0.5153, 0.2041, 1],
    [0.1832, 0.0661, 1],
    [0.5015, 0.9812, -1],
    [0.5024, 0.5274, -1]
]
train_data = np.array(train_data_raw)

# Extrair apenas os dados com presenca de radiacao (d = 1) para o k-means
X_rad = np.array([row[:2] for row in train_data if row[2] == 1])

from typing import Tuple, List

# Implementacao do K-Means manual
def run_kmeans(X: np.ndarray, n_clusters: int = 2, max_iters: int = 100, seed: int = 42) -> Tuple[np.ndarray, List[float]]:
    np.random.seed(seed)
    # Inicializando centros com dois pontos aleatorios do conjunto
    indices = np.random.choice(len(X), n_clusters, replace=False)
    centers = X[indices]
    
    for _ in range(max_iters):
        # Atribuir aos clusters
        labels = np.array([np.argmin([np.sum((x - c)**2) for c in centers]) for x in X])
        # Atualizar centros
        new_centers = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else centers[i] for i in range(n_clusters)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
        
    variances = []
    for i in range(n_clusters):
        points = X[labels == i]
        # Variancia = soma das distancias quadraticas / numero de pontos
        var = np.mean(np.sum((points - centers[i])**2, axis=1))
        variances.append(var)
        
    return centers, variances

# Definir funcao de ativacao RBF
def gaussian_rbf(x: np.ndarray, c: np.ndarray, var: float) -> float:
    return float(np.exp(-np.sum((x - c)**2) / (2 * var)))

# Treinamento da camada de saida (Regra Delta)
def train_rbf_output(
    H_train: np.ndarray, 
    d_train: np.ndarray, 
    eta: float = 0.01, 
    epsilon: float = 1e-7, 
    seed: int = 42
) -> Tuple[np.ndarray, int]:
    np.random.seed(seed)
    W = np.random.rand(H_train.shape[1]) # [w0, w1, w2]
    epoch = 0
    prev_mse = float('inf')
    
    while True:
        for i in range(len(H_train)):
            h = H_train[i]
            d = d_train[i]
            
            y = np.dot(W, h)
            error = d - y
            
            W = W + eta * error * h
            
        y_all = np.dot(H_train, W)
        mse = np.mean((d_train - y_all)**2)
        
        if abs(prev_mse - mse) <= epsilon:
            break
            
        prev_mse = mse
        epoch += 1
        
    return W, epoch

# Fluxo de treinamento principal
centers, variances = run_kmeans(X_rad, 2, seed=42)
print("Centros dos clusters:", centers)
print("Variancias:", variances)

# Preparar dados para o treinamento da camada de saida
H_train = []
for x in train_data[:, :2]:
    h1 = gaussian_rbf(x, centers[0], variances[0])
    h2 = gaussian_rbf(x, centers[1], variances[1])
    H_train.append([1.0, h1, h2]) # bias = 1.0
H_train = np.array(H_train)
d_train = train_data[:, 2]

print("\nIniciando treinamento da camada de saida...")
W, epoch = train_rbf_output(H_train, d_train, eta=0.01, epsilon=1e-7, seed=42)
print(f"Convergiu na epoca {epoch}")
print("Pesos finais:", W)

# Validacao
test_data_raw = [
    [0.8705, 0.9329, -1],
    [0.0388, 0.2703, 1],
    [0.8236, 0.4458, -1],
    [0.7075, 0.1502, 1],
    [0.9587, 0.8663, -1],
    [0.6115, 0.9365, -1],
    [0.3534, 0.3646, 1],
    [0.3268, 0.2766, 1],
    [0.6129, 0.4518, -1],
    [0.9948, 0.4962, -1]
]
test_data = np.array(test_data_raw)

H_test = []
for x in test_data[:, :2]:
    h1 = gaussian_rbf(x, centers[0], variances[0])
    h2 = gaussian_rbf(x, centers[1], variances[1])
    H_test.append([1.0, h1, h2])
H_test = np.array(H_test)

y_test_linear = np.dot(H_test, W)
y_test_pos = np.sign(y_test_linear)
# Tratar 0 como 1 se houver
y_test_pos[y_test_pos == 0] = 1

print("\nResultados do Teste:")
correct = 0
for i in range(len(test_data)):
    print(f"Amostra {i+1}: x1={test_data[i,0]:.4f}, x2={test_data[i,1]:.4f}, d={test_data[i,2]}, y={y_test_linear[i]:.4f}, y_pos={y_test_pos[i]}")
    if y_test_pos[i] == test_data[i, 2]:
        correct += 1
        
print(f"Taxa de acerto: {correct/len(test_data)*100:.2f}%")
