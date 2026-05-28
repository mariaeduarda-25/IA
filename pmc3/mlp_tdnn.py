import numpy as np
import matplotlib.pyplot as plt

# Dados de treinamento
train_data = [
    0.1701, 0.1023, 0.4405, 0.3609, 0.7192, 0.2258, 0.3175, 0.0127, 0.4290, 0.0544,
    0.8000, 0.0450, 0.4268, 0.0112, 0.3218, 0.2185, 0.7240, 0.3516, 0.4420, 0.0984,
    0.1747, 0.3964, 0.5114, 0.6183, 0.3330, 0.2398, 0.0508, 0.4497, 0.2178, 0.7762,
    0.1078, 0.3773, 0.0001, 0.3877, 0.0821, 0.7836, 0.1887, 0.4483, 0.0424, 0.2539,
    0.3164, 0.6386, 0.4862, 0.4068, 0.1611, 0.1101, 0.4372, 0.3795, 0.7092, 0.2400,
    0.3087, 0.0159, 0.4330, 0.0733, 0.7995, 0.0262, 0.4223, 0.0085, 0.3303, 0.2037,
    0.7332, 0.3328, 0.4445, 0.0909, 0.1838, 0.3888, 0.5277, 0.6042, 0.3435, 0.2304,
    0.0568, 0.4500, 0.2371, 0.7705, 0.1246, 0.3701, 0.0006, 0.3943, 0.0646, 0.7878,
    0.1694, 0.4468, 0.0372, 0.2632, 0.3048, 0.6516, 0.4690, 0.4132, 0.1523, 0.1182,
    0.4334, 0.3978, 0.6987, 0.2538, 0.2998, 0.0195, 0.4366, 0.0924, 0.7984, 0.0077
]

# Dados de teste
test_data = [
    0.4173, 0.0062, 0.3387, 0.1886, 0.7418, 0.3138, 0.4466, 0.0835, 0.1930, 0.3807,
    0.5438, 0.5897, 0.3536, 0.2210, 0.0631, 0.4499, 0.2564, 0.7642, 0.1411, 0.3626
]

# Combina para usar no TDNN na fase de teste
all_data = train_data + test_data

from typing import Tuple, List, Optional

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)

def train_mlp(
    X: np.ndarray, 
    Y: np.ndarray, 
    p: int, 
    N1: int, 
    eta: float = 0.1, 
    alpha: float = 0.8, 
    epsilon: float = 0.5e-6, 
    max_epochs: int = 100000, 
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, List[float]]:
    """
    Treina uma rede neural do tipo TDNN (MLP com atrasos) usando backpropagation com momentum.
    
    Parâmetros:
        X: Dados de entrada (amostras x lags).
        Y: Valores desejados de saída (amostras x 1).
        p: Quantidade de atrasos (lags).
        N1: Neurônios na camada oculta.
        eta: Taxa de aprendizado.
        alpha: Fator de momentum.
        epsilon: Critério de parada do erro.
        max_epochs: Limite de épocas de treinamento.
        seed: Semente aleatória opcional.
        
    Retorna:
        W1, b1, W2, b2: Pesos e biases.
        epochs: Número total de épocas executadas.
        mse_history: Histórico de EQM por época.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1 input layer with p neurons
    # 1 hidden layer with N1 neurons
    # 1 output layer with 1 neuron
    
    W1 = np.random.rand(p, N1)
    b1 = np.random.rand(1, N1)
    W2 = np.random.rand(N1, 1)
    b2 = np.random.rand(1, 1)
    
    dW1 = np.zeros_like(W1)
    db1 = np.zeros_like(b1)
    dW2 = np.zeros_like(W2)
    db2 = np.zeros_like(b2)
    
    mse_history = []
    
    for epoch in range(max_epochs):
        # Forward pass
        hidden_input = np.dot(X, W1) + b1
        hidden_output = sigmoid(hidden_input)
        
        final_input = np.dot(hidden_output, W2) + b2
        final_output = sigmoid(final_input)
        
        # Error
        error = Y - final_output
        mse = np.mean(error ** 2)
        mse_history.append(mse)
        
        if mse <= epsilon:
            break
            
        # Backward pass
        d_output = error * sigmoid_derivative(final_output)
        d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)
        
        # Calculate gradients
        grad_W2 = hidden_output.T.dot(d_output) / X.shape[0]
        grad_b2 = np.sum(d_output, axis=0, keepdims=True) / X.shape[0]
        grad_W1 = X.T.dot(d_hidden) / X.shape[0]
        grad_b1 = np.sum(d_hidden, axis=0, keepdims=True) / X.shape[0]
        
        # Update weights with momentum
        dW2 = eta * grad_W2 + alpha * dW2
        db2 = eta * grad_b2 + alpha * db2
        dW1 = eta * grad_W1 + alpha * dW1
        db1 = eta * grad_b1 + alpha * db1
        
        W2 += dW2
        b2 += db2
        W1 += dW1
        b1 += db1
        
    return W1, b1, W2, b2, epoch + 1, mse_history

def predict(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    return sigmoid(final_input)

def create_dataset(data: List[float], p: int) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    for i in range(p, len(data)):
        # x(t-1) ... x(t-p)
        x = data[i-p:i][::-1] # reverse so x_1 is t-1
        X.append(x)
        Y.append([data[i]])
    return np.array(X), np.array(Y)

networks = [
    {"name": "Rede 1", "p": 5, "N1": 10},
    {"name": "Rede 2", "p": 10, "N1": 15},
    {"name": "Rede 3", "p": 15, "N1": 25}
]

results = {}

for net in networks:
    print(f"Treinando {net['name']}...")
    X_train, Y_train = create_dataset(train_data, net["p"])
    
    net_results = []
    for i in range(3):
        seed = 42 + i + net["p"]*10
        print(f"  Treinamento {i+1}...")
        W1, b1, W2, b2, epochs, mse_hist = train_mlp(X_train, Y_train, net["p"], net["N1"], eta=0.1, alpha=0.8, epsilon=0.5e-6, seed=seed)
        net_results.append({
            "W1": W1, "b1": b1, "W2": W2, "b2": b2,
            "epochs": epochs,
            "mse": mse_hist[-1],
            "mse_hist": mse_hist
        })
    results[net['name']] = net_results

# Prepare validation outputs
print("\n--- Validação ---")
# Validation sets start at t=101. 
# For t=101, inputs are t=100, t=99... t=101-p.
validation_predictions = {}
for net in networks:
    p = net["p"]
    X_val = []
    for t in range(100, 120): # indices in all_data: 100 is t=101
        x = all_data[t-p:t][::-1]
        X_val.append(x)
    X_val = np.array(X_val)
    
    val_res = []
    for i in range(3):
        model = results[net['name']][i]
        preds = predict(X_val, model["W1"], model["b1"], model["W2"], model["b2"])
        val_res.append(preds.flatten())
    validation_predictions[net['name']] = val_res

# Calculate Relative Error and Variance
validation_stats = {}
for net in networks:
    validation_stats[net['name']] = []
    for i in range(3):
        preds = validation_predictions[net['name']][i]
        actual = np.array(test_data)
        rel_errors = np.abs(actual - preds) / actual
        mean_rel_error = np.mean(rel_errors)
        variance = np.var(rel_errors)
        validation_stats[net['name']].append({
            "mean_rel_error": mean_rel_error,
            "variance": variance
        })

# Imprimir tabelas para relatorio
print("EQM e Épocas")
for net in networks:
    print(net['name'])
    for i in range(3):
        print(f"  T{i+1}: EQM={results[net['name']][i]['mse']:.6f}, Epocas={results[net['name']][i]['epochs']}")

print("\nValidação T1, T2, T3")
for t in range(20):
    row = f"t={101+t} | {test_data[t]:.4f} | "
    for net in networks:
        for i in range(3):
            row += f"{validation_predictions[net['name']][i][t]:.4f} | "
    print(row)

print("\nErros")
for net in networks:
    print(net['name'])
    for i in range(3):
        print(f"  T{i+1}: ERM={validation_stats[net['name']][i]['mean_rel_error']*100:.4f}%, Var={validation_stats[net['name']][i]['variance']*10000:.4f} (10^-4)")

# Plotting EQM
plt.figure(figsize=(15, 5))
for i, net in enumerate(networks):
    best_idx = np.argmin([res["mse"] for res in results[net['name']]])
    best_hist = results[net['name']][best_idx]["mse_hist"]
    
    plt.subplot(1, 3, i+1)
    plt.plot(best_hist, color='blue')
    plt.title(f"{net['name']} - T{best_idx+1} (Melhor)")
    plt.xlabel("Épocas")
    plt.ylabel("EQM")
    plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_eqm_pmc3.png")

# Plotting Estimations
plt.figure(figsize=(15, 5))
t_axis = range(101, 121)
for i, net in enumerate(networks):
    best_idx = np.argmin([res["mse"] for res in results[net['name']]])
    preds = validation_predictions[net['name']][best_idx]
    
    plt.subplot(1, 3, i+1)
    plt.plot(t_axis, test_data, label='Desejado', marker='o', color='green')
    plt.plot(t_axis, preds, label=f'Estimado (T{best_idx+1})', marker='x', color='red')
    plt.title(f"{net['name']} - T{best_idx+1} (Melhor)")
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_estimativas_pmc3.png")

print("\nMelhor Rede:")
best_nets = []
for net in networks:
    best_idx = np.argmin([stats["mean_rel_error"] for stats in validation_stats[net['name']]])
    best_nets.append((net['name'], best_idx, validation_stats[net['name']][best_idx]["mean_rel_error"]))
best_nets.sort(key=lambda x: x[2])
print(f"A melhor topologia em termos de ERM no teste é a {best_nets[0][0]}, Treinamento T{best_nets[0][1]+1}")
