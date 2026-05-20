import numpy as np
import csv
import random
import time
from typing import Tuple, List

def sigmoid(x: np.ndarray) -> np.ndarray:
    # Usando np.clip para evitar overflow
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    sx = sigmoid(x)
    return sx * (1.0 - sx)

def train_mlp(
    X: np.ndarray, 
    D: np.ndarray, 
    num_hidden: int = 10, 
    eta: float = 0.1, 
    alpha: float = 0.0, 
    epsilon: float = 1e-6, 
    max_epochs: int = 100000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float, List[float]]:
    """
    Treina um Perceptron Multicamadas (MLP) com suporte a momentum.
    
    Parâmetros:
        X: Matriz NumPy de entradas (amostras x características).
        D: Matriz NumPy de saídas desejadas (amostras x saídas).
        num_hidden: Número de neurônios na camada oculta.
        eta: Taxa de aprendizado.
        alpha: Fator de momentum.
        epsilon: Critério de parada do erro quadrático médio (EQM).
        max_epochs: Número máximo de épocas permitidas.
        
    Retorna:
        W1, b1, W2, b2: Pesos e biases das camadas oculta e de saída.
        epoch: Número de épocas executadas.
        MSE: Erro quadrático médio final.
        tempo: Tempo total de processamento em segundos.
        mse_history: Histórico do EQM por época.
    """
    num_samples, num_inputs = X.shape
    num_outputs = D.shape[1]
    
    # Inicializando pesos com valores aleatórios entre 0 e 1
    # Pesos da camada de entrada para a camada oculta
    W1 = np.random.rand(num_inputs, num_hidden)
    # Bias da camada oculta
    b1 = np.random.rand(1, num_hidden)
    
    # Pesos da camada oculta para a camada de saída
    W2 = np.random.rand(num_hidden, num_outputs)
    # Bias da camada de saída
    b2 = np.random.rand(1, num_outputs)
    
    # For momentum
    vW1 = np.zeros_like(W1)
    vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2)
    vb2 = np.zeros_like(b2)
    
    epoch = 0
    mse_history = []
    
    start_time = time.time()
    
    while epoch < max_epochs:
        # Forward pass (Batch)
        # Camada Oculta
        V1 = np.dot(X, W1) + b1
        Y1 = sigmoid(V1)
        
        # Camada de Saída
        V2 = np.dot(Y1, W2) + b2
        Y2 = sigmoid(V2)
        
        # Cálculo do Erro
        Error = D - Y2
        MSE = np.mean(Error ** 2)
        mse_history.append(MSE)
        
        # Critério de Parada
        if MSE < epsilon:
            break
            
        # Backward pass (Backpropagation)
        # Delta da camada de saída
        delta_output = Error * Y2 * (1.0 - Y2)
        
        # Delta da camada oculta
        delta_hidden = np.dot(delta_output, W2.T) * Y1 * (1.0 - Y1)
        
        # Gradients
        gradW2 = np.dot(Y1.T, delta_output) / num_samples
        gradb2 = np.sum(delta_output, axis=0, keepdims=True) / num_samples
        gradW1 = np.dot(X.T, delta_hidden) / num_samples
        gradb1 = np.sum(delta_hidden, axis=0, keepdims=True) / num_samples
        
        # Atualização com Momentum
        vW2 = eta * gradW2 + alpha * vW2
        vb2 = eta * gradb2 + alpha * vb2
        vW1 = eta * gradW1 + alpha * vW1
        vb1 = eta * gradb1 + alpha * vb1
        
        W2 += vW2
        b2 += vb2
        W1 += vW1
        b1 += vb1
        
        epoch += 1
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return W1, b1, W2, b2, epoch, MSE, elapsed_time, mse_history

def main():
    print("===================================================================")
    print(" Treinamento de MLP para Estimativa de Energia Absorvida (PMC1)")
    print("===================================================================")
    print("Parâmetros:")
    print("- Topologia: 3 entradas, 10 ocultos, 1 saída (3-10-1)")
    print("- Ativação: Logística (sigmoide) em todas as camadas")
    print("- Taxa de aprendizado (eta): 0.1")
    print("- Precisão (epsilon): 1e-6")
    print("-------------------------------------------------------------------\n")

    # Carregar dados usando csv em vez de pandas
    X_list = []
    D_list = []
    
    with open('dataset_pmc1.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Pular cabeçalho
        for row in reader:
            # x1, x2, x3 estão nos índices 1, 2, 3
            X_list.append([float(row[1]), float(row[2]), float(row[3])])
            # d está no índice 4
            D_list.append([float(row[4])])
            
    X = np.array(X_list)
    D = np.array(D_list)
    
    num_trainings = 5
    eta = 0.1
    epsilon = 1e-6
    max_epochs = 100000
    
    results = []
    
    for i in range(num_trainings):
        # Reiniciar a seed para garantir pesos iniciais diferentes mas reprodutíveis se necessário
        # Não setar semente ou setar com base no tempo garante que será aleatório
        np.random.seed(int(time.time()) + i)
        
        print(f"Iniciando Treinamento {i+1}/{num_trainings}...")
        
        W1, b1, W2, b2, epoch, mse, duration, mse_history = train_mlp(X, D, num_hidden=10, eta=eta, epsilon=epsilon, max_epochs=max_epochs)
        
        results.append({
            'treinamento': i + 1,
            'epocas': epoch,
            'mse_final': mse,
            'tempo_segundos': duration,
            'mse_history': mse_history
        })
        
        print(f" > Concluído em {epoch} épocas | MSE Final: {mse:.8f} | Tempo: {duration:.2f} s\n")

    print("===================================================================")
    print(" Resumo dos 5 Treinamentos")
    print("===================================================================")
    for r in results:
        print(f"Treinamento {r['treinamento']}: {r['epocas']} épocas alcançadas, MSE = {r['mse_final']:.8f}")

if __name__ == "__main__":
    main()
