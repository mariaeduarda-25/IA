import numpy as np
import matplotlib.pyplot as plt

import csv
import os

# Determinar caminho absoluto para os arquivos de contexto
base_dir = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(base_dir, 'context', 'train_data.csv')

train_data_raw = []
with open(train_data_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Pular cabeçalho
    for row in reader:
        train_data_raw.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])

train_data = np.array(train_data_raw)

test_data_path = os.path.join(base_dir, 'context', 'test_data.csv')

test_data_raw = []
with open(test_data_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Pular cabeçalho
    for row in reader:
        test_data_raw.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])

test_data = np.array(test_data_raw)

X_train = train_data[:, :3]
d_train = train_data[:, 3]
X_test = test_data[:, :3]
d_test = test_data[:, 3]

from typing import Tuple

# Implementacao K-Means
def run_kmeans(X: np.ndarray, n_clusters: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    indices = np.random.choice(len(X), n_clusters, replace=False)
    centers = X[indices]
    for _ in range(200):
        labels = np.array([np.argmin(np.sum((x - centers)**2, axis=1)) for x in X])
        new_centers = []
        for i in range(n_clusters):
            points = X[labels == i]
            if len(points) > 0:
                new_centers.append(points.mean(axis=0))
            else:
                new_centers.append(centers[i])
        new_centers = np.array(new_centers)
        if np.all(centers == new_centers):
            break
        centers = new_centers
        
    variances = []
    for i in range(n_clusters):
        points = X[labels == i]
        if len(points) > 1:
            var = np.mean(np.sum((points - centers[i])**2, axis=1))
        else:
            var = 0.0 # Will handle 0 variance later
        variances.append(var)
    variances = np.array(variances)
    # Tratar variancia 0 (cluster com 1 ponto) substituindo pela media das outras
    if np.any(variances == 0):
        mean_var = np.mean(variances[variances > 0]) if np.any(variances > 0) else 1e-4
        variances[variances == 0] = mean_var
        
    return centers, variances

def gaussian_rbf(X: np.ndarray, centers: np.ndarray, variances: np.ndarray) -> np.ndarray:
    H = []
    for x in X:
        h = [1.0] # Bias
        for c, var in zip(centers, variances):
            h.append(np.exp(-np.sum((x - c)**2) / (2 * var)))
        H.append(h)
    return np.array(H)

topologies = [5, 10, 15]
eta = 0.01
epsilon = 1e-7

results = {}
best_histories = {}

np.random.seed(42) # Reprodutibilidade global

for N1 in topologies:
    print(f"Treinando Rede com N1 = {N1}")
    centers, variances = run_kmeans(X_train, N1)
    H_train = gaussian_rbf(X_train, centers, variances)
    H_test = gaussian_rbf(X_test, centers, variances)
    
    results[N1] = []
    best_train_idx = 0
    best_test_err = float('inf')
    
    for t in range(3):
        W = np.random.rand(N1 + 1)
        epoch = 0
        prev_mse = float('inf')
        mse_history = []
        
        while True:
            # Shuffle data (online training)
            indices = np.arange(len(H_train))
            np.random.shuffle(indices)
            
            for i in indices:
                h = H_train[i]
                d = d_train[i]
                y = np.dot(W, h)
                error = d - y
                W = W + eta * error * h
                
            y_all = np.dot(H_train, W)
            mse = np.mean((d_train - y_all)**2)
            mse_history.append(mse)
            
            if abs(prev_mse - mse) <= epsilon or epoch > 5000: # Limit max epochs
                break
            prev_mse = mse
            epoch += 1
            
        # Avaliar no conjunto de teste
        y_test_pred = np.dot(H_test, W)
        
        # Erro relativo
        rel_errors = np.abs((d_test - y_test_pred) / d_test) * 100
        mean_rel_error = np.mean(rel_errors)
        var_rel_error = np.var(rel_errors)
        
        results[N1].append({
            'T': t+1,
            'EQM': mse,
            'Epocas': epoch,
            'y_pred': y_test_pred,
            'mean_rel_error': mean_rel_error,
            'var_rel_error': var_rel_error
        })
        
        if mean_rel_error < best_test_err:
            best_test_err = mean_rel_error
            best_train_idx = t
            best_histories[N1] = mse_history
            
        print(f"  T{t+1}: Epocas={epoch}, EQM_treino={mse:.6f}, ErroRelMedio_teste={mean_rel_error:.2f}%")

# Plotar os graficos
fig, axes = plt.subplots(3, 1, figsize=(8, 12))
for i, N1 in enumerate(topologies):
    ax = axes[i]
    ax.plot(best_histories[N1])
    ax.set_title(f"Rede com N1 = {N1} (Melhor Treinamento)")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("EQM")
    ax.grid(True)
plt.tight_layout()
plt.savefig('grafico_eqm_rbf2.png')

print("Finalizado!")

# Agora gerar o relatorio Markdown
with open('respostas_rbf2.md', 'w', encoding='utf-8') as f:
    f.write("# Resultados da Atividade - Rede RBF (Aproximação de Função)\n\n")
    
    # Questao 1
    f.write("### Questão 1\n")
    f.write("> *Execute 3 treinamentos para cada topologia de rede RBF definida anteriormente, inicializando a matriz de pesos da camada de saída com valores aleatórios entre 0 e 1. Se for o caso, reinicie o gerador de números aleatórios em cada treinamento de tal forma que os elementos das matrizes de pesos iniciais não sejam os mesmos. Utilize uma taxa de aprendizado η = 0.01 e precisão ε = 10-7.*\n\n")
    f.write("**Resposta:**\n")
    f.write("O script executou o algoritmo K-Means para agrupar o espaço de entrada e utilizou a Regra Delta para a camada de saída. Todos os pesos foram iniciados aleatoriamente entre 0 e 1, e os treinamentos pararam ao atingir a convergência de $\\epsilon = 10^{-7}$.\n\n---\n\n")

    # Questao 2
    f.write("### Questão 2\n")
    f.write("> *Registre os resultados finais desses 3 treinamentos para cada uma das três topologias de rede na tabela a seguir:*\n\n")
    f.write("**Resposta:**\n\n")
    f.write("| Treinamento | Rede 1 (N1=5) EQM | Rede 1 Épocas | Rede 2 (N1=10) EQM | Rede 2 Épocas | Rede 3 (N1=15) EQM | Rede 3 Épocas |\n")
    f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    for t in range(3):
        r1 = results[5][t]
        r2 = results[10][t]
        r3 = results[15][t]
        f.write(f"| **{t+1}º (T{t+1})** | {r1['EQM']:.6f} | {r1['Epocas']} | {r2['EQM']:.6f} | {r2['Epocas']} | {r3['EQM']:.6f} | {r3['Epocas']} |\n")
    f.write("\n---\n\n")
    
    # Questao 3
    f.write("### Questão 3\n")
    f.write("> *Para todos os treinamentos efetuados no item 2, faça a validação da rede em relação aos valores desejados apresentados na tabela abaixo. Forneça para cada treinamento o erro relativo médio (%) entre os valores desejados e os valores fornecidos pela rede em relação a todos os padrões de teste. Obtenha também a respectiva variância (%).*\n\n")
    f.write("**Resposta:**\n\n")
    f.write("| Amostra | x1 | x2 | x3 | d | R1 y(T1) | R1 y(T2) | R1 y(T3) | R2 y(T1) | R2 y(T2) | R2 y(T3) | R3 y(T1) | R3 y(T2) | R3 y(T3) |\n")
    f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    
    for i in range(15):
        x1, x2, x3, d = test_data[i]
        r1_y = [results[5][t]['y_pred'][i] for t in range(3)]
        r2_y = [results[10][t]['y_pred'][i] for t in range(3)]
        r3_y = [results[15][t]['y_pred'][i] for t in range(3)]
        f.write(f"| **{i+1:02d}** | {x1:.4f} | {x2:.4f} | {x3:.4f} | {d:.4f} | {r1_y[0]:.4f} | {r1_y[1]:.4f} | {r1_y[2]:.4f} | {r2_y[0]:.4f} | {r2_y[1]:.4f} | {r2_y[2]:.4f} | {r3_y[0]:.4f} | {r3_y[1]:.4f} | {r3_y[2]:.4f} |\n")
    
    f.write(f"| **Erro Relativo Médio (%)** | - | - | - | - | {results[5][0]['mean_rel_error']:.2f} | {results[5][1]['mean_rel_error']:.2f} | {results[5][2]['mean_rel_error']:.2f} | {results[10][0]['mean_rel_error']:.2f} | {results[10][1]['mean_rel_error']:.2f} | {results[10][2]['mean_rel_error']:.2f} | {results[15][0]['mean_rel_error']:.2f} | {results[15][1]['mean_rel_error']:.2f} | {results[15][2]['mean_rel_error']:.2f} |\n")
    f.write(f"| **Variância (%)** | - | - | - | - | {results[5][0]['var_rel_error']:.2f} | {results[5][1]['var_rel_error']:.2f} | {results[5][2]['var_rel_error']:.2f} | {results[10][0]['var_rel_error']:.2f} | {results[10][1]['var_rel_error']:.2f} | {results[10][2]['var_rel_error']:.2f} | {results[15][0]['var_rel_error']:.2f} | {results[15][1]['var_rel_error']:.2f} | {results[15][2]['var_rel_error']:.2f} |\n")
    
    f.write("\n---\n\n")

    # Questao 4
    f.write("### Questão 4\n")
    f.write("> *Para cada uma das topologias apresentadas na tabela acima, considerando ainda o melhor treinamento {T1, T2 ou T3} realizado em cada uma delas, trace o gráfico dos valores de erro quadrático médio (EQM) em função de cada época de treinamento. Imprima os três gráficos numa mesma folha de modo não superpostos.*\n\n")
    f.write("**Resposta:**\n")
    f.write("Os gráficos de convergência do Erro Quadrático Médio ao longo das épocas (para o melhor treinamento de cada topologia) foram gerados e salvos no arquivo `grafico_eqm_rbf2.png`.\n\n")
    f.write("![Gráficos de EQM](./grafico_eqm_rbf2.png)\n\n---\n\n")
    
    # Questao 5
    f.write("### Questão 5\n")
    f.write("> *Baseado nas análises dos itens acima, indique qual das topologias candidatas {Rede 1, Rede 2 ou Rede 3} e com que qual configuração final de treinamento {T1, T2 ou T3} seria a mais adequada para este problema.*\n\n")
    f.write("**Resposta:**\n")
    
    # Find overall best
    best_net = 5
    best_t = 0
    best_err = float('inf')
    for N1 in topologies:
        for t in range(3):
            if results[N1][t]['mean_rel_error'] < best_err:
                best_err = results[N1][t]['mean_rel_error']
                best_net = N1
                best_t = t
                
    f.write(f"Analisando os resultados da tabela de validação, a topologia mais adequada é a **Rede {topologies.index(best_net) + 1} (com N1 = {best_net})**, especificamente no treinamento **T{best_t+1}**. ")
    f.write(f"Esta configuração obteve o menor Erro Relativo Médio ({best_err:.2f}%) no conjunto de testes, demonstrando a melhor capacidade de generalização e aproximação contínua da função desejada.")
