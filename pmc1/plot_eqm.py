import numpy as np
import csv
import matplotlib.pyplot as plt
from mlp_energia import train_mlp
import time

def main():
    print("Iniciando treinamentos para gerar os gráficos...")
    
    # Carregar dados
    X_list = []
    D_list = []
    
    with open('dataset_pmc1.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Pular cabeçalho
        for row in reader:
            X_list.append([float(row[1]), float(row[2]), float(row[3])])
            D_list.append([float(row[4])])
            
    X = np.array(X_list)
    D = np.array(D_list)
    
    eta = 0.1
    epsilon = 1e-6
    max_epochs = 100000
    
    # Rodar 2 vezes
    histories = []
    
    for i in range(2):
        np.random.seed(int(time.time()) + i)
        print(f"Executando Treinamento T{i+1}...")
        _, _, _, _, epoch, mse, duration, mse_history = train_mlp(X, D, num_hidden=10, eta=eta, epsilon=epsilon, max_epochs=max_epochs)
        histories.append(mse_history)
        print(f" Treinamento T{i+1} concluído. MSE={mse:.6f}, {epoch} épocas.")

    # Plotar
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(histories[0], color='blue')
    ax1.set_title("Treinamento T1: Evolução do Erro Quadrático Médio (EQM)")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("EQM")
    ax1.grid(True)
    
    ax2.plot(histories[1], color='red')
    ax2.set_title("Treinamento T2: Evolução do Erro Quadrático Médio (EQM)")
    ax2.set_xlabel("Épocas")
    ax2.set_ylabel("EQM")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("graficos_eqm.png")
    print("Gráficos salvos em graficos_eqm.png")

if __name__ == "__main__":
    main()
