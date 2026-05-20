import random
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from typing import List, Tuple

def treinar_adaline(
    X: List[List[float]], 
    d: List[int], 
    eta: float, 
    epsilon: float, 
    max_epocas: int
) -> Tuple[List[float], List[float], int, List[float]]:
    """
    Treina uma rede neural ADALINE (sem uso de bibliotecas externas como numpy).
    
    Parâmetros:
    X : list de lists
        Matriz de entradas (amostras x características).
    d : list
        Vetor de saídas desejadas (-1 para Válvula A, +1 para Válvula B).
    eta : float
        Taxa de aprendizado.
    epsilon : float
        Precisão para o critério de parada.
    max_epocas : int
        Número máximo de épocas permitidas.
        
    Retorna:
    w_inicial : list
        Vetor de pesos iniciais.
    w_final : list
        Vetor de pesos finais.
    epocas : int
        Número de épocas até a convergência.
    historico_eqm : list
        Histórico do Erro Quadrático Médio em cada época.
    """
    num_amostras = len(X)
    num_caracteristicas = len(X[0])
    
    # Adicionando a entrada do bias (x0 = -1) na matriz X.
    # Pode-se usar x0 = 1 ou x0 = -1, utilizando -1 conforme adotado frequentemente na literatura.
    X_bias = [[-1] + amostra for amostra in X]
    
    # Inicializando o vetor de pesos (w) com valores aleatórios entre 0 e 1.
    w = [random.uniform(0, 1) for _ in range(num_caracteristicas + 1)]
    w_inicial = list(w) # Salva uma cópia dos pesos iniciais
    
    eqm_anterior = float('inf')
    epocas = 0
    historico_eqm = []
    
    while epocas < max_epocas:
        # Atualização estocástica dos pesos (LMS - Least Mean Squares)
        for i in range(num_amostras):
            x_i = X_bias[i]
            d_i = d[i]
            
            # Cálculo da saída linear do neurônio (ativação) u = w * x
            u = sum(w[j] * x_i[j] for j in range(len(w)))
            
            # Atualização dos pesos (Regra Delta)
            erro = d_i - u
            for j in range(len(w)):
                w[j] = w[j] + eta * erro * x_i[j]
        
        # Após atualizar com todas as amostras na época, calcula-se o EQM da época
        soma_erros_quadrados = 0
        for i in range(num_amostras):
            u_i = sum(w[j] * X_bias[i][j] for j in range(len(w)))
            soma_erros_quadrados += (d[i] - u_i) ** 2
            
        eqm_atual = soma_erros_quadrados / num_amostras
        historico_eqm.append(eqm_atual)
        
        # Critério de parada: a variação do Erro Quadrático Médio é menor que a precisão
        if abs(eqm_atual - eqm_anterior) < epsilon:
            break
            
        eqm_anterior = eqm_atual
        epocas += 1
        
    return w_inicial, w, epocas, historico_eqm


if __name__ == "__main__":
    # ---------------------------------------------------------
    # DADOS DE TREINAMENTO (ANEXO)
    # ---------------------------------------------------------
    X = [
        [0.4329, -1.3719, 0.7022, -0.8535],
        [0.3024, 0.2286, 0.8630, 2.7909],
        [0.1349, -0.6445, 1.0530, 0.5687],
        [0.3374, -1.7163, 0.3670, -0.6283],
        [1.1434, -0.0485, 0.6637, 1.2606],
        [1.3749, -0.5071, 0.4464, 1.3009],
        [0.7221, -0.7587, 0.7681, -0.5592],
        [0.4403, -0.8072, 0.5154, -0.3129],
        [-0.5231, 0.3548, 0.2538, 1.5776],
        [0.3255, -2.0000, 0.7112, -1.1209],
        [0.5824, 1.3915, -0.2291, 4.1735],
        [0.1340, 0.6081, 0.4450, 3.2230],
        [0.1480, -0.2988, 0.4778, 0.8649],
        [0.7359, 0.1869, -0.0872, 2.3584],
        [0.7115, -1.1469, 0.3394, 0.9573],
        [0.8251, -1.2840, 0.8452, 1.2382],
        [0.1569, 0.3712, 0.8825, 1.7633],
        [0.0033, 0.6835, 0.5389, 2.8249],
        [0.4243, 0.8313, 0.2634, 3.5855],
        [1.0490, 0.1326, 0.9138, 1.9792],
        [1.4276, 0.5331, -0.0145, 3.7286],
        [0.5971, 1.4865, 0.2904, 4.6069],
        [0.8475, 2.1479, 0.3179, 5.8235],
        [1.3967, -0.4171, 0.6443, 1.3927],
        [0.0044, 1.5378, 0.6099, 4.7755],
        [0.2201, -0.5668, 0.0515, 0.7829],
        [0.6300, -1.2480, 0.8591, 0.8093],
        [-0.2479, 0.8960, 0.0547, 1.7381],
        [-0.3088, -0.0929, 0.8659, 1.5483],
        [-0.5180, 1.4974, 0.5453, 2.3993],
        [0.6833, 0.8266, 0.0829, 2.8864],
        [0.4353, -1.4066, 0.4207, -0.4879],
        [-0.1069, -3.2329, 0.1856, -2.4572],
        [0.4662, 0.6261, 0.7304, 3.4370],
        [0.8298, -1.4089, 0.3119, 1.3235]
    ]
    d = [1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1]
    
    # ---------------------------------------------------------
    # PARÂMETROS DA REDE
    # ---------------------------------------------------------
    taxa_aprendizado = 0.0025
    precisao = 1e-6
    max_epocas = 10000
    
    print("Iniciando os 5 treinamentos da rede ADALINE...")
    print(f"Taxa de aprendizado: {taxa_aprendizado} | Precisão: {precisao}\n")
    
    # Questão 1 e 2: Executar 5 treinamentos e preencher a tabela
    resultados = []
    
    for i in range(1, 6):
        w_ini, w_fin, num_epocas, historico = treinar_adaline(X, d, taxa_aprendizado, precisao, max_epocas)
        resultados.append((i, w_ini, w_fin, num_epocas, historico))
        
    # Imprimindo a tabela formatada (Markdown)
    print("\nResultados dos 5 Treinamentos:")
    print("| Treinamento | Inicial w0 | Inicial w1 | Inicial w2 | Inicial w3 | Inicial w4 | Final w0 | Final w1 | Final w2 | Final w3 | Final w4 | Épocas |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for res in resultados:
        t, w_i, w_f, ep, hist = res
        print(f"| {t}º (T{t}) | {w_i[0]:.4f} | {w_i[1]:.4f} | {w_i[2]:.4f} | {w_i[3]:.4f} | {w_i[4]:.4f} | {w_f[0]:.4f} | {w_f[1]:.4f} | {w_f[2]:.4f} | {w_f[3]:.4f} | {w_f[4]:.4f} | {ep} |")
        
    # ---------------------------------------------------------
    # CLASSIFICAÇÃO DAS AMOSTRAS DE TESTE
    # ---------------------------------------------------------
    amostras_teste = [
        [0.9694, 0.6909, 0.4334, 3.4965],
        [0.5427, 1.3832, 0.6390, 4.0352],
        [0.6081, -0.9196, 0.5925, 0.1016],
        [-0.1618, 0.4694, 0.2030, 3.0117],
        [0.1870, -0.2578, 0.6124, 1.7749],
        [0.4891, -0.5276, 0.4378, 0.6439],
        [0.3777, 2.0149, 0.7423, 3.3932],
        [1.1498, -0.4067, 0.2469, 1.5866],
        [0.9325, 1.0950, 1.0359, 3.3591],
        [0.5060, 1.3317, 0.9222, 3.7174],
        [0.0497, -2.0656, 0.6124, -0.6585],
        [0.4004, 3.5369, 0.9766, 5.3532],
        [-0.1874, 1.3343, 0.5374, 3.2189],
        [0.5060, 1.3317, 0.9222, 3.7174],
        [1.6375, -0.7911, 0.7537, 0.5515]
    ]

    print("\nClassificação das Novas Amostras de Teste:")
    print("| Amostra | x1 | x2 | x3 | x4 | y (T1) | y (T2) | y (T3) | y (T4) | y (T5) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    
    for idx, amostra in enumerate(amostras_teste):
        linha = f"| {idx+1} | {amostra[0]:.4f} | {amostra[1]:.4f} | {amostra[2]:.4f} | {amostra[3]:.4f} "
        
        # Testando a amostra para cada um dos 5 treinamentos
        # O ADALINE usa um bias x0 = -1 no nosso código
        x_in = [-1] + amostra
        
        for res in resultados:
            w_fin = res[2] # Vetor de pesos finais
            
            # Cálculo de u = w * x
            u = sum(w_fin[j] * x_in[j] for j in range(len(w_fin)))
            
            # Ativação degrau bipolar/sinal
            y = 1 if u >= 0 else -1
            linha += f"| {y} "
            
        linha += "|"
        print(linha)
        
    # ---------------------------------------------------------
    # GRÁFICO DO EQM PARA T1 E T2
    # ---------------------------------------------------------
    if HAS_MATPLOTLIB:
        print("\nGerando o gráfico do EQM para os Treinamentos 1 e 2...")
        hist_t1 = resultados[0][4]
        hist_t2 = resultados[1][4]
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(hist_t1) + 1), hist_t1, label='Treinamento 1 (T1)', color='blue')
        plt.plot(range(1, len(hist_t2) + 1), hist_t2, label='Treinamento 2 (T2)', color='red')
        
        plt.title('Erro Quadrático Médio (EQM) vs Épocas de Treinamento')
        plt.xlabel('Épocas')
        plt.ylabel('Erro Quadrático Médio (EQM)')
        plt.legend()
        plt.grid(True)
        # Salva a imagem localmente (útil em alguns terminais) e também tenta exibir
        plt.savefig('grafico_eqm.png')
        print("Gráfico salvo como 'grafico_eqm.png'. A janela com o gráfico abrirá agora (feche-a para finalizar o script).")
        plt.show()
    else:
        print("\n[Aviso] A biblioteca 'matplotlib' não está instalada.")
        print("Para visualizar o gráfico do EQM (T1 e T2), instale com o comando: pip install matplotlib")

