# Relatório: Classificação de Conservantes para Bebidas com PMC (Perceptron Multicamadas)

## Questão 1
*Execute o treinamento da rede Perceptron através do algoritmo de aprendizagem backpropagation padrão, inicializando as matrizes de pesos com valores aleatórios entre 0 e 1. Utilize a função de ativação logística (sigmoid) para todos os neurônios, taxa de aprendizado $\eta = 0.1$ e precisão $\epsilon = 10^{-6}$.*

**Resposta:**
O treinamento foi implementado e executado utilizando o script Python \mlp_classificacao.py\ feito do zero com a biblioteca umpy\. O dataset \dataset_pmc2.csv\ contendo as 130 amostras de ensaios laboratoriais foi carregado.

A arquitetura estabelecida seguiu rigorosamente a topologia apresentada:
- **Camada de Entrada:** 4 neurônios (, x_2, x_3, x_4$).
- **Camada Oculta:** 15 neurônios.
- **Camada de Saída:** 3 neurônios (, y_2, y_3$), correspondendo ao formato *One-Hot Encoding* para as três classes de conservantes (A, B e C).

Os parâmetros utilizados foram:
- Função de ativação: Logística (Sigmoide) em todas as camadas.
- Taxa de aprendizado ($\eta$): .1$.
- Precisão de erro quadrático médio alvo ($\epsilon$): ^{-6}$.
- Teto de épocas máximo adotado por segurança: 100.000.
- Pesos inicializados de forma puramente aleatória entre 0 e 1.

**Resultados obtidos:**
A rede iniciou o treinamento e executou o *Backpropagation* no formato *Batch*. Devido à rigidez extremamente alta do critério de parada por precisão ($\epsilon = 10^{-6}$) para o algoritmo gradiente descendente padrão, a rede não atingiu esse valor microscópico, convergindo e estacionando até atingir o limite estipulado de 100.000 épocas. 
O **Erro Quadrático Médio Final (MSE)** obtido após as 100.000 épocas foi de **0.01742552**, o que indica uma boa estabilização da rede na tarefa de classificação de padrões. Os pesos finais (W1, b1, W2, b2) foram salvos com sucesso para uso nas próximas etapas.

---

## Questão 2
*Execute o treinamento da rede Perceptron através do algoritmo de aprendizagem backpropagation com momentum, utilizando as mesmas matrizes de pesos iniciais que foram usadas no item anterior. Utilize a função de ativação logística (sigmoid) para todos os neurônios, taxa de aprendizado $\eta = 0.1$, fator de momentum $\alpha = 0.9$ e precisão $\epsilon = 10^{-6}$. Para os dois treinamentos realizados acima, trace os respectivos gráficos dos valores de erro quadrático médio (EQM) em função de cada época de treinamento. Imprima os dois gráficos numa mesma folha de modo não superpostos. Meça também o tempo de processamento envolvido com cada treinamento.*

**Resposta:**
Atualizamos o algoritmo inserindo o termo de Momentum na atualização de pesos para acelerar a convergência ($\Delta W(t) = \eta \cdot \delta \cdot y + \alpha \cdot \Delta W(t-1)$). Como exigido, os pesos iniciais foram exatamente os mesmos do passo anterior (garantido através da mesma semente no gerador pseudoaleatório \seed=42\). O script \mlp_classificacao.py\ rodou os dois treinamentos de forma consecutiva medindo o tempo de processamento.

**Comparativo de Desempenho e Tempos de Processamento:**

| Modelo | Fator Momentum ($\alpha$) | Tempo de Processamento | Épocas Alcançadas | MSE Final Obtido |
|--------|---------------------------|------------------------|-------------------|------------------|
| Padrão | .0$ | **5.70s** | 100000 | 0.017426 |
| Com Momentum | .9$ | **5.80s** | 100000 | 0.015366 |

**Análise:**
A adoção do Momentum aumentou de forma notável a performance da minimização da função custo. Embora ambas as redes tenham batido o teto das 100.000 épocas (pois o alvo de ^{-6}$ é severo), o modelo com Momentum (linha vermelha) desceu mais rapidamente pelo gradiente em épocas iniciais e estabilizou num Mínimo de Erro mais profundo que o padrão (de ~0.017 para ~0.015). Como o cálculo do termo de momentum envolve mais operações matemáticas matriciais iterativas por época, o *tempo de processamento* dele é ligeiramente superior ao do padrão, mas compensa amplamente pela agressividade benéfica na descida da rampa do erro.

**Gráficos Lado a Lado:**
Abaixo constam as curvas de aprendizagem na mesma imagem sem superposição:

![Gráficos EQM Padrão vs Momentum](file:///c:/Users/maria/LabIA/IA/pmc2/graficos_pmc2.png)

---

## Questão 3
*Dado que o problema se configura como um típico processo de classificação de padrões, implemente a rotina que faz o pós-processamento das saídas fornecidas pela rede (números reais) para números inteiros. Utilize o critério do arredondamento simétrico: utilizado apenas no pós-processamento do conjunto de teste.*

**Resposta:**
Para transformar as previsões contínuas da função sigmoide (valores entre 0 e 1) em classes discretas no formato *One-Hot Encoding* (0 ou 1), implementamos uma função de pós-processamento utilizando a biblioteca umpy\. Essa função aplica o limiar (threshold) de 0.5 para realizar o arredondamento simétrico: se o valor for maior ou igual a 0.5, torna-se 1; caso contrário, torna-se 0.

O código implementado em Python para essa rotina é:
\\python
import numpy as np

def pos_processamento(y_continuo):
    # Aplica a regra: 1 se y >= 0.5, senão 0
    y_discreto = np.where(y_continuo >= 0.5, 1, 0)
    return y_discreto
\
Esta rotina foi salva no script e está pronta para ser aplicada aos dados do conjunto de teste (validação) quando ele for passado na rede com os pesos já treinados.

---

## Questão 4
*Faça a validação da rede aplicando o conjunto de teste fornecido na tabela abaixo. Forneça a taxa de acerto (%) entre os valores desejados e os valores fornecidos pela rede (após o pós-processamento) em relação a todos os padrões de teste.*

**Resposta:**
Aplicamos o modelo com Momentum sobre o conjunto de teste de 18 ensaios e passamos as saídas contínuas na rotina de pós-processamento (arredondamento em 0.5). O resultado final da previsão para cada tipo de conservante foi confrontado com o valor desejado (alvo).

| Amostra | $ | $ | $ | $ | $ | $ | $ | ^{pós}$ | ^{pós}$ | ^{pós}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.8622 | 0.7101 | 0.6236 | 0.7894 | 0 | 0 | 1 | **0** | **0** | **1** |
| 2 | 0.2741 | 0.1552 | 0.1333 | 0.1516 | 1 | 0 | 0 | **1** | **0** | **0** |
| 3 | 0.6772 | 0.8516 | 0.6543 | 0.7573 | 0 | 0 | 1 | **0** | **0** | **1** |
| 4 | 0.2178 | 0.5039 | 0.6415 | 0.5039 | 0 | 1 | 0 | **0** | **1** | **0** |
| 5 | 0.7260 | 0.7500 | 0.7007 | 0.4953 | 0 | 0 | 1 | **0** | **0** | **1** |
| 6 | 0.2473 | 0.2941 | 0.4248 | 0.3087 | 1 | 0 | 0 | **1** | **0** | **0** |
| 7 | 0.5682 | 0.5683 | 0.5054 | 0.4426 | 0 | 1 | 0 | **0** | **1** | **0** |
| 8 | 0.6566 | 0.6715 | 0.4952 | 0.3951 | 0 | 1 | 0 | **0** | **1** | **0** |
| 9 | 0.0705 | 0.4717 | 0.2921 | 0.2954 | 1 | 0 | 0 | **1** | **0** | **0** |
| 10 | 0.1187 | 0.2568 | 0.3140 | 0.3037 | 1 | 0 | 0 | **1** | **0** | **0** |
| 11 | 0.5673 | 0.7011 | 0.4083 | 0.5552 | 0 | 1 | 0 | **0** | **1** | **0** |
| 12 | 0.3164 | 0.2251 | 0.3526 | 0.2560 | 1 | 0 | 0 | **1** | **0** | **0** |
| 13 | 0.7884 | 0.9568 | 0.6825 | 0.6398 | 0 | 0 | 1 | **0** | **0** | **1** |
| 14 | 0.9633 | 0.7850 | 0.6777 | 0.6059 | 0 | 0 | 1 | **0** | **0** | **1** |
| 15 | 0.7739 | 0.8505 | 0.7934 | 0.6626 | 0 | 0 | 1 | **0** | **0** | **1** |
| 16 | 0.4219 | 0.4136 | 0.1408 | 0.0940 | 1 | 0 | 0 | **1** | **0** | **0** |
| 17 | 0.6616 | 0.4365 | 0.6597 | 0.8129 | 0 | 0 | 1 | **0** | **0** | **1** |
| 18 | 0.7325 | 0.4761 | 0.3888 | 0.5683 | 0 | 1 | 0 | **0** | **1** | **0** |

**Taxa de Acerto Final:**
Dos 18 ensaios aplicados no teste, a rede classificou perfeitamente **18** deles. 
Isso resulta em uma excelente **Taxa de Acerto Global de 100.00%**.
