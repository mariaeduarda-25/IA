# Resultados da Atividade - Rede RBF (Identificação de Radiação)

### Questão 1
> *Execute o treinamento da camada escondida através do algoritmo “k-means”. Em se tratando de um problema de classificação de padrões, compute os centros dos dois clusters levando-se em consideração apenas aqueles padrões com presença de radiação. Após o treinamento, forneça os valores das coordenadas do centro de cada cluster e sua respectiva variância.*

**Resposta:**
O algoritmo k-means foi aplicado com $K = 2$ filtrando apenas os padrões do conjunto de treinamento onde existe a presença de radiação ($d = 1$). A variância foi calculada com base na distância quadrática média dos pontos em relação ao centro.

| Cluster | Centro | Variância |
|:---:|:---:|:---:|
| **1** | (0.1648, 0.6121) | 0.0298 |
| **2** | (0.3990, 0.1571) | 0.0385 |

---

### Questão 2
> *Após o treinamento da camada intermediária execute o treinamento da camada de saída usando a regra delta generalizada. Utilize uma taxa de aprendizado η = 0.01 e precisão de ε = 10-7. No final da convergência forneça os valores dos pesos referente ao neurônio da camada de saída.*

**Resposta:**
Aplicando a Regra Delta na camada de saída (com $\eta = 0.01$ e $\epsilon = 10^{-7}$), a rede convergiu na época **328**. Os pesos finais obtidos foram:

| Peso | Valor |
|:---:|:---:|
| **W2 1,0** *(Bias)* | -1.0026 |
| **W2 1,1** | 2.3780 |
| **W2 1,2** | 2.6977 |

---

### Questão 3
> *Dado que o problema se configura como um típico processo de classificação de padrões, implemente a rotina que faz o pós-processamento das saídas fornecidas pela rede (números reais) para números inteiros. Utilize a função sinal, ou seja: função utilizada apenas no pós-processamento do conjunto de teste.*

**Resposta:**
A rotina de pós-processamento foi devidamente implementada no script `rbf.py`. A rede foi programada para aplicar a função sinal ($sign$) nas respostas lineares ($y$) do conjunto de teste, retornando $1$ para valores positivos e $-1$ para valores nulos ou negativos, permitindo a classificação final.

---

### Questão 4
> *Faça a validação da rede aplicando o conjunto de teste fornecido na tabela abaixo. Forneça a taxa de acerto (%) entre os valores desejados e os valores fornecidos pela rede (após o pós-processamento) em relação a todos os padrões de teste.*

**Resposta:**
Abaixo está o preenchimento da tabela aplicando a rede treinada nas amostras de teste:

| Amostra | x1 | x2 | d | y | ypós |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.8705 | 0.9329 | -1 | -1.0025 | -1 |
| **2** | 0.0388 | 0.2703 | 1 | -0.3231 | -1 |
| **3** | 0.8236 | 0.4458 | -1 | -0.9140 | -1 |
| **4** | 0.7075 | 0.1502 | 1 | -0.2201 | -1 |
| **5** | 0.9587 | 0.8663 | -1 | -1.0026 | -1 |
| **6** | 0.6115 | 0.9365 | -1 | -0.9878 | -1 |
| **7** | 0.3534 | 0.3646 | 1 | 0.9665 | 1 |
| **8** | 0.3268 | 0.2766 | 1 | 1.3232 | 1 |
| **9** | 0.6129 | 0.4518 | -1 | -0.4682 | -1 |
| **10** | 0.9948 | 0.4962 | -1 | -0.9966 | -1 |

**Taxa de Acerto (%):** 80,00%

---

### Questão 5
> *Se for o caso, explique quais estratégias poderemos adotar para tentar aumentar a taxa de acerto desta RBF.*

**Resposta:**
A rede não obteve 100% de acerto (errou as amostras 2 e 4 do teste). Algumas estratégias que podem ser adotadas para tentar aumentar essa taxa de acerto incluem:

1.  **Aumentar o Número de Centros (Clusters):** Aumentar o número de neurônios da camada escondida (ex: 4 ou 6 clusters em vez de apenas 2) daria maior grau de liberdade para a rede mapear fronteiras de decisão mais complexas no espaço dimensional.
2.  **Usar Todo o Conjunto para Computar Centros:** Na atividade os centros foram computados levando em consideração apenas as amostras da classe 1 (presença de radiação). Em RBFs, normalmente os centros das funções gaussianas são distribuídos por todo o espaço de entrada, considerando todas as classes, para cobrir melhor o domínio de dados.
3.  **Ajuste Fino do Fator de Dispersão ($\sigma^2$):** A forma que as variâncias foram calculadas pode ser alterada. Testar uma variância uniforme para todos os neurônios de base ou heurísticas (como $d_{max}/\sqrt{2K}$) pode suavizar as curvas de ativação, melhorando a capacidade de generalização da rede.
4.  **Aumentar a Base de Dados de Treinamento:** 40 amostras podem ser insuficientes dependendo da complexidade das fronteiras de radiação. Mais exemplos podem ajudar a rede a generalizar melhor e evitar sobreajuste (overfitting).
