# Relatório: Estimativa de Energia Absorvida com PMC (Perceptron Multicamadas)

## Questão 1
*Execute 5 treinamentos para a rede PERCEPTRON inicializando as matrizes de pesos em cada treinamento com valores aleatórios...*

**Resposta:**
O treinamento foi executado através do script Python \mlp_energia.py\ desenvolvido do zero usando a biblioteca umpy\. O dataset \dataset_pmc1.csv\ foi gerado a partir dos dados brutos com as 200 amostras (onde $ é a energia absorvida $).
A rede neural tem a topologia **3-10-1** e utilizou a função logística (sigmoide) em todas as camadas. Os parâmetros aplicados foram: taxa de aprendizado $\eta = 0.1$, precisão $\epsilon = 10^{-6}$ e número máximo de épocas fixado em 100.000. Foram executados 5 treinamentos independentes inicializando os pesos de forma aleatória entre 0 e 1 (usando seeds diferentes no gerador).

---

## Questão 2
*Registre os resultados finais desses 5 treinamentos na tabela.*

**Resposta:**
Os 5 treinamentos convergiram parando no critério do teto máximo estipulado de épocas, já que a precisão de $\epsilon = 10^{-6}$ é extremamente rigorosa.

| Treinamento | Erro Quadrático Médio | Número de Épocas |
| ----------- | --------------------- | ---------------- |
| 1º (T1)     | 0.00017215            | 100000           |
| 2º (T2)     | 0.00025296            | 100000           |
| 3º (T3)     | 0.00019851            | 100000           |
| 4º (T4)     | 0.00051204            | 100000           |
| 5º (T5)     | 0.00017441            | 100000           |

---

## Questão 3
*Baseado na tabela do item 2, explique de forma detalhada por que tanto o erro quadrático médio quanto o número de épocas variam de treinamento para treinamento. (No caso o número de épocas ficou igual).*

**Resposta:**
**1. Sobre o número de épocas estar igual (100.000):**
Como o valor de precisão $\epsilon = 10^{-6}$ é uma tolerância extremamente pequena, a rede atingiu o limite de segurança de épocas (100.000) antes de bater o alvo do erro. Ela convergiu para um mínimo na casa de .0014$, esgotando o tempo limite em todas as rodadas.

**2. Sobre a variação do Erro Quadrático Médio (e das épocas, se não houvesse teto):**
Isso ocorre devido à **inicialização aleatória dos pesos sinápticos**. A superfície de erro do Perceptron Multicamadas (MLP) é não-convexa, cheia de vales e mínimos locais. Como a rede começa em um ponto inicial aleatório diferente a cada treinamento, a trajetória ladeira abaixo tomada pelo algoritmo de retropropagação (Gradiente Descendente) segue caminhos distintos. Por seguir caminhos diferentes, a rede acaba "estacionando" em mínimos locais distintos (ex: .00145$ vs .00142$). Sem o limite de épocas, um caminho poderia ser mais rápido que outro, causando variação também no número de épocas.

---

## Questão 4
*Para os dois treinamentos acima com maiores números de épocas, trace os respectivos gráficos dos valores de erro quadrático médio (EQM).*

**Resposta:**
Como todos atingiram 100.000 épocas, escolhemos os treinamentos **T1** e **T2**. Os gráficos estão na mesma figura em *subplots* não superpostos.

![Evolução do Erro Quadrático Médio - T1 e T2](file:///c:/Users/maria/LabIA/IA/pmc1/graficos_eqm.png)

---

## Questão 5
*Para todos os treinamentos efetuados no item 2, faça a validação da rede aplicando o conjunto de teste fornecido...*

**Resposta:**
Aplicamos os pesos dos modelos T1 a T5 às 20 amostras do conjunto de teste não visto e obtivemos os valores de rede e as taxas de erro:

| Amostra | $x_1$ | $x_2$ | $x_3$ | $d$ | $y_{rede}$ (T1) | $y_{rede}$ (T2) | $y_{rede}$ (T3) | $y_{rede}$ (T4) | $y_{rede}$ (T5) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0611 | 0.2860 | 0.7464 | 0.4831 | 0.4606 | 0.4673 | 0.4602 | 0.4635 | 0.4789 |
| 2 | 0.5102 | 0.7464 | 0.0860 | 0.5965 | 0.5863 | 0.5886 | 0.5944 | 0.5840 | 0.5921 |
| 3 | 0.0004 | 0.6916 | 0.5006 | 0.5318 | 0.5192 | 0.5206 | 0.5264 | 0.5140 | 0.5130 |
| 4 | 0.9430 | 0.4476 | 0.2648 | 0.6843 | 0.7040 | 0.7100 | 0.6976 | 0.7038 | 0.6953 |
| 5 | 0.1399 | 0.1610 | 0.2477 | 0.2872 | 0.3024 | 0.2974 | 0.2981 | 0.2946 | 0.2989 |
| 6 | 0.6423 | 0.3229 | 0.8567 | 0.7663 | 0.7663 | 0.7634 | 0.7675 | 0.7713 | 0.7715 |
| 7 | 0.6492 | 0.0007 | 0.6422 | 0.5666 | 0.5545 | 0.5532 | 0.5640 | 0.5407 | 0.5485 |
| 8 | 0.1818 | 0.5078 | 0.9046 | 0.6601 | 0.6657 | 0.6618 | 0.6641 | 0.6603 | 0.6678 |
| 9 | 0.7382 | 0.2647 | 0.1916 | 0.5427 | 0.5381 | 0.5371 | 0.5354 | 0.5402 | 0.5310 |
| 10 | 0.3879 | 0.1307 | 0.8656 | 0.5836 | 0.5789 | 0.5775 | 0.5778 | 0.5890 | 0.5726 |
| 11 | 0.1903 | 0.6523 | 0.7820 | 0.6950 | 0.6974 | 0.7014 | 0.6990 | 0.6981 | 0.6990 |
| 12 | 0.8401 | 0.4490 | 0.2719 | 0.6790 | 0.6895 | 0.6914 | 0.6871 | 0.6901 | 0.6837 |
| 13 | 0.0029 | 0.3264 | 0.2476 | 0.2956 | 0.3045 | 0.2986 | 0.3030 | 0.3041 | 0.3031 |
| 14 | 0.7088 | 0.9342 | 0.2763 | 0.7742 | 0.7872 | 0.7808 | 0.7867 | 0.7861 | 0.7847 |
| 15 | 0.1283 | 0.1882 | 0.7253 | 0.4662 | 0.4482 | 0.4536 | 0.4433 | 0.4552 | 0.4570 |
| 16 | 0.8882 | 0.3077 | 0.8931 | 0.8093 | 0.8186 | 0.8162 | 0.8136 | 0.8159 | 0.8201 |
| 17 | 0.2225 | 0.9182 | 0.7820 | 0.7581 | 0.7716 | 0.7786 | 0.7711 | 0.7730 | 0.7739 |
| 18 | 0.1957 | 0.8423 | 0.3085 | 0.5826 | 0.5839 | 0.5793 | 0.5808 | 0.5840 | 0.5813 |
| 19 | 0.9991 | 0.5914 | 0.3933 | 0.7938 | 0.8107 | 0.8186 | 0.8079 | 0.8100 | 0.8099 |
| 20 | 0.2299 | 0.1524 | 0.7353 | 0.5012 | 0.4888 | 0.4916 | 0.4839 | 0.4964 | 0.4884 |
| **Erro Relativo Médio (%)** | - | - | - | - | **1.9903%** | **1.7760%** | **1.7005%** | **1.8220%** | **1.7652%** |
| **Variância (%)** | - | - | - | - | **1.9393%** | **1.1624%** | **2.0293%** | **1.5770%** | **1.0415%** |

---

## Questão 6
*Baseado nas análises da tabela acima indique qual das configurações finais de treinamento {T1, T2, T3, T4 ou T5} seria a mais adequada para o sistema de ressonância magnética, ou seja, qual delas está oferecendo a melhor generalização.*

**Resposta:**
Baseado nos resultados do conjunto de teste (dados não vistos durante o treinamento), as configurações finais mais adequadas são a **T3** ou a **T5**.

**Justificativa:**
A capacidade de generalização de um modelo é medida pelo seu desempenho em dados inéditos:
1. **Erro Relativo Médio:** O treinamento **T3** apresentou a menor taxa média de erro relativo global (**1.7005%**), seguido de perto pelo T5 (**1.7652%**). Ambos mostram excelente aproximação em média.
2. **Variância:** A variância do erro para o **T5** (**1.0415%**) é significativamente menor que a de todas as outras execuções (incluindo o T3 que obteve **2.0293%**). Isso indica que, embora o T3 possua um erro médio ligeiramente inferior, o modelo **T5** é o mais estável e uniforme, com menor oscilação nas suas previsões sobre dados novos.

Portanto, dependendo se o foco do sistema é a minimização estrita do erro médio (indicando **T3**) ou a maior consistência das previsões (indicando **T5**), ambas as topologias representam a excelência da generalização da rede.
