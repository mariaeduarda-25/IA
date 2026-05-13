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
| 1º (T1)     | 0.00145594            | 100000           |
| 2º (T2)     | 0.00145546            | 100000           |
| 3º (T3)     | 0.00144852            | 100000           |
| 4º (T4)     | 0.00144426            | 100000           |
| 5º (T5)     | 0.00142899            | 100000           |

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

| Amostra | $ | $ | $ | $ | {rede}$ (T1) | {rede}$ (T2) | {rede}$ (T3) | {rede}$ (T4) | {rede}$ (T5) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0611 | 0.2860 | 0.7464 | 0.4831 | 0.4849 | 0.4887 | 0.4873 | 0.4875 | 0.4821 |
| 2 | 0.5102 | 0.7464 | 0.0860 | 0.5965 | 0.5924 | 0.5899 | 0.5940 | 0.5963 | 0.5927 |
| 3 | 0.0004 | 0.6916 | 0.5006 | 0.5318 | 0.5262 | 0.5256 | 0.5298 | 0.5320 | 0.5280 |
| 4 | 0.9430 | 0.4476 | 0.2648 | 0.6843 | 0.7110 | 0.7124 | 0.7124 | 0.7118 | 0.7108 |
| 5 | 0.1399 | 0.1610 | 0.2477 | 0.2872 | 0.2898 | 0.2961 | 0.2848 | 0.2818 | 0.2935 |
| 6 | 0.6423 | 0.3229 | 0.8567 | 0.7663 | 0.7588 | 0.7583 | 0.7587 | 0.7620 | 0.7621 |
| 7 | 0.6492 | 0.0007 | 0.6422 | 0.5666 | 0.5693 | 0.5682 | 0.5704 | 0.5699 | 0.5649 |
| 8 | 0.1818 | 0.5078 | 0.9046 | 0.6601 | 0.6818 | 0.6801 | 0.6836 | 0.6843 | 0.6812 |
| 9 | 0.7382 | 0.2647 | 0.1916 | 0.5427 | 0.5324 | 0.5284 | 0.5346 | 0.5340 | 0.5303 |
| 10 | 0.3879 | 0.1307 | 0.8656 | 0.5836 | 0.6032 | 0.6034 | 0.6047 | 0.6069 | 0.6007 |
| 11 | 0.1903 | 0.6523 | 0.7820 | 0.6950 | 0.6938 | 0.6926 | 0.6953 | 0.6948 | 0.6924 |
| 12 | 0.8401 | 0.4490 | 0.2719 | 0.6790 | 0.6772 | 0.6773 | 0.6793 | 0.6793 | 0.6765 |
| 13 | 0.0029 | 0.3264 | 0.2476 | 0.2956 | 0.2980 | 0.3041 | 0.2944 | 0.2950 | 0.3056 |
| 14 | 0.7088 | 0.9342 | 0.2763 | 0.7742 | 0.7901 | 0.7942 | 0.7876 | 0.7850 | 0.7877 |
| 15 | 0.1283 | 0.1882 | 0.7253 | 0.4662 | 0.4648 | 0.4688 | 0.4662 | 0.4652 | 0.4606 |
| 16 | 0.8882 | 0.3077 | 0.8931 | 0.8093 | 0.8283 | 0.8288 | 0.8263 | 0.8295 | 0.8338 |
| 17 | 0.2225 | 0.9182 | 0.7820 | 0.7581 | 0.7875 | 0.7866 | 0.7849 | 0.7799 | 0.7842 |
| 18 | 0.1957 | 0.8423 | 0.3085 | 0.5826 | 0.5912 | 0.5893 | 0.5930 | 0.5947 | 0.5914 |
| 19 | 0.9991 | 0.5914 | 0.3933 | 0.7938 | 0.8074 | 0.8113 | 0.8062 | 0.8056 | 0.8087 |
| 20 | 0.2299 | 0.1524 | 0.7353 | 0.5012 | 0.4957 | 0.4982 | 0.4973 | 0.4966 | 0.4909 |
| **Erro Relativo Médio (%)** | - | - | - | - | **1.5505%** | **1.8858%** | **1.4454%** | **1.4512%** | **1.7948%** |
| **Variância (%)** | - | - | - | - | **1.4216%** | **1.4893%** | **1.6314%** | **1.7459%** | **1.4046%** |

---

## Questão 6
*Baseado nas análises da tabela acima indique qual das configurações finais de treinamento {T1, T2, T3, T4 ou T5} seria a mais adequada para o sistema de ressonância magnética, ou seja, qual delas está oferecendo a melhor generalização.*

**Resposta:**
Baseado nos resultados do conjunto de teste (dados não vistos durante o treinamento), a configuração final mais adequada é a **T3**.

**Justificativa:**
A capacidade de generalização de um modelo é medida pelo seu desempenho em dados inéditos. Olhando para a tabela:
1. **Erro Relativo Médio:** O treinamento **T3** apresentou a menor taxa média de erro relativo global (**1.4454%**), seguido de perto pelo T4 (1.4512%). Isso significa que, na média, as estimativas do T3 são as que mais se aproximam da energia absorvida real.
2. **Variância:** A variância do erro para o **T3** (**1.6314%**) também é menor que a do seu concorrente mais próximo, o T4 (1.7459%). Isso indica que os erros do T3 oscilam menos, oferecendo previsões mais consistentes e estáveis ao longo de todo o espectro de novas amostras.

Portanto, por apresentar o **menor erro médio** combinado com uma **boa estabilidade (menor variância dentre os melhores)**, o modelo **T3** é o que melhor generalizou a função matemática não-linear subjacente ao sistema de ressonância magnética.
