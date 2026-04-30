# Resultados da Atividade - Perceptron com Regra de Hebb

### 2. Registre os resultados dos 5 treinamentos na tabela abaixo:

Ao executar o algoritmo do Perceptron para a classificação dos óleos 5 vezes, os pesos foram inicializados com valores aleatórios entre 0 e 1 em cada execução. A taxa de aprendizagem utilizada foi $\eta = 0.01$. *(O item 1 da atividade também está demonstrado aqui)*.

| Treinamento | Inicial $w_0$ | Inicial $w_1$ | Inicial $w_2$ | Inicial $w_3$ | Final $w_0$ | Final $w_1$ | Final $w_2$ | Final $w_3$ | Épocas |
|---|---|---|---|---|---|---|---|---|---|
| 1º (T1) | 0.8532 | 0.6667 | 0.3064 | 0.0699 | 3.1132 | 1.5977 | 2.5156 | -0.7430 | 388 |
| 2º (T2) | 0.9620 | 0.2268 | 0.2479 | 0.1899 | 2.9020 | 1.4017 | 2.4056 | -0.6951 | 320 |
| 3º (T3) | 0.5515 | 0.2929 | 0.4303 | 0.6034 | 3.0515 | 1.5573 | 2.4743 | -0.7302 | 387 |
| 4º (T4) | 0.4885 | 0.7580 | 0.7713 | 0.7649 | 3.0685 | 1.5292 | 2.4634 | -0.7295 | 364 |
| 5º (T5) | 0.5979 | 0.3370 | 0.6002 | 0.0047 | 3.0979 | 1.5713 | 2.5034 | -0.7403 | 380 |

*(Obs: Os valores de pesos finais podem variar levemente de acordo com a semente aleatória gerada, os dados acima são de uma das execuções do código em Python)*

---

### 3. Após o treinamento do perceptron aplique o mesmo na classificação automática das seguintes amostras de óleo, indicando na tabela abaixo os resultados das saídas (Classes) referentes aos cinco processos de treinamento realizados no item 1:

Abaixo, a tabela preenchida com a classificação das 10 novas amostras utilizando os vetores de pesos finais obtidos nos 5 treinamentos:

| Amostra | $x_1$ | $x_2$ | $x_3$ | $y$ (T1) | $y$ (T2) | $y$ (T3) | $y$ (T4) | $y$ (T5) |
|---|---|---|---|---|---|---|---|---|
| 1 | -0.3565 | 0.0620 | 5.9891 | -1 | -1 | -1 | -1 | -1 |
| 2 | -0.7842 | 1.1267 | 5.5912 | 1 | 1 | 1 | 1 | 1 |
| 3 | 0.3012 | 0.5611 | 5.8234 | 1 | 1 | 1 | 1 | 1 |
| 4 | 0.7757 | 1.0648 | 8.0677 | 1 | 1 | 1 | 1 | 1 |
| 5 | 0.1570 | 0.8028 | 6.3040 | 1 | 1 | 1 | 1 | 1 |
| 6 | -0.7014 | 1.0316 | 3.6005 | 1 | 1 | 1 | 1 | 1 |
| 7 | 0.3748 | 0.1536 | 6.1537 | -1 | -1 | -1 | -1 | -1 |
| 8 | -0.6920 | 0.9404 | 4.4058 | 1 | 1 | 1 | 1 | 1 |
| 9 | -1.3970 | 0.7141 | 4.9263 | -1 | -1 | -1 | -1 | -1 |
| 10 | -1.8842 | -0.2805 | 1.2548 | -1 | -1 | -1 | -1 | -1 |

---

### 4. Explique por que o número de épocas de treinamento varia a cada vez que executamos o treinamento do perceptron.

**Resposta:** 
O número de épocas varia a cada execução porque os pesos iniciais das conexões sinápticas ($w_0, w_1, w_2, w_3$) são gerados de forma aleatória no início de cada treinamento. Como o algoritmo parte de pontos diferentes no "espaço de pesos" a cada vez, o caminho percorrido e a quantidade de ajustes iterativos necessários (épocas) para encontrar a reta/hiperplano que separa perfeitamente as duas classes (C1 e C2) também serão diferentes.

---

### 5. Qual a principal limitação do perceptron quando aplicado em problemas de classificação de padrões?

**Resposta:**
A principal limitação do Perceptron de camada única é que ele só consegue classificar corretamente padrões em problemas que sejam **linearmente separáveis**. Isso significa que ele só funciona se for possível traçar uma reta (ou um hiperplano, em problemas com mais dimensões) que divida e separe perfeitamente as classes de dados. Se os dados possuírem uma distribuição não linear (como no clássico problema lógico XOR), o algoritmo do Perceptron não conseguirá convergir e nunca encontrará uma solução capaz de separar as classes sem erros.
