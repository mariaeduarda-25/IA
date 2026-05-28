# Resultados da Atividade - Perceptron com Regra de Hebb Supervisionada

### 2. Registre os resultados dos 5 treinamentos na tabela abaixo:

Ao executar o algoritmo utilizando a **Regra de Hebb Supervisionada** 5 vezes, os pesos foram inicializados com valores aleatórios entre 0 e 1 em cada execução. A taxa de aprendizagem utilizada foi $\eta = 0.01$. Sob esta regra, a atualização ocorre de forma incondicional para cada padrão da época, completando o treinamento em **exatamente 1 época**.

| Treinamento | Inicial $w_0$ | Inicial $w_1$ | Inicial $w_2$ | Inicial $w_3$ | Final $w_0$ | Final $w_1$ | Final $w_2$ | Final $w_3$ | Épocas |
|---|---|---|---|---|---|---|---|---|---|
| 1º (T1) | 0.9690 | 0.8212 | 0.7806 | 0.5587 | 0.9890 | 0.8153 | 0.8424 | 0.5408 | 1 |
| 2º (T2) | 0.3683 | 0.3972 | 0.2196 | 0.8457 | 0.3883 | 0.3913 | 0.2814 | 0.8278 | 1 |
| 3º (T3) | 0.4479 | 0.6988 | 0.9220 | 0.4202 | 0.4679 | 0.6929 | 0.9839 | 0.4022 | 1 |
| 4º (T4) | 0.7137 | 0.8709 | 0.7618 | 0.8935 | 0.7337 | 0.8650 | 0.8236 | 0.8755 | 1 |
| 5º (T5) | 0.7998 | 0.4076 | 0.2011 | 0.0293 | 0.8198 | 0.4017 | 0.2630 | 0.0114 | 1 |

---

### 3. Após o treinamento do perceptron aplique o mesmo na classificação automática das seguintes amostras de óleo, indicando na tabela abaixo os resultados das saídas (Classes) referentes aos cinco processos de treinamento realizados no item 1:

Abaixo, a tabela preenchida com a classificação das 10 novas amostras utilizando os vetores de pesos finais obtidos nos 5 treinamentos da Regra de Hebb Supervisionada:

| Amostra | $x_1$ | $x_2$ | $x_3$ | $y$ (T1) | $y$ (T2) | $y$ (T3) | $y$ (T4) | $y$ (T5) |
|---|---|---|---|---|---|---|---|---|
| 1 | -0.3565 | 0.0620 | 5.9891 | 1 | 1 | 1 | 1 | 1 |
| 2 | -0.7842 | 1.1267 | 5.5912 | 1 | 1 | 1 | 1 | 1 |
| 3 | 0.3012 | 0.5611 | 5.8234 | 1 | 1 | 1 | 1 | 1 |
| 4 | 0.7757 | 1.0648 | 8.0677 | 1 | 1 | 1 | 1 | 1 |
| 5 | 0.1570 | 0.8028 | 6.3040 | 1 | 1 | 1 | 1 | 1 |
| 6 | -0.7014 | 1.0316 | 3.6005 | 1 | 1 | 1 | 1 | 1 |
| 7 | 0.3748 | 0.1536 | 6.1537 | 1 | 1 | 1 | 1 | 1 |
| 8 | -0.6920 | 0.9404 | 4.4058 | 1 | 1 | 1 | 1 | 1 |
| 9 | -1.3970 | 0.7141 | 4.9263 | 1 | 1 | 1 | 1 | 1 |
| 10 | -1.8842 | -0.2805 | 1.2548 | -1 | 1 | -1 | -1 | 1 |

---

### 4. Explique por que o número de épocas de treinamento varia a cada vez que executamos o treinamento do perceptron.

**Resposta:** 
Na **Regra de Hebb Supervisionada (Associação por Correlação)**, a atualização dos pesos é incondicional ($\Delta w_j = \eta \cdot d_i \cdot x_{ij}$), ou seja, ocorre independentemente da saída real $y$ da rede. Como não há correção de erro baseada no feedback, o algoritmo processa cada amostra apenas uma única vez, completando o treinamento em **exatamente 1 época**, sem qualquer variação.

Contudo, no algoritmo do **Perceptron clássico (Regra do Erro Padrão / Correção de Erros)**, o número de épocas de treinamento varia a cada execução. Isso ocorre porque o ajuste dos pesos só é acionado se a rede cometer um erro de classificação ($\Delta w_j = \eta \cdot (d_i - y_i) \cdot x_{ij}$). Como os pesos iniciais são gerados aleatoriamente em cada treinamento, a rede parte de um ponto diferente no espaço de estados de pesos a cada vez. Portanto, o número de erros iniciais e o trajeto de ajustes iterativos necessários para encontrar a reta/hiperplano divisor de classes (convergência) variam de execução para execução.

---

### 5. Qual a principal limitação do perceptron quando aplicado em problemas de classificação de padrões?

**Resposta:**
A principal limitação do Perceptron de camada única é que ele só consegue classificar corretamente padrões em problemas que sejam **linearmente separáveis**. Isso significa que ele só funciona se for possível traçar uma reta (ou um hiperplano, em problemas com mais dimensões) que divida e separe perfeitamente as classes de dados. Se os dados possuírem uma distribuição não linear (como no clássico problema lógico XOR), o algoritmo do Perceptron não conseguirá convergir e nunca encontrará uma solução capaz de separar as classes sem erros.
No caso específico da Regra de Hebb Supervisionada, há ainda a limitação adicional de que ela não realiza correção de erro iterativa, dependendo de que os vetores de entrada sejam ortogonais ou linearmente independentes para garantir a classificação perfeita de forma determinística em um único passo.
