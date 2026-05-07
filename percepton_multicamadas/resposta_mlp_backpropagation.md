# Algoritmo Backpropagation - Perceptron Multicamadas com Conexão Direta

Esta topologia apresenta uma particularidade interessante: além das conexões tradicionais (Entrada $\rightarrow$ Oculta e Oculta $\rightarrow$ Saída), existe uma **conexão direta** da camada de entrada para a camada de saída, representada pela matriz $W3$. 

Abaixo está a demonstração detalhada da sequência matemática e do ajuste de pesos utilizando o algoritmo **Backpropagation** (Retropropagação do Erro).

---

## 1. Definição das Variáveis e Dimensões

*   **$X_p$**: Vetor de entrada do padrão $p$ (dimensão $N \times 1$).
*   **$d_p$**: Saída desejada (alvo) para o padrão $p$ (escalar, pois há apenas 1 neurônio na saída).
*   **$W1$**: Matriz de pesos entre a Camada de Entrada e a Camada Oculta (dimensão $N_1 \times N$).
*   **$W2$**: Matriz de pesos (vetor linha) entre a Camada Oculta e a Camada de Saída (dimensão $1 \times N_1$).
*   **$W3$**: Matriz de pesos (vetor linha) da conexão direta entre a Camada de Entrada e a Camada de Saída (dimensão $1 \times N$).
*   **$f(\cdot)$**: Função de ativação (ex: Sigmoide ou Tangente Hiperbólica).
*   **$f'(\cdot)$**: Derivada da função de ativação.
*   **$\eta$**: Taxa de aprendizagem.

*(Nota: Na prática, adiciona-se o bias $x_0 = 1$ ou $-1$ às camadas, o que adicionaria uma coluna às matrizes de pesos. Para focar na topologia pedida, a demonstração foca nos pesos sinápticos listados).*

---

## 2. Passo 1: Fase Forward (Propagação do Sinal)

Apresenta-se um padrão de treinamento $X_p$ à rede e calcula-se a saída propagando o sinal camada por camada, da esquerda para a direita.

### A) Processamento na Camada Oculta (2ª Camada)
Calcula-se o campo local induzido ($Net_H$) e o sinal de saída ($Y_H$) dos $N_1$ neurônios da camada oculta:

$$Net_H = W1 \cdot X_p$$
$$Y_H = f(Net_H)$$

### B) Processamento na Camada de Saída (3ª Camada)
O único neurônio de saída recebe estímulos de **duas** fontes: da camada oculta (via $W2$) e diretamente da camada de entrada (via $W3$). O campo local induzido ($Net_O$) será a soma desses estímulos:

$$Net_O = (W2 \cdot Y_H) + (W3 \cdot X_p)$$
$$Y_O = f(Net_O)$$
*(Onde $Y_O$ é a saída final predita pela rede para o padrão $p$)*.

---

## 3. Passo 2: Fase Backward (Retropropagação do Erro)

Calculamos o erro na saída e o retropropagamos da direita para a esquerda para encontrar os Gradientes Locais ($\delta$).

### A) Erro e Gradiente Local na Camada de Saída
O erro absoluto é a diferença entre a saída desejada e a predita:
$$e = d_p - Y_O$$

O gradiente local do neurônio de saída ($\delta_O$) é o erro multiplicado pela derivada da função de ativação calculada em $Net_O$:
$$\delta_O = e \cdot f'(Net_O)$$

### B) Gradiente Local na Camada Oculta
Os neurônios da camada oculta não têm um erro $e$ direto (pois não sabemos qual deveria ser a saída exata da camada oculta). O erro é "jogado para trás" a partir da camada de saída através dos pesos $W2$.

O vetor de gradientes locais da camada oculta ($\delta_H$) é:
$$\delta_H = f'(Net_H) \circ (W2^T \cdot \delta_O)$$
*(O símbolo $\circ$ representa a multiplicação elemento a elemento, também conhecida como Produto de Hadamard).*

---

## 4. Passo 3: Ajuste das Matrizes de Pesos (Regra Delta Generalizada)

Com os gradientes locais calculados, atualizamos todas as matrizes de pesos. A regra geral é: $W_{novo} = W_{atual} + \eta \cdot \delta \cdot (Sinal\_de\_Entrada)^T$.

### A) Ajuste da Matriz W2 (Oculta $\rightarrow$ Saída)
A entrada para os pesos $W2$ é a saída da camada oculta ($Y_H$).
$$\Delta W2 = \eta \cdot \delta_O \cdot Y_H^T$$
$$W2_{novo} = W2_{atual} + \Delta W2$$

### B) Ajuste da Matriz W3 (Entrada $\rightarrow$ Saída)
A entrada para os pesos $W3$ é o vetor original do padrão ($X_p$). Como esta é uma conexão direta com o neurônio de saída, usamos o gradiente da saída ($\delta_O$).
$$\Delta W3 = \eta \cdot \delta_O \cdot X_p^T$$
$$W3_{novo} = W3_{atual} + \Delta W3$$

### C) Ajuste da Matriz W1 (Entrada $\rightarrow$ Oculta)
A entrada para os pesos $W1$ é o vetor original do padrão ($X_p$). O gradiente utilizado é o da camada oculta ($\delta_H$).
$$\Delta W1 = \eta \cdot \delta_H \cdot X_p^T$$
$$W1_{novo} = W1_{atual} + \Delta W1$$

---

## 5. Iteração e Condição de Parada
1. Repete-se o processo (Fase Forward $\rightarrow$ Fase Backward $\rightarrow$ Ajuste de Pesos) para cada um dos **$P$ padrões** do Conjunto de Treinamento. Uma passagem completa pelos $P$ padrões constitui **1 Época**.
2. Calcula-se o Erro Quadrático Médio (EQM) de todos os $P$ padrões.
3. O algoritmo continua iterando épocas sucessivas até que o EQM atinja um valor menor que a precisão estipulada ($\epsilon$) ou alcance o limite máximo de épocas definido.
