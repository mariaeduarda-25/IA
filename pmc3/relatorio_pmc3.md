# Relatório: Previsão do Mercado Financeiro com TDNN (Perceptron Multicamadas)

## Questões 1 e 2
*Execute 3 treinamentos para cada rede perceptron acima inicializando as matrizes de pesos em cada treinamento com valores aleatórios entre 0 e 1... Registre os resultados finais desses 3 treinamentos para cada uma das três topologias de rede na tabela a seguir:*

**Resposta:**
O treinamento foi executado através de um script Python utilizando a topologia *Time Delay Neural Network* (TDNN). O dataset foi ajustado para mapear janelas deslizantes de $p$ observações passadas para prever o valor atual $f(t)$.
As três topologias candidatas foram testadas utilizando a função logística (sigmoide) nas camadas, taxa de aprendizado $\eta = 0.1$, fator de momentum $\alpha = 0.8$, precisão de $\epsilon = 0.5 \times 10^{-6}$ e até 100.000 épocas. Os pesos foram inicializados aleatoriamente entre 0 e 1, com sementes diferentes a cada treinamento.

**Resultados do Treinamento:**

| Treinamento | Rede 1 ($p=5, N1=10$) | Rede 2 ($p=10, N1=15$) | Rede 3 ($p=15, N1=25$) |
| ----------- | ----------------------- | ------------------------ | ------------------------ |
|             | **EQM** \| **Épocas** | **EQM** \| **Épocas**  | **EQM** \| **Épocas**  |
| 1º (T1)     | 0.001835 \| 100000      | 0.000897 \| 100000       | 0.494518 \| 100000       |
| 2º (T2)     | 0.001831 \| 100000      | 0.000948 \| 100000       | 0.001101 \| 100000       |
| 3º (T3)     | 0.001791 \| 100000      | 0.001220 \| 100000       | 0.001289 \| 100000       |

*Nota: A Rede 3 (T1) convergiu para um mínimo local ruim (saturação dos pesos), resultando num EQM elevado comparado aos outros treinamentos.*

---

## Questão 3
*Para todos os treinamentos efetuados no item 2, faça a validação da rede em relação aos valores desejados apresentados na tabela abaixo. Forneça para cada treinamento o erro relativo médio e a respectiva variância.*

**Resposta:**

| Amostra | Desejado $f(t)$ | Rede 1 (T1) | Rede 1 (T2) | Rede 1 (T3) | Rede 2 (T1) | Rede 2 (T2) | Rede 2 (T3) | Rede 3 (T1) | Rede 3 (T2) | Rede 3 (T3) |
|---|---|---|---|---|---|---|---|---|---|---|
| t = 101 | 0.4173 | 0.4528 | 0.4554 | 0.4630 | 0.4212 | 0.4202 | 0.4350 | 1.0000 | 0.4364 | 0.4456 |
| t = 102 | 0.0062 | 0.0192 | 0.0191 | 0.0168 | 0.0119 | 0.0066 | 0.0077 | 1.0000 | 0.0085 | 0.0107 |
| t = 103 | 0.3387 | 0.3839 | 0.3814 | 0.3829 | 0.3671 | 0.3787 | 0.3787 | 1.0000 | 0.3858 | 0.3901 |
| t = 104 | 0.1886 | 0.2148 | 0.2482 | 0.2508 | 0.1430 | 0.1371 | 0.1338 | 1.0000 | 0.1302 | 0.1340 |
| t = 105 | 0.7418 | 0.7159 | 0.6908 | 0.7031 | 0.7187 | 0.7314 | 0.7364 | 1.0000 | 0.7650 | 0.7623 |
| t = 106 | 0.3138 | 0.2043 | 0.1958 | 0.1967 | 0.2475 | 0.2484 | 0.2344 | 1.0000 | 0.2443 | 0.2369 |
| t = 107 | 0.4466 | 0.4299 | 0.4172 | 0.4143 | 0.4496 | 0.4524 | 0.4506 | 1.0000 | 0.4442 | 0.4525 |
| t = 108 | 0.0835 | 0.0667 | 0.0843 | 0.0803 | 0.1007 | 0.0939 | 0.1000 | 1.0000 | 0.0947 | 0.0876 |
| t = 109 | 0.1930 | 0.2081 | 0.2085 | 0.2025 | 0.2151 | 0.2130 | 0.2250 | 1.0000 | 0.1978 | 0.1987 |
| t = 110 | 0.3807 | 0.3128 | 0.2965 | 0.2968 | 0.4560 | 0.4541 | 0.4724 | 1.0000 | 0.4311 | 0.4592 |
| t = 111 | 0.5438 | 0.5372 | 0.5614 | 0.5584 | 0.5528 | 0.5426 | 0.5509 | 1.0000 | 0.5214 | 0.5248 |
| t = 112 | 0.5897 | 0.6404 | 0.6277 | 0.6276 | 0.6040 | 0.6051 | 0.6173 | 1.0000 | 0.6142 | 0.6105 |
| t = 113 | 0.3536 | 0.3870 | 0.3769 | 0.3810 | 0.3298 | 0.3273 | 0.3137 | 1.0000 | 0.3338 | 0.3206 |
| t = 114 | 0.2210 | 0.1867 | 0.2093 | 0.2142 | 0.2285 | 0.2372 | 0.2372 | 1.0000 | 0.2370 | 0.2370 |
| t = 115 | 0.0631 | 0.1355 | 0.1256 | 0.1305 | 0.0565 | 0.0474 | 0.0437 | 1.0000 | 0.0566 | 0.0462 |
| t = 116 | 0.4499 | 0.4484 | 0.4381 | 0.4398 | 0.4111 | 0.4123 | 0.3991 | 1.0000 | 0.4260 | 0.4158 |
| t = 117 | 0.2564 | 0.2431 | 0.2424 | 0.2413 | 0.2482 | 0.2556 | 0.2541 | 1.0000 | 0.2623 | 0.2614 |
| t = 118 | 0.7642 | 0.7700 | 0.7800 | 0.7648 | 0.7788 | 0.7683 | 0.7583 | 1.0000 | 0.7468 | 0.7403 |
| t = 119 | 0.1411 | 0.1455 | 0.1418 | 0.1401 | 0.1385 | 0.1440 | 0.1467 | 1.0000 | 0.1468 | 0.1550 |
| t = 120 | 0.3626 | 0.3145 | 0.3249 | 0.3250 | 0.3595 | 0.3492 | 0.3495 | 1.0000 | 0.3477 | 0.3530 |
| **Erro Relativo Médio (%)** | - | **25.25%** | **24.26%** | **22.73%** | **12.21%** | **8.44%** | **11.59%** | **1112.56%** | **9.52%** | **12.77%** |
| **Variância (x $10^{-4}$)** | - | **2360.83** | **2244.47** | **1697.63** | **397.14** | **69.44** | **103.00** | **11838138** | **92.45** | **257.11** |

---

## Questão 4
*Para cada uma das topologias... trace o gráfico dos valores de erro quadrático médio (EQM) em função de cada época de treinamento. Imprima os três gráficos numa mesma folha de modo não superpostos.*

**Resposta:**
Os gráficos representam o melhor treinamento de cada topologia (T3 para Rede 1, T1 para Rede 2, T2 para Rede 3 - de acordo com os menores valores de EQM no treinamento).

![Evolução do Erro Quadrático Médio - Melhores Treinamentos](file:///home/alunos/Desktop/www/LabIA/IA/pmc3/grafico_eqm_pmc3.png)

---

## Questão 5
*Para cada uma das topologias... trace o gráfico dos valores desejados e dos valores estimados pela respectiva rede em função do domínio de estimação considerado (t=101..120). Imprima os três gráficos numa mesma folha de modo não superpostos.*

**Resposta:**

![Valores Desejados vs Estimados - t=101..120](file:///home/alunos/Desktop/www/LabIA/IA/pmc3/grafico_estimativas_pmc3.png)

---

## Questão 6
*Baseado nas análises dos itens acima, indique qual das topologias candidatas {Rede 1, Rede 2 ou Rede 3} e com que qual configuração final de treinamento {T1 , T2 ou T3} seria a mais adequada para realização de previsões neste processo.*

**Resposta:**
A configuração mais adequada é a **Rede 2, com o Treinamento T2**.

**Justificativa:**
Ao analisar a validação sobre os dados inéditos ($t=101..120$), a Rede 2 apresentou a melhor capacidade de generalização e previsão da série temporal. Especificamente, o **Treinamento T2 da Rede 2** obteve:
1. O **menor Erro Relativo Médio** global (**8.44%**), indicando que as estimativas se aproximaram com grande acurácia dos valores reais do mercado.
2. A **menor Variância** de erro (69.44 $\times 10^{-4}$), garantindo que não existam desvios ou picos drásticos de previsão. 

A Rede 1 se mostrou muito simples (baixa capacidade, resultando em erro ~22-25%) e a Rede 3 apresentou uma pequena queda na capacidade de generalização e tendência a travar em mínimos locais péssimos (como no T1).

---

## Questão 7
*Em relação aos algoritmos de treinamento que são variantes do algoritmo backpropagation, investigue e comente sobre as principais características e vantagens dos seguintes algoritmos: Resilient-Propagation (RProp) e Levenberg-Marquardt (LM).*

**Resposta:**

**1. Resilient-Propagation (RProp)**
- **Características:** É um algoritmo de otimização que utiliza apenas o sinal (direção) do gradiente da função de erro para atualizar os pesos, ignorando sua magnitude. Cada peso da rede tem seu próprio tamanho de passo (taxa de atualização) adaptativo. Se o gradiente mantém o mesmo sinal em épocas consecutivas, o tamanho do passo aumenta (acelerando a convergência). Se o gradiente muda de sinal, significa que o passo foi grande demais, passando pelo mínimo, e portanto o tamanho do passo é reduzido.
- **Vantagens:** Supera de forma muito eficaz o problema de *vanishing gradients* (desaparecimento do gradiente), comum no uso de funções de ativação como a sigmoide em redes mais profundas, já que a magnitude pequena do gradiente não afeta a velocidade do passo. É um algoritmo extremamente rápido para problemas de reconhecimento de padrões e não exige que o desenvolvedor perca tempo ajustando manualmente a taxa de aprendizado, pois os passos são auto-ajustáveis.

**2. Levenberg-Marquardt (LM)**
- **Características:** Trata-se de uma aproximação do método de Newton (que utiliza derivadas de segunda ordem) desenhada especificamente para minimizar funções de erro baseadas em soma de quadrados (Erro Quadrático). Em vez de calcular a Matriz Hessiana completa (que é extremamente custosa computacionalmente), o LM calcula a Matriz Jacobiana para aproximar a Hessiana. O algoritmo atua como uma mistura (interpolando dinamicamente através de um fator de amortecimento $\mu$) entre o método de descida de gradiente padrão (quando está longe do mínimo) e o método de Gauss-Newton (quando está próximo do mínimo de erro).
- **Vantagens:** É considerado um dos algoritmos de treinamento mais rápidos disponíveis para redes neurais *feedforward* de tamanho moderado (até algumas centenas de pesos). Converge de forma imensamente mais rápida e com precisão muito maior que métodos de primeira ordem (como o backpropagation com momentum). É o estado da arte para problemas de regressão e aproximação de funções (fitting) onde se exige altíssima precisão no EQM.
