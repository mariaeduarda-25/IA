# Respostas da Atividade - Rede de Kohonen (SOM)

Esta atividade apresenta os resultados obtidos com o treinamento e a simulação de uma **Rede de Kohonen (Self-Organizing Map - SOM)** para detecção de similaridades e agrupamento de amostras imperfeitas de borracha em um processo industrial.

---

## 1. Mapeamento das Regiões do Grid 4x4 (Questão 1)

O grid topológico bidimensional de tamanho 4x4 (16 neurônios) foi treinado utilizando as 120 amostras de dados contidas no apêndice da atividade, com os seguintes parâmetros configurados:
*   **Neurônios no Grid:** 16 neurônios dispostos em 4x4, mapeados com coordenadas $(r, c)$ para $r, c \in \{0, 1, 2, 3\}$.
*   **Taxa de Aprendizado ($\eta$):** $0.001$ (fixo).
*   **Raio de Vizinhança ($R$):** $1$ (vizinhos Manhattan directos: distância $\le 1$).
*   **Épocas de Treinamento:** $10.000$ épocas (para garantir a convergência estável dos pesos).

De acordo com o enunciado, as amostras pertencem a três classes conhecidas:
*   **Classe A:** Amostras 1 a 20 ($x_1 \approx 0.25$, $x_2 \approx 0.25$, $x_3 \approx 0.20$)
*   **Classe B:** Amostras 21 a 60 ($x_1 \approx 0.75$, $x_2 \approx 0.25$, $x_3 \approx 0.75$)
*   **Classe C:** Amostras 61 a 120 ($x_1 \approx 0.50$, $x_2 \approx 0.75$, $x_3 \approx 0.50$)

### Organização Topológica do Grid 4x4
Após o treinamento, o mapeamento obtido para cada neurônio (representando a contagem de amostras de cada classe que ativaram o neurônio correspondente como vencedor) foi o seguinte:

```
+--------------------------+--------------------------+--------------------------+--------------------------+
| Neurônio 01 (N01): [C]   | Neurônio 02 (N02): [C]   | Neurônio 03 (N03): [B]   | Neurônio 04 (N04): [B]   |
| (A:0, B:0, C:11)         | (A:0, B:0, C:11)         | (A:0, B:1, C:0)          | (A:0, B:18, C:0)         |
+--------------------------+--------------------------+--------------------------+--------------------------+
| Neurônio 05 (N05): [C]   | Neurônio 06 (N06): [C]   | Neurônio 07 (N07): [B]   | Neurônio 08 (N08): [B]   |
| (A:0, B:0, C:11)         | (A:0, B:0, C:6)          | (A:0, B:2, C:0)          | (A:0, B:19, C:0)         |
+--------------------------+--------------------------+--------------------------+--------------------------+
| Neurônio 09 (N09): [C]   | Neurônio 10 (N10): [-]   | Neurônio 11 (N11): [A]   | Neurônio 12 (N12): [-]   |
| (A:0, B:0, C:7)          | (A:0, B:0, C:0) - Inat   | (A:5, B:0, C:0)          | (A:0, B:0, C:0) - Inat   |
+--------------------------+--------------------------+--------------------------+--------------------------+
| Neurônio 13 (N13): [C]   | Neurônio 14 (N14): [-]   | Neurônio 15 (N15): [A]   | Neurônio 16 (N16): [A]   |
| (A:0, B:0, C:14)         | (A:0, B:0, C:0) - Inat   | (A:6, B:0, C:0)          | (A:9, B:0, C:0)          |
+--------------------------+--------------------------+--------------------------+--------------------------+
```

### Regiões Definidas para as Classes A, B e C:
O mapa se organizou de forma perfeitamente contígua e separada espacialmente:
*   **Classe A:** Ativada pelos neurônios **11, 15 e 16** (canto inferior direito do grid).
*   **Classe B:** Ativada pelos neurônios **3, 4, 7 e 8** (canto superior direito do grid).
*   **Classe C:** Ativada pelos neurônios **1, 2, 5, 6, 9 e 13** (toda a metade e lateral esquerda do grid).
*   **Neurônios Inativos (sem amostras ativas):** **10, 12 e 14**. Eles agem como fronteiras/transições neutras na topologia do mapa.

---

## 2. Classificação das Amostras de Teste (Questão 2)

Utilizando a rede treinada e o mapeamento espacial obtido acima, classificamos as 12 amostras de teste fornecidas. O resultado da classificação foi:

| Amostra | $x_1$ | $x_2$ | $x_3$ | Neurônio Vencedor | Classe Atribuída |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.2471 | 0.1778 | 0.2905 | Neurônio 11 | **Classe A** |
| **2** | 0.8240 | 0.2223 | 0.7041 | Neurônio 08 | **Classe B** |
| **3** | 0.4960 | 0.7231 | 0.5866 | Neurônio 06 | **Classe C** |
| **4** | 0.2923 | 0.2041 | 0.2234 | Neurônio 11 | **Classe A** |
| **5** | 0.8118 | 0.2668 | 0.7484 | Neurônio 08 | **Classe B** |
| **6** | 0.4837 | 0.8200 | 0.4792 | Neurônio 05 | **Classe C** |
| **7** | 0.3248 | 0.2629 | 0.2375 | Neurônio 11 | **Classe A** |
| **8** | 0.7209 | 0.2116 | 0.7821 | Neurônio 04 | **Classe B** |
| **9** | 0.5259 | 0.6522 | 0.5957 | Neurônio 02 | **Classe C** |
| **10** | 0.2075 | 0.1669 | 0.1745 | Neurônio 15 | **Classe A** |
| **11** | 0.7830 | 0.3171 | 0.7888 | Neurônio 04 | **Classe B** |
| **12** | 0.5393 | 0.7510 | 0.5682 | Neurônio 02 | **Classe C** |

---

## 3. Demonstração Matemática: Regra de Hebb por Minimização de Erro (Questão 3)

Deseja-se demonstrar que a regra de alteração de pesos "Norma Euclidiana" para um padrão $\mathbf{x}$ é obtida a partir da minimização da função do erro quadrático médio de um neurônio vencedor $j$:

$$E = \frac{1}{2} \sum_{i=1}^n (x_i - w_{ji})^2$$

### Demonstração:

Para minimizar a função de erro $E$ em relação a um peso específico $w_{ji}$ do neurônio vencedor $j$, aplicamos o método do **Gradiente Descendente**. A regra do gradiente estabelece que a variação do peso ($\Delta w_{ji}$) deve ser proporcional à direção oposta do gradiente da função de erro:

$$\Delta w_{ji} = -\eta \frac{\partial E}{\partial w_{ji}}$$

onde $\eta$ representa a taxa de aprendizado positivo ($0 < \eta \le 1$).

Calculamos a derivada parcial da função de erro $E$ em relação a $w_{ji}$:

$$\frac{\partial E}{\partial w_{ji}} = \frac{\partial}{\partial w_{ji}} \left[ \frac{1}{2} \sum_{k=1}^n (x_k - w_{jk})^2 \right]$$

Pela linearidade da derivada, todos os termos da soma com índice $k \neq i$ possuem derivada nula com relação a $w_{ji}$ (já que $w_{jk}$ é independente de $w_{ji}$ se $k \neq i$). Portanto, a derivada simplifica-se para:

$$\frac{\partial E}{\partial w_{ji}} = \frac{\partial}{\partial w_{ji}} \left[ \frac{1}{2} (x_i - w_{ji})^2 \right]$$

Aplicando a **regra da cadeia**:
1. Derivada da parte externa: $\frac{1}{2} \cdot 2(x_i - w_{ji}) = (x_i - w_{ji})$
2. Derivada da parte interna com relação a $w_{ji}$: $\frac{\partial}{\partial w_{ji}} (x_i - w_{ji}) = -1$

Multiplicando os termos:

$$\frac{\partial E}{\partial w_{ji}} = (x_i - w_{ji}) \cdot (-1) = -(x_i - w_{ji})$$

Substituindo essa derivada parcial de volta na regra do gradiente descendente:

$$\Delta w_{ji} = -\eta \cdot [-(x_i - w_{ji})]$$

$$\Delta w_{ji} = \eta \cdot (x_i - w_{ji})$$

Assim, a regra de atualização dos pesos no instante $t+1$ será:

$$w_{ji}(t+1) = w_{ji}(t) + \Delta w_{ji} = w_{ji}(t) + \eta (x_i - w_{ji}(t))$$

Essa equação corresponde exatamente à regra de aprendizado do neurônio vencedor em mapas auto-organizáveis de Kohonen (e cooperativamente estendida aos seus vizinhos topológicos dentro do raio de vizinhança $R$).

**Fim da Demonstração.**
