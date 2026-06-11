# Respostas da Atividade - Rede ART-1

Esta atividade apresenta a implementação e o treinamento de uma rede neural **ART-1 (Adaptive Resonance Theory 1)** para classificar e agrupar 10 situações de comportamento industrial representadas por 16 variáveis de status binárias.

---

## 1. Dados de Entrada (Situações de Processo)
Cada situação é um vetor binário de 16 dimensões, onde cada dimensão $x_1$ a $x_{16}$ indica o status (0 ou 1) de uma variável do processo industrial:

| Situação | x1 | x2 | x3 | x4 | x5 | x6 | x7 | x8 | x9 | x10 | x11 | x12 | x13 | x14 | x15 | x16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Situação 1** | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| **Situação 2** | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| **Situação 3** | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| **Situação 4** | 1 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| **Situação 5** | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| **Situação 6** | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| **Situação 7** | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 0 |
| **Situação 8** | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| **Situação 9** | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| **Situação 10** | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |

---

## 2. Resultados das Simulações com Diferentes Vigilâncias ($\rho$)

### 2.50 Vigilância $\rho = 0.5$
- **Classes Ativas**: 4
- **Épocas até a convergência**: 3

| Classe | Situações Agrupadas | Protótipo / Diagnóstico (Variáveis Ativas) |
|---|---|---|
| **Classe A** | Situação 9 | x7, x9, x10, x12, x16 |
| **Classe B** | Situação 2, Situação 4, Situação 5, Situação 10 | x3, x5, x7, x10, x11 |
| **Classe C** | Situação 7 | x1, x3, x5, x6, x8, x9, x10, x11, x13, x14, x15 |
| **Classe D** | Situação 1, Situação 3, Situação 6, Situação 8 | x4, x7, x9, x10, x12, x13, x15, x16 |

### 2.80 Vigilância $\rho = 0.8$
- **Classes Ativas**: 5
- **Épocas até a convergência**: 2

| Classe | Situações Agrupadas | Protótipo / Diagnóstico (Variáveis Ativas) |
|---|---|---|
| **Classe A** | Situação 1, Situação 6 | x2, x4, x7, x9, x10, x12, x13, x14, x15, x16 |
| **Classe B** | Situação 2, Situação 7 | x1, x3, x5, x6, x8, x9, x10, x11, x13 |
| **Classe C** | Situação 3, Situação 8 | x1, x3, x4, x5, x6, x7, x9, x10, x12, x13, x15, x16 |
| **Classe D** | Situação 4, Situação 9 | x2, x3, x5, x7, x9, x10, x12, x14 |
| **Classe E** | Situação 5, Situação 10 | x3, x4, x5, x6, x7, x8, x10, x11, x16 |

### 2.90 Vigilância $\rho = 0.9$
- **Classes Ativas**: 7
- **Épocas até a convergência**: 2

| Classe | Situações Agrupadas | Protótipo / Diagnóstico (Variáveis Ativas) |
|---|---|---|
| **Classe A** | Situação 1, Situação 6 | x2, x4, x7, x9, x10, x12, x13, x14, x15, x16 |
| **Classe B** | Situação 2 | x1, x3, x5, x6, x7, x8, x9, x10, x11, x13 |
| **Classe C** | Situação 3, Situação 8 | x1, x3, x4, x5, x6, x7, x9, x10, x12, x13, x15, x16 |
| **Classe D** | Situação 4 | x1, x2, x3, x5, x7, x9, x10, x11, x12, x14 |
| **Classe E** | Situação 5, Situação 10 | x3, x4, x5, x6, x7, x8, x10, x11, x16 |
| **Classe F** | Situação 7 | x1, x3, x5, x6, x8, x9, x10, x11, x13, x14, x15 |
| **Classe G** | Situação 9 | x2, x3, x5, x7, x9, x10, x12, x14, x16 |

### 2.99 Vigilância $\rho = 0.99$
- **Classes Ativas**: 8
- **Épocas até a convergência**: 2

| Classe | Situações Agrupadas | Protótipo / Diagnóstico (Variáveis Ativas) |
|---|---|---|
| **Classe A** | Situação 1 | x2, x4, x5, x7, x9, x10, x12, x13, x14, x15, x16 |
| **Classe B** | Situação 2 | x1, x3, x5, x6, x7, x8, x9, x10, x11, x13 |
| **Classe C** | Situação 3, Situação 8 | x1, x3, x4, x5, x6, x7, x9, x10, x12, x13, x15, x16 |
| **Classe D** | Situação 4 | x1, x2, x3, x5, x7, x9, x10, x11, x12, x14 |
| **Classe E** | Situação 5, Situação 10 | x3, x4, x5, x6, x7, x8, x10, x11, x16 |
| **Classe F** | Situação 6 | x1, x2, x4, x7, x9, x10, x12, x13, x14, x15, x16 |
| **Classe G** | Situação 7 | x1, x3, x5, x6, x8, x9, x10, x11, x13, x14, x15 |
| **Classe H** | Situação 9 | x2, x3, x5, x7, x9, x10, x12, x14, x16 |

---

## 3. Análise Comparativa do Parâmetro de Vigilância ($\rho$)

O parâmetro de vigilância ($\rho$) é o responsável direto pela granularidade dos agrupamentos realizados pela rede ART-1. A análise de cada cenário revela:

1. **$\rho = 0.5$ (Vigilância Baixa)**:
   - Ocorre um agrupamento mais amplo e generalista. A rede tolera uma quantidade considerável de diferenças entre os vetores de entrada de uma mesma classe.
   - Resulta em **4 classes ativas**. As situações que compartilham algumas poucas variáveis são agrupadas na mesma classe. Por exemplo, quase todas as situações que possuem várias variáveis em comum são colocadas juntas, gerando classes muito populosas.

2. **$\rho = 0.8$ (Vigilância Média-Alta)**:
   - Aumenta o rigor da rede. Os padrões precisam apresentar uma similaridade substancial para serem classificados na mesma categoria.
   - Resulta em **5 classes ativas**. A rede começa a separar situações que antes estavam juntas devido a variáveis de status discordantes.

3. **$\rho = 0.9$ (Vigilância Alta)**:
   - A rede torna-se bastante seletiva. Qualquer divergência menor em uma ou duas variáveis importantes faz com que a situação seja enviada para uma classe diferente ou crie uma nova classe.
   - Resulta em **7 classes ativas**.

4. **$\rho = 0.99$ (Vigilância Extremamente Alta)**:
   - A rede exige uma correspondência praticamente perfeita (identidade) entre as variáveis de status.
   - Resulta em **8 classes ativas**. Apenas situações absolutamente idênticas ficam agrupadas. Neste caso:
     * As situações **3 e 8** (que são idênticas) formam a mesma classe.
     * As situações **5 e 10** (que também são idênticas) formam a mesma classe.
     * Todas as demais situações, por possuírem alguma diferença de pelo menos 1 bit, são alocadas em classes individuais de elemento único.

## 4. Diagnóstico para Manutenção Industrial

A representação das classes por meio dos vetores de pesos top-down ($W_b$ ou templates) serve como um **diagnóstico lógico**. O protótipo resultante de cada agrupamento representa o operador de **AND lógico** das situações contidas nele.
Isto significa que, se uma classe agrupa Situações de falhas ou comportamentos anômalos parecidos, o protótipo identifica as variáveis de status que estão **comprovadamente ativas em todos os casos daquela classe**, indicando a causa raiz comum do problema.

- Em $\rho$ intermediários (como $0.8$), podemos agrupar falhas semelhantes e obter um diagnóstico genérico do subsistema afetado (ex: sensores redundantes ativados).
- Em $\rho$ elevados (como $0.99$), obtemos diagnósticos ultraespecíficos de falhas isoladas de componentes individuais.
