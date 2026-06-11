# Respostas da Atividade - Rede LVQ-1

Esta atividade apresenta a implementação, o treinamento e os resultados obtidos com uma rede **Learning Vector Quantization 1 (LVQ-1)** utilizada para classificar perfis de demanda de energia elétrica com base em leituras realizadas das 7 às 12 horas (vetores de 6 dimensões).

Configuração da simulação:
- **Taxa de aprendizado inicial ($\alpha_0$):** $0.05$ (decaimento linear até $0.0$)
- **Épocas de treinamento:** $1000$ épocas
- **Reprodutibilidade:** Executado com semente fixa (`seed = 42`) e embaralhamento dos padrões por época.

---

## 1. Pesos Iniciais e Finais dos Protótipos

Para analisar o efeito do posicionamento inicial, o modelo foi treinado sob duas estratégias de inicialização:

### Cenário A: Inicialização por Centróide da Classe (Recomendado)
Nesta estratégia, o protótipo de cada classe é inicializado como a média aritmética de todas as amostras pertencentes àquela classe no conjunto de treinamento.

| Protótipo | 7h | 8h | 9h | 10h | 11h | 12h |
|---|---|---|---|---|---|---|
| **W1 Inicial (Méd)** | 2.3424 | 1.4871 | 1.9424 | 1.2456 | 2.3315 | 1.8151 |
| **W1 Final** | 2.3424 | 1.4871 | 1.9423 | 1.2456 | 2.3315 | 1.8150 |
| **W2 Inicial (Méd)** | 1.0641 | 0.1305 | 1.2496 | 5.3630 | 3.1519 | 2.3546 |
| **W2 Final** | 1.0641 | 0.1305 | 1.2496 | 5.3629 | 3.1518 | 2.3546 |
| **W3 Inicial (Méd)** | 1.4055 | 2.2811 | 1.0344 | 2.4214 | 1.7341 | 5.0960 |
| **W3 Final** | 1.4055 | 2.2810 | 1.0344 | 2.4214 | 1.7341 | 5.0960 |
| **W4 Inicial (Méd)** | 2.9488 | 1.4922 | 4.6610 | 1.3814 | 4.2524 | 6.8548 |
| **W4 Final** | 2.9488 | 1.4922 | 4.6610 | 1.3814 | 4.2524 | 6.8548 |

### Cenário B: Inicialização pela Primeira Amostra da Classe
Nesta estratégia clássica, o protótipo de cada classe é inicializado com os valores da primeira amostra correspondente daquela classe (Amostras 1, 5, 9 e 13).

| Protótipo | 7h | 8h | 9h | 10h | 11h | 12h |
|---|---|---|---|---|---|---|
| **W1 Inicial (Amostra)** | 2.3976 | 1.5328 | 1.9044 | 1.1937 | 2.4184 | 1.8649 |
| **W1 Final** | 2.3424 | 1.4871 | 1.9423 | 1.2456 | 2.3315 | 1.8150 |
| **W2 Inicial (Amostra)** | 1.1201 | 0.0587 | 1.3154 | 5.3783 | 3.1849 | 2.4276 |
| **W2 Final** | 1.0641 | 0.1305 | 1.2496 | 5.3629 | 3.1518 | 2.3546 |
| **W3 Inicial (Amostra)** | 1.4871 | 2.3448 | 0.9918 | 2.3160 | 1.6783 | 5.0850 |
| **W3 Final** | 1.4055 | 2.2810 | 1.0344 | 2.4214 | 1.7341 | 5.0960 |
| **W4 Inicial (Amostra)** | 2.9364 | 1.5233 | 4.6109 | 1.3160 | 4.2700 | 6.8749 |
| **W4 Final** | 2.9488 | 1.4922 | 4.6610 | 1.3814 | 4.2524 | 6.8548 |

---

## 2. Classificação dos Novos Dias (Dados de Teste)

Após o treinamento, as duas redes foram aplicadas aos dados de teste. Ambas convergiram para a mesma classificação perfeita, indicando que os limites de decisão gerados são estáveis e concordantes:

| Dia | 7h | 8h | 9h | 10h | 11h | 12h | Classe Atribuída (Centróide) | Classe Atribuída (1ª Amostra) |
|---|---|---|---|---|---|---|---|---|
| **1** | 2.9817 | 1.5656 | 4.8391 | 1.4311 | 4.1916 | 6.9718 | **Classe 4** | **Classe 4** |
| **2** | 1.5537 | 2.2615 | 1.3169 | 2.5873 | 1.7570 | 5.0958 | **Classe 3** | **Classe 3** |
| **3** | 1.2240 | 0.2445 | 1.3595 | 5.4192 | 3.2027 | 2.5675 | **Classe 2** | **Classe 2** |
| **4** | 2.5828 | 1.5146 | 2.1119 | 1.2859 | 2.3414 | 1.8695 | **Classe 1** | **Classe 1** |
| **5** | 2.4168 | 1.4857 | 1.8959 | 1.3013 | 2.4500 | 1.7868 | **Classe 1** | **Classe 1** |
| **6** | 1.0604 | 0.2276 | 1.2806 | 5.4732 | 3.2133 | 2.4839 | **Classe 2** | **Classe 2** |
| **7** | 1.5246 | 2.4254 | 1.1353 | 2.5325 | 1.7569 | 5.2640 | **Classe 3** | **Classe 3** |
| **8** | 3.0565 | 1.6259 | 4.7743 | 1.3654 | 4.2904 | 6.9808 | **Classe 4** | **Classe 4** |

---

## 3. Análise dos Resultados e Comportamento da LVQ-1

### 3.1. Classificação dos Perfis
A rede LVQ-1 mapeou de forma consistente os novos dias de teste:
- **Classe 1 (Demanda Estável Baixa):** Dias 4 e 5 representam o perfil da Classe 1, caracterizado por demanda estável em torno de 2.4 - 2.5 MW no início da manhã e final da manhã, sem picos drásticos.
- **Classe 2 (Pico Intermediário ao Meio-Dia):** Dias 3 e 6. Apresentam um pico notável de consumo próximo a 5.4 MW especificamente às 10 horas, com queda abrupta na sequência.
- **Classe 3 (Pico às 8h e 12h):** Dias 2 e 7. Caracterizados por um consumo elevado no início às 8h (2.3 - 2.4 MW) e uma subida substancial no final da manhã às 12h (5.1 - 5.2 MW).
- **Classe 4 (Demanda Crescente Acentuada):** Dias 1 e 8. Mostram alto consumo geral com patamares crescentes, iniciando em 3.0 MW e disparando até quase 7.0 MW às 12h.

### 3.2. Efeito da Inicialização dos Protótipos
A inicialização por centróides resulta em protótipos que começam exatamente no centro de massa das classes, exigindo menos deslocamento angular e de distância espacial para convergir. A inicialização por amostra individual, embora clássica, pode herdar ruídos ou desvios individuais da amostra selecionada.
No entanto, devido ao fato de que as classes deste conjunto de dados estão muito bem separadas geometricamente no espaço $\mathbb{R}^6$, ambas as inicializações conduziram a protótipos finais que definem as mesmas fronteiras de decisão corretas para todas as amostras de teste.

### 3.3. Papel da Regra de Repulsão
A regra de repulsão ($w_J \leftarrow w_J - \alpha(x - w_J)$) é a característica crucial que diferencia a rede LVQ-1 (aprendizado supervisionado) de modelos não supervisionados como a rede de Kohonen. Se um protótipo incorretamente ganha a competição por uma amostra de outra classe, ele é 'empurrado' para longe daquele limite. Isso otimiza o posicionamento do protótipo ao longo da fronteira de decisão (hiperplano de separação), maximizando a margem de acerto para a classificação futura de novos dados.
