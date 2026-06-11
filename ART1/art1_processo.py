import numpy as np
import os

class ART1:
    def __init__(self, n_features, max_categories=20, rho=0.5, L=2.0):
        """
        Inicializa a Rede ART-1.
        
        Parâmetros:
        n_features : int
            Tamanho do vetor de entrada (F1).
        max_categories : int
            Número máximo de categorias na camada de reconhecimento (F2).
        rho : float
            Parâmetro de vigilância (0 < rho <= 1).
        L : float
            Parâmetro constante (L > 1), tipicamente 2.0.
        """
        self.n_features = n_features
        self.max_categories = max_categories
        self.rho = rho
        self.L = L
        
        # Pesos bottom-up (F1 -> F2): Wf de dimensão (max_categories, n_features)
        # Inicializados conforme a regra padrão: L / (L - 1 + n_features)
        self.Wf = np.ones((max_categories, n_features)) * (self.L / (self.L - 1 + n_features))
        
        # Pesos top-down (F2 -> F1): Wb de dimensão (n_features, max_categories)
        # Inicializados com 1.0 (permite que qualquer entrada ressoe inicialmente)
        self.Wb = np.ones((n_features, max_categories))
        
        # Vetor para controlar quais categorias já foram associadas a algum padrão
        self.committed = np.zeros(max_categories, dtype=bool)

    def train_pattern(self, x):
        """
        Apresenta um padrão x à rede, realiza o teste de vigilância,
        atualiza os pesos se houver ressonância e retorna a categoria vencedora.
        """
        x = np.array(x, dtype=float)
        x_sum = np.sum(x)
        
        if x_sum == 0:
            return -1  # Padrão nulo
            
        disabled_nodes = set()
        
        while True:
            # Calcular ativações bottom-up (Y)
            activations = []
            for j in range(self.max_categories):
                if j in disabled_nodes:
                    activations.append(-1.0)
                else:
                    # u_j = Wf_j * x
                    act = np.dot(self.Wf[j], x)
                    activations.append(act)
            
            # Seleciona a categoria com maior ativação
            J = np.argmax(activations)
            
            if activations[J] < 0:
                raise ValueError("Capacidade máxima de categorias excedida!")
            
            # Teste de Vigilância
            # Interseção lógica entre a entrada x e o template top-down Wb[:, J]
            # Como x e Wb são binários, a multiplicação elemento a elemento realiza o AND lógico
            intersection = x * self.Wb[:, J]
            intersection_sum = np.sum(intersection)
            
            match_ratio = intersection_sum / x_sum
            
            if match_ratio >= self.rho:
                # Ressonância estabelecida!
                # Atualiza os pesos top-down (template)
                self.Wb[:, J] = intersection
                # Atualiza os pesos bottom-up
                self.Wf[J, :] = (self.L * intersection) / (self.L - 1 + intersection_sum)
                self.committed[J] = True
                return J
            else:
                # Mismatch! O neurônio J é inibido temporariamente para este padrão
                disabled_nodes.add(J)

def run_simulation(data, rho, L=2.0, max_epochs=100):
    n_samples = len(data)
    n_features = len(data[0])
    
    net = ART1(n_features=n_features, max_categories=20, rho=rho, L=L)
    
    # Lista para armazenar a atribuição de categoria para cada amostra
    assignments = [-1] * n_samples
    
    # Loop de épocas até convergência
    for epoch in range(max_epochs):
        changed = False
        current_assignments = []
        
        for i, x in enumerate(data):
            category = net.train_pattern(x)
            current_assignments.append(category)
            if category != assignments[i]:
                changed = True
                
        assignments = current_assignments
        if not changed:
            break
            
    # Mapear categorias para as situações nelas contidas
    clusters = {}
    for i, cat in enumerate(assignments):
        if cat not in clusters:
            clusters[cat] = []
        clusters[cat].append(i + 1)  # Indexação 1-based para situações
        
    return net, clusters, assignments, epoch + 1

if __name__ == "__main__":
    # 10 situações com 16 variáveis de status cada
    situations = [
        [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1], # Situação 1
        [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0], # Situação 2
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], # Situação 3
        [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0], # Situação 4
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1], # Situação 5
        [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1], # Situação 6
        [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0], # Situação 7
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], # Situação 8
        [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], # Situação 9
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1]  # Situação 10
    ]
    
    vigilance_levels = [0.5, 0.8, 0.9, 0.99]
    simulation_results = {}
    
    print("Iniciando simulação da rede ART-1 para detecção e agrupamento de situações...")
    
    for rho in vigilance_levels:
        net, clusters, assignments, epochs = run_simulation(situations, rho)
        
        # Mapeando os nomes das classes de forma amigável (Classe A, Classe B, etc.)
        sorted_keys = sorted(clusters.keys())
        class_mapping = {key: f"Classe {chr(65 + idx)}" for idx, key in enumerate(sorted_keys)}
        
        friendly_clusters = {}
        for key, val in clusters.items():
            friendly_clusters[class_mapping[key]] = {
                "situations": val,
                "template": net.Wb[:, key].astype(int).tolist()
            }
            
        simulation_results[rho] = {
            "num_classes": len(clusters),
            "clusters": friendly_clusters,
            "epochs": epochs
        }
        
        print(f"\nSimulação com Vigilância rho = {rho}:")
        print(f"  - Convergiu em {epochs} épocas.")
        print(f"  - Número de classes ativas: {len(clusters)}")
        for cls_name, info in friendly_clusters.items():
            sits = ", ".join(f"Situação {s}" for s in info["situations"])
            template_ones = [i+1 for i, val in enumerate(info["template"]) if val == 1]
            print(f"    * {cls_name}: [{sits}] | Variáveis de status ativas no diagnóstico: x{template_ones}")

    # Gerar o arquivo respostas_art1.md automaticamente
    md_content = []
    md_content.append("# Respostas da Atividade - Rede ART-1")
    md_content.append("")
    md_content.append("Esta atividade apresenta a implementação e o treinamento de uma rede neural **ART-1 (Adaptive Resonance Theory 1)** para classificar e agrupar 10 situações de comportamento industrial representadas por 16 variáveis de status binárias.")
    md_content.append("")
    md_content.append("---")
    md_content.append("")
    md_content.append("## 1. Dados de Entrada (Situações de Processo)")
    md_content.append("Cada situação é um vetor binário de 16 dimensões, onde cada dimensão $x_1$ a $x_{16}$ indica o status (0 ou 1) de uma variável do processo industrial:")
    md_content.append("")
    md_content.append("| Situação | x1 | x2 | x3 | x4 | x5 | x6 | x7 | x8 | x9 | x10 | x11 | x12 | x13 | x14 | x15 | x16 |")
    md_content.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for idx, sit in enumerate(situations):
        vals = " | ".join(str(v) for v in sit)
        md_content.append(f"| **Situação {idx+1}** | {vals} |")
    md_content.append("")
    md_content.append("---")
    md_content.append("")
    md_content.append("## 2. Resultados das Simulações com Diferentes Vigilâncias ($\\rho$)")
    md_content.append("")
    
    for rho in vigilance_levels:
        res = simulation_results[rho]
        md_content.append(f"### 2.{int(rho*100) if rho != 0.99 else 99} Vigilância $\\rho = {rho}$")
        md_content.append(f"- **Classes Ativas**: {res['num_classes']}")
        md_content.append(f"- **Épocas até a convergência**: {res['epochs']}")
        md_content.append("")
        md_content.append("| Classe | Situações Agrupadas | Protótipo / Diagnóstico (Variáveis Ativas) |")
        md_content.append("|---|---|---|")
        
        for cls_name, info in sorted(res['clusters'].items()):
            sits = ", ".join(f"Situação {s}" for s in info["situations"])
            # Achar índices das variáveis ativas no protótipo (1-based)
            template_ones = [f"x{i+1}" for i, val in enumerate(info["template"]) if val == 1]
            vars_str = ", ".join(template_ones) if template_ones else "Nenhuma (Vetor Nulo)"
            md_content.append(f"| **{cls_name}** | {sits} | {vars_str} |")
        md_content.append("")
        
    md_content.append("---")
    md_content.append("")
    md_content.append("## 3. Análise Comparativa do Parâmetro de Vigilância ($\\rho$)")
    md_content.append("")
    md_content.append("O parâmetro de vigilância ($\\rho$) é o responsável direto pela granularidade dos agrupamentos realizados pela rede ART-1. A análise de cada cenário revela:")
    md_content.append("")
    md_content.append("1. **$\\rho = 0.5$ (Vigilância Baixa)**:")
    md_content.append("   - Ocorre um agrupamento mais amplo e generalista. A rede tolera uma quantidade considerável de diferenças entre os vetores de entrada de uma mesma classe.")
    md_content.append(f"   - Resulta em **{simulation_results[0.5]['num_classes']} classes ativas**. As situações que compartilham algumas poucas variáveis são agrupadas na mesma classe. Por exemplo, quase todas as situações que possuem várias variáveis em comum são colocadas juntas, gerando classes muito populosas.")
    md_content.append("")
    md_content.append("2. **$\\rho = 0.8$ (Vigilância Média-Alta)**:")
    md_content.append("   - Aumenta o rigor da rede. Os padrões precisam apresentar uma similaridade substancial para serem classificados na mesma categoria.")
    md_content.append(f"   - Resulta em **{simulation_results[0.8]['num_classes']} classes ativas**. A rede começa a separar situações que antes estavam juntas devido a variáveis de status discordantes.")
    md_content.append("")
    md_content.append("3. **$\\rho = 0.9$ (Vigilância Alta)**:")
    md_content.append("   - A rede torna-se bastante seletiva. Qualquer divergência menor em uma ou duas variáveis importantes faz com que a situação seja enviada para uma classe diferente ou crie uma nova classe.")
    md_content.append(f"   - Resulta em **{simulation_results[0.9]['num_classes']} classes ativas**.")
    md_content.append("")
    md_content.append("4. **$\\rho = 0.99$ (Vigilância Extremamente Alta)**:")
    md_content.append("   - A rede exige uma correspondência praticamente perfeita (identidade) entre as variáveis de status.")
    md_content.append(f"   - Resulta em **{simulation_results[0.99]['num_classes']} classes ativas**. Apenas situações absolutamente idênticas ficam agrupadas. Neste caso:")
    md_content.append("     * As situações **3 e 8** (que são idênticas) formam a mesma classe.")
    md_content.append("     * As situações **5 e 10** (que também são idênticas) formam a mesma classe.")
    md_content.append("     * Todas as demais situações, por possuírem alguma diferença de pelo menos 1 bit, são alocadas em classes individuais de elemento único.")
    md_content.append("")
    md_content.append("## 4. Diagnóstico para Manutenção Industrial")
    md_content.append("")
    md_content.append("A representação das classes por meio dos vetores de pesos top-down ($W_b$ ou templates) serve como um **diagnóstico lógico**. O protótipo resultante de cada agrupamento representa o operador de **AND lógico** das situações contidas nele.")
    md_content.append("Isto significa que, se uma classe agrupa Situações de falhas ou comportamentos anômalos parecidos, o protótipo identifica as variáveis de status que estão **comprovadamente ativas em todos os casos daquela classe**, indicando a causa raiz comum do problema.")
    md_content.append("")
    md_content.append("- Em $\\rho$ intermediários (como $0.8$), podemos agrupar falhas semelhantes e obter um diagnóstico genérico do subsistema afetado (ex: sensores redundantes ativados).")
    md_content.append("- Em $\\rho$ elevados (como $0.99$), obtemos diagnósticos ultraespecíficos de falhas isoladas de componentes individuais.")
    md_content.append("")
    
    # Escrever respostas_art1.md
    output_path = os.path.join("c:\\Users\\maria\\atividades\\IA\\ART1", "respostas_art1.md")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))
        print(f"\nArquivo respostas_art1.md gerado com sucesso em: {output_path}")
    except Exception as e:
        print(f"Erro ao salvar arquivo respostas_art1.md: {e}")
