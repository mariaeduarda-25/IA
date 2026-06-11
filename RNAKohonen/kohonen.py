import csv
import numpy as np
from typing import List, Tuple, Dict

class KohonenSOM:
    def __init__(self, grid_size: int = 4, input_dim: int = 3, lr: float = 0.001, radius: int = 1) -> None:
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.lr = lr
        self.radius = radius
        # Inicializar os pesos aleatoriamente no intervalo [0, 1]
        # Usamos uma semente fixa para reprodutibilidade dos resultados
        np.random.seed(42)
        self.weights = np.random.uniform(0.1, 0.9, (grid_size * grid_size, input_dim))
        
    def find_winner(self, x: np.ndarray) -> int:
        """
        Encontra o neurônio vencedor (Best Matching Unit - BMU)
        calculando a menor distância euclidiana.
        """
        distances = np.sum((self.weights - x) ** 2, axis=1)
        return int(np.argmin(distances))
        
    def get_neighbors(self, winner_idx: int) -> List[int]:
        """
        Retorna a lista de índices de neurônios vizinhos na grade 2D
        usando distância de Manhattan com raio <= self.radius.
        """
        winner_r = winner_idx // self.grid_size
        winner_c = winner_idx % self.grid_size
        
        neighbors = []
        for i in range(self.grid_size * self.grid_size):
            r = i // self.grid_size
            c = i % self.grid_size
            dist = abs(r - winner_r) + abs(c - winner_c)
            if dist <= self.radius:
                neighbors.append(i)
        return neighbors
        
    def train_step(self, x: np.ndarray) -> None:
        winner = self.find_winner(x)
        neighbors = self.get_neighbors(winner)
        for n in neighbors:
            # Regra de Hebb com decaimento / coerção para vizinhos (norma euclidiana minimizada)
            self.weights[n] += self.lr * (x - self.weights[n])
            
    def train(self, X: np.ndarray, epochs: int = 5000) -> None:
        for epoch in range(epochs):
            # Embaralhar as amostras a cada época para evitar viés de ordem
            indices = np.random.permutation(len(X))
            for idx in indices:
                self.train_step(X[idx])

def load_data(filepath: str) -> Tuple[np.ndarray, List[int]]:
    X = []
    ids = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            if not row:
                continue
            ids.append(int(row[0]))
            X.append([float(row[1]), float(row[2]), float(row[3])])
    return np.array(X, dtype=np.float64), ids

def main():
    train_path = "/home/alunos/Desktop/www/atvIA/IA/RNAKohonen /context/train_data.csv"
    test_path = "/home/alunos/Desktop/www/atvIA/IA/RNAKohonen /context/test_data.csv"
    
    # 1. Carregar dados
    X_train, train_ids = load_data(train_path)
    X_test, test_ids = load_data(test_path)
    
    print(f"Dados de treinamento carregados: {X_train.shape}")
    print(f"Dados de teste carregados: {X_test.shape}\n")
    
    # 2. Instanciar e treinar rede
    # Parâmetros: grid 4x4, input 3, taxa de aprendizado 0.001, raio de vizinhança 1
    som = KohonenSOM(grid_size=4, input_dim=3, lr=0.001, radius=1)
    
    # Treinar por 10000 épocas para garantir convergência completa e estável dos pesos
    epochs = 10000
    print(f"Treinando a Rede de Kohonen por {epochs} épocas...")
    som.train(X_train, epochs=epochs)
    print("Treinamento finalizado.\n")
    
    # 3. Mapear as classes A, B e C nos neurônios do Grid 4x4
    # Classe A: amostras 1-20 (ids 1 a 20)
    # Classe B: amostras 21-60 (ids 21 a 60)
    # Classe C: amostras 61-120 (ids 61 a 120)
    
    neuron_class_counts = {i: {"A": 0, "B": 0, "C": 0} for i in range(16)}
    
    for x, sample_id in zip(X_train, train_ids):
        winner = som.find_winner(x)
        if 1 <= sample_id <= 20:
            cls = "A"
        elif 21 <= sample_id <= 60:
            cls = "B"
        elif 61 <= sample_id <= 120:
            cls = "C"
        else:
            cls = "Desconhecida"
        neuron_class_counts[winner][cls] += 1
        
    print("=== Mapeamento de Classes no Grid 4x4 ===")
    print("Indica a contagem de amostras de cada classe que ativaram cada neurônio:")
    grid_mapping = np.empty((4, 4), dtype=object)
    for r in range(4):
        row_str = []
        for c in range(4):
            idx = r * 4 + c
            counts = neuron_class_counts[idx]
            # Determinar a classe majoritária do neurônio
            if counts["A"] > 0 and counts["B"] == 0 and counts["C"] == 0:
                cell_label = "A"
            elif counts["B"] > 0 and counts["A"] == 0 and counts["C"] == 0:
                cell_label = "B"
            elif counts["C"] > 0 and counts["A"] == 0 and counts["B"] == 0:
                cell_label = "C"
            elif sum(counts.values()) == 0:
                cell_label = "-"
            else:
                # Caso de mistura (raro se rede bem treinada)
                cell_label = max(counts, key=counts.get)
            grid_mapping[r, c] = cell_label
            row_str.append(f"N{idx+1:02d}:[{cell_label}] (A:{counts['A']}, B:{counts['B']}, C:{counts['C']})")
        print("  |  ".join(row_str))
    print()
    
    # Mostrar resumo das regiões do grid para cada classe
    class_neurons = {"A": [], "B": [], "C": [], "Inativo": []}
    for i in range(16):
        r = i // 4
        c = i % 4
        lbl = grid_mapping[r, c]
        if lbl == "-":
            class_neurons["Inativo"].append(i + 1)
        else:
            class_neurons[lbl].append(i + 1)
            
    print("=== Regiões do Grid 4x4 ===")
    print(f"• Classe A (Amostras 1-20): Neurônios {class_neurons['A']}")
    print(f"• Classe B (Amostras 21-60): Neurônios {class_neurons['B']}")
    print(f"• Classe C (Amostras 61-120): Neurônios {class_neurons['C']}")
    if class_neurons["Inativo"]:
        print(f"• Sem Amostras Ativas: Neurônios {class_neurons['Inativo']}")
    print()
    
    # 4. Classificar amostras de teste
    print("=== Classificação das Amostras de Teste ===")
    print(f"{'Amostra':<10}{'x1':<10}{'x2':<10}{'x3':<10}{'Neurônio Vencedor':<20}{'Classe Atribuída':<15}")
    print("-" * 75)
    for x, test_id in zip(X_test, test_ids):
        winner = som.find_winner(x)
        winner_r = winner // 4
        winner_c = winner % 4
        assigned_class = grid_mapping[winner_r, winner_c]
        
        # Se o neurônio for inativo, buscar a classe do vizinho mais próximo ou classificar diretamente pela distância
        if assigned_class == "-":
            # Classificar calculando a classe da amostra de treino mais próxima
            # (ou distância direta aos centros dos neurônios ativos)
            active_distances = []
            for i in range(16):
                r = i // 4
                c = i % 4
                if grid_mapping[r, c] != "-":
                    dist = np.sum((som.weights[i] - x) ** 2)
                    active_distances.append((dist, grid_mapping[r, c]))
            active_distances.sort()
            assigned_class = active_distances[0][1]
            
        print(f"{test_id:<10}{x[0]:<10.4f}{x[1]:<10.4f}{x[2]:<10.4f}{winner+1:<20}{assigned_class:<15}")

if __name__ == "__main__":
    main()
