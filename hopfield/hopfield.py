import numpy as np
from typing import List, Tuple, Optional

# Definir os padrões originais de 45 bits (9 linhas x 5 colunas)
# Pixel branco = -1, Pixel escuro = 1

p1 = np.array([
    [-1, -1,  1,  1, -1],
    [-1,  1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1]
]).flatten()

p2 = np.array([
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [ 1,  1, -1, -1, -1],
    [ 1,  1, -1, -1, -1],
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1]
]).flatten()

p3 = np.array([
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1]
]).flatten()

p4 = np.array([
    [ 1,  1, -1,  1,  1],
    [ 1,  1, -1,  1,  1],
    [ 1,  1, -1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1]
]).flatten()

PATTERNS = [p1, p2, p3, p4]
PATTERN_NAMES = ["Dígito 1", "Dígito 2", "Dígito 3", "Dígito 4"]

class HopfieldNetwork:
    """
    Rede de Hopfield com 45 neurônios para armazenar e recuperar
    padrões binários de tamanho 9x5.
    """
    def __init__(self, num_neurons: int = 45) -> None:
        self.num_neurons = num_neurons
        self.W = np.zeros((num_neurons, num_neurons), dtype=np.float64)
        
    def train(self, patterns: List[np.ndarray]) -> None:
        """
        Treina a rede de Hopfield usando a regra de Hebbian de produto externo.
        Zera a diagonal principal para desativar auto-associação.
        """
        self.W = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float64)
        for p in patterns:
            self.W += np.outer(p, p)
        # Zera diagonal
        np.fill_diagonal(self.W, 0.0)
        
    def add_noise(self, state: np.ndarray, noise_level: float = 0.20, seed: Optional[int] = None) -> np.ndarray:
        """
        Inverte aleatoriamente a polaridade de um percentual (noise_level) de bits.
        """
        if seed is not None:
            np.random.seed(seed)
        noisy_state = state.copy()
        num_flips = int(round(self.num_neurons * noise_level))
        flip_indices = np.random.choice(self.num_neurons, num_flips, replace=False)
        for idx in flip_indices:
            noisy_state[idx] = -noisy_state[idx]
        return noisy_state
        
    def reconstruct(self, initial_state: np.ndarray, max_epochs: int = 100, seed: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Executa a decodificação assíncrona até estabilizar ou atingir o limite.
        Retorna o estado final estável e o número de épocas percorridas.
        """
        if seed is not None:
            np.random.seed(seed)
        state = initial_state.copy().astype(np.float64)
        
        for epoch in range(max_epochs):
            prev_state = state.copy()
            # Ordem aleatória de atualização para garantia de convergência assíncrona
            order = np.random.permutation(self.num_neurons)
            for i in order:
                activation = np.dot(self.W[i], state)
                # Função de ativação limite (Equivalente a tanh com beta -> inf)
                state[i] = 1.0 if activation >= 0.0 else -1.0
                
            if np.array_equal(state, prev_state):
                return state, epoch + 1
        return state, max_epochs

def get_grid_str(state: np.ndarray) -> str:
    """
    Retorna uma representação visual 9x5 em formato string.
    """
    lines = []
    for r in range(9):
        row = state[r*5 : (r+1)*5]
        line_str = " ".join(["#" if val == 1 else "." for val in row])
        lines.append(line_str)
    return "\n".join(lines)

def run_simulation() -> None:
    print("===================================================================")
    print(" Simulação da Rede de Hopfield - Memória Associativa (45 Neurônios)")
    print("===================================================================")
    
    # 1. Instanciar e Treinar a Rede
    net = HopfieldNetwork(num_neurons=45)
    net.train(PATTERNS)
    print("Rede treinada com os 4 padrões de dígitos (1, 2, 3, 4).\n")
    
    # 2. Rodar 12 Simulações (3 por padrão)
    results_report = []
    
    for p_idx, pattern in enumerate(PATTERNS):
        name = PATTERN_NAMES[p_idx]
        print(f"--- Simulações para {name} ---")
        for sim_idx in range(3):
            # Usar sementes fixas para reprodutibilidade das distorções
            seed = 42 + p_idx * 10 + sim_idx
            
            # Adicionar 20% de ruído (9 bits invertidos)
            noisy_state = net.add_noise(pattern, noise_level=0.20, seed=seed)
            
            # Recuperar imagem
            recovered_state, epochs = net.reconstruct(noisy_state, seed=seed)
            
            # Verificar se a recuperação foi perfeita
            success = np.array_equal(recovered_state, pattern)
            
            print(f"  Situação {sim_idx + 1}:")
            print("  [Distorcido]      [Recuperado]")
            
            noisy_lines = get_grid_str(noisy_state).split('\n')
            rec_lines = get_grid_str(recovered_state).split('\n')
            
            for nl, rl in zip(noisy_lines, rec_lines):
                print(f"  {nl}     {rl}")
                
            print(f"  Convergência em {epochs} épocas. Sucesso: {success}\n")
            
            results_report.append({
                "pattern": name,
                "sim": sim_idx + 1,
                "original": pattern,
                "noisy": noisy_state,
                "recovered": recovered_state,
                "epochs": epochs,
                "success": success
            })
            
    # 3. Teste com nível de ruído excessivo (ex: 45% e 60%)
    print("--- Teste de Limites de Ruído ---")
    for noise in [0.45, 0.60]:
        seed = 100
        noisy_state = net.add_noise(p3, noise_level=noise, seed=seed)
        recovered_state, epochs = net.reconstruct(noisy_state, seed=seed)
        success = np.array_equal(recovered_state, p3)
        print(f"  Ruído: {noise*100:.0f}% | Sucesso de recuperação do Dígito 3: {success} ({epochs} épocas)")
        if not success:
            # Encontrar qual padrão mais se assemelha à saída ou se é um estado espúrio
            matches = [np.array_equal(recovered_state, p) for p in PATTERNS]
            if any(matches):
                matched_name = PATTERN_NAMES[matches.index(True)]
                print(f"    -> Rede convergiu incorretamente para o atrator: {matched_name}")
            else:
                print("    -> Rede convergiu para um estado espúrio (mínimo local incorreto).")
    print()

if __name__ == "__main__":
    run_simulation()
