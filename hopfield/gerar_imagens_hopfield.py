import os
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, num_neurons: int = 45) -> None:
        self.num_neurons = num_neurons
        self.W = np.zeros((num_neurons, num_neurons), dtype=np.float64)
        
    def train(self, patterns) -> None:
        self.W = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float64)
        for p in patterns:
            self.W += np.outer(p, p)
        np.fill_diagonal(self.W, 0.0)
        
    def add_noise(self, state, noise_level: float = 0.20, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        noisy_state = state.copy()
        num_flips = int(round(self.num_neurons * noise_level))
        flip_indices = np.random.choice(self.num_neurons, num_flips, replace=False)
        for idx in flip_indices:
            noisy_state[idx] = -noisy_state[idx]
        return noisy_state
        
    def reconstruct(self, initial_state, max_epochs: int = 100, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        state = initial_state.copy().astype(np.float64)
        
        for epoch in range(max_epochs):
            prev_state = state.copy()
            order = np.random.permutation(self.num_neurons)
            for i in order:
                activation = np.dot(self.W[i], state)
                state[i] = 1.0 if activation >= 0.0 else -1.0
                
            if np.array_equal(state, prev_state):
                return state, epoch + 1
        return state, max_epochs

def plot_grid(ax, state, title):
    grid = state.reshape((9, 5))
    ax.imshow(grid, cmap="gray_r", vmin=-1, vmax=1)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    # Adicionar gridlines discretas
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 9, 1), minor=True)
    ax.grid(which='minor', color='#999999', linestyle='-', linewidth=0.5)

def main():
    # Criar pasta para imagens se não existir
    os.makedirs("imagens", exist_ok=True)
    
    # 1. Instanciar e Treinar a Rede
    net = HopfieldNetwork(num_neurons=45)
    net.train(PATTERNS)
    
    # Executar as simulações e salvar as imagens
    for p_idx, pattern in enumerate(PATTERNS):
        name = PATTERN_NAMES[p_idx]
        for sim_idx in range(3):
            seed = 42 + p_idx * 10 + sim_idx
            noisy_state = net.add_noise(pattern, noise_level=0.20, seed=seed)
            recovered_state, epochs = net.reconstruct(noisy_state, seed=seed)
            
            # Plotar a imagem
            fig, axes = plt.subplots(1, 3, figsize=(6, 2.8))
            plot_grid(axes[0], pattern, "Original")
            plot_grid(axes[1], noisy_state, "Com Ruído (20%)")
            plot_grid(axes[2], recovered_state, f"Recuperado ({epochs} ep.)")
            
            fig.suptitle(f"{name} - Situação {sim_idx + 1}", fontsize=12, fontweight="bold", y=0.98)
            plt.tight_layout()
            
            img_path = f"imagens/digit_{p_idx+1}_sim_{sim_idx+1}.png"
            plt.savefig(img_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Salva: {img_path}")
            
    # Executar simulações adicionais com ruído excessivo para demonstrar limites
    for noise in [0.45, 0.60]:
        seed = 100
        noisy_state = net.add_noise(p3, noise_level=noise, seed=seed)
        recovered_state, epochs = net.reconstruct(noisy_state, seed=seed)
        
        matched_name = "Estado Espúrio"
        for idx, p in enumerate(PATTERNS):
            if np.array_equal(recovered_state, p):
                matched_name = PATTERN_NAMES[idx]
                break
                
        fig, axes = plt.subplots(1, 3, figsize=(6, 2.8))
        plot_grid(axes[0], p3, "Original (Dígito 3)")
        plot_grid(axes[1], noisy_state, f"Ruído ({int(noise*100)}%)")
        plot_grid(axes[2], recovered_state, f"Resultado ({matched_name})")
        fig.suptitle(f"Dígito 3 - Teste de Ruído a {int(noise*100)}%", fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout()
        
        img_path = f"imagens/digit_3_noise_{int(noise*100)}.png"
        plt.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Salva: {img_path}")

if __name__ == "__main__":
    main()
