import numpy as np
import csv
from mlp_energia import train_mlp, sigmoid

def main():
    # Load train data
    X_list = []
    D_list = []
    with open('dataset_pmc1.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            X_list.append([float(row[1]), float(row[2]), float(row[3])])
            D_list.append([float(row[4])])
    X_train = np.array(X_list)
    D_train = np.array(D_list)

    # Load test data
    X_test_list = []
    D_test_list = []
    with open('test_data.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                X_test_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
                D_test_list.append([float(parts[4])])
    X_test = np.array(X_test_list)
    D_test = np.array(D_test_list)

    # Train 5 models
    models = []
    print("Training 5 models...")
    for i in range(5):
        np.random.seed(42 + i)
        W1, b1, W2, b2, _, _, _, _ = train_mlp(X_train, D_train, num_hidden=10, eta=0.1, epsilon=1e-6, max_epochs=100000)
        models.append((W1, b1, W2, b2))
        print(f"Model {i+1} trained.")

    # Predict on test data
    y_preds = []
    for W1, b1, W2, b2 in models:
        V1 = np.dot(X_test, W1) + b1
        Y1 = sigmoid(V1)
        V2 = np.dot(Y1, W2) + b2
        Y2 = sigmoid(V2)
        y_preds.append(Y2)

    # Generate table
    print("\n| Amostra | $x_1$ | $x_2$ | $x_3$ | $d$ | $y_{rede}$ (T1) | $y_{rede}$ (T2) | $y_{rede}$ (T3) | $y_{rede}$ (T4) | $y_{rede}$ (T5) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for i in range(len(X_test)):
        row = f"| {i+1} | {X_test[i][0]:.4f} | {X_test[i][1]:.4f} | {X_test[i][2]:.4f} | {D_test[i][0]:.4f} "
        for j in range(5):
            row += f"| {y_preds[j][i][0]:.4f} "
        row += "|"
        print(row)

    # Calculate relative error and variance
    mean_errs = []
    var_errs = []
    for j in range(5):
        # Rel error = |d - y| / |d| * 100
        errors = np.abs(D_test - y_preds[j]) / np.abs(D_test) * 100
        mean_err = np.mean(errors)
        var_err = np.var(errors)
        mean_errs.append(mean_err)
        var_errs.append(var_err)

    print(f"| **Erro Relativo Médio (%)** | - | - | - | - | **{mean_errs[0]:.4f}%** | **{mean_errs[1]:.4f}%** | **{mean_errs[2]:.4f}%** | **{mean_errs[3]:.4f}%** | **{mean_errs[4]:.4f}%** |")
    print(f"| **Variância (%)** | - | - | - | - | **{var_errs[0]:.4f}%** | **{var_errs[1]:.4f}%** | **{var_errs[2]:.4f}%** | **{var_errs[3]:.4f}%** | **{var_errs[4]:.4f}%** |")

if __name__ == "__main__":
    main()
