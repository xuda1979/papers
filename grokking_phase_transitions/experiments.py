import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Configuration
P = 53
HIDDEN_DIM = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1.0  # High weight decay is key for grokking
EPOCHS = 10000
TRAIN_FRACTION = 0.5
SEED = 42

np.random.seed(SEED)

def one_hot(x, p):
    res = np.zeros((len(x), p))
    res[np.arange(len(x)), x] = 1
    return res

# Dataset Generation: Modular Addition
# Input: [one_hot(a), one_hot(b)]
# Output: one_hot((a + b) % P)
pairs = np.array([[i, j] for i in range(P) for j in range(P)])
labels = (pairs[:, 0] + pairs[:, 1]) % P

# Shuffle and Split
perm = np.random.permutation(len(pairs))
pairs = pairs[perm]
labels = labels[perm]

n_train = int(len(pairs) * TRAIN_FRACTION)
train_pairs = pairs[:n_train]
train_labels = labels[:n_train]
test_pairs = pairs[n_train:]
test_labels = labels[n_train:]

# Precompute inputs
X_train = np.concatenate([one_hot(train_pairs[:, 0], P), one_hot(train_pairs[:, 1], P)], axis=1)
Y_train = one_hot(train_labels, P)
X_test = np.concatenate([one_hot(test_pairs[:, 0], P), one_hot(test_pairs[:, 1], P)], axis=1)
Y_test = test_labels # Keep as indices for accuracy check

print(f"Task: Modular Addition (mod {P})")
print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# Model Parameters
# W1: (2*P, H), b1: (H)
# W2: (H, P), b2: (P)
scale = 1.0 / np.sqrt(2*P)
W1 = np.random.randn(2*P, HIDDEN_DIM) * scale
b1 = np.zeros(HIDDEN_DIM)
scale2 = 1.0 / np.sqrt(HIDDEN_DIM)
W2 = np.random.randn(HIDDEN_DIM, P) * scale2
b2 = np.zeros(P)

params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

# Adam Optimizer State
m = {k: np.zeros_like(v) for k, v in params.items()}
v = {k: np.zeros_like(v) for k, v in params.items()}
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    # Stabilize softmax
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def forward(params, X):
    z1 = X @ params['W1'] + params['b1']
    a1 = relu(z1)
    z2 = a1 @ params['W2'] + params['b2']
    probs = softmax(z2)
    return probs, a1, z1

def cross_entropy_loss(probs, Y_one_hot):
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.mean(np.sum(Y_one_hot * np.log(probs), axis=1))

def accuracy(probs, Y_indices):
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == Y_indices)

# Training Loop
train_losses = []
test_accs = []
train_accs = []
norms = []

print("Starting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    # Forward
    probs, a1, z1 = forward(params, X_train)

    # Loss
    loss = cross_entropy_loss(probs, Y_train)

    # Backward
    N = X_train.shape[0]
    dz2 = probs - Y_train # (N, P)
    dW2 = (a1.T @ dz2) / N # (H, P)
    db2 = np.mean(dz2, axis=0) # (P)

    da1 = dz2 @ params['W2'].T # (N, H)
    dz1 = da1 * (z1 > 0) # ReLU derivative
    dW1 = (X_train.T @ dz1) / N # (2P, H)
    db1 = np.mean(dz1, axis=0) # (H)

    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    # AdamW Step
    # Adam update + Weight Decay (decoupled)
    # theta = theta - lr * grad_scaling - lr * wd * theta

    for k in params.keys():
        # Weight decay
        params[k] -= LEARNING_RATE * WEIGHT_DECAY * params[k]

        # Adam
        m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
        v[k] = beta2 * v[k] + (1 - beta2) * (grads[k]**2)

        m_hat = m[k] / (1 - beta1**(epoch+1))
        v_hat = v[k] / (1 - beta2**(epoch+1))

        params[k] -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + epsilon)

    # Logging
    if epoch % 100 == 0:
        # Evaluate
        train_acc = accuracy(probs, np.argmax(Y_train, axis=1))

        test_probs, _, _ = forward(params, X_test)
        test_acc = accuracy(test_probs, Y_test)

        param_norm = np.sqrt(np.sum(params['W1']**2) + np.sum(params['W2']**2))

        train_losses.append(loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        norms.append(param_norm)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss {loss:.4f}, Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}, Norm {param_norm:.2f}")

        # Stop if solved
        if train_acc > 0.99 and test_acc > 0.99:
            print(f"Converged at epoch {epoch}")
            break

total_time = time.time() - start_time
print(f"Training finished in {total_time:.2f}s")

# Plotting
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "grokking_plot.png")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Accuracy', color='blue')
plt.plot(test_accs, label='Test Accuracy', color='green')
plt.xlabel('Epochs (x100)')
plt.ylabel('Accuracy')
plt.title('Grokking: Generalization Lag')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(norms, label='L2 Norm of Weights', color='red')
plt.xlabel('Epochs (x100)')
plt.ylabel('Norm')
plt.title('Weight Norm Dynamics')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(save_path)
print(f"Plot saved to {save_path}")
