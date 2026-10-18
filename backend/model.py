"""
model.py - a small neural network written from scratch with NumPy.

Architecture (a Multi-Layer Perceptron):

    input (1029 numbers)
       -> Dense 128 + ReLU
       -> Dense 64  + ReLU
       -> Dense 4   + Softmax   (question / answer / header / other)

It contains everything asked for in the project brief:
forward propagation, activation functions, loss function,
backpropagation, an optimizer (Adam) and prediction.
"""
import json
import numpy as np


def relu(z):
    """ReLU activation: keeps positive numbers, turns negatives into 0."""
    return np.maximum(0, z)


def softmax(z):
    """Softmax turns 4 raw scores into 4 probabilities that add up to 1."""
    z = z - z.max(axis=1, keepdims=True)  # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class FormFieldNet:
    def __init__(self, layer_sizes, seed=42):
        """layer_sizes example: [1029, 128, 64, 4]"""
        self.layer_sizes = list(layer_sizes)
        rng = np.random.RandomState(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialisation works well with ReLU
            self.W.append((rng.randn(n_in, n_out) * np.sqrt(2.0 / n_in)).astype(np.float32))
            self.b.append(np.zeros(n_out, dtype=np.float32))
        # Adam optimizer memory
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]
        self._t = 0

    # ------------------------------------------------ forward propagation
    def forward(self, X):
        """Pass the data through all layers. Returns probabilities and cached layer inputs."""
        activations = [X]
        a = X
        last = len(self.W) - 1
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            a = softmax(z) if i == last else relu(z)
            activations.append(a)
        return a, activations

    # ------------------------------------------------------- loss function
    @staticmethod
    def loss(probs, y, class_weights=None):
        """
        Cross-entropy loss: -log(probability given to the correct class).
        class_weights make rare classes count more (the dataset is imbalanced).
        """
        n = len(y)
        nll = -np.log(probs[np.arange(n), y] + 1e-9)
        if class_weights is not None:
            nll = nll * class_weights[y]
        return float(nll.mean())

    def compute_loss(self, X, y, class_weights=None):
        probs, _ = self.forward(X)
        return self.loss(probs, y, class_weights)

    # ------------------------------------------------------ backpropagation
    def backward(self, activations, y, class_weights=None):
        """Compute the gradient of the loss for every weight and bias."""
        n = len(y)
        probs = activations[-1]
        # For softmax + cross-entropy the output error is (probs - one_hot)
        delta = probs.copy()
        delta[np.arange(n), y] -= 1.0
        if class_weights is not None:
            delta *= class_weights[y][:, None]
        delta /= n

        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            grads_W[i] = activations[i].T @ delta
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (activations[i] > 0)  # ReLU derivative
        return grads_W, grads_b

    # --------------------------------------------------- optimizer (Adam)
    def update(self, grads_W, grads_b, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        params = self.W + self.b
        grads = grads_W + grads_b
        for k, (p, g) in enumerate(zip(params, grads)):
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * (g * g)
            m_hat = self._m[k] / (1 - beta1 ** self._t)
            v_hat = self._v[k] / (1 - beta2 ** self._t)
            p -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32)  # in-place

    def train_step(self, X, y, class_weights=None, lr=0.001):
        """One mini-batch step: forward -> loss -> backward -> update."""
        probs, acts = self.forward(X)
        batch_loss = self.loss(probs, y, class_weights)
        gW, gb = self.backward(acts, y, class_weights)
        self.update(gW, gb, lr)
        return batch_loss

    # ------------------------------------------------------------ prediction
    def predict_proba(self, X):
        probs, _ = self.forward(X)
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    # ----------------------------------------------------------- save / load
    def save(self, path, extra_info=None):
        arrays = {"layer_sizes": np.array(self.layer_sizes), "info": np.array(json.dumps(extra_info or {}))}
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            arrays[f"W{i}"] = w
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        net = cls([int(v) for v in data["layer_sizes"]])
        for i in range(len(net.W)):
            net.W[i] = data[f"W{i}"].astype(np.float32)
            net.b[i] = data[f"b{i}"].astype(np.float32)
        net.info = json.loads(str(data["info"]))
        return net
