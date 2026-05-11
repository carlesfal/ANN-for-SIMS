"""
Custom backpropagation neural network for regression.

Pure NumPy implementation with:
- Dense layers (arbitrary depth/width)
- ReLU hidden activations, linear output
- L2 weight regularization
- Dropout (inverted, training-only)
- Adam optimiser
- Mini-batch SGD
- Early stopping on a validation set
- Model save / load (via numpy npz)
"""

import copy
import numpy as np


# ------------------------------------------------------------------ #
#  Activation helpers                                                  #
# ------------------------------------------------------------------ #

def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(z.dtype)


# ------------------------------------------------------------------ #
#  Dense layer                                                         #
# ------------------------------------------------------------------ #

class DenseLayer:
    """Single fully-connected layer with optional L2 regularization."""

    def __init__(self, n_in, n_out, activation="relu", l2_reg=0.0,
                 seed=None):
        rng = np.random.RandomState(seed)
        # He initialisation for ReLU layers
        scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
        self.W = rng.randn(n_in, n_out).astype(np.float64) * scale
        self.b = np.zeros((1, n_out), dtype=np.float64)

        self.activation = activation  # "relu" or "linear"
        self.l2_reg = l2_reg

        # Cache for forward/backward
        self.z = None   # pre-activation
        self.a = None   # post-activation (input to next layer)
        self.a_in = None  # input to this layer

        # Gradients
        self.dW = None
        self.db = None

    def forward(self, X):
        self.a_in = X
        self.z = X @ self.W + self.b
        if self.activation == "relu":
            self.a = relu(self.z)
        else:
            self.a = self.z.copy()
        return self.a

    def backward(self, dA):
        """
        Given dL/dA (gradient of loss w.r.t. this layer's output),
        compute dL/dW, dL/db, and return dL/dA_prev.
        """
        m = dA.shape[0]

        if self.activation == "relu":
            dZ = dA * relu_derivative(self.z)
        else:
            dZ = dA

        self.dW = (self.a_in.T @ dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m

        # L2 regularisation gradient
        if self.l2_reg > 0:
            self.dW += self.l2_reg * self.W

        dA_prev = dZ @ self.W.T
        return dA_prev


# ------------------------------------------------------------------ #
#  Dropout layer (inverted dropout)                                    #
# ------------------------------------------------------------------ #

class DropoutLayer:
    """Inverted dropout — scales activations at train time so that
    inference needs no adjustment."""

    def __init__(self, rate=0.0, seed=None):
        self.rate = rate          # probability of *dropping* a unit
        self.keep_prob = 1.0 - rate
        self.mask = None
        self.training = True
        self._rng = np.random.RandomState(seed)

    def forward(self, X):
        if self.training and self.rate > 0:
            self.mask = (self._rng.rand(*X.shape) < self.keep_prob).astype(X.dtype)
            return (X * self.mask) / self.keep_prob
        return X

    def backward(self, dA):
        if self.training and self.rate > 0 and self.mask is not None:
            return (dA * self.mask) / self.keep_prob
        return dA


# ------------------------------------------------------------------ #
#  Adam optimiser (per-parameter)                                      #
# ------------------------------------------------------------------ #

class AdamOptimizer:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self._states = {}  # keyed by id(param)

    def step(self, param, grad):
        pid = id(param)
        if pid not in self._states:
            self._states[pid] = {
                "m": np.zeros_like(param),
                "v": np.zeros_like(param),
            }
        self.t += 1
        s = self._states[pid]
        s["m"] = self.beta1 * s["m"] + (1 - self.beta1) * grad
        s["v"] = self.beta2 * s["v"] + (1 - self.beta2) * (grad ** 2)

        m_hat = s["m"] / (1 - self.beta1 ** self.t)
        v_hat = s["v"] / (1 - self.beta2 ** self.t)

        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# ------------------------------------------------------------------ #
#  Neural Network (assembles layers)                                   #
# ------------------------------------------------------------------ #

class NeuralNetwork:
    """
    Feed-forward neural network trained with backpropagation + Adam.

    Parameters
    ----------
    layer_sizes : list[int]
        Number of units per hidden layer, e.g. [256, 128, 64].
    n_inputs : int
        Dimensionality of the input features.
    dropout_rates : list[float] or float
        Per-layer dropout rate. A scalar is broadcast to all hidden layers.
    l2_reg : float
        L2 regularisation coefficient applied to every Dense weight matrix.
    learning_rate : float
        Adam learning rate.
    seed : int or None
        Reproducibility seed.
    """

    def __init__(self, layer_sizes, n_inputs, dropout_rates=0.0,
                 l2_reg=0.0, learning_rate=1e-3, seed=42):
        self.seed = seed
        self.learning_rate = learning_rate

        if isinstance(dropout_rates, (int, float)):
            dropout_rates = [float(dropout_rates)] * len(layer_sizes)

        self.layers = []          # interleaved Dense + Dropout
        self._dense_layers = []   # Dense-only references (convenience)
        rng = np.random.RandomState(seed)

        prev_size = n_inputs
        for i, units in enumerate(layer_sizes):
            layer_seed = int(rng.randint(0, 2**31))
            dense = DenseLayer(prev_size, units, activation="relu",
                               l2_reg=l2_reg, seed=layer_seed)
            self.layers.append(dense)
            self._dense_layers.append(dense)

            drop_rate = dropout_rates[i] if i < len(dropout_rates) else 0.0
            if drop_rate > 0:
                drop = DropoutLayer(rate=drop_rate,
                                    seed=int(rng.randint(0, 2**31)))
                self.layers.append(drop)

            prev_size = units

        # Output layer (linear, single unit for regression)
        out_seed = int(rng.randint(0, 2**31))
        output_layer = DenseLayer(prev_size, 1, activation="linear",
                                  l2_reg=0.0, seed=out_seed)
        self.layers.append(output_layer)
        self._dense_layers.append(output_layer)

        self.optimizer = AdamOptimizer(lr=learning_rate)

        # Training history
        self.history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

    # ---- mode toggle ------------------------------------------------ #

    def train_mode(self):
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                layer.training = True

    def eval_mode(self):
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                layer.training = False

    # ---- forward / predict ------------------------------------------ #

    def _forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, X):
        self.eval_mode()
        return self._forward(X)

    # ---- loss ------------------------------------------------------- #

    def _mse_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        data_loss = np.mean((y_pred - y_true) ** 2)

        # L2 penalty
        reg_loss = 0.0
        for layer in self._dense_layers:
            if layer.l2_reg > 0:
                reg_loss += layer.l2_reg * np.sum(layer.W ** 2)

        return data_loss + reg_loss

    @staticmethod
    def _mae(y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))

    # ---- backward --------------------------------------------------- #

    def _backward(self, y_pred, y_true):
        m = y_true.shape[0]
        # dL/d(y_pred) for MSE  = 2*(y_pred - y_true)/m
        dA = 2.0 * (y_pred - y_true) / m

        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    # ---- parameter update ------------------------------------------- #

    def _update_params(self):
        for layer in self._dense_layers:
            self.optimizer.step(layer.W, layer.dW)
            self.optimizer.step(layer.b, layer.db)

    # ---- training loop ---------------------------------------------- #

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=200, batch_size=32, patience=20,
            restore_best_weights=True, verbose=1):
        """
        Train the network with mini-batch backpropagation + Adam.

        Parameters
        ----------
        X_train, y_train : ndarray  (n, p) and (n, 1)
        X_val, y_val : ndarray or None
        epochs : int
        batch_size : int
        patience : int
            Early stopping patience (ignored if no validation data).
        restore_best_weights : bool
            If True, restore the weights from the epoch with the lowest
            validation loss when early stopping triggers (or at the end).
        verbose : int
            0 = silent, 1 = per-epoch summary.
        """
        n = X_train.shape[0]
        best_val_loss = np.inf
        wait = 0
        best_weights = None

        self.history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

        rng = np.random.RandomState(self.seed)

        for epoch in range(1, epochs + 1):
            self.train_mode()

            # Shuffle
            perm = rng.permutation(n)
            X_shuf = X_train[perm]
            y_shuf = y_train[perm]

            epoch_losses = []

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb = X_shuf[start:end]
                yb = y_shuf[start:end]

                y_pred = self._forward(Xb)
                loss = self._mse_loss(y_pred, yb)
                epoch_losses.append(loss)

                self._backward(y_pred, yb)
                self._update_params()

            # Epoch metrics (full train set, eval mode)
            self.eval_mode()
            y_train_pred = self._forward(X_train)
            train_loss = self._mse_loss(y_train_pred, y_train)
            train_mae = self._mae(y_train_pred, y_train)
            self.history["loss"].append(float(train_loss))
            self.history["mae"].append(float(train_mae))

            line = f"Epoch {epoch}/{epochs} - loss: {train_loss:.6f} - mae: {train_mae:.6f}"

            if X_val is not None and y_val is not None:
                y_val_pred = self._forward(X_val)
                val_loss = self._mse_loss(y_val_pred, y_val)
                val_mae = self._mae(y_val_pred, y_val)
                self.history["val_loss"].append(float(val_loss))
                self.history["val_mae"].append(float(val_mae))
                line += f" - val_loss: {val_loss:.6f} - val_mae: {val_mae:.6f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                    if restore_best_weights:
                        best_weights = self._get_weights()
                else:
                    wait += 1
                    if patience and wait >= patience:
                        if verbose:
                            print(f"{line}  ** early stop **")
                        break

            if verbose:
                print(line)

        # Restore best weights
        if restore_best_weights and best_weights is not None:
            self._set_weights(best_weights)

        return self.history

    # ---- weight snapshot / restore ---------------------------------- #

    def _get_weights(self):
        return [(layer.W.copy(), layer.b.copy()) for layer in self._dense_layers]

    def _set_weights(self, weights):
        for layer, (W, b) in zip(self._dense_layers, weights):
            layer.W = W.copy()
            layer.b = b.copy()

    def get_weights(self):
        return self._get_weights()

    def set_weights(self, weights):
        self._set_weights(weights)

    # ---- persistence ------------------------------------------------ #

    def save(self, path):
        """Save weights to a .npz file."""
        data = {}
        for i, layer in enumerate(self._dense_layers):
            data[f"W_{i}"] = layer.W
            data[f"b_{i}"] = layer.b
        np.savez(path, **data)

    def load(self, path):
        """Load weights from a .npz file."""
        if not path.endswith(".npz"):
            path += ".npz"
        data = np.load(path)
        for i, layer in enumerate(self._dense_layers):
            layer.W = data[f"W_{i}"]
            layer.b = data[f"b_{i}"]

    # ---- summary ---------------------------------------------------- #

    def summary(self):
        """Print a Keras-style summary of the architecture."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"{'Layer':<25} {'Output Shape':<20} {'Params':>10}")
        lines.append("=" * 60)
        total_params = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                n_params = layer.W.size + layer.b.size
                total_params += n_params
                act = layer.activation
                shape = f"(None, {layer.W.shape[1]})"
                name = f"Dense-{i} ({act})"
                lines.append(f"{name:<25} {shape:<20} {n_params:>10}")
            elif isinstance(layer, DropoutLayer):
                name = f"Dropout-{i} (rate={layer.rate:.1f})"
                lines.append(f"{name:<25} {'—':<20} {'0':>10}")
        lines.append("=" * 60)
        lines.append(f"Total trainable parameters: {total_params}")
        lines.append("=" * 60)
        text = "\n".join(lines)
        print(text)
        return text


# ------------------------------------------------------------------ #
#  Builder function (mirrors Keras-Tuner best_hp interface)            #
# ------------------------------------------------------------------ #

def build_backprop_model(best_hp, n_inputs):
    """
    Build a NeuralNetwork from a Keras-Tuner HyperParameters object
    (or a plain dict with the same keys).

    Expected keys:
        num_layers, units_0 … units_{num_layers-1},
        dropout_0 … dropout_{num_layers-1},
        l2_reg, lr
    """
    if hasattr(best_hp, "values"):
        hp = best_hp.values
    else:
        hp = best_hp

    n_layers = int(hp["num_layers"])
    layer_sizes = [int(hp[f"units_{i}"]) for i in range(n_layers)]
    dropout_rates = [float(hp.get(f"dropout_{i}", 0.0)) for i in range(n_layers)]
    l2_reg = float(hp.get("l2_reg", 0.0))
    lr = float(hp.get("lr", 1e-3))

    return NeuralNetwork(
        layer_sizes=layer_sizes,
        n_inputs=n_inputs,
        dropout_rates=dropout_rates,
        l2_reg=l2_reg,
        learning_rate=lr,
        seed=42,
    )
