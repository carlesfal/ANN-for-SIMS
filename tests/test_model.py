from __future__ import annotations

import importlib.util
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from app.models.ann_model import ANN
    from app.models.trainer import Trainer, TrainingConfig


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for model tests")
class TestModel(unittest.TestCase):
    def test_ann_forward_shape(self) -> None:
        model = ANN(input_dim=1, output_dim=1, hidden_layers=[8, 4], activation="relu")
        x = torch.randn(10, 1)
        y = model(x)
        self.assertEqual(tuple(y.shape), (10, 1))

    def test_training_runs(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.random((100, 1))
        y = 3 * x.squeeze() + 1

        config = TrainingConfig(
            hidden_layers=[8, 4],
            activation="relu",
            optimizer="adam",
            learning_rate=0.01,
            epochs=5,
            batch_size=16,
        )
        trainer = Trainer()
        model, history = trainer.train(x[:80], y[:80], x[80:], y[80:], config)

        self.assertIsNotNone(model)
        self.assertEqual(len(history["train_loss"]), config.epochs)
        self.assertEqual(len(history["val_loss"]), config.epochs)


if __name__ == "__main__":
    unittest.main()
