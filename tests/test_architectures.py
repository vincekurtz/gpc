from pathlib import Path

import cloudpickle
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

from gpc.architectures import (
    MLP,
    DenoisingCNN,
    DenoisingMLP,
    PositionalEmbedding,
)


def test_mlp_construction() -> None:
    """Create a simple MLP and verify sizes."""
    batch_size = 10
    input_size = 2
    output_size = 3

    # Create the model
    model = MLP([input_size, 128, 32, output_size], nnx.Rngs(0))

    # Make sure the model is constructed correctly
    input = jnp.zeros((batch_size, input_size))
    output = model(input)
    assert output.shape == (batch_size, output_size)

    # Print a summary of the model
    nnx.display(model)


def test_mlp_save_load() -> None:
    """Verify that we can pickle an MLP."""
    layer_sizes = [2, 3, 4]
    mlp = MLP(layer_sizes, rngs=nnx.Rngs(1))
    dummy_input = jnp.ones((2,))
    original_output = mlp(dummy_input)

    # Create a temporary path for saving stuff
    local_dir = Path("_test_mlp")
    local_dir.mkdir(parents=True, exist_ok=True)

    model_path = local_dir / "mlp.pkl"
    with Path(model_path).open("wb") as f:
        cloudpickle.dump(mlp, f)

    with Path(model_path).open("rb") as f:
        model_restored = cloudpickle.load(f)

    # Check that the model is still functional
    restored_output = model_restored(dummy_input)
    assert jnp.allclose(original_output, restored_output)

    # Remove the temporary directory
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


def test_denoising_mlp() -> None:
    """Test the denoising MLP."""
    num_steps = 5
    action_dim = 3
    obs_dim = 4

    # Define the network architecture
    net = DenoisingMLP(action_dim, obs_dim, num_steps, (32, 32), nnx.Rngs(0))

    # Test on some data
    U = jnp.ones((num_steps, action_dim))
    y = jnp.ones(obs_dim)
    t = jnp.ones(1)
    U_out = net(U, y, t)
    assert U_out.shape == (num_steps, action_dim)

    # Test on some batched data
    U = jnp.ones((14, 24, num_steps, action_dim))
    y = jnp.ones((14, 24, obs_dim))
    t = jnp.ones((14, 24, 1))
    U_out = net(U, y, t)
    assert U_out.shape == (14, 24, num_steps, action_dim)


def test_positional_embedding() -> None:
    """Test our sinusoidal positional embedding."""
    dim = 8
    emb = PositionalEmbedding(dim)

    e = emb(jnp.zeros(1)[0])
    assert e.shape == (dim,)

    t = jnp.zeros((24, 14, 1))
    e = emb(t)
    assert e.shape == (24, 14, dim)

    t = jnp.linspace(0, 1, 100)
    e = emb(t)
    assert e.shape == (100, dim)

    if __name__ == "__main__":
        # Visualize the positional embedding
        plt.plot(t, e)
        plt.xlabel("Time")
        plt.ylabel("Positional Embedding")
        plt.title("Sinusoidal Positional Embedding")
        plt.show()


def test_denoising_cnn() -> None:
    """Test the denoising CNN."""
    num_steps = 5
    action_dim = 3
    obs_dim = 4

    # Define the network architecture
    net = DenoisingCNN(action_dim, obs_dim, num_steps, [32, 32], nnx.Rngs(0))

    # Test on some data
    U = jnp.ones((num_steps, action_dim))
    y = jnp.ones(obs_dim)
    t = jnp.ones(1)
    U_out = net(U, y, t)
    assert U_out.shape == (num_steps, action_dim)

    # Test on some batched data
    U = jnp.ones((14, 24, num_steps, action_dim))
    y = jnp.ones((14, 24, obs_dim))
    t = jnp.ones((14, 24, 1))
    U_out = net(U, y, t)
    assert U_out.shape == (14, 24, num_steps, action_dim)


if __name__ == "__main__":
    test_mlp_construction()
    test_mlp_save_load()
    test_denoising_mlp()
    test_positional_embedding()
    test_denoising_cnn()
