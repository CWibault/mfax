import jax
import jax.numpy as jnp


def distribute(discrete_states, continuous_state):
    """
    Differentiable (piecewise-linear) distribution of a scalar onto a 1D grid.

    Notes
    - We treat the selected bin index as non-differentiable (stop-gradient),
      and differentiate only through the linear interpolation weight. This
      makes the output differentiable a.e. w.r.t. `continuous_state`.
    Args:
        discrete_states: Discrete states to round the continuous states to. Shape (n_discrete_states,).
        continuous_state: Scalar continuous state to round.
    Returns:
        discretized_state_idxs: Indices of the discretized states. Shape (2,).
        probs: Probabilities of the discretized states. Shape (2,).
    """
    discrete_states = jnp.asarray(discrete_states)
    x = jnp.asarray(continuous_state)

    # --- bin selection (non-differentiable): choose i such that x ∈ [s[i], s[i+1]] ---
    i = jnp.searchsorted(discrete_states, x, side="right") - 1
    i = jnp.clip(i, 0, discrete_states.shape[0] - 2).astype(jnp.int32)
    i = jax.lax.stop_gradient(i)

    s0 = discrete_states[i]
    s1 = discrete_states[i + 1]
    denom = jnp.maximum(s1 - s0, jnp.finfo(s0.dtype).eps)

    # --- linearly interpolated weight for upper grid point ---
    t = (x - s0) / denom
    t = jnp.clip(t, 0.0, 1.0)

    # --- weights correspond to indices [i, i+1] as [lower, upper] ---
    probs = jnp.stack([1.0 - t, t], axis=0).astype(jnp.float32)
    idxs = jnp.stack([i, i + 1], axis=0)
    return idxs, probs
