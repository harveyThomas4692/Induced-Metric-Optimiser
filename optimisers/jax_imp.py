import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from typing import NamedTuple, Optional


class SGDState(NamedTuple):
    step: jnp.ndarray
    momentum: jnp.ndarray  # Momentum buffer
    metric_ema: jnp.ndarray  # EMA of metric scale
    rms_ema: Optional[jnp.ndarray] = None  # EMA of RMS for custom_sgd_rms


def custom_sgd(learning_rate=0.1, momentum=0.9, xi=0.1, beta=0.1, weight_decay=0.0):
    """Optimized custom SGD with momentum and metric modification.
    """
    
    # Pre-compute constants
    neg_lr = -learning_rate
    one_minus_beta = 1 - beta
    one_minus_momentum = 1 - momentum

    def init(params):
        """Initialise optimiser state."""
        return SGDState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
            metric_ema=jnp.zeros([]),
            rms_ema=None
        )
    
    @jax.jit
    def update(grads, state, params=None):
        """Update parameters using SGD with momentum and metric modification."""
        step = state.step + 1
        
        # Use optax's optimised norm computation
        grad_norm_sq = optax.tree.norm(grads, ord=2, squared=True)
        trace = xi * grad_norm_sq
        
        # Update EMA of metric scale
        new_metric_ema = beta * state.metric_ema + one_minus_beta * trace
        
        # Bias correction and metric scale computation
        metric_corrected = new_metric_ema / (1 - beta ** step)
        metric_scale = 1 / (1 + jnp.abs(metric_corrected))
        
        
        # Combined momentum update and gradient scaling
        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + one_minus_momentum*g, 
            state.momentum, 
            grads
        )
        
        # Apply learning rate and weight decay (optax expects negative updates)
        updates = jax.tree.map(lambda m, p: neg_lr * metric_scale * m /(1 - momentum ** step) - learning_rate * weight_decay * p, new_momentum, params)

        new_state = SGDState(step=step, momentum=new_momentum, metric_ema=new_metric_ema, rms_ema=None)
        
        return updates, new_state
    
    return optax.GradientTransformation(init, update)

def custom_sgd_log(learning_rate=0.1, momentum=0.9, xi=0.1, beta=0.1, weight_decay=0.0):
    """Optimised custom SGD with loss-based metric modification."""
    
    # Pre-compute constants
    neg_lr = -learning_rate
    one_minus_beta = 1 - beta
    one_minus_momentum = 1 - momentum

    def init(params):
        """Initialise optimiser state."""
        return SGDState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
            metric_ema=jnp.zeros([]),
            rms_ema=None
        )
    
    @jax.jit
    def update(grads, state, loss, params=None):
        """Update parameters using SGD with momentum and metric modification."""
        step = state.step + 1

        # Use optax's optimised norm computation
        grad_norm_sq = optax.tree.norm(grads, ord=2, squared=True)
        trace = xi * grad_norm_sq
        
        # Update EMA of metric scale
        new_metric_ema = beta * state.metric_ema + one_minus_beta * trace
        
        # Bias correction and loss-based metric scale computation
        metric_corrected = new_metric_ema / (1 - beta ** step)
        metric_scale = loss / (jnp.square(loss) + metric_corrected)
        
        # Combined momentum update and gradient scaling
        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + one_minus_momentum * g, 
            state.momentum, 
            grads
        )
        
        # Apply learning rate and weight decay (optax expects negative updates)
        updates = jax.tree.map(lambda m, p: neg_lr * metric_scale * m / (1 - momentum ** step) - learning_rate * weight_decay * p, new_momentum, params)

        new_state = SGDState(step=step, momentum=new_momentum, metric_ema=new_metric_ema, rms_ema=None)
        
        return updates, new_state
    
    return optax.GradientTransformation(init, update)

def custom_sgd_rms(learning_rate=0.1, momentum=0.9, xi=0.1, beta=0.1, beta_rms=0.99, weight_decay=0.0, eps=1e-8):
    """custom SGD with momentum and metric modification. Gradients are scaled by RMS."""
    
    # Pre-compute constants
    neg_lr = -learning_rate
    one_minus_beta = 1 - beta
    one_minus_beta_rms = 1 - beta_rms
    one_minus_momentum = 1 - momentum
    
    def init(params):
        """Initialise optimiser state."""
        return SGDState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
            metric_ema=jnp.zeros([]),
            rms_ema=jax.tree.map(jnp.zeros_like, params)
        )
    
    @jax.jit
    def update(grads, state, params=None):
        """Update parameters using SGD with momentum and metric modification."""
        step = state.step + 1

        # Update EMA of RMS (Root Mean Square) - per parameter
        new_rms_ema = jax.tree.map(
            lambda r, g: beta_rms * r + one_minus_beta_rms * (g ** 2),
            state.rms_ema, grads
        )

        # Bias correction for RMS
        rms_corrected = jax.tree.map(
            lambda r: r / (1 - beta_rms ** step),
            new_rms_ema
        )

        # Calculate gradient norm
        grad_norm_sq = jax.tree_util.tree_reduce(
        lambda acc, g_r_pair: acc + jnp.sum(g_r_pair),
        jax.tree.map(lambda g, r: g ** 2 / (jnp.sqrt(r) + eps), grads, rms_corrected),
        initializer=0.0)
        trace = xi * grad_norm_sq
        
        # Update EMA of metric scale
        new_metric_ema = beta * state.metric_ema + one_minus_beta * trace

        # Bias correction and metric scale computation
        metric_corrected = new_metric_ema / (1 - beta ** step)
        metric_scale = 1 / (1 + jnp.abs(metric_corrected))

        # Combined momentum update and gradient scaling with RMS normalisation
        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + one_minus_momentum * g, 
            state.momentum, 
            grads
        )
        
        # Apply learning rate and weight decay (optax expects negative updates)
        updates = jax.tree.map(lambda m, p, r: neg_lr * metric_scale * m / ((1 - momentum ** step)*(jnp.sqrt(r) + eps))
                               - learning_rate * weight_decay * p, new_momentum, params, rms_corrected)

        new_state = SGDState(step=step, momentum=new_momentum, metric_ema=new_metric_ema, rms_ema=new_rms_ema)
        
        return updates, new_state
    
    return optax.GradientTransformation(init, update)