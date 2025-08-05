"""
JAX implementations of custom SGD optimizers with induced metric modifications.

This module provides JAX-based optimizers using the optax framework:
- custom_sgd: Basic custom SGD with momentum and metric modification
- custom_sgd_log: Custom SGD with loss-based metric scaling
- custom_sgd_rms: Custom SGD with RMS gradient scaling

Usage examples:

# Basic custom SGD
optimizer = custom_sgd(learning_rate=0.01, momentum=0.9, xi=0.1, beta=0.1)
opt_state = optimizer.init(params)
updates, opt_state = optimizer.update(grads, opt_state, params)

# Custom SGD with loss-based metric (requires passing loss to update())
optimizer = custom_sgd_log(learning_rate=0.01, momentum=0.9, xi=0.1, beta=0.1)
opt_state = optimizer.init(params)
updates, opt_state = optimizer.update(grads, opt_state, loss, params)

# Custom SGD with RMS scaling
optimizer = custom_sgd_rms(learning_rate=0.01, momentum=0.9, xi=0.1, beta=0.1, beta_rms=0.99)
opt_state = optimizer.init(params)
updates, opt_state = optimizer.update(grads, opt_state, params)

Note: All optimizers follow the optax GradientTransformation interface for compatibility
with JAX training loops.
"""

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from typing import NamedTuple, Optional


class SGDState(NamedTuple):
    """State container for custom SGD optimizers.
    
    Attributes:
        step: Current optimization step count
        momentum: Momentum buffer storing accumulated gradients
        metric_ema: Exponential moving average of the metric scale
        rms_ema: Optional EMA of squared gradients for RMS scaling (used in custom_sgd_rms)
    """
    step: jnp.ndarray
    momentum: jnp.ndarray  # Momentum buffer
    metric_ema: jnp.ndarray  # EMA of metric scale
    rms_ema: Optional[jnp.ndarray] = None  # EMA of RMS for custom_sgd_rms


def custom_sgd(learning_rate=0.1, momentum=0.9, xi=0.1, beta=0.8, weight_decay=0.0):
    """Custom SGD optimizer with momentum and metric modification.
    
    This optimizer implements SGD with momentum while dynamically adjusting the learning
    rate based on gradient norms. The metric modification scales updates based on an
    exponential moving average of gradient magnitudes.
    
    Args:
        learning_rate: Base learning rate (default: 0.1)
        momentum: Momentum factor for gradient accumulation (default: 0.9)
        xi: Scaling factor for gradient norm in metric computation (default: 0.1)
        beta: EMA decay rate for metric scale tracking (default: 0.8)
        weight_decay: Weight decay (L2 regularization) factor (default: 0.0)
        
    Returns:
        optax.GradientTransformation: JAX optimizer that can be used with optax
    """
    
    # Pre-compute constants for efficiency
    neg_lr = -learning_rate  # Negative for parameter updates
    one_minus_beta = 1 - beta  # For EMA computation
    one_minus_momentum = 1 - momentum  # For momentum computation

    def init(params):
        """Initialize optimizer state.
        
        Args:
            params: Model parameters to optimize
            
        Returns:
            SGDState: Initial optimizer state
        """
        return SGDState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
            metric_ema=jnp.zeros([]),
            rms_ema=None
        )
    
    @jax.jit
    def update(grads, state, params=None):
        """Update parameters using custom SGD with momentum and metric modification.
        
        Args:
            grads: Current gradients
            state: Current optimizer state
            params: Current parameter values (optional, used for weight decay)
            
        Returns:
            Tuple of (updates, new_state) where updates are parameter changes
        """
        step = state.step + 1
        
        # Calculate gradient norm squared across all parameters
        grad_norm_sq = optax.tree.norm(grads, ord=2, squared=True)
        trace = xi * grad_norm_sq
        
        # Update EMA of metric scale
        new_metric_ema = beta * state.metric_ema + one_minus_beta * trace
        
        # Apply bias correction and compute metric scale
        metric_corrected = new_metric_ema / (1 - beta ** step)
        metric_scale = 1 / (1 + jnp.abs(metric_corrected))
        
        # Update momentum buffer with current gradients
        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + one_minus_momentum*g, 
            state.momentum, 
            grads
        )
        
        # Apply bias correction to momentum and compute parameter updates
        # Include learning rate, metric scaling, and weight decay
        updates = jax.tree.map(lambda m, p: neg_lr * metric_scale * m /(1 - momentum ** step) - learning_rate * weight_decay * p, new_momentum, params)

        # Create new optimizer state
        new_state = SGDState(step=step, momentum=new_momentum, metric_ema=new_metric_ema, rms_ema=None)
        
        return updates, new_state
    
    return optax.GradientTransformation(init, update)

def custom_sgd_log(learning_rate=0.1, momentum=0.9, xi=0.1, beta=0.8, weight_decay=0.0):
    """Custom SGD optimizer with loss-based metric modification.
    
    This optimizer extends the basic custom SGD by incorporating the loss value
    into the metric scaling computation. The metric scale is computed as a function
    of both the gradient norms and the current loss value.
    
    Args:
        learning_rate: Base learning rate (default: 0.1)
        momentum: Momentum factor for gradient accumulation (default: 0.9)
        xi: Scaling factor for gradient norm in metric computation (default: 0.1)
        beta: EMA decay rate for metric scale tracking (default: 0.8)
        weight_decay: Weight decay (L2 regularization) factor (default: 0.0)
        
    Returns:
        optax.GradientTransformation: JAX optimizer that can be used with optax
    """
    
    # Pre-compute constants for efficiency
    neg_lr = -learning_rate  # Negative for parameter updates
    one_minus_beta = 1 - beta  # For EMA computation
    one_minus_momentum = 1 - momentum  # For momentum computation

    def init(params):
        """Initialize optimizer state.
        
        Args:
            params: Model parameters to optimize
            
        Returns:
            SGDState: Initial optimizer state
        """
        return SGDState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
            metric_ema=jnp.zeros([]),
            rms_ema=None
        )
    
    @jax.jit
    def update(grads, state, loss, params=None):
        """Update parameters using custom SGD with loss-based metric modification.
        
        Args:
            grads: Current gradients
            state: Current optimizer state
            loss: Current loss value (required for metric computation)
            params: Current parameter values (optional, used for weight decay)
            
        Returns:
            Tuple of (updates, new_state) where updates are parameter changes
        """
        step = state.step + 1

        # Calculate gradient norm squared across all parameters
        grad_norm_sq = optax.tree.norm(grads, ord=2, squared=True)
        trace = xi * grad_norm_sq
        
        # Update EMA of metric scale
        new_metric_ema = beta * state.metric_ema + one_minus_beta * trace
        
        # Apply bias correction and compute loss-based metric scale
        metric_corrected = new_metric_ema / (1 - beta ** step)
        metric_scale = loss / (jnp.square(loss) + metric_corrected)
        
        # Update momentum buffer with current gradients
        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + one_minus_momentum * g, 
            state.momentum, 
            grads
        )
        
        # Apply bias correction to momentum and compute parameter updates
        # Include learning rate, metric scaling, and weight decay
        updates = jax.tree.map(lambda m, p: neg_lr * metric_scale * m / (1 - momentum ** step) - learning_rate * weight_decay * p, new_momentum, params)

        # Create new optimizer state
        new_state = SGDState(step=step, momentum=new_momentum, metric_ema=new_metric_ema, rms_ema=None)
        
        return updates, new_state
    
    return optax.GradientTransformation(init, update)

def custom_sgd_rms(learning_rate=0.1, momentum=0.9, xi=0.1, beta=0.8, beta_rms=0.99, weight_decay=0.0, eps=1e-8):
    """Custom SGD optimizer with momentum, metric modification, and RMS scaling.
    
    This optimizer combines momentum-based SGD with adaptive gradient scaling similar
    to RMSprop. Gradients are normalized by their RMS (root mean square) values, and
    the learning rate is further modulated by a metric based on RMS-scaled gradient norms.
    
    Args:
        learning_rate: Base learning rate (default: 0.1)
        momentum: Momentum factor for gradient accumulation (default: 0.9)
        xi: Scaling factor for gradient norm in metric computation (default: 0.1)
        beta: EMA decay rate for metric scale tracking (default: 0.8)
        beta_rms: EMA decay rate for RMS computation (default: 0.99)
        weight_decay: Weight decay (L2 regularization) factor (default: 0.0)
        eps: Small constant for numerical stability (default: 1e-8)
        
    Returns:
        optax.GradientTransformation: JAX optimizer that can be used with optax
    """
    
    # Pre-compute constants for efficiency
    neg_lr = -learning_rate  # Negative for parameter updates
    one_minus_beta = 1 - beta  # For metric EMA computation
    one_minus_beta_rms = 1 - beta_rms  # For RMS EMA computation
    one_minus_momentum = 1 - momentum  # For momentum computation
    
    def init(params):
        """Initialize optimizer state.
        
        Args:
            params: Model parameters to optimize
            
        Returns:
            SGDState: Initial optimizer state with RMS tracking
        """
        return SGDState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
            metric_ema=jnp.zeros([]),
            rms_ema=jax.tree.map(jnp.zeros_like, params)
        )
    
    @jax.jit
    def update(grads, state, params=None):
        """Update parameters using custom SGD with momentum, metric modification, and RMS scaling.
        
        Args:
            grads: Current gradients
            state: Current optimizer state
            params: Current parameter values (optional, used for weight decay)
            
        Returns:
            Tuple of (updates, new_state) where updates are parameter changes
        """
        step = state.step + 1

        # Update EMA of squared gradients (RMS computation) for each parameter
        new_rms_ema = jax.tree.map(
            lambda r, g: beta_rms * r + one_minus_beta_rms * (g ** 2),
            state.rms_ema, grads
        )

        # Apply bias correction to RMS estimates
        rms_corrected = jax.tree.map(
            lambda r: r / (1 - beta_rms ** step),
            new_rms_ema
        )

        # Calculate RMS-scaled gradient norm for metric computation
        grad_norm_sq = jax.tree_util.tree_reduce(
        lambda acc, g_r_pair: acc + jnp.sum(g_r_pair),
        jax.tree.map(lambda g, r: g ** 2 / (jnp.sqrt(r) + eps), grads, rms_corrected),
        initializer=0.0)
        trace = xi * grad_norm_sq
        
        # Update EMA of metric scale
        new_metric_ema = beta * state.metric_ema + one_minus_beta * trace

        # Apply bias correction and compute metric scale
        metric_corrected = new_metric_ema / (1 - beta ** step)
        metric_scale = 1 / (1 + jnp.abs(metric_corrected))

        # Update momentum buffer with current gradients
        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + one_minus_momentum * g, 
            state.momentum, 
            grads
        )
        
        # Apply bias correction to momentum and compute parameter updates
        # Include learning rate, metric scaling, RMS normalization, and weight decay
        updates = jax.tree.map(lambda m, p, r: neg_lr * metric_scale * m / ((1 - momentum ** step)*(jnp.sqrt(r) + eps))
                               - learning_rate * weight_decay * p, new_momentum, params, rms_corrected)

        # Create new optimizer state with updated RMS tracking
        new_state = SGDState(step=step, momentum=new_momentum, metric_ema=new_metric_ema, rms_ema=new_rms_ema)
        
        return updates, new_state
    
    return optax.GradientTransformation(init, update)