import jax
import jax.numpy as jnp

import optax
import flax
from flax import linen as nn

import numpy as np
import matplotlib.pyplot as plt

import wandb

from tqdm import tqdm

from typing import NamedTuple

from itertools import combinations_with_replacement, product
import time
import argparse

# Import shared models
from shared_models import custom_sgd, custom_sgd_log, custom_sgd_rms, MLP, SGDState, _compute_grad_norm_squared
import pandas as pd

# Print JAX device information
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")


# Parse command line arguments
parser = argparse.ArgumentParser(description='Regression Hyperparameter Sweep')
parser.add_argument('--optimiser', type=str, required=True, 
                    choices=['adam', 'adamw', 'sgd', 'sgd_metric', 'sgd_log_metric', 'sgd_rms', 'muon'],
                    help='Optimiser to use for training')
parser.add_argument('--num_runs', type=int, default=50,
                    help='Number of runs for the sweep')
parser.add_argument('--index', type=int, default=0,)
parser.add_argument('--val_freq', type=int, default=1,
                    help='Frequency of validation loss computation (every N epochs)')
parser.add_argument('--search', type=str, default='bayes',
                    choices=['bayes', 'grid', 'random'],
                    help='Search method for hyperparameter tuning')
args = parser.parse_args()

# Generate synthetic regression data
point_key = jax.random.PRNGKey(0)
func_key = jax.random.PRNGKey(1)
order = 6
degree = 6

# Define a random polynomial of order 4 variables and degree 3
def random_polynomial(key, order, degree):
    """Generate coefficients for a random polynomial with cross terms"""
    # Generate all possible combinations of powers for multivariate polynomial
    
    # Create all possible power combinations up to degree
    power_combinations = []
    for total_degree in range(degree + 1):
        # Generate all ways to distribute total_degree among order variables
        for powers in product(range(total_degree + 1), repeat=order):
            if sum(powers) == total_degree:
                power_combinations.append(powers)
    
    num_terms = len(power_combinations)
    coeffs = jax.random.normal(key, (num_terms,))
    return coeffs, power_combinations

def evaluate_polynomial(coeffs, power_combinations, x):
    """Evaluate polynomial with cross terms at given points"""
    result = jnp.zeros(x.shape[0])
    
    for i, (coeff, powers) in enumerate(zip(coeffs, power_combinations)):
        term = jnp.ones(x.shape[0])
        for j, power in enumerate(powers):
            if power > 0:
                term *= x[:, j] ** power
        result += coeff * term
    
    return result

# Generate random polynomial coefficients with cross terms
poly_coeffs, power_combinations = random_polynomial(func_key, order, degree)

# Preprocess data
x = jax.random.normal(point_key, (13312, 4))  # Power of 2 aligned data
y = evaluate_polynomial(poly_coeffs, power_combinations, x)
# Normalize y
y_mean = jnp.mean(y)
y_std = jnp.std(y)
y = (y - y_mean) / y_std

# Train/test split
x_train = x[:8192]  # 8192 = 2^13, divisible by all batch sizes
y_train = y[:8192]
x_test = x[8192:]  # 5120 test samples 
y_test = y[8192:]

def create_data_loaders(x_train, y_train, x_test, y_test, batch_size, seed):
    """Create data loaders for training and testing"""
    # Shuffle training data
    key = jax.random.PRNGKey(seed)
    perm = jax.random.permutation(key, len(x_train))
    x_train = x_train[perm]
    y_train = y_train[perm]
    
    # Calculate number of complete batches
    n_train_batches = len(x_train) // batch_size
    n_test_batches = len(x_test) // batch_size
    
    # Create training batches
    train_batches = []
    for i in range(n_train_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        train_batches.append((x_train[start_idx:end_idx], y_train[start_idx:end_idx]))
    
    # Create test batches
    test_batches = []
    for i in range(n_test_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        test_batches.append((x_test[start_idx:end_idx], y_test[start_idx:end_idx]))
    
    return train_batches, test_batches


# Define the loss function
def mse_fn(params, x, y, model):
    """Compute MSE for a batch"""
    predictions = model.apply(params, x).squeeze()
    return jnp.mean((predictions - y) ** 2)

# Compute MSE over all batches
def compute_full_mse(params, data_batches, model):
    """Compute MSE over all batches"""
    total_mse = 0.0
    total_samples = 0
    
    for x_batch, y_batch in data_batches:
        batch_mse = mse_fn(params, x_batch, y_batch, model)
        batch_size = len(x_batch)
        total_mse += batch_mse * batch_size
        total_samples += batch_size
    
    return total_mse / total_samples


# Training functions for each optimizer
def train_adam(config, seed):
    """Train MLP with Adam optimizer"""
    
    # Initialize model and data
    model = MLP(features=32, output_dim=1)
    train_batches, test_batches = create_data_loaders(x_train, y_train, x_test, y_test, 
                                                     config['batch_size'], seed)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 4))
    params = model.init(key, dummy_input)
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate=config['learning_rate'], 
                          b1=config['beta1'], b2=config['beta2'],
                          eps=config['eps'])
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    train_mses = []
    test_mses = []
    
    min_val_loss = float('inf')
    min_loss_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(lambda p: mse_fn(p, x, y, model))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    train_time = 0.0
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Training
        train_time += time.time()
        for x_batch, y_batch in train_batches:
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            epoch_losses.append(loss)
        train_time -= time.time()
        
        avg_loss = float(jnp.mean(jnp.array(epoch_losses)))
        
        # Calculate accuracy every args.val_freq epochs for better monitoring
        if epoch % args.val_freq == 0 or epoch == n_epochs - 1:
            train_mse = float(compute_full_mse(params, train_batches, model))
            test_mse = float(compute_full_mse(params, test_batches, model))
            if test_mse < min_val_loss:
                min_val_loss = test_mse
                min_loss_epoch = epoch
            
            train_losses.append(avg_loss)
            train_mses.append(train_mse)
            test_mses.append(test_mse)
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "min_val_loss": min_val_loss,
                "min_loss_epoch": min_loss_epoch,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
            })
    
    return {
        'min_val_loss': min_val_loss,
        'min_loss_epoch': min_loss_epoch,
        'final_train_mse': train_mses[-1] if train_mses else float('inf'),
        'final_test_mse': test_mses[-1] if test_mses else float('inf'),
        'train_losses': train_losses,
        'train_mses': train_mses,
        'test_mses': test_mses,
    }


def train_sgd(config, seed):
    """Train MLP with SGD optimizer"""
    
    # Initialize model and data
    model = MLP(features=32, output_dim=1)
    train_batches, test_batches = create_data_loaders(x_train, y_train, x_test, y_test, 
                                                     config['batch_size'], seed)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 4))
    params = model.init(key, dummy_input)
    
    # Initialize optimizer
    optimizer = optax.sgd(learning_rate=config['learning_rate'], 
                         momentum=config['momentum'])
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    train_mses = []
    test_mses = []
    
    min_val_loss = float('inf')
    min_loss_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(lambda p: mse_fn(p, x, y, model))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    train_time = 0.0
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Training
        train_time += time.time()
        for x_batch, y_batch in train_batches:
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            epoch_losses.append(loss)
        train_time -= time.time()
        
        avg_loss = float(jnp.mean(jnp.array(epoch_losses)))
        
        # Calculate accuracy every args.val_freq epochs for better monitoring
        if epoch % args.val_freq == 0 or epoch == n_epochs - 1:
            train_mse = float(compute_full_mse(params, train_batches, model))
            test_mse = float(compute_full_mse(params, test_batches, model))
            if test_mse < min_val_loss:
                min_val_loss = test_mse
                min_loss_epoch = epoch
            
            train_losses.append(avg_loss)
            train_mses.append(train_mse)
            test_mses.append(test_mse)
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "min_val_loss": min_val_loss,
                "min_loss_epoch": min_loss_epoch,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
            })
    
    return {
        'min_val_loss': min_val_loss,
        'min_loss_epoch': min_loss_epoch,
        'final_train_mse': train_mses[-1] if train_mses else float('inf'),
        'final_test_mse': test_mses[-1] if test_mses else float('inf'),
        'train_losses': train_losses,
        'train_mses': train_mses,
        'test_mses': test_mses,
    }


def train_sgd_metric(config, seed):
    """Train MLP with custom SGD metric optimizer"""
    
    # Initialize model and data
    model = MLP(features=32, output_dim=1)
    train_batches, test_batches = create_data_loaders(x_train, y_train, x_test, y_test, 
                                                     config['batch_size'], seed)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 4))
    params = model.init(key, dummy_input)
    
    # Initialize optimizer
    optimizer = custom_sgd(learning_rate=config['learning_rate'], 
                          momentum=config['momentum'],
                          xi=config['xi'], beta=config['beta'],
                          weight_decay=config['weight_decay'])
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    train_mses = []
    test_mses = []
    
    min_val_loss = float('inf')
    min_loss_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(lambda p: mse_fn(p, x, y, model))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    train_time = 0.0
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Training
        train_time += time.time()
        for x_batch, y_batch in train_batches:
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            epoch_losses.append(loss)
        train_time -= time.time()
        
        avg_loss = float(jnp.mean(jnp.array(epoch_losses)))
        
        # Calculate accuracy every args.val_freq epochs for better monitoring
        if epoch % args.val_freq == 0 or epoch == n_epochs - 1:
            train_mse = float(compute_full_mse(params, train_batches, model))
            test_mse = float(compute_full_mse(params, test_batches, model))
            if test_mse < min_val_loss:
                min_val_loss = test_mse
                min_loss_epoch = epoch
            
            train_losses.append(avg_loss)
            train_mses.append(train_mse)
            test_mses.append(test_mse)
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "min_val_loss": min_val_loss,
                "min_loss_epoch": min_loss_epoch,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
            })
    
    return {
        'min_val_loss': min_val_loss,
        'min_loss_epoch': min_loss_epoch,
        'final_train_mse': train_mses[-1] if train_mses else float('inf'),
        'final_test_mse': test_mses[-1] if test_mses else float('inf'),
        'train_losses': train_losses,
        'train_mses': train_mses,
        'test_mses': test_mses,
    }


def train_sgd_log_metric(config, seed):
    """Train MLP with custom SGD log metric optimizer"""
    
    # Initialize model and data
    model = MLP(features=32, output_dim=1)
    train_batches, test_batches = create_data_loaders(x_train, y_train, x_test, y_test, 
                                                     config['batch_size'], seed)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 4))
    params = model.init(key, dummy_input)
    
    # Initialize optimizer
    optimizer = custom_sgd_log(learning_rate=config['learning_rate'], 
                              momentum=config['momentum'],
                              xi=config['xi'], beta=config['beta'],
                              weight_decay=config['weight_decay'])
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    train_mses = []
    test_mses = []
    
    min_val_loss = float('inf')
    min_loss_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(lambda p: mse_fn(p, x, y, model))(params)
        updates, opt_state = optimizer.update(grads, opt_state, loss, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    train_time = 0.0
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Training
        train_time += time.time()
        for x_batch, y_batch in train_batches:
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            epoch_losses.append(loss)
        train_time -= time.time()
        
        avg_loss = float(jnp.mean(jnp.array(epoch_losses)))
        
        # Calculate accuracy every args.val_freq epochs for better monitoring
        if epoch % args.val_freq == 0 or epoch == n_epochs - 1:
            train_mse = float(compute_full_mse(params, train_batches, model))
            test_mse = float(compute_full_mse(params, test_batches, model))
            if test_mse < min_val_loss:
                min_val_loss = test_mse
                min_loss_epoch = epoch
            
            train_losses.append(avg_loss)
            train_mses.append(train_mse)
            test_mses.append(test_mse)
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "min_val_loss": min_val_loss,
                "min_loss_epoch": min_loss_epoch,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
            })
    
    return {
        'min_val_loss': min_val_loss,
        'min_loss_epoch': min_loss_epoch,
        'final_train_mse': train_mses[-1] if train_mses else float('inf'),
        'final_test_mse': test_mses[-1] if test_mses else float('inf'),
        'train_losses': train_losses,
        'train_mses': train_mses,
        'test_mses': test_mses,
    }


def train_sgd_rms(config, seed):
    """Train MLP with custom SGD RMS optimizer"""
    
    # Initialize model and data
    model = MLP(features=32, output_dim=1)
    train_batches, test_batches = create_data_loaders(x_train, y_train, x_test, y_test, 
                                                     config['batch_size'], seed)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 4))
    params = model.init(key, dummy_input)
    
    # Initialize optimizer
    optimizer = custom_sgd_rms(learning_rate=config['learning_rate'], 
                              momentum=config['momentum'],
                              xi=config['xi'], beta=config['beta'],
                              beta_rms=config['beta_rms'], eps=config['eps'],
                              weight_decay=config['weight_decay'])
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    train_mses = []
    test_mses = []
    
    min_val_loss = float('inf')
    min_loss_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(lambda p: mse_fn(p, x, y, model))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    train_time = 0.0
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Training
        train_time += time.time()
        for x_batch, y_batch in train_batches:
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            epoch_losses.append(loss)
        train_time -= time.time()
        
        avg_loss = float(jnp.mean(jnp.array(epoch_losses)))
        
        # Calculate accuracy every args.val_freq epochs for better monitoring
        if epoch % args.val_freq == 0 or epoch == n_epochs - 1:
            train_mse = float(compute_full_mse(params, train_batches, model))
            test_mse = float(compute_full_mse(params, test_batches, model))
            if test_mse < min_val_loss:
                min_val_loss = test_mse
                min_loss_epoch = epoch
            
            train_losses.append(avg_loss)
            train_mses.append(train_mse)
            test_mses.append(test_mse)
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "min_val_loss": min_val_loss,
                "min_loss_epoch": min_loss_epoch,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
            })
    
    return {
        'min_val_loss': min_val_loss,
        'min_loss_epoch': min_loss_epoch,
        'final_train_mse': train_mses[-1] if train_mses else float('inf'),
        'final_test_mse': test_mses[-1] if test_mses else float('inf'),
        'train_losses': train_losses,
        'train_mses': train_mses,
        'test_mses': test_mses,
    }


def train_adamw(config, seed):
    """Train MLP with AdamW optimizer"""
    
    # Initialize model and data
    model = MLP(features=32, output_dim=1)
    train_batches, test_batches = create_data_loaders(x_train, y_train, x_test, y_test, 
                                                     config['batch_size'], seed)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 4))
    params = model.init(key, dummy_input)
    
    # Initialize optimizer
    optimizer = optax.adamw(learning_rate=config['learning_rate'], 
                           b1=config['beta1'], b2=config['beta2'],
                           eps=config['eps'], weight_decay=config['weight_decay'])
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    train_mses = []
    test_mses = []
    
    min_val_loss = float('inf')
    min_loss_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(lambda p: mse_fn(p, x, y, model))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    train_time = 0.0
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Training
        train_time += time.time()
        for x_batch, y_batch in train_batches:
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            epoch_losses.append(loss)
        train_time -= time.time()
        
        avg_loss = float(jnp.mean(jnp.array(epoch_losses)))
        
        # Calculate accuracy every args.val_freq epochs for better monitoring
        if epoch % args.val_freq == 0 or epoch == n_epochs - 1:
            train_mse = float(compute_full_mse(params, train_batches, model))
            test_mse = float(compute_full_mse(params, test_batches, model))
            if test_mse < min_val_loss:
                min_val_loss = test_mse
                min_loss_epoch = epoch
            
            train_losses.append(avg_loss)
            train_mses.append(train_mse)
            test_mses.append(test_mse)
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "min_val_loss": min_val_loss,
                "min_loss_epoch": min_loss_epoch,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
            })
    
    return {
        'min_val_loss': min_val_loss,
        'min_loss_epoch': min_loss_epoch,
        'final_train_mse': train_mses[-1] if train_mses else float('inf'),
        'final_test_mse': test_mses[-1] if test_mses else float('inf'),
        'train_losses': train_losses,
        'train_mses': train_mses,
        'test_mses': test_mses,
    }


def train_muon(config, seed):
    """Train MLP with Muon optimizer"""
    
    # Initialize model and data
    model = MLP(features=32, output_dim=1)
    train_batches, test_batches = create_data_loaders(x_train, y_train, x_test, y_test, 
                                                     config['batch_size'], seed)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 4))
    params = model.init(key, dummy_input)
    
    # Initialize Muon optimizer
    optimizer = optax.contrib.muon(
        learning_rate=config['learning_rate'],
        adam_b1=config['adam_b1'],
        adam_b2=config['adam_b2'],
        eps=config['eps'],
        beta=config['beta'],
        weight_decay=config['weight_decay']
    )
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    train_mses = []
    test_mses = []
    
    min_val_loss = float('inf')
    min_loss_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(lambda p: mse_fn(p, x, y, model))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    train_time = 0.0
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Training
        train_time += time.time()
        for x_batch, y_batch in train_batches:
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            epoch_losses.append(loss)
        train_time -= time.time()
        
        avg_loss = float(jnp.mean(jnp.array(epoch_losses)))
        
        # Calculate accuracy every args.val_freq epochs for better monitoring
        if epoch % args.val_freq == 0 or epoch == n_epochs - 1:
            train_mse = float(compute_full_mse(params, train_batches, model))
            test_mse = float(compute_full_mse(params, test_batches, model))
            if test_mse < min_val_loss:
                min_val_loss = test_mse
                min_loss_epoch = epoch
            
            train_losses.append(avg_loss)
            train_mses.append(train_mse)
            test_mses.append(test_mse)
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "min_val_loss": min_val_loss,
                "min_loss_epoch": min_loss_epoch,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
            })
    
    return {
        'min_val_loss': min_val_loss,
        'min_loss_epoch': min_loss_epoch,
        'final_train_mse': train_mses[-1] if train_mses else float('inf'),
        'final_test_mse': test_mses[-1] if test_mses else float('inf'),
        'train_losses': train_losses,
        'train_mses': train_mses,
        'test_mses': test_mses,
    }



def get_sweep_config(optimizer_name):
    """Get W&B sweep configuration for the given optimizer"""
    
    base_config = {
        'name': f'regression_sweep_{optimizer_name}_{args.index}',
        'method': args.search,
        'metric': {
            'name': 'min_val_loss',
            'goal': 'minimize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 20
        }
    }
    
    if optimizer_name == 'adam':
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1},
            'beta1': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'beta2': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
            'eps': {'values': [1e-8]},
            'batch_size': {'values': [1024]},
            'n_epochs': {'value': 200}
        }
    elif optimizer_name == 'adamw':
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1},
            'beta1': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'beta2': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
            'eps': {'values': [1e-8]},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'batch_size': {'values': [1024]},
            'n_epochs': {'value': 200}
        }
    elif optimizer_name == 'sgd':
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1},
            'momentum': {'distribution': 'uniform', 'min': 0.1, 'max': 0.99},
            'batch_size': {'values': [1024]},
            'n_epochs': {'value': 200}
        }
    elif optimizer_name in ['sgd_metric', 'sgd_log_metric']:
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1},
            'momentum': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'xi': {'distribution': 'log_uniform_values', 'min': 1e-3, 'max': 1e-1},
            'beta': {'distribution': 'uniform', 'min': 0.6, 'max': 0.9},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'batch_size': {'values': [1024]},
            'n_epochs': {'value': 200}
        }
    elif optimizer_name == 'sgd_rms':
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1},
            'momentum': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'xi': {'distribution': 'log_uniform_values', 'min': 1e-3, 'max': 1e-1},
            'beta': {'distribution': 'uniform', 'min': 0.6, 'max': 0.9},
            'beta_rms': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
            'eps': {'values': [1e-8]},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'batch_size': {'values': [1024]},
            'n_epochs': {'value': 200}
        }
    elif optimizer_name == 'muon':
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1},
            'adam_b1': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'adam_b2': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
            'eps': {'values': [1e-8]},
            'beta': {'distribution': 'uniform', 'min': 0.9, 'max': 0.99},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'batch_size': {'values': [1024]},
            'n_epochs': {'value': 200}
        }
    
    return base_config


def train():
    """Training function for W&B sweep"""
    
    # Add tags
    tags = [args.optimiser, "regression", f"run_{args.index}", args.search]
    
    # Initialize wandb
    wandb.init(project="induced_metric", tags=tags)
    
    # Get config from wandb
    config = wandb.config
    
    # Select training function
    train_functions = {
        'adam': train_adam,
        'adamw': train_adamw,
        'sgd': train_sgd,
        'sgd_metric': train_sgd_metric,
        'sgd_log_metric': train_sgd_log_metric,
        'sgd_rms': train_sgd_rms,
        'muon': train_muon
    }
    
    train_fn = train_functions[args.optimiser]
    
    # Generate a random seed for this run
    seed = np.random.randint(1, 1000000)
    
    # Time the training
    start_time = time.time()
    
    # Run training
    results = train_fn(config, seed)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Log final results
    wandb.log({
        'final_min_val_loss': results['min_val_loss'],
        'final_min_loss_epoch': results['min_loss_epoch'],
        'final_train_mse': results['final_train_mse'],
        'final_test_mse': results['final_test_mse'],
        'total_training_time_sec': training_time,
        'total_training_time_min': training_time / 60,
        'seed': seed,
        'optimizer': args.optimiser,
        'architecture': 'MLP'
    })
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best validation loss: {results['min_val_loss']:.6f}")


if __name__ == "__main__":
    # Get sweep configuration
    sweep_config = get_sweep_config(args.optimiser)
    
    # Create or get sweep
    sweep_name = f"regression_{args.optimiser}_sweep"
    
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="induced_metric")
    
    print(f"Starting sweep: {sweep_name}")
    print(f"Sweep ID: {sweep_id}")
    print(f"Number of runs: {args.num_runs}")
    
    # Start the sweep agent
    wandb.agent(sweep_id, train, count=args.num_runs)
    
    print("Sweep completed!")