import jax
import jax.numpy as jnp

import optax
import flax
from flax import linen as nn

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
parser = argparse.ArgumentParser(description='MNIST MLP Hyperparameter Sweep')
parser.add_argument('--optimiser', type=str, required=True, 
                    choices=['adam', 'adamw', 'sgd', 'sgd_metric', 'sgd_log_metric', 'sgd_rms', 'muon'],
                    help='Optimiser to use for training')
parser.add_argument('--num_runs', type=int, default=50,
                    help='Number of runs for the sweep')
parser.add_argument('--index', type=int, default=0,)
parser.add_argument('--val_freq', type=int, default=1,
                    help='Frequency of validation accuracy computation (every N epochs)')
parser.add_argument('--search', type=str, default='bayes',
                    choices=['bayes', 'grid', 'random'],
                    help='Search method for hyperparameter tuning')
args = parser.parse_args()

def load_mnist():
    """Load and preprocess MNIST dataset"""
    import tensorflow as tf
    
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Preprocess data for MLP (flatten)
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    
    return x_train, y_train, x_test, y_test

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


def loss_fn(params, x, y, model):
    """Compute loss for a batch"""
    logits = model.apply(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def accuracy_fn(params, x, y, model):
    """Compute accuracy for a batch"""
    logits = model.apply(params, x)
    predictions = jnp.argmax(logits, axis=1)
    return jnp.mean(predictions == y)


def compute_full_accuracy(params, data_batches, model):
    """Compute accuracy over all batches"""
    total_correct = 0
    total_samples = 0
    
    for x_batch, y_batch in data_batches:
        batch_acc = accuracy_fn(params, x_batch, y_batch, model)
        batch_size = len(x_batch)
        total_correct += batch_acc * batch_size
        total_samples += batch_size
    
    return total_correct / total_samples


# Training functions for each optimizer
def train_adam(config, seed):
    """Train MLP with Adam optimizer"""
    
    # Load data
    x_train, y_train, x_test, y_test = load_mnist()
    
    # Initialize model and data
    model = MLP(features=64, output_dim=10)
    train_batches, test_batches = create_data_loaders(x_train, y_train, x_test, y_test, 
                                                     config['batch_size'], seed)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 784))
    params = model.init(key, dummy_input)
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate=config['learning_rate'], 
                          b1=config['beta1'], b2=config['beta2'],
                          eps=config['eps'])
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    train_accs = []
    test_accs = []
    
    max_val_acc = 0.0
    max_acc_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
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
            train_acc = float(compute_full_accuracy(params, train_batches, model))
            test_acc = float(compute_full_accuracy(params, test_batches, model))
            if test_acc > max_val_acc:
                max_val_acc = test_acc
                max_acc_epoch = epoch
            
            train_losses.append(avg_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "max_val_acc": max_val_acc,
                "max_acc_epoch": max_acc_epoch,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
            })
    
    return {
        'max_val_acc': max_val_acc,
        'max_acc_epoch': max_acc_epoch,
        'final_train_acc': train_accs[-1] if train_accs else 0.0,
        'final_test_acc': test_accs[-1] if test_accs else 0.0,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
    }


def train_sgd(config, seed):
    """Train MLP with SGD optimizer"""
    
    # Load data
    x_train, y_train, x_test, y_test = load_mnist()
    
    # Initialize model and data
    model = MLP(features=64, output_dim=10)
    train_batches, test_batches = create_data_loaders(x_train, y_train, x_test, y_test, 
                                                     config['batch_size'], seed)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 784))
    params = model.init(key, dummy_input)
    
    # Initialize optimizer
    optimizer = optax.sgd(learning_rate=config['learning_rate'], 
                         momentum=config['momentum'])
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    train_accs = []
    test_accs = []
    
    max_val_acc = 0.0
    max_acc_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
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
            train_acc = float(compute_full_accuracy(params, train_batches, model))
            test_acc = float(compute_full_accuracy(params, test_batches, model))
            if test_acc > max_val_acc:
                max_val_acc = test_acc
                max_acc_epoch = epoch
            
            train_losses.append(avg_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "max_val_acc": max_val_acc,
                "max_acc_epoch": max_acc_epoch,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
            })
    
    return {
        'max_val_acc': max_val_acc,
        'max_acc_epoch': max_acc_epoch,
        'final_train_acc': train_accs[-1] if train_accs else 0.0,
        'final_test_acc': test_accs[-1] if test_accs else 0.0,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
    }


def train_sgd_metric(config, seed):
    """Train MLP with custom SGD metric optimizer"""
    
    # Load data
    x_train, y_train, x_test, y_test = load_mnist()
    
    # Initialize model and data
    model = MLP(features=64, output_dim=10)
    train_batches, test_batches = create_data_loaders(x_train, y_train, x_test, y_test, 
                                                     config['batch_size'], seed)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 784))
    params = model.init(key, dummy_input)
    
    # Initialize optimizer
    optimizer = custom_sgd(learning_rate=config['learning_rate'], 
                          momentum=config['momentum'],
                          xi=config['xi'], beta=config['beta'],
                          weight_decay=config['weight_decay'])
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    train_accs = []
    test_accs = []
    
    max_val_acc = 0.0
    max_acc_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
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
            train_acc = float(compute_full_accuracy(params, train_batches, model))
            test_acc = float(compute_full_accuracy(params, test_batches, model))
            if test_acc > max_val_acc:
                max_val_acc = test_acc
                max_acc_epoch = epoch
            
            train_losses.append(avg_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "max_val_acc": max_val_acc,
                "max_acc_epoch": max_acc_epoch,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
            })
    
    return {
        'max_val_acc': max_val_acc,
        'max_acc_epoch': max_acc_epoch,
        'final_train_acc': train_accs[-1] if train_accs else 0.0,
        'final_test_acc': test_accs[-1] if test_accs else 0.0,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
    }


def train_sgd_log_metric(config, seed):
    """Train MLP with custom SGD log metric optimizer"""
    
    # Load data
    x_train, y_train, x_test, y_test = load_mnist()
    
    # Initialize model and data
    model = MLP(features=64, output_dim=10)
    train_batches, test_batches = create_data_loaders(x_train, y_train, x_test, y_test, 
                                                     config['batch_size'], seed)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 784))
    params = model.init(key, dummy_input)
    
    # Initialize optimizer
    optimizer = custom_sgd_log(learning_rate=config['learning_rate'], 
                              momentum=config['momentum'],
                              xi=config['xi'], beta=config['beta'],
                              weight_decay=config['weight_decay'])
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    train_accs = []
    test_accs = []
    
    max_val_acc = 0.0
    max_acc_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
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
            train_acc = float(compute_full_accuracy(params, train_batches, model))
            test_acc = float(compute_full_accuracy(params, test_batches, model))
            if test_acc > max_val_acc:
                max_val_acc = test_acc
                max_acc_epoch = epoch
            
            train_losses.append(avg_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "max_val_acc": max_val_acc,
                "max_acc_epoch": max_acc_epoch,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
            })
    
    return {
        'max_val_acc': max_val_acc,
        'max_acc_epoch': max_acc_epoch,
        'final_train_acc': train_accs[-1] if train_accs else 0.0,
        'final_test_acc': test_accs[-1] if test_accs else 0.0,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
    }


def train_adamw(config, seed):
    """Train MLP with AdamW optimizer"""
    
    # Load data
    x_train, y_train, x_test, y_test = load_mnist()
    
    # Initialize model and data
    model = MLP(features=64, output_dim=10)
    train_batches, test_batches = create_data_loaders(x_train, y_train, x_test, y_test, 
                                                     config['batch_size'], seed)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 784))
    params = model.init(key, dummy_input)
    
    # Initialize optimizer
    optimizer = optax.adamw(learning_rate=config['learning_rate'], 
                           b1=config['beta1'], b2=config['beta2'],
                           eps=config['eps'], weight_decay=config['weight_decay'])
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    train_accs = []
    test_accs = []
    
    max_val_acc = 0.0
    max_acc_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
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
            train_acc = float(compute_full_accuracy(params, train_batches, model))
            test_acc = float(compute_full_accuracy(params, test_batches, model))
            if test_acc > max_val_acc:
                max_val_acc = test_acc
                max_acc_epoch = epoch
            
            train_losses.append(avg_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "max_val_acc": max_val_acc,
                "max_acc_epoch": max_acc_epoch,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
            })
    
    return {
        'max_val_acc': max_val_acc,
        'max_acc_epoch': max_acc_epoch,
        'final_train_acc': train_accs[-1] if train_accs else 0.0,
        'final_test_acc': test_accs[-1] if test_accs else 0.0,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
    }


def train_muon(config, seed):
    """Train MLP with Muon optimizer"""
    
    # Load data
    x_train, y_train, x_test, y_test = load_mnist()
    
    # Initialize model and data
    model = MLP(features=64, output_dim=10)
    train_batches, test_batches = create_data_loaders(x_train, y_train, x_test, y_test, 
                                                     config['batch_size'], seed)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 784))
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
    train_accs = []
    test_accs = []
    
    max_val_acc = 0.0
    max_acc_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
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
            train_acc = float(compute_full_accuracy(params, train_batches, model))
            test_acc = float(compute_full_accuracy(params, test_batches, model))
            if test_acc > max_val_acc:
                max_val_acc = test_acc
                max_acc_epoch = epoch
            
            train_losses.append(avg_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "max_val_acc": max_val_acc,
                "max_acc_epoch": max_acc_epoch,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
            })
    
    return {
        'max_val_acc': max_val_acc,
        'max_acc_epoch': max_acc_epoch,
        'final_train_acc': train_accs[-1] if train_accs else 0.0,
        'final_test_acc': test_accs[-1] if test_accs else 0.0,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
    }


def train_sgd_rms(config, seed):
    """Train MLP with custom SGD RMS optimizer"""
    
    # Load data
    x_train, y_train, x_test, y_test = load_mnist()
    
    # Initialize model and data
    model = MLP(features=64, output_dim=10)
    train_batches, test_batches = create_data_loaders(x_train, y_train, x_test, y_test, 
                                                     config['batch_size'], seed)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 784))
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
    train_accs = []
    test_accs = []
    
    max_val_acc = 0.0
    max_acc_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
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
            train_acc = float(compute_full_accuracy(params, train_batches, model))
            test_acc = float(compute_full_accuracy(params, test_batches, model))
            if test_acc > max_val_acc:
                max_val_acc = test_acc
                max_acc_epoch = epoch
            
            train_losses.append(avg_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "max_val_acc": max_val_acc,
                "max_acc_epoch": max_acc_epoch,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
            })
    
    return {
        'max_val_acc': max_val_acc,
        'max_acc_epoch': max_acc_epoch,
        'final_train_acc': train_accs[-1] if train_accs else 0.0,
        'final_test_acc': test_accs[-1] if test_accs else 0.0,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
    }


def get_sweep_config(optimizer_name):
    """Get W&B sweep configuration for the given optimizer"""
    
    base_config = {
        'name': f'mnist_mlp_sweep_{optimizer_name}_{args.index}',
        'method': 'random',
        'metric': {
            'name': 'max_val_acc',
            'goal': 'maximize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 20
        }
    }
    
    if optimizer_name == 'adam':
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},
            'beta1': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'beta2': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
            'eps': {'distribution': 'log_uniform_values', 'min': 1e-10, 'max': 1e-6},
            'batch_size': {'values': [1024]},
            'n_epochs': {'value': 200}
        }
    elif optimizer_name == 'adamw':
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},
            'beta1': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'beta2': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
            'eps': {'distribution': 'log_uniform_values', 'min': 1e-10, 'max': 1e-6},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'batch_size': {'values': [1024]},
            'n_epochs': {'value': 200}
        }
    elif optimizer_name == 'sgd':
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},
            'momentum': {'distribution': 'uniform', 'min': 0.0, 'max': 0.99},
            'batch_size': {'values': [1024]},
            'n_epochs': {'value': 200}
        }
    elif optimizer_name in ['sgd_metric', 'sgd_log_metric']:
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1},
            'momentum': {'distribution': 'uniform', 'min': 0.0, 'max': 0.99},
            'xi': {'distribution': 'log_uniform_values', 'min': 1e-3, 'max': 1e1},
            'beta': {'distribution': 'uniform', 'min': 0, 'max': 0.3},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'batch_size': {'values': [1024]},
            'n_epochs': {'value': 200}
        }
    elif optimizer_name == 'sgd_rms':
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1},
            'momentum': {'distribution': 'uniform', 'min': 0.0, 'max': 0.99},
            'xi': {'distribution': 'log_uniform_values', 'min': 1e-3, 'max': 1e1},
            'beta': {'distribution': 'uniform', 'min': 0, 'max': 0.3},
            'beta_rms': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
            'eps': {'distribution': 'log_uniform_values', 'min': 1e-10, 'max': 1e-6},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'batch_size': {'values': [1024]},
            'n_epochs': {'value': 200}
        }
    elif optimizer_name == 'muon':
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},
            'adam_b1': {'distribution': 'uniform', 'min': 0.85, 'max': 0.95},
            'adam_b2': {'distribution': 'uniform', 'min': 0.99, 'max': 0.9999},
            'eps': {'distribution': 'log_uniform_values', 'min': 1e-8, 'max': 1e-6},
            'beta': {'distribution': 'uniform', 'min': 0.9, 'max': 0.99},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'batch_size': {'values': [1024]},
            'n_epochs': {'value': 200}
        }
    
    return base_config


def train():
    """Training function for W&B sweep"""
    
    # Add tags
    tags = [args.optimiser, "mnist_mlp", f"run_{args.index}", args.search]
    
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
        'final_max_val_acc': results['max_val_acc'],
        'final_max_acc_epoch': results['max_acc_epoch'],
        'final_train_acc': results['final_train_acc'],
        'final_test_acc': results['final_test_acc'],
        'total_training_time_sec': training_time,
        'total_training_time_min': training_time / 60,
        'seed': seed,
        'optimizer': args.optimiser,
        'architecture': 'MLP'
    })
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best validation accuracy: {results['max_val_acc']:.6f}")


if __name__ == "__main__":
    # Get sweep configuration
    sweep_config = get_sweep_config(args.optimiser)
    
    # Create or get sweep
    sweep_name = f"mnist_mlp_{args.optimiser}_sweep"
    
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="induced_metric")
    
    print(f"Starting sweep: {sweep_name}")
    print(f"Sweep ID: {sweep_id}")
    print(f"Number of runs: {args.num_runs}")
    
    # Start the sweep agent
    wandb.agent(sweep_id, train, count=args.num_runs)
    
    print("Sweep completed!")