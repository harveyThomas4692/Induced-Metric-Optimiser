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
import os
import requests

# Import shared models
from shared_models import custom_sgd, custom_sgd_log, custom_sgd_rms, MiniGPT, SGDState, _compute_grad_norm_squared
import pandas as pd

# Print JAX device information
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")


# Parse command line arguments
parser = argparse.ArgumentParser(description='Tiny Shakespeare MiniGPT Hyperparameter Sweep')
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

# Fixed architecture configuration
ARCHITECTURE_CONFIG = {
    'batch_size': 256,
    'n_epochs': 100,
    'seq_len': 256,  # Increased for better language modeling
    'embed_dim': 128,  # Fixed architecture  
    'num_heads': 4,  # Fixed architecture
    'num_layers': 4,  # Fixed architecture
    'dropout_rate': 0.1  # Fixed architecture
}

def download_shakespeare():
    """Download the tiny Shakespeare dataset"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "tinyshakespeare.txt"
    
    if not os.path.exists(filename):
        print("Downloading tiny Shakespeare dataset...")
        response = requests.get(url)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Download complete!")
    
    return filename

def load_shakespeare(seq_len=128, val_split=0.1):
    """Load and preprocess the tiny Shakespeare dataset for character-level modeling"""
    
    # Download dataset if not exists
    filename = download_shakespeare()
    
    # Read the text
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Text length: {len(text)} characters")
    
    # Create character-to-index mapping
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"First 10 chars: {chars[:10]}")
    
    # Convert text to indices
    data = jnp.array([char_to_idx[ch] for ch in text])
    
    # Split into train and validation
    split_idx = int(len(data) * (1 - val_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data, vocab_size, char_to_idx, idx_to_char

def create_text_batches(data, seq_len, batch_size, seed):
    """Create batches for language modeling (input, target) pairs"""
    key = jax.random.PRNGKey(seed)
    
    # Calculate number of sequences we can fit
    num_sequences = (len(data) - 1) // seq_len
    
    # Create input-target pairs
    inputs = []
    targets = []
    
    for i in range(num_sequences):
        start_idx = i * seq_len
        end_idx = start_idx + seq_len
        if end_idx < len(data):
            inputs.append(data[start_idx:end_idx])
            targets.append(data[start_idx + 1:end_idx + 1])
    
    inputs = jnp.array(inputs)
    targets = jnp.array(targets)
    
    # Shuffle the data
    perm = jax.random.permutation(key, len(inputs))
    inputs = inputs[perm]
    targets = targets[perm]
    
    # Create batches
    num_batches = len(inputs) // batch_size
    batches = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batches.append((inputs[start_idx:end_idx], targets[start_idx:end_idx]))
    
    return batches

def loss_fn(params, x, y, model, key):
    """Compute cross-entropy loss for language modeling"""
    logits = model.apply(params, x, train=True, rngs={'dropout': key})
    # Flatten for cross-entropy calculation
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = y.reshape(-1)
    return optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat).mean()

def perplexity_fn(params, data_batches, model):
    """Compute perplexity over all batches"""
    total_loss = 0.0
    total_tokens = 0
    
    for x_batch, y_batch in data_batches:
        # Use deterministic=True for evaluation (no dropout)
        logits = model.apply(params, x_batch, train=False)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = y_batch.reshape(-1)
        batch_loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat).mean()
        total_loss += batch_loss * y_batch.size
        total_tokens += y_batch.size
    
    avg_loss = total_loss / total_tokens
    return jnp.exp(avg_loss)

# Training functions for each optimizer
def train_adam(config, seed):
    """Train MiniGPT with Adam optimizer"""
    
    # Load data
    train_data, val_data, vocab_size, char_to_idx, idx_to_char = load_shakespeare(
        seq_len=config['seq_len'], val_split=0.1
    )
    
    # Initialize model
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate'],
        max_seq_len=config['seq_len']
    )
    
    # Create data loaders
    train_batches = create_text_batches(train_data, config['seq_len'], config['batch_size'], seed)
    val_batches = create_text_batches(val_data, config['seq_len'], config['batch_size'], seed + 1)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, config['seq_len']), dtype=jnp.int32)
    init_key, dropout_key = jax.random.split(key, 2)
    params = model.init(init_key, dummy_input, train=True, rngs={'dropout': dropout_key})
    
    # Initialize optimizer
    optimizer = optax.adam(
        learning_rate=config['learning_rate'], 
        b1=config['beta1'], 
        b2=config['beta2'],
        eps=config['eps']
    )
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    val_perplexities = []
    
    min_val_perplexity = float('inf')
    min_perp_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def update_step(params, opt_state, x, y, key):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model, key))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Pre-generate all epoch keys and batch keys
    epoch_keys = []
    batch_keys_per_epoch = []
    for epoch in range(n_epochs):
        epoch_key = jax.random.PRNGKey(seed + epoch)
        epoch_keys.append(epoch_key)
        batch_keys = [jax.random.fold_in(epoch_key, i) for i in range(len(train_batches))]
        batch_keys_per_epoch.append(batch_keys)
    
    train_time = 0.0
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Training epoch
        train_time += time.time()
        for i, (x_batch, y_batch) in enumerate(train_batches):
            batch_key = batch_keys_per_epoch[epoch][i]
            params, opt_state, loss = update_step(params, opt_state, x_batch, y_batch, batch_key)
            epoch_losses.append(loss)
        train_time -= time.time()
        
        avg_train_loss = jnp.mean(jnp.array(epoch_losses))
        train_losses.append(float(avg_train_loss))
        
        # Validation every val_freq epochs
        if (epoch + 1) % args.val_freq == 0:
            val_perplexity = perplexity_fn(params, val_batches, model)
            val_perplexities.append(float(val_perplexity))
            
            if val_perplexity < min_val_perplexity:
                min_val_perplexity = val_perplexity
                min_perp_epoch = epoch
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_perplexity': val_perplexity,
                'min_val_perplexity': min_val_perplexity,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss
            })
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, time={train_time:.2f}s")
    
    return {
        'min_val_perplexity': float(min_val_perplexity),
        'min_perp_epoch': min_perp_epoch,
        'final_train_loss': train_losses[-1] if train_losses else 0.0,
        'final_val_perplexity': val_perplexities[-1] if val_perplexities else float('inf'),
        'train_losses': train_losses,
        'val_perplexities': val_perplexities
    }

def train_sgd(config, seed):
    """Train MiniGPT with SGD optimizer"""
    
    # Load data
    train_data, val_data, vocab_size, char_to_idx, idx_to_char = load_shakespeare(
        seq_len=config['seq_len'], val_split=0.1
    )
    
    # Initialize model
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate'],
        max_seq_len=config['seq_len']
    )
    
    # Create data loaders
    train_batches = create_text_batches(train_data, config['seq_len'], config['batch_size'], seed)
    val_batches = create_text_batches(val_data, config['seq_len'], config['batch_size'], seed + 1)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, config['seq_len']), dtype=jnp.int32)
    params = model.init(key, dummy_input, train=True)
    
    # Initialize optimizer
    optimizer = optax.sgd(
        learning_rate=config['learning_rate'], 
        momentum=config['momentum']
    )
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    val_perplexities = []
    
    min_val_perplexity = float('inf')
    min_perp_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def update_step(params, opt_state, x, y, key):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model, key))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Pre-generate all epoch keys and batch keys
    epoch_keys = []
    batch_keys_per_epoch = []
    for epoch in range(n_epochs):
        epoch_key = jax.random.PRNGKey(seed + epoch)
        epoch_keys.append(epoch_key)
        batch_keys = [jax.random.fold_in(epoch_key, i) for i in range(len(train_batches))]
        batch_keys_per_epoch.append(batch_keys)
    
    train_time = 0.0
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Training epoch
        train_time += time.time()
        for i, (x_batch, y_batch) in enumerate(train_batches):
            batch_key = batch_keys_per_epoch[epoch][i]
            params, opt_state, loss = update_step(params, opt_state, x_batch, y_batch, batch_key)
            epoch_losses.append(loss)
        train_time -= time.time()
        
        avg_train_loss = jnp.mean(jnp.array(epoch_losses))
        train_losses.append(float(avg_train_loss))
        
        # Validation every val_freq epochs
        if (epoch + 1) % args.val_freq == 0:
            val_perplexity = perplexity_fn(params, val_batches, model)
            val_perplexities.append(float(val_perplexity))
            
            if val_perplexity < min_val_perplexity:
                min_val_perplexity = val_perplexity
                min_perp_epoch = epoch
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_perplexity': val_perplexity,
                'min_val_perplexity': min_val_perplexity,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss
            })
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, time={train_time:.2f}s")
    
    return {
        'min_val_perplexity': float(min_val_perplexity),
        'min_perp_epoch': min_perp_epoch,
        'final_train_loss': train_losses[-1] if train_losses else 0.0,
        'final_val_perplexity': val_perplexities[-1] if val_perplexities else float('inf'),
        'train_losses': train_losses,
        'val_perplexities': val_perplexities
    }

def train_sgd_metric(config, seed):
    """Train MiniGPT with custom SGD metric optimizer"""
    
    # Load data
    train_data, val_data, vocab_size, char_to_idx, idx_to_char = load_shakespeare(
        seq_len=config['seq_len'], val_split=0.1
    )
    
    # Initialize model
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate'],
        max_seq_len=config['seq_len']
    )
    
    # Create data loaders
    train_batches = create_text_batches(train_data, config['seq_len'], config['batch_size'], seed)
    val_batches = create_text_batches(val_data, config['seq_len'], config['batch_size'], seed + 1)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, config['seq_len']), dtype=jnp.int32)
    params = model.init(key, dummy_input, train=True)
    
    # Initialize optimizer
    optimizer = custom_sgd(
        learning_rate=config['learning_rate'], 
        momentum=config['momentum'],
        xi=config['xi'], 
        beta=config['beta'],
        weight_decay=config['weight_decay']
    )
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    val_perplexities = []
    
    min_val_perplexity = float('inf')
    min_perp_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def update_step(params, opt_state, x, y, key):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model, key))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Pre-generate all epoch keys and batch keys
    epoch_keys = []
    batch_keys_per_epoch = []
    for epoch in range(n_epochs):
        epoch_key = jax.random.PRNGKey(seed + epoch)
        epoch_keys.append(epoch_key)
        batch_keys = [jax.random.fold_in(epoch_key, i) for i in range(len(train_batches))]
        batch_keys_per_epoch.append(batch_keys)
    
    train_time = 0.0
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Training epoch
        train_time += time.time()
        for i, (x_batch, y_batch) in enumerate(train_batches):
            batch_key = batch_keys_per_epoch[epoch][i]
            params, opt_state, loss = update_step(params, opt_state, x_batch, y_batch, batch_key)
            epoch_losses.append(loss)
        train_time -= time.time()
        
        avg_train_loss = jnp.mean(jnp.array(epoch_losses))
        train_losses.append(float(avg_train_loss))
        
        # Validation every val_freq epochs
        if (epoch + 1) % args.val_freq == 0:
            val_perplexity = perplexity_fn(params, val_batches, model)
            val_perplexities.append(float(val_perplexity))
            
            if val_perplexity < min_val_perplexity:
                min_val_perplexity = val_perplexity
                min_perp_epoch = epoch
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_perplexity': val_perplexity,
                'min_val_perplexity': min_val_perplexity,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss
            })
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, time={train_time:.2f}s")
    
    return {
        'min_val_perplexity': float(min_val_perplexity),
        'min_perp_epoch': min_perp_epoch,
        'final_train_loss': train_losses[-1] if train_losses else 0.0,
        'final_val_perplexity': val_perplexities[-1] if val_perplexities else float('inf'),
        'train_losses': train_losses,
        'val_perplexities': val_perplexities
    }

def train_sgd_log_metric(config, seed):
    """Train MiniGPT with custom SGD log metric optimizer"""
    
    # Load data
    train_data, val_data, vocab_size, char_to_idx, idx_to_char = load_shakespeare(
        seq_len=config['seq_len'], val_split=0.1
    )
    
    # Initialize model
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate'],
        max_seq_len=config['seq_len']
    )
    
    # Create data loaders
    train_batches = create_text_batches(train_data, config['seq_len'], config['batch_size'], seed)
    val_batches = create_text_batches(val_data, config['seq_len'], config['batch_size'], seed + 1)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, config['seq_len']), dtype=jnp.int32)
    params = model.init(key, dummy_input, train=True)
    
    # Initialize optimizer
    optimizer = custom_sgd_log(
        learning_rate=config['learning_rate'], 
        momentum=config['momentum'],
        xi=config['xi'], 
        beta=config['beta'],
        weight_decay=config['weight_decay']
    )
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    val_perplexities = []
    
    min_val_perplexity = float('inf')
    min_perp_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit  
    def update_step(params, opt_state, x, y, key):
        def loss_and_grad_fn(p):
            loss = loss_fn(p, x, y, model, key)
            return loss, loss
        
        (loss, loss_for_opt), grads = jax.value_and_grad(loss_and_grad_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, loss_for_opt, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Pre-generate all epoch keys and batch keys
    epoch_keys = []
    batch_keys_per_epoch = []
    for epoch in range(n_epochs):
        epoch_key = jax.random.PRNGKey(seed + epoch)
        epoch_keys.append(epoch_key)
        batch_keys = [jax.random.fold_in(epoch_key, i) for i in range(len(train_batches))]
        batch_keys_per_epoch.append(batch_keys)
    
    train_time = 0.0
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Training epoch
        train_time += time.time()
        for i, (x_batch, y_batch) in enumerate(train_batches):
            batch_key = batch_keys_per_epoch[epoch][i]
            params, opt_state, loss = update_step(params, opt_state, x_batch, y_batch, batch_key)
            epoch_losses.append(loss)
        train_time -= time.time()
        
        avg_train_loss = jnp.mean(jnp.array(epoch_losses))
        train_losses.append(float(avg_train_loss))
        
        # Validation every val_freq epochs
        if (epoch + 1) % args.val_freq == 0:
            val_perplexity = perplexity_fn(params, val_batches, model)
            val_perplexities.append(float(val_perplexity))
            
            if val_perplexity < min_val_perplexity:
                min_val_perplexity = val_perplexity
                min_perp_epoch = epoch
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_perplexity': val_perplexity,
                'min_val_perplexity': min_val_perplexity,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss
            })
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, time={train_time:.2f}s")
    
    return {
        'min_val_perplexity': float(min_val_perplexity),
        'min_perp_epoch': min_perp_epoch,
        'final_train_loss': train_losses[-1] if train_losses else 0.0,
        'final_val_perplexity': val_perplexities[-1] if val_perplexities else float('inf'),
        'train_losses': train_losses,
        'val_perplexities': val_perplexities
    }

def train_adamw(config, seed):
    """Train MiniGPT with AdamW optimizer"""
    
    # Load data
    train_data, val_data, vocab_size, char_to_idx, idx_to_char = load_shakespeare(
        seq_len=config['seq_len'], val_split=0.1
    )
    
    # Initialize model
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate'],
        max_seq_len=config['seq_len']
    )
    
    # Create data loaders
    train_batches = create_text_batches(train_data, config['seq_len'], config['batch_size'], seed)
    val_batches = create_text_batches(val_data, config['seq_len'], config['batch_size'], seed + 1)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, config['seq_len']), dtype=jnp.int32)
    params = model.init(key, dummy_input, train=True)
    
    # Initialize optimizer
    optimizer = optax.adamw(
        learning_rate=config['learning_rate'], 
        b1=config['beta1'], 
        b2=config['beta2'],
        eps=config['eps'], 
        weight_decay=config['weight_decay']
    )
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    val_perplexities = []
    
    min_val_perplexity = float('inf')
    min_perp_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def update_step(params, opt_state, x, y, key):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model, key))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Pre-generate all epoch keys and batch keys
    epoch_keys = []
    batch_keys_per_epoch = []
    for epoch in range(n_epochs):
        epoch_key = jax.random.PRNGKey(seed + epoch)
        epoch_keys.append(epoch_key)
        batch_keys = [jax.random.fold_in(epoch_key, i) for i in range(len(train_batches))]
        batch_keys_per_epoch.append(batch_keys)
    
    train_time = 0.0
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Training epoch
        train_time += time.time()
        for i, (x_batch, y_batch) in enumerate(train_batches):
            batch_key = batch_keys_per_epoch[epoch][i]
            params, opt_state, loss = update_step(params, opt_state, x_batch, y_batch, batch_key)
            epoch_losses.append(loss)
        train_time -= time.time()
        
        avg_train_loss = jnp.mean(jnp.array(epoch_losses))
        train_losses.append(float(avg_train_loss))
        
        # Validation every val_freq epochs
        if (epoch + 1) % args.val_freq == 0:
            val_perplexity = perplexity_fn(params, val_batches, model)
            val_perplexities.append(float(val_perplexity))
            
            if val_perplexity < min_val_perplexity:
                min_val_perplexity = val_perplexity
                min_perp_epoch = epoch
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_perplexity': val_perplexity,
                'min_val_perplexity': min_val_perplexity,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss
            })
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, time={train_time:.2f}s")
    
    return {
        'min_val_perplexity': float(min_val_perplexity),
        'min_perp_epoch': min_perp_epoch,
        'final_train_loss': train_losses[-1] if train_losses else 0.0,
        'final_val_perplexity': val_perplexities[-1] if val_perplexities else float('inf'),
        'train_losses': train_losses,
        'val_perplexities': val_perplexities
    }

def train_muon(config, seed):
    """Train MiniGPT with Muon optimizer"""
    
    # Load data
    train_data, val_data, vocab_size, char_to_idx, idx_to_char = load_shakespeare(
        seq_len=config['seq_len'], val_split=0.1
    )
    
    # Initialize model
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate'],
        max_seq_len=config['seq_len']
    )
    
    # Create data loaders
    train_batches = create_text_batches(train_data, config['seq_len'], config['batch_size'], seed)
    val_batches = create_text_batches(val_data, config['seq_len'], config['batch_size'], seed + 1)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, config['seq_len']), dtype=jnp.int32)
    params = model.init(key, dummy_input, train=True)
    
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
    val_perplexities = []
    
    min_val_perplexity = float('inf')
    min_perp_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def update_step(params, opt_state, x, y, key):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model, key))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Pre-generate all epoch keys and batch keys
    epoch_keys = []
    batch_keys_per_epoch = []
    for epoch in range(n_epochs):
        epoch_key = jax.random.PRNGKey(seed + epoch)
        epoch_keys.append(epoch_key)
        batch_keys = [jax.random.fold_in(epoch_key, i) for i in range(len(train_batches))]
        batch_keys_per_epoch.append(batch_keys)
    
    train_time = 0.0
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Training epoch
        train_time += time.time()
        for i, (x_batch, y_batch) in enumerate(train_batches):
            batch_key = batch_keys_per_epoch[epoch][i]
            params, opt_state, loss = update_step(params, opt_state, x_batch, y_batch, batch_key)
            epoch_losses.append(loss)
        train_time -= time.time()
        
        avg_train_loss = jnp.mean(jnp.array(epoch_losses))
        train_losses.append(float(avg_train_loss))
        
        # Validation every val_freq epochs
        if (epoch + 1) % args.val_freq == 0:
            val_perplexity = perplexity_fn(params, val_batches, model)
            val_perplexities.append(float(val_perplexity))
            
            if val_perplexity < min_val_perplexity:
                min_val_perplexity = val_perplexity
                min_perp_epoch = epoch
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_perplexity': val_perplexity,
                'min_val_perplexity': min_val_perplexity,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss
            })
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, time={train_time:.2f}s")
    
    return {
        'min_val_perplexity': float(min_val_perplexity),
        'min_perp_epoch': min_perp_epoch,
        'final_train_loss': train_losses[-1] if train_losses else 0.0,
        'final_val_perplexity': val_perplexities[-1] if val_perplexities else float('inf'),
        'train_losses': train_losses,
        'val_perplexities': val_perplexities
    }

def train_sgd_rms(config, seed):
    """Train MiniGPT with custom SGD RMS optimizer"""
    
    # Load data
    train_data, val_data, vocab_size, char_to_idx, idx_to_char = load_shakespeare(
        seq_len=config['seq_len'], val_split=0.1
    )
    
    # Initialize model
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate'],
        max_seq_len=config['seq_len']
    )
    
    # Create data loaders
    train_batches = create_text_batches(train_data, config['seq_len'], config['batch_size'], seed)
    val_batches = create_text_batches(val_data, config['seq_len'], config['batch_size'], seed + 1)
    
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, config['seq_len']), dtype=jnp.int32)
    params = model.init(key, dummy_input, train=True)
    
    # Initialize optimizer
    optimizer = custom_sgd_rms(
        learning_rate=config['learning_rate'], 
        momentum=config['momentum'],
        xi=config['xi'], 
        beta=config['beta'],
        beta_rms=config['beta_rms'], 
        eps=config['eps'],
        weight_decay=config['weight_decay']
    )
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    val_perplexities = []
    
    min_val_perplexity = float('inf')
    min_perp_epoch = 0
    n_epochs = config['n_epochs']
    
    @jax.jit
    def update_step(params, opt_state, x, y, key):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model, key))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Pre-generate all epoch keys and batch keys
    epoch_keys = []
    batch_keys_per_epoch = []
    for epoch in range(n_epochs):
        epoch_key = jax.random.PRNGKey(seed + epoch)
        epoch_keys.append(epoch_key)
        batch_keys = [jax.random.fold_in(epoch_key, i) for i in range(len(train_batches))]
        batch_keys_per_epoch.append(batch_keys)
    
    train_time = 0.0
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Training epoch
        train_time += time.time()
        for i, (x_batch, y_batch) in enumerate(train_batches):
            batch_key = batch_keys_per_epoch[epoch][i]
            params, opt_state, loss = update_step(params, opt_state, x_batch, y_batch, batch_key)
            epoch_losses.append(loss)
        train_time -= time.time()
        
        avg_train_loss = jnp.mean(jnp.array(epoch_losses))
        train_losses.append(float(avg_train_loss))
        
        # Validation every val_freq epochs
        if (epoch + 1) % args.val_freq == 0:
            val_perplexity = perplexity_fn(params, val_batches, model)
            val_perplexities.append(float(val_perplexity))
            
            if val_perplexity < min_val_perplexity:
                min_val_perplexity = val_perplexity
                min_perp_epoch = epoch
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_perplexity': val_perplexity,
                'min_val_perplexity': min_val_perplexity,
                'train_time_seconds': train_time
            })
        else:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss
            })
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, time={train_time:.2f}s")
    
    return {
        'min_val_perplexity': float(min_val_perplexity),
        'min_perp_epoch': min_perp_epoch,
        'final_train_loss': train_losses[-1] if train_losses else 0.0,
        'final_val_perplexity': val_perplexities[-1] if val_perplexities else float('inf'),
        'train_losses': train_losses,
        'val_perplexities': val_perplexities
    }

def get_sweep_config(optimizer_name):
    """Get W&B sweep configuration for the given optimizer"""
    
    base_config = {
        'name': f'shakespeare_minigpt_sweep_{optimizer_name}_{args.index}',
        'method': args.search,
        'metric': {
            'name': 'min_val_perplexity',
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
            'n_epochs': {'value': 200}
        }
    elif optimizer_name == 'adamw':
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1},
            'beta1': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'beta2': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
            'eps': {'values': [1e-8]},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'n_epochs': {'value': 200}
        }
    elif optimizer_name == 'sgd':
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1},
            'momentum': {'distribution': 'uniform', 'min': 0.1, 'max': 0.99},
            'n_epochs': {'value': 200}
        }
    elif optimizer_name in ['sgd_metric', 'sgd_log_metric']:
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1},
            'momentum': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'xi': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-3},
            'beta': {'distribution': 'uniform', 'min': 0.6, 'max': 0.9},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'n_epochs': {'value': 200}
        }
    elif optimizer_name == 'sgd_rms':
        base_config['parameters'] = {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1},
            'momentum': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'xi': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-3},
            'beta': {'distribution': 'uniform', 'min': 0.6, 'max': 0.9},
            'beta_rms': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
            'eps': {'values': [1e-8]},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
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
            'n_epochs': {'value': 200}
        }
    
    return base_config

def train():
    """Training function for W&B sweep"""
    
    # Add tags
    tags = [args.optimiser, "shakespeare_minigpt", f"run_{args.index}", args.search]
    
    # Initialize wandb
    wandb.init(project="induced_metric", tags=tags)
    
    # Get config from wandb and merge with fixed architecture
    config = dict(wandb.config)
    config.update(ARCHITECTURE_CONFIG)  # Add fixed architecture parameters
    
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
    
    # Run training (single seed per run, like CIFAR)
    results = train_fn(dict(config), seed)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Log final results
    wandb.log({
        'final_min_val_perplexity': results['min_val_perplexity'],
        'final_min_perp_epoch': results['min_perp_epoch'],
        'final_train_loss': results['final_train_loss'],
        'final_val_perplexity': results['final_val_perplexity'],
        'total_training_time_sec': training_time,
        'total_training_time_min': training_time / 60,
        'seed': seed,
        'optimizer': args.optimiser,
        'architecture': 'MiniGPT'
    })
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best validation perplexity: {results['min_val_perplexity']:.6f}")
    
    wandb.finish()

if __name__ == "__main__":
    print(f"Starting sweep for optimizer: {args.optimiser}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Run index: {args.index}")
    print(f"Search method: {args.search}")
    
    # Get sweep configuration
    sweep_config = get_sweep_config(args.optimiser)
    
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="induced_metric")
    
    print(f"Sweep ID: {sweep_id}")
    
    # Start the sweep agent
    wandb.agent(sweep_id, train, count=args.num_runs)
    
    print("Sweep completed!")
