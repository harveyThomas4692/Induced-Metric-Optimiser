"""
PyTorch implementations of custom SGD optimizers with induced metric modifications.

This module provides PyTorch equivalents of the JAX optimizers defined in jax_imp.py:
- CustomSGD: Equivalent to custom_sgd() 
- CustomSGDLog: Equivalent to custom_sgd_log()
- CustomSGDRMS: Equivalent to custom_sgd_rms()

Usage examples:

# Basic custom SGD
optimizer = CustomSGD(model.parameters(), lr=0.01, momentum=0.9, xi=0.1, beta=0.1)

# Custom SGD with loss-based metric (requires passing loss to step())
optimizer = CustomSGDLog(model.parameters(), lr=0.01, momentum=0.9, xi=0.1, beta=0.1)
loss = criterion(outputs, targets)
optimizer.step(loss_value=loss.item())

# Custom SGD with RMS scaling
optimizer = CustomSGDRMS(model.parameters(), lr=0.01, momentum=0.9, xi=0.1, beta=0.1, beta_rms=0.99)

Note: The PyTorch implementations maintain the same mathematical behavior as the JAX versions
but follow PyTorch's optimizer interface conventions.
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Optional, Union


class CustomSGD(Optimizer):
    """PyTorch implementation of custom SGD with momentum and metric modification.
    
    This optimizer corresponds to the custom_sgd function in the JAX implementation.
    """
    
    def __init__(self, params, lr=0.1, momentum=0.9, xi=0.1, beta=0.8, weight_decay=0.0):
        """Initialize the CustomSGD optimizer.
        
        Args:
            params: Model parameters to optimize
            lr: Learning rate (default: 0.1)
            momentum: Momentum factor (default: 0.9)
            xi: Scaling factor for gradient norm in metric (default: 0.1)
            beta: EMA decay rate for metric scale (default: 0.8)
            weight_decay: Weight decay factor (default: 0.0)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if xi < 0.0:
            raise ValueError(f"Invalid xi value: {xi}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, momentum=momentum, xi=xi, beta=beta, weight_decay=weight_decay)
        super(CustomSGD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            xi = group['xi']
            beta = group['beta']
            weight_decay = group['weight_decay']
            
            # Calculate gradient norm squared across all parameters in this group
            grad_norm_sq = 0.0
            for p in group['params']:
                if p.grad is not None:
                    grad_norm_sq += torch.sum(p.grad.data ** 2)
            
            trace = xi * grad_norm_sq
            
            # Get or initialize state
            if len(self.state) == 0:
                for p in group['params']:
                    self.state[p] = dict(
                        step=0,
                        momentum_buffer=torch.zeros_like(p.data),
                    )
                # Global state for this parameter group
                self.state['global'] = dict(metric_ema=0.0)
            
            # Update global metric EMA
            global_state = self.state['global']
            step = max([self.state[p]['step'] for p in group['params'] if p.grad is not None], default=0) + 1
            
            # Update EMA of metric scale
            metric_ema = beta * global_state['metric_ema'] + (1 - beta) * trace
            global_state['metric_ema'] = metric_ema
            
            # Bias correction and metric scale computation
            metric_corrected = metric_ema / (1 - beta ** step)
            metric_scale = 1 / (1 + abs(metric_corrected))
            
            # Update parameters
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                state['step'] = step
                
                # Momentum buffer
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                
                # Apply bias correction to momentum
                bias_corrected_momentum = buf / (1 - momentum ** step)
                
                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
                
                # Apply update with metric scaling
                p.data.add_(bias_corrected_momentum, alpha=-lr * metric_scale)
        
        return loss


class CustomSGDLog(Optimizer):
    """PyTorch implementation of custom SGD with loss-based metric modification.
    
    This optimizer corresponds to the custom_sgd_log function in the JAX implementation.
    """
    
    def __init__(self, params, lr=0.1, momentum=0.9, xi=0.1, beta=0.8, weight_decay=0.0):
        """Initialize the CustomSGDLog optimizer.
        
        Args:
            params: Model parameters to optimize
            lr: Learning rate (default: 0.1)
            momentum: Momentum factor (default: 0.9)
            xi: Scaling factor for gradient norm in metric (default: 0.1)
            beta: EMA decay rate for metric scale (default: 0.8)
            weight_decay: Weight decay factor (default: 0.0)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if xi < 0.0:
            raise ValueError(f"Invalid xi value: {xi}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, momentum=momentum, xi=xi, beta=beta, weight_decay=weight_decay)
        super(CustomSGDLog, self).__init__(params, defaults)
    
    def step(self, loss_value, closure=None):
        """Perform a single optimization step.
        
        Args:
            loss_value: Current loss value (required for metric computation)
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if loss_value is None:
            raise ValueError("loss_value is required for CustomSGDLog optimizer")
        
        # Convert loss_value to tensor if it's a scalar
        if not isinstance(loss_value, torch.Tensor):
            loss_value = torch.tensor(loss_value, dtype=torch.float32)
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            xi = group['xi']
            beta = group['beta']
            weight_decay = group['weight_decay']
            
            # Calculate gradient norm squared across all parameters in this group
            grad_norm_sq = 0.0
            for p in group['params']:
                if p.grad is not None:
                    grad_norm_sq += torch.sum(p.grad.data ** 2)
            
            trace = xi * grad_norm_sq
            
            # Get or initialize state
            if len(self.state) == 0:
                for p in group['params']:
                    self.state[p] = dict(
                        step=0,
                        momentum_buffer=torch.zeros_like(p.data),
                    )
                # Global state for this parameter group
                self.state['global'] = dict(metric_ema=0.0)
            
            # Update global metric EMA
            global_state = self.state['global']
            step = max([self.state[p]['step'] for p in group['params'] if p.grad is not None], default=0) + 1
            
            # Update EMA of metric scale
            metric_ema = beta * global_state['metric_ema'] + (1 - beta) * trace
            global_state['metric_ema'] = metric_ema
            
            # Bias correction and loss-based metric scale computation
            metric_corrected = metric_ema / (1 - beta ** step)
            metric_scale = loss_value / (loss_value ** 2 + metric_corrected)
            
            # Update parameters
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                state['step'] = step
                
                # Momentum buffer
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                
                # Apply bias correction to momentum
                bias_corrected_momentum = buf / (1 - momentum ** step)
                
                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
                
                # Apply update with metric scaling
                p.data.add_(bias_corrected_momentum, alpha=-lr * metric_scale)
        
        return loss


class CustomSGDRMS(Optimizer):
    """PyTorch implementation of custom SGD with momentum, metric modification, and RMS scaling.
    
    This optimizer corresponds to the custom_sgd_rms function in the JAX implementation.
    """
    
    def __init__(self, params, lr=0.1, momentum=0.9, xi=0.1, beta=0.8, beta_rms=0.99, weight_decay=0.0, eps=1e-8):
        """Initialize the CustomSGDRMS optimizer.
        
        Args:
            params: Model parameters to optimize
            lr: Learning rate (default: 0.1)
            momentum: Momentum factor (default: 0.9)
            xi: Scaling factor for gradient norm in metric (default: 0.1)
            beta: EMA decay rate for metric scale (default: 0.8)
            beta_rms: EMA decay rate for RMS (default: 0.99)
            weight_decay: Weight decay factor (default: 0.0)
            eps: Small constant for numerical stability (default: 1e-8)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if xi < 0.0:
            raise ValueError(f"Invalid xi value: {xi}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= beta_rms <= 1.0:
            raise ValueError(f"Invalid beta_rms parameter: {beta_rms}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")
            
        defaults = dict(lr=lr, momentum=momentum, xi=xi, beta=beta, beta_rms=beta_rms, 
                       weight_decay=weight_decay, eps=eps)
        super(CustomSGDRMS, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            xi = group['xi']
            beta = group['beta']
            beta_rms = group['beta_rms']
            weight_decay = group['weight_decay']
            eps = group['eps']
            
            # Get or initialize state
            if len(self.state) == 0:
                for p in group['params']:
                    self.state[p] = dict(
                        step=0,
                        momentum_buffer=torch.zeros_like(p.data),
                        rms_ema=torch.zeros_like(p.data),
                    )
                # Global state for this parameter group
                self.state['global'] = dict(metric_ema=0.0)
            
            # Update step counter
            step = max([self.state[p]['step'] for p in group['params'] if p.grad is not None], default=0) + 1
            
            # Update RMS EMA for each parameter and calculate RMS-corrected gradient norm
            grad_norm_sq = 0.0
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                state['step'] = step
                grad = p.grad.data
                
                # Update EMA of RMS (Root Mean Square) - per parameter
                rms_ema = state['rms_ema']
                rms_ema.mul_(beta_rms).addcmul_(grad, grad, value=1 - beta_rms)
                
                # Bias correction for RMS
                rms_corrected = rms_ema / (1 - beta_rms ** step)
                
                # Calculate contribution to gradient norm with RMS scaling (matching JAX implementation)
                grad_norm_contribution = grad ** 2 / (torch.sqrt(rms_corrected) + eps)
                grad_norm_sq += torch.sum(grad_norm_contribution)
            
            trace = xi * grad_norm_sq
            
            # Update global metric EMA
            global_state = self.state['global']
            metric_ema = beta * global_state['metric_ema'] + (1 - beta) * trace
            global_state['metric_ema'] = metric_ema
            
            # Bias correction and metric scale computation
            metric_corrected = metric_ema / (1 - beta ** step)
            metric_scale = 1 / (1 + abs(metric_corrected))
            
            # Update parameters
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Get bias-corrected RMS
                rms_corrected = state['rms_ema'] / (1 - beta_rms ** step)
                
                # Momentum buffer
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                
                # Apply bias correction to momentum
                bias_corrected_momentum = buf / (1 - momentum ** step)
                
                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
                
                # Apply update with metric scaling and RMS normalization
                rms_denom = torch.sqrt(rms_corrected) + eps
                scaled_update = bias_corrected_momentum / rms_denom
                p.data.add_(scaled_update, alpha=-lr * metric_scale)
        
        return loss
