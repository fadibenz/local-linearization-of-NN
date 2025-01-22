# Neural Network Feature Visualization

This notebook explores the visualization of features learned by neural networks through local linearization techniques. 
It provides hands-on experience with gradient visualization, singular value decomposition (SVD) analysis, 
and feature exploration in neural networks. 
The implementation helps understand generalized linear models (GLMs) by examining how neural networks can be viewed as GLMs during training through feature linearization.

## Key Components

- Data Generation: Creates synthetic piecewise linear functions with noise
- Network Architectures: Implements single-hidden layer networks with varying widths 
- Visualization Tools: Plots gradients, SVD components, and network predictions
- Training Framework: Includes SGD optimization with different learning rate configurations

## Main Tasks

1. Gradient Visualization
   - Visualize partial derivatives w.r.t weights and biases
   - Compare initialization vs trained states

2. SVD Analysis
   - Compute SVD of feature matrices
   - Visualize principal features and singular values

3. Two-Layer Extension
   - Implement deeper architecture
   - Extend visualization framework

## Requirements

```python
torch==1.13
ipympl
torchviz
matplotlib
numpy
```

## Usage

Run cells sequentially. Interactive widgets allow exploration of training progress and feature visualization.

## Acknowledgements

Starter code taken from CS182 (Deep Neural Networks) course at UC Berkeley. Course materials are used with permission for educational purposes.