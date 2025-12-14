# Numpy Neural Network From Scratch

A fully connected Neural Network implemented from scratch in Python, now upgraded with **NumPy** for vectorized operations. This project demonstrates the core concepts of deep learning—forward propagation, backpropagation, and gradient descent—without relying on high-level DL frameworks like PyTorch or TensorFlow.

##  Key Features

- **Vectorised Implementation**: Replaced original loop-based logic with efficient matrix operations using NumPy.
- **Dynamic Architecture**: Easily configurable layer sizes and depths. Current setup: `[32, 16, 1]`.
- **Mini-Batch Gradient Descent**: Implements batch processing for stable and faster convergence.
- **Custom Backpropagation**: Gradients are calculated manually using the chain rule (no autograd!).

##  Current Task: Function Approximation

The network is currently trained to predict the output of a complex non-linear function:

$$
y = \sin(3x_1) + x_2^2 \cdot \cos(3x_3) + x_4^3
$$

### Performance
- **Time spent building**: 30h+
- **Accuracy**: Capable of achieving extremely low error rates (MME $\approx$ 0.0001) on synthetic data.

##  Installation & Usage

### Prerequisites
You will need Python installed along with `numpy`.

```bash
pip install numpy
```

### Running the Project
Navigate to the `NN` directory and run the main script:

```bash
python main.py
```

This will generate a synthetic dataset, train the network for a set number of epochs, and output the final Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

##  Future Plans

- [] Implement training on a real-world dataset (e.g., Boston House Prices).
- [] Compare performance with a standard Scikit-learn MLP implementation.
- [] Further optimize the current backpropagation algorithm.

##  Author's Note

> "This project was for learning purposes, which I think it fulfilled. It started as a pure Python implementation and is now vectorized. If you find this repo, I wish you fun with my half-working NN!"

---
*Created by skacel*
