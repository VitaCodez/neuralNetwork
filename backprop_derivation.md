# Vectorized Backpropagation Derivation & Implementation Plan

This document outlines the mathematical derivation and logical steps to implement vectorized backpropagation for your neural network.

## 1. Goal
The objective is to calculate the "error" for every layer efficiently using matrices, instead of looping through individual neurons. Once we have the error for a layer, we can easily find the gradients to update weights and biases.

## 2. Notation
- **L**: The last layer (output layer).
- **l**: Any current layer we are processing.
- **l + 1**: The next layer (closer to output).
- **Activations (a)**: Output of a layer after applying tanh.
- **Weighted Inputs (z)**: Input to a layer before applying tanh.
- **y**: The True Output (target).
- **(*)**: Element-wise multiplication.
- **(@)**: Matrix multiplication.

---

## 3. The Derivation

### Step A: Output Layer Error
The error for the last layer is straightforward. It is the difference between what the network predicted and what the true value is, scaled by the slope (derivative) of the activation function.

Formula:
**Error_L = (Prediction - Target) * Derivative_of_Activation**

In your code terms (assuming tanh):
`Error_L = (Activation - True_Output) * (1 - Activation^2)`

### Step B: Hidden Layer Error
For hidden layers, we don't have a "Target". Instead, the error is the weighted sum of the errors from the layer ahead of it.

Formula:
**Error_l = (Weights_Next_Transposed @ Error_Next) * Derivative_of_Activation**

In simpler terms:
1.  Take the weights of the **next layer** (Weights_Next).
2.  **Transpose** them (swap rows/cols) so dimensions align.
3.  **Matrix Multiply** (@) them by the **error of the next layer** (Error_Next).
4.  **Element-wise Multiply** (*) the result by the derivative of the current layer's activation `(1 - Activation^2)`.

--- 

## 4. Implementation Algorithm

Here is the recipe for your code loop.

1.  **Iterate Backwards**:
    Start a loop from the last layer index down to the first.
    *Tip: use `reversed(range(len(self.weights)))`*

2.  **Check if Output Layer**:
    - If it is the last layer:
        - Use "Step A" above.
        - Store this result as `next_layer_error`.

3.  **If Hidden Layer**:
    - If it is NOT the last layer:
        - Retrieve the `next_layer_error` you calculated in the *previous loop iteration*.
        - Retrieve the weights of the layer ahead (the weights connecting current layer to the next).
        - Calculate the current error using "Step B" above.
        - *Crucial*: You need `Weights_Next.T @ next_layer_error`.
        - Update `next_layer_error` to be this new current error (so it's ready for the next iteration).

4.  **Store Gradients**:
    - Once you have the error for the current layer, the gradient for weights is:
      `Gradient_Weights = Error @ Input_from_Previous_Layer.T`
    - The gradient for biases is just the Error itself.
