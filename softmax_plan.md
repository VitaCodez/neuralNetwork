# Softmax and Cross-Entropy Implementation Plan

This guide outlines how to upgrade your neural network from a simple regressor (predicting continuous numbers) to a classifier (predicting probabilities for classes) using **Softmax** and **Cross-Entropy Loss**.

## 1. The Core Concepts

### Softmax
Instead of outputting raw numbers (like `2.5` or `-1.2`), we want our network to output **probabilities**.
- All outputs must be between `0` and `1`.
- The sum of all outputs for a single sample must be `1.0`.

### Cross-Entropy Loss
This is the standard "cost function" for classification.
- It penalizes the network heavily if it tries to say the correct answer has low probability.
- It pairs perfectly with Softmax.

## 2. Implementation Steps

### Step 1: Create the Softmax Helper
You need a function that converts raw scores into probabilities. Since you are using batch processing (matrices), you must be careful with dimensions.

**Logic:**
```python
def softmax(z):
    # z shape: (output_neurons, batch_size)
    
    # 1. Shift values for numerical stability (prevents overflow errors)
    # subtracting the max value doesn't change the result but keeps numbers small
    shift_z = z - np.max(z, axis=0, keepdims=True)
    
    # 2. Exponentiate
    exp_z = np.exp(shift_z)
    
    # 3. Normalize (divide by sum)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
```

### Step 2: Update the Forward Pass (`activate`)
In `network.py`, modify your `activate` method to use Softmax ONLY on the very last layer.

**Current Logic:**
- Loop through layers.
- If last layer: use Linear (no activation).
- Else: use Tanh.

**New Logic:**
- Loop through layers.
- If last layer: apply **softmax(out)**.
- Else: use **np.tanh(out)**.

### Step 3: The "Magic" of Backpropagation
You might think the derivative of Softmax + Cross-Entropy is complicated. It is not!

The gradient for the final layer simplifies to:
`error = predicted_probability - target_one_hot`

**What this means for your code:**
In `backpropagate`, the calculation for the initial `error` term (the one for the last layer) essentially **stays the same**:
```python
# Before (MSE):
error.append(layer_output - true_output)

# Now (Cross-Entropy):
error.append(layer_output - true_output) 
```
*Note: Make sure `true_output` is the correct shape!*

### Step 4: Update the Teacher (`teacher.py`)

#### A. One-Hot Encoding
Your network outputs a vector of probabilities (e.g., 10 neurons for digits 0-9). 
Your training data `correct_train` currently might look like a single number: `[5]`.
It needs to look like a vector: `[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]`.

You need to ensure your dataset handles this conversion before passing it to `backpropagate`.

#### B. Testing Metric (Accuracy vs Loss)
For classification, "Mean Squared Error" is hard to interpret. It's better to print **Accuracy**.

**Logic for Accuracy:**
1. Get the network prediction: `pred_class = np.argmax(activations[-1], axis=0)`
2. Get the true class: `true_class = np.argmax(true_output, axis=0)`
3. Compare: `matches = pred_class == true_class`
4. Accuracy: `np.mean(matches)`

## Summary Checklist

- [ ] Add `softmax` method to `neuronNetwork` class.
- [ ] Update `activate` to use `softmax` for the final layer output.
- [ ] Ensure `backpropagate` is receiving One-Hot Encoded target data.
- [ ] Update `teacher.py` to calculate and print **Accuracy** instead of raw error.
