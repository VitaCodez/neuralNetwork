# Mini-Batch Implementation Plan

You have successfully vectorized a single sample pass. Now, we will upgrade the system to handle **mini-batches** (processing multiple examples at once). This significantly speeds up training and smooths out the gradient updates.

## 1. Concept
Instead of passing a single column vector $(3, 1)$ as input, you will pass a matrix $(3, N)$, where $N$ is your batch size (e.g., 32).
*   **Weights** stay the same size.
*   **Biases** stay the same size but will need to "broadcast" across the batch.
*   **Activations** become matrices of shape $(layer\_size, batch\_size)$.

---

## 2. Changes in `network.py`

### Step A: Shape Awareness
Most of your matrix multiplication code `weights @ x + bias` explicitly supports batches already! 
*   If `x` is $(3, 32)$ and `W` is $(5, 3)$, then `W @ x` becomes $(5, 32)$.
*   NumPy automatically handles the bias addition `(5, 32) + (5, 1)` by adding the bias column to every column of your batch.

### Step B: Averaging the Gradient
When you calculate gradients for a batch, you will get a gradient "matrix" that represents the sum (or raw values) for all examples. You usually want the **average** gradient to keep your updates stable regardless of batch size.

**The Logic:**
1.  **Gradients List**: Initialize an accumulation list for your gradients (zeros).
2.  **Loop**: For each sample in the batch:
    *   Forward Pass
    *   Backpropagate -> Get `sample_gradients`
    *   Add `sample_gradients` to your accumulation list.
3.  **Finish**: Divide the accumulated gradients by `Batch_Size`.
4.  **Update**: Apply this *averaged* gradient to your weights.

*Alternative (Advanced): You can actually do full matrix-backprop without a loop, but start with the "Loop over batch, Accumulate Gradients" approach first. It is easier to debug.*

---

## 3. Changes in `teacher.py` (The Training Loop)

### Step A: Batch Generator
You need a helper to chop your training data into chunks.
*   Create a method `get_batches(data, batch_size)`.
*   It should shuffle the data (optional but recommended) and yield chunks of size `batch_size`.

### Step B: The New Fit Loop
Refactor your `Fit` method to use nested loops:
```text
For each Epoch:
    Shuffle Data
    For each Batch in Data:
        1. Initialize `total_gradients` = 0
        2. For each Example in Batch:
               gradients = network.backpropagate(example)
               total_gradients += gradients
        3. average_gradients = total_gradients / batch_size
        4. network.update_weights(average_gradients)
```

---

## 4. Checklist

- [ ] **Data Structure**: Ensure your input data can be sliced into batches.
- [ ] **Accumulator**: You need a way to add a list of numpy arrays to another list of numpy arrays (element-wise addition).
- [ ] **Averaging**: Make sure to divide by the actual batch size (the last batch might be smaller!).
