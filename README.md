# Assignment 2: Classify
contributed by < [`陳彥廷`](https://github.com/CYT701/ComputerArchitecture2024) >

## Artificial Neural Network (ANN)

![image](https://hackmd.io/_uploads/HJwugdmlyg.png)

An **artificial neural network** is an interconnected group of nodes, inspired by a simplification of neurons in a brain. Here, each circular node represents an artificial neuron and an arrow represents a connection from the output of one artificial neuron to the input of another.


### ReLU (Rectified Linear Unit)
#### Activation Function

The Activation Function standardizes the calculated input by regulating its "range of output values" and the "relationship between output values".

#### ReLU
![image](https://hackmd.io/_uploads/SkHrC5Ng1x.png)
The mathematical expression for ReLU is:

\begin{equation}
f(x) = max(0, x)
\end{equation}

***Characteristics of ReLU:***
1. **Non-linearity:** Despite its simple formula, ReLU is a nonlinear function. This nonlinearity allows neural networks to learn and represent complex relationships, which is one reason for their powerful capabilities.
2. **High computational efficiency:** The ReLU computation is very simple, involving just a comparison between the input value and 0, so its computational cost is low, making it very efficient in practice.
3. **Addresses the vanishing gradient problem:** In activation functions like Sigmoid and Tanh, when the input is very large or very small, the output gradient becomes very close to 0, which slows down gradient updates during backpropagation, affecting the learning speed of the neural network. With ReLU, the gradient is 1 when x > 0, avoiding the vanishing gradient problem and speeding up convergence.

***Drawbacks of ReLU:***
1. "Dying neuron" problem: Since ReLU outputs 0 for inputs less than 0, the gradient for those neurons is also 0, preventing them from updating. If many neurons receive negative inputs, these neurons can "die," meaning they will never be activated and will not learn new features.
2. Unbounded output: The output of the ReLU function is unbounded in the positive region. If the input becomes very large, the output can also become very large, which may lead to extremely large weights and affect the model's stability.

### ArgMax
ArgMax is a common mathematical operation used to find the position (index) where a function or sequence reaches its maximum value, rather than the value itself. In simple terms, ArgMax returns the input that gives the highest output of a function. 
For example, for the sequence [2, 5, 1, 8, 3]:

\begin{equation}
ArgMax([2, 5, 1, 8, 3]) = 4
\end{equation}

because the value `8`, which is the maximum, is at index `4`. So, ArgMax returns `4`, the index, not the maximum value itself.

***Key Characteristics of ArgMax:***
1. **Returns the index of the maximum value:** ArgMax returns the position of the maximum value, unlike the Max function which returns the value itself.
2. **Uniqueness:** If there are multiple instances of the maximum value, ArgMax typically returns the index of the first occurrence.
3. **Used in classification problems:** ArgMax is commonly used in machine learning classification tasks. For example, in multi-class classification problems, the final output layer usually provides a probability distribution (like from a Softmax layer), where each output corresponds to a class. ArgMax is used to select the output with the highest probability, determining the predicted class.

***ArgMax in Neural Networks:***
1. **Final decision in classifiers:** In multi-class classification, the neural network's final layer (often a Softmax layer) produces a set of probabilities for each class. ArgMax is used to select the class with the highest probability as the final prediction.
2. **Action selection in reinforcement learning:** In reinforcement learning, ArgMax is used to select the best action from a set of action values. The agent selects the action that maximizes the expected reward.

***Important Considerations:***
1. **Not suitable for gradient-based learning:** ArgMax is a non-continuous operation, which means its derivative is 0. This makes it unsuitable for use in layers where gradient-based methods like backpropagation are required. In neural networks, continuous and differentiable functions (such as ReLU) are used in the hidden layers to allow gradient updates.

2. **Loss of information:** ArgMax only focuses on the maximum value and ignores other values in the input, which could also carry useful information. For example, in classification, if two class probabilities are very close, ArgMax will only select the higher one, ignoring how close they are.

## Functionality of essential operations
### ReLU
Convert values less than 0 in the input matrix to 0.

### ArgMax
Return the index of the maximum value in the input matrix.

### Dot Product
Given two input matrices A and B, multiply each element of A with the corresponding element at the same position in B, then sum all the results. Ensure to account for the **stride** in both matrices A and B during the computation.

### Matrix Multiplication
Given two input matrices A and B, compute the matrix multiplication by calculating the dot product of each row of A with each column of B. Consider whether matrix B has a **stride**, and account for it during the computation.

#### Stride
In matrix operations (especially in convolutional neural networks and image processing), stride refers to the distance the convolution kernel (or filter) moves on the input matrix during each step.

### Read Matrix
Read matrix data from a binary file.

### Write Matrix
Write a matrix to a file in binary format. The matrix is stored in a fixed structure, including the number of rows, the number of columns, and the matrix data itself (stored in row-major order).

### Classification
This is a neural network-based classifier that uses matrix operations to compute results and perform classification. It utilizes matrix multiplication for the hidden layer, with the ReLU activation function applied. Another matrix multiplication is used to compute the scores, and finally, the `argmax` function is employed to select the highest score.
## Challenges
### Dot Product
At first, when implementing the dot function, my initial idea was to replace all the multiplications with additions. Following this approach, there are three places where multiplication is required: the dot product operation and handling the strides of the first and second input matrices. However, using accumulation for these operations would result in poor efficiency. Therefore, when dealing with strides, I used additional registers to directly multiply the stride of each matrix by 4 (left-shifting by 2 bits) during each loop iteration and added the result back to the pointer referencing the matrix elements. This approach effectively calculates the offset directly, saving the accumulation required for stride computation and improving the efficiency of the code.
:::danger
**FIXME: Can pass all the test cases in `test_dot`, but an error occurs in `classify_3` when testing `test_classify`.**
:::
:::warning
In the following code, I use an accumulation method to compute the dot product. The register `t0` is used to store the result of the dot product, while `t2` and `t3` respectively store the two numbers to be multiplied. A loop is used to repeatedly add the value in `t2` to `t0` and decrement `t3` by one. The loop terminates when `t3` is less than or equal to zero.
This causes error in `classify_3`
```cpp!
mul_dot:
    add t0, t0, t2            
    addi t3, t3, -1
    bgt t3, zero, mul_dot 
```
Due to an error occurring during the `classify_3` test, I modified the code to the following form. The register `t6` is used to store the product of `t2` and `t3`, which is then accumulated into `t0`. Using this code, all test cases can be passed successfully.
```cpp!
mul_dot:
    li t6, 0
    mul t6, t2, t3
    add t0, t0, t6
```
:::
### Matrix Multiplication
In this code, the first problem I encountered was that it used many registers and relied on the stack to store some values that did not need to be modified. As a result, I spent a considerable amount of time understanding the code.
## References
- [Assignment2: Complete Applications](https://hackmd.io/@sysprog/2024-arch-homework2#Project-Overview)
- [Artificial Neural Network (ANN)](https://en.wikipedia.org/wiki/Neural_network_(machine_learning))
- [何謂 Artificial Neural Network?](https://r23456999.medium.com/%E4%BD%95%E8%AC%82-artificial-neural-netwrok-33c546c94794)
- [【ML09】Activation Function 是什麼?](https://medium.com/%E6%B7%B1%E6%80%9D%E5%BF%83%E6%80%9D/ml08-activation-function-%E6%98%AF%E4%BB%80%E9%BA%BC-15ec78fa1ce4)
- [Lab 3: RISC-V, Venus](https://cs61c.org/fa24/labs/lab03/#setup)
- [Venus Reference](https://cs61c.org/fa24/resources/venus-reference/#venus-web-interface)
- [Common Errors](https://cs61c.org/fa24/resources/common-errors/#venus)
- [mnist](https://www.tensorflow.org/datasets/catalog/mnist)

