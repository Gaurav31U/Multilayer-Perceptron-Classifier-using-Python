# Multi-Layer Perceptron (MLP) for Iris Dataset

## Description
This project implements a simple Multi-Layer Perceptron (MLP) from scratch using numpy. The MLP is trained on the Iris dataset to classify iris flowers into three different species.

## Installation
1. Clone the repository:
    ```shell
    git clone https://github.com/Gaurav31U/Multilayer-Perceptron-Classifier-using-Python.git
    cd Multilayer-Perceptron-Classifier-using-Python
    ```

2. Install the required packages:
    ```shell
    pip install numpy pandas
    ```

3. Download the Iris dataset and place it in the same directory as the script:
    ```shell
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -O iris.csv
    ```

## Usage
1. Run the script to train the MLP:
    ```shell
    python mlp_iris.py
    ```

## Code Explanation
- `mlp` Class:
  - `__init__(self, m, n, p, eta, epoaches)`: Initializes the MLP with input size `m`, hidden layer size `n`, output size `p`, learning rate `eta`, and number of epochs `epoaches`.
  - `sigmoid(self, y)`: Computes the sigmoid activation function.
  - `forward_prop(self, inp)`: Performs forward propagation to compute the output.
  - `back_prop(self, d, inp)`: Performs backward propagation to update the weights.
  - `train(self, X, Y)`: Trains the MLP on the input data `X` and labels `Y`.

- Data Handling:
  - Reads the Iris dataset from a CSV file, converts categorical labels to numeric codes, and shuffles the data.
  - Splits the data into input features `X` and labels `Y`.
