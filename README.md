# Fast Surrogate Model for Electronic Cooling

## Overview

This project develops a surrogate model to quickly predict thermal behavior in electronic circuits, streamlining the design process by reducing the time and cost associated with traditional simulations.


### Machine Learning Model

- **Architecture**: A Fully Connected Neural Network (FCNN) with:
  - **Input Layer**: Handles inputs like power load and ambient temperature.
  - **Hidden Layers**: Two layers, each with 64 neurons, using ReLU activation to capture complex patterns.
  - **Output Layer**: Single neuron predicting the circuit temperature.

- **Training Process**:
  - Process data in batches, calculate predictions, measure errors, and adjust weights using backpropagation. Repeat across epochs until the model's predictions stabilize.

## Challenges and Solutions

- **Fluctuating Loss**: Observed during training; addressed by increasing the batch size for more stability.
