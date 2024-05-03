# Fast Surrogate Model for Electronic Cooling

## Overview

This project develops a surrogate model to quickly predict thermal behavior in electronic circuits, streamlining the design process by reducing the time and cost associated with traditional simulations.

## Goals

- Create a fast and accurate surrogate model to emulate detailed thermal simulations of electronic circuits.
- Automate the data collection and simulation processes for model training and validation.

## System Design

### Data Preparation

- **Simulation Data**: Identify scenarios involving variations in power loads and ambient temperatures.
- **Data Collection**: Develop a methodology to gather or generate comprehensive simulation data.

### Machine Learning Model

- **Architecture**: A Fully Connected Neural Network (FCNN) with:
  - **Input Layer**: Handles inputs like power load and ambient temperature.
  - **Hidden Layers**: Two layers, each with 64 neurons, using ReLU activation to capture complex patterns.
  - **Output Layer**: Single neuron predicting the circuit temperature.

- **Training Process**:
  - Process data in batches, calculate predictions, measure errors, and adjust weights using backpropagation. Repeat across epochs until the model's predictions stabilize.

## Challenges and Solutions

- **Fluctuating Loss**: Observed during training; addressed by increasing the batch size for more stability.

## Conclusion

The project successfully demonstrates how machine learning can efficiently replicate and potentially replace extensive physical simulations in electronic design, offering a faster, cost-effective alternative.
