# Fast Surrogate Model for Electronic Cooling Simulation

## System Design Overview

This project aims to develop a surrogate model that predicts thermal behavior in electronic circuits more quickly than traditional simulations. Using machine learning techniques, we construct a model that can efficiently approximate the outcomes of complex thermal simulations. This surrogate model facilitates rapid testing and iteration in electronic design processes, which is crucial for the thermal management of high-performance electronics.

## Rationale

### Increasing Efficiency and Reducing Costs

Traditional thermal simulations for electronic circuits are computationally intensive and time-consuming, which leads to increasing costs and extended development cycles in electronics designs.

## Goal

The primary goal of this project is to combine statistical, machine learning, and reduced order modeling techniques to:

1. Construct a fast surrogate model that can accurately mimic the results of detailed thermal simulations of electronic circuits.
2. Automate data collection and simulation processes for training and validating the surrogate model.

## Project Design

### Step 1: Define the Simulation Domain

#### Simulation Data

Identify and describe typical scenarios for electronics cooling simulations, which include varying power loads and ambient temperatures.

#### Data Collection

Design a method to collect or generate simulation data that covers a wide range of these scenarios.

### Key Scenarios

- **Power Load Variations**: Different levels of power consumption impact the heat generation within electronic circuits.
- **Ambient Temperature Variations**: Different external temperatures affect the cooling efficiency.

### Step 2: Develop the Machine Learning Model

#### Model Architecture

**Overview**:

We utilize a Fully Connected Neural Network (FCNN) designed to handle the functional relationship between the input features (power load and ambient temperature) and the output (circuit temperature).

- **Input Layer**: Receives two input features corresponding to the power load and ambient temperature.
- **Hidden Layers**: Consists of two hidden layers, each with 64 neurons. These layers use ReLU (Rectified Linear Unit) activation functions, introducing non-linearity into the model to learn more complex patterns in the data.
- **Output Layer**: A single neuron outputs the predicted circuit temperature, providing a direct measure of the cooling performance.

#### Training Process

**Overview**:

The model is trained using a dataset generated in Step 1. Training involves adjusting the model weights to minimize the error between the predicted temperatures and actual simulated temperatures.

**Our Training Loop**:

- **Batch Processing**: Iterate through the data in batches to optimize resource usage.
- **Forward Pass**: Compute the predicted outputs using the current model state.
- **Loss Calculation**: Determine the error by comparing the predicted outputs to true outputs.
- **Backpropagation**: Update the model weights based on the gradient of the loss function, aiming to reduce the loss in subsequent iterations.
- **Epoch Iteration**: Repeat the process for a set number of epochs until the loss converges to a minimum or stops improving significantly.

### Step 3: Reduced Order Modeling

**Reduced Complexity**:

Apply techniques to reduce the dimensionality of simulation data, making the learning process faster and more efficient. However, Principal Component Analysis (PCA) was not needed due to the minimal set of input features in our project.

## Problems Encountered

1. **Fluctuating Loss During Training**
   - **Cause**: The fluctuation in the loss could be due to several factors, including the small batch size, which led to higher variance in the gradient estimates.
   - **Fix**: Increased the batch size to stabilize the loss.

## Conclusion

This project demonstrates the feasibility of using machine learning techniques to create effective surrogate models for simulating electronic cooling processes. The model allows for rapid iterations and could significantly reduce the time and cost associated with traditional simulation methods.
