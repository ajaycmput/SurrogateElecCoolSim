import sys
import numpy as np
import pandas as pd

def generate_data(num_samples=1000):
    np.random.seed(0)
    # Randomly generate power loads (in watts)
    power_loads = np.random.uniform(low=50, high=150, size=num_samples)
    
    # Randomly generate ambient temperatures (in degrees Celsius)
    ambient_temps = np.random.uniform(low=10, high=35, size=num_samples)
    
    # Assume a linear relationship
    circuit_temps = 0.5 * power_loads + 0.4 * ambient_temps + np.random.normal(loc=0, scale=5, size=num_samples)
    
    # Create a DataFrame
    data = pd.DataFrame({
        'Power Load (W)': power_loads,
        'Ambient Temp (C)': ambient_temps,
        'Circuit Temp (C)': circuit_temps
    })
    
    return data

def main(num_samples=1000):
    data = generate_data(num_samples)
    data.to_csv('../data/electronics_cooling_simulation_data.csv', index=False)
    print("Data generated and saved to CSV.")

if __name__ == "__main__":
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    main(num_samples)
