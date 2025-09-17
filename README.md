# Test-Negative Design Simulation with Antibody Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains code for simulating test-negative design studies using network-based epidemic models with realistic antibody dynamics. The simulation framework models two pathogens (test-negative and test-positive) with differential vaccine effects, antibody-mediated protection, and waning immunity over time.

**Key Features:**
- Network-based epidemic modeling with two co-circulating pathogens
- Realistic antibody dynamics with vaccination and infection-induced immunity
- Antibody waning over time affecting protection levels
- Multiple analytical approaches: Cox models, difference-in-differences, test-negative design
- Spline-based dose-response modeling of antibody protection
- Monte Carlo simulation for uncertainty quantification
- Parameter sensitivity analysis for antibody dynamics

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/[your-username]/antibody-tnd.git
   cd antibody-tnd
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Required Dependencies

The simulation requires the following Python packages:
- `numpy` - Numerical computing
- `pandas` - Data manipulation and analysis
- `matplotlib` - Plotting and visualization
- `networkx` - Network graph creation and analysis
- `lifelines` - Survival analysis (Cox proportional hazards models)
- `scikit-learn` - Machine learning utilities and spline transformations
- `seaborn` - Statistical data visualization
- `tqdm` - Progress bars
- `jupyter` - Jupyter notebook support

## Quick Start

### Basic Test-Negative Design Simulation

Run a basic simulation with antibody dynamics:

```python
from utils import run_simulation

# Define simulation parameters
PARAMS = {
    'num_nodes': 10000,
    'infection_prob': (0.03, 0.04),  # test-negative, test-positive
    'vaccine_prob': 0.35,
    'antibody_max_vacc': 0.8,        # max antibody from vaccination
    'antibody_max_inf': 0.9,         # max antibody from infection
    'antibody_waning': 0.015,        # 1.5% waning per time step
    'steps': 100
}

# Run simulation
G, results = run_simulation(**PARAMS)

print(f"Test-positive pathogen VE: {(1 - results['hr_test_positive']) * 100:.1f}%")
print(f"Difference-in-differences VE: {(1 - results['hr_difference_in_differences']) * 100:.1f}%")
```

### Interactive Analysis

For comprehensive analysis, open the Jupyter notebook:

```bash
jupyter notebook test_negative_antibody_simulation.ipynb
```

The notebook includes:
- Complete simulation examples with antibody dynamics
- Antibody level visualization and dose-response curves
- Comparison of different analytical approaches
- Antibody waning rate analysis
- Spline-based protection function modeling
- Monte Carlo uncertainty quantification

## Repository Structure

```
antibody-tnd/
├── utils.py                                    # Core simulation functions
├── test_negative_antibody_simulation.ipynb     # Interactive analysis notebook
├── README.md                                   # This file
├── requirements.txt                            # Python dependencies
├── LICENSE                                     # MIT license
└── examples/                                   # Additional example scripts (optional)
```

## Core Functions

### `run_simulation()`
Main function to execute a complete test-negative design simulation with antibody dynamics.

**Key Parameters:**
- `num_nodes`: Population size (default: 1000)
- `infection_prob`: Tuple of base infection probabilities (test_neg, test_pos)
- `vaccine_prob`: Probability of vaccination (default: 0.5)
- `antibody_max_vacc`: Maximum antibody level from vaccination (default: 0.8)
- `antibody_max_inf`: Maximum antibody level from infection (default: 0.9)
- `antibody_waning`: Antibody decline per time step (default: 0.01)

### `estimate_models()`
Estimates protection functions using restricted cubic splines for antibody levels.

**Returns:**
- Protection curves across antibody levels for each analytical approach
- Vaccination-based hazard ratios for comparison
- Spline-based dose-response relationships

### Antibody Functions
- `update_antibody_levels()`: Handles antibody waning each time step
- `get_protection_factor()`: Converts antibody levels to protection (0-1 scale)
- `assign_vaccine()`: Sets initial antibody levels from vaccination

## Simulation Model

### Network Structure
- **Nodes**: Represent individuals with antibody levels and vaccination status
- **Edges**: Represent social contacts for disease transmission
- **Graph Type**: Erdős–Rényi random graph (configurable)

### Disease States (Two Pathogens)
- **S1/S2**: Susceptible to test-negative/test-positive pathogen
- **I1/I2**: Infectious with test-negative/test-positive pathogen
- **T1/T2**: Tested positive for test-negative/test-positive pathogen

### Antibody Dynamics
- **Initial levels**: Set by vaccination (random 0 to max_vacc) or infection (max_inf)
- **Protection**: Linear relationship between antibody level and infection probability
- **Waning**: Antibody levels decrease by fixed amount each time step
- **Reinfection**: Possible when antibody levels wane sufficiently

### Vaccine Effects
- **Pathogen-specific**: Vaccine only affects test-positive pathogen antibodies
- **Individual variation**: Random antibody levels from vaccination (0 to max)
- **Differential protection**: Test-negative pathogen unaffected by vaccination

### Confounding Structure
- **X**: Measured confounder (affects infection, testing, vaccination)
- **U**: Unmeasured confounder (affects infection, testing, vaccination)
- **Time-varying**: Unmeasured confounding activates after step 20

## Example Use Cases

### 1. Compare Antibody Waning Rates
```python
waning_rates = [0.0, 0.005, 0.015, 0.03, 0.05]

for rate in waning_rates:
    params = PARAMS.copy()
    params['antibody_waning'] = rate
    G, results = run_simulation(**params, plot=False)
    print(f"Waning {rate*100:.1f}%: VE = {(1-results['hr_test_positive'])*100:.1f}%")
```

### 2. Analyze Antibody Protection Curves
```python
from utils import estimate_models

# Run simulation and analyze antibody relationships
G, _ = run_simulation(**PARAMS, plot=False)
spline_results = estimate_models(G, print_results=True)

# Examine protection at different antibody levels
print("Antibody levels:", spline_results['antibody_grid'])
print("Test-pos protection:", spline_results['protection_test_positive'])
```

### 3. Test Antibody Parameter Sensitivity
```python
antibody_max_values = [0.4, 0.6, 0.8, 1.0]
for max_ab in antibody_max_values:
    params = PARAMS.copy()
    params['antibody_max_vacc'] = max_ab
    G, results = run_simulation(**params, plot=False)
    print(f"Max antibody {max_ab}: VE = {(1-results['hr_test_positive'])*100:.1f}%")
```

## Output and Results

### Visualizations
- **Epidemic curves**: Disease progression for both pathogens
- **Antibody dynamics**: Population average antibody levels over time
- **Protection curves**: Spline-based dose-response relationships
- **Network plots**: Final states showing antibody levels and vaccination status
- **Distribution plots**: Antibody levels by vaccination status

### Statistical Output
- **Protection functions**: 1 - Relative Risk across antibody levels
- **Hazard ratios**: Traditional vaccination-based estimates
- **Confidence intervals**: From Monte Carlo simulation
- **Model comparisons**: Multiple analytical approaches (Cox, DiD, TND)

### Analytical Approaches
1. **Test-negative pathogen analysis**: Should show minimal vaccine effect
2. **Test-positive pathogen analysis**: Should show vaccine benefit
3. **Difference-in-differences**: Isolates true vaccine effect
4. **Test-negative design**: Compares test outcomes among tested individuals
5. **Spline-based modeling**: Flexible dose-response relationships

## Key Insights

### Expected Results
- **Test-negative pathogen VE ≈ 0%**: Vaccine doesn't protect against this pathogen
- **Test-positive pathogen VE > 0%**: Vaccine provides protection
- **Antibody dose-response**: Higher antibody levels → better protection
- **Waning effects**: Faster waning → reduced overall vaccine effectiveness

### Model Features
- **Realistic immunity**: Antibody levels provide graduated protection (not binary)
- **Waning immunity**: Protection decreases over time, allowing reinfection
- **Individual variation**: People have different antibody responses to vaccination
- **Pathogen specificity**: Vaccine only protects against one pathogen (test-positive)

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration inquiries, please contact [your-email] or open an issue on GitHub.

## Acknowledgments

- Built using NetworkX for graph operations
- Survival analysis implemented with lifelines package
- Spline modeling using scikit-learn SplineTransformer
- Visualization powered by matplotlib and seaborn
- Test-negative design methodology following established epidemiological practices
