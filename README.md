# Test-Negative Design Simulation with Antibody Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains code for simulating **test-negative design studies** using network-based epidemic models with realistic **antibody dynamics**. The simulation framework models two pathogens (test-negative and test-positive) with differential vaccine effects, antibody-mediated protection, and waning immunity over time.

**Primary Focus: Test-Negative Design with Antibody Levels**
- Uses antibody concentrations measured at time of testing as the primary exposure variable
- Generates spline-based dose-response curves for antibody-protection relationships
- Compares antibody-based TND with traditional vaccination-based TND approaches
- Incorporates realistic antibody waning and temporal dynamics

**Key Features:**
- Network-based epidemic modeling with two co-circulating pathogens
- Individual-level antibody dynamics with vaccination and infection-induced immunity
- Antibody waning over time affecting protection levels
- **Test-negative design as primary analytical approach**
- Spline-based dose-response modeling of antibody-protection relationships
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
- `scikit-learn` - Machine learning utilities and spline transformations
- `seaborn` - Statistical data visualization
- `tqdm` - Progress bars
- `jupyter` - Jupyter notebook support

## Quick Start

### Basic Test-Negative Design Simulation

Run a basic simulation focusing on antibody-based test-negative design:

```python
from utils import run_simulation

# Define simulation parameters focused on antibody dynamics
PARAMS = {
    'num_nodes': 50000,                   # Population size
    'infection_prob': (0.03, 0.04),      # Base infection probabilities (test-neg, test-pos)
    'vaccine_prob': 0.35,                # Probability of vaccination
    'antibody_max_vacc': 1.0,            # Max antibody from vaccination
    'antibody_max_inf': 1.0,             # Max antibody from infection
    'antibody_waning': 0.015,            # Antibody waning per time step (1.5%)
    'steps': 60                          # Simulation duration
}

# Run simulation
G, results = run_simulation(**PARAMS)

# Primary result: Test-negative design with vaccination
if results['hr_test_negative_design'] is not None:
    ve_tnd = (1 - results['hr_test_negative_design']) * 100
    print(f"TND Vaccine Effectiveness (vaccination): {ve_tnd:.1f}%")

# Advanced result: Antibody-based protection curve
if results['protection_test_negative_design'] is not None:
    antibody_levels = results['antibody_grid']
    protection = results['protection_test_negative_design']
    print(f"Antibody protection curve available across levels {antibody_levels}")
```

### Interactive Analysis

For comprehensive analysis, open the Jupyter notebook:

```bash
jupyter notebook test_negative_antibody_simulation.ipynb
```

The notebook includes:
- Complete test-negative design simulation examples with antibody dynamics
- Antibody level visualization and dose-response curve analysis
- Comparison of vaccination-based vs antibody-based TND approaches
- Antibody waning rate sensitivity analysis
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

**Returns:**
- `G`: Final network graph with individual antibody levels and outcomes
- `results`: Dictionary containing TND estimates and time series data

### `estimate_models()`
Estimates protection functions using **test-negative design** with antibody levels measured at time of testing.

**Primary Analysis:**
- **Antibody-based TND**: Uses spline-transformed antibody levels as continuous exposure
- **Vaccination-based TND**: Traditional binary vaccination exposure for comparison

**Returns:**
- `protection_test_negative_design`: Protection curve across antibody levels (1 - Odds Ratio)
- `hr_vaccination_tnd`: Hazard ratio for vaccination-based TND
- `antibody_grid`: Grid of antibody levels for dose-response curve

### Antibody Functions
- `update_antibody_levels()`: Handles antibody waning each time step
- `get_protection_factor()`: Converts antibody levels to protection (0-1 scale)
- `assign_vaccine()`: Sets initial antibody levels from vaccination (random 0 to max_vacc)

## Simulation Model

### Network Structure
- **Nodes**: Represent individuals with antibody levels, vaccination status, and infection history
- **Edges**: Represent social contacts for disease transmission
- **Graph Type**: Erdős–Rényi random graph (configurable density)

### Disease States (Two Pathogens)
- **S1/S2**: Susceptible to test-negative/test-positive pathogen
- **I1/I2**: Infectious with test-negative/test-positive pathogen
- **T1/T2**: Tested positive for test-negative/test-positive pathogen

### Antibody Dynamics
- **Initial levels**: 
  - Vaccination: Random uniform distribution (0 to max_vacc) for test-positive pathogen only
  - Infection: Maximum level (max_inf) for respective pathogen
- **Protection**: Linear relationship - higher antibody = lower infection probability
- **Waning**: Fixed absolute decrease per time step (antibody_waning parameter)
- **Reinfection**: Possible when antibody levels wane sufficiently low

### Vaccine Effects
- **Pathogen-specific**: Vaccine only affects test-positive pathogen antibodies
- **Individual variation**: Random antibody levels from vaccination create heterogeneity
- **No direct efficacy**: Protection mediated entirely through antibody levels

### Test-Negative Design Analysis
- **Primary outcome**: Test result (positive vs negative) at time of testing
- **Exposure**: Antibody levels measured at time of testing (not final levels)
- **Spline modeling**: Flexible dose-response using restricted cubic splines
- **Confounding control**: Includes measured confounder X in regression models

### Confounding Structure
- **X**: Measured confounder (affects infection, testing, vaccination probabilities)
- **U**: Unmeasured confounder (affects infection, testing, vaccination probabilities)
- **Individual heterogeneity**: Each person has unique confounder values and derived probabilities

## Example Use Cases

### 1. Compare Antibody Waning Rates
```python
waning_rates = [0.0, 0.005, 0.015, 0.03, 0.05]

for rate in waning_rates:
    params = PARAMS.copy()
    params['antibody_waning'] = rate
    G, results = run_simulation(**params, plot=False)
    
    if results['hr_test_negative_design'] is not None:
        ve = (1 - results['hr_test_negative_design']) * 100
        print(f"Waning {rate*100:.1f}%/step: TND VE = {ve:.1f}%")
```

### 2. Analyze Antibody-Based Protection Curves
```python
from utils import estimate_models

# Run simulation and analyze antibody relationships
G, results = run_simulation(**PARAMS, plot=False)

# Examine protection at different antibody levels
if results['protection_test_negative_design'] is not None:
    antibody_levels = results['antibody_grid']
    protection = results['protection_test_negative_design']
    
    for ab, prot in zip(antibody_levels, protection):
        ve_percent = prot * 100
        print(f"Antibody {ab:.1f}: Protection = {prot:.3f} (VE = {ve_percent:.1f}%)")
```

### 3. Monte Carlo Uncertainty Quantification
```python
import numpy as np
from tqdm import tqdm

n_sims = 50
ve_estimates = []

for i in tqdm(range(n_sims)):
    np.random.seed(i * 123)
    G, results = run_simulation(**PARAMS, plot=False, print_progress=False)
    
    if results['hr_test_negative_design'] is not None:
        ve = (1 - results['hr_test_negative_design']) * 100
        ve_estimates.append(ve)

if len(ve_estimates) > 0:
    print(f"TND VE: {np.mean(ve_estimates):.1f}% ± {np.std(ve_estimates):.1f}%")
    print(f"95% CI: [{np.percentile(ve_estimates, 2.5):.1f}%, {np.percentile(ve_estimates, 97.5):.1f}%]")
```

## Output and Results

### Primary Outputs
- **Test-negative design estimates**: Vaccine effectiveness using antibody levels at testing
- **Antibody protection curves**: Spline-based dose-response relationships (0-1 antibody scale)
- **Time series data**: Epidemic progression, testing rates, and antibody dynamics over time

### Visualizations
- **Epidemic curves**: Disease progression for both pathogens over time
- **Antibody dynamics**: Population average antibody levels with waning effects
- **Protection curves**: Dose-response relationship between antibody levels and protection
- **Testing patterns**: Number of individuals tested by vaccination status and pathogen
- **Distribution plots**: Final antibody levels by vaccination status and infection history

### Statistical Output
- **Protection functions**: 1 - Odds Ratio across antibody levels from spline regression
- **Vaccination-based TND**: Traditional binary exposure hazard ratios for comparison
- **Confidence intervals**: From Monte Carlo simulation across multiple runs
- **Success rates**: Proportion of simulations with sufficient data for analysis

## Key Methodological Features

### Focus on Test-Negative Design
- **Antibody timing**: Uses levels measured exactly when testing occurs
- **Appropriate controls**: Test-negative cases serve as controls for test-positive cases
- **Spline flexibility**: Non-linear dose-response relationships between antibodies and protection
- **Confounding adjustment**: Includes measured covariates in regression models

### Realistic Antibody Dynamics
- **Vaccination variability**: Random antibody levels from vaccination (not fixed efficacy)
- **Infection-induced immunity**: Natural infection provides antibody boost
- **Temporal precision**: Antibody levels captured at exact time of testing
- **Waning effects**: Gradual decline in protection over time

### Simulation Advantages
- **Ground truth**: Known antibody-protection relationships for validation
- **Parameter control**: Systematic testing of antibody waning rates and dynamics
- **Uncertainty quantification**: Monte Carlo estimation of confidence intervals
- **Method comparison**: Direct comparison of antibody vs vaccination-based approaches

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

- Test-negative design methodology following established epidemiological practices
- Spline modeling using scikit-learn SplineTransformer for flexible dose-response curves
- Network simulation powered by NetworkX for realistic disease transmission
- Visualization created with matplotlib and seaborn
- Antibody dynamics inspired by immunological literature on vaccine-induced and natural immunity
