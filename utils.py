import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
from scipy.interpolate import splev, splrep

def generate_random_graph(num_nodes, edges_per_node):
    """Generates a random graph with a specified number of nodes and expected edges."""
    probability = edges_per_node / (num_nodes - 1)
    G = nx.erdos_renyi_graph(num_nodes, probability)
    return G

def assign_initial_states(G, infection_prob, recovery_prob, testing_prob, vaccine_prob, vaccine_efficacy,
                         antibody_max_vacc, antibody_max_inf, antibody_waning):
    """Assigns initial states to nodes in the graph."""
    for node in G.nodes():
        G.nodes[node]['S1'] = True  # Susceptible to test negative pathogen
        G.nodes[node]['S2'] = True  # Susceptible to test positive pathogen
        G.nodes[node]['I1'] = False  # Infectious test negative pathogen
        G.nodes[node]['I2'] = False  # Infectious test positive pathogen
        G.nodes[node]['T1'] = False  # Tested for test negative pathogen
        G.nodes[node]['T2'] = False  # Tested for test positive pathogen
        G.nodes[node]['U'] = random.random() # Unmeasured confounder
        G.nodes[node]['X'] = random.random() # Measured confounder
        G.nodes[node]['time1'] = 0  # Event time for test negative pathogen
        G.nodes[node]['time2'] = 0  # Event time for test positive pathogen
        G.nodes[node]['vaccinated'] = False  # Not vaccinated initially
        G.nodes[node]['vaccine_prob'] = vaccine_prob # Probability of being vaccinated
        G.nodes[node]['infection_prob_1'] = infection_prob[0] # Infection probability for test negative pathogen
        G.nodes[node]['infection_prob_2'] = infection_prob[1] # Infection probability for test positive pathogen
        G.nodes[node]['recovery_prob_1'] = recovery_prob[0] # Probability of recovery for test negative pathogen
        G.nodes[node]['recovery_prob_2'] = recovery_prob[1] # Probability of recovery for test positive pathogen
        G.nodes[node]['testing_prob_1'] = testing_prob[0] # Probability of testing for test negative pathogen
        G.nodes[node]['testing_prob_2'] = testing_prob[1] # Probability of testing for test positive pathogen
        G.nodes[node]['vaccine_efficacy_1'] = vaccine_efficacy[0] # Efficacy of vaccine for test negative pathogen
        G.nodes[node]['vaccine_efficacy_2'] = vaccine_efficacy[1] # Efficacy of vaccine for test positive pathogen
        
        # Antibody levels (0 = no protection, 1 = complete protection)
        G.nodes[node]['antibody_level_1'] = 0.0  # Antibody level for test negative pathogen (not affected by vaccination)
        G.nodes[node]['antibody_level_2'] = 0.0  # Antibody level for test positive pathogen (affected by vaccination and infection)
        
        # Antibody parameters
        G.nodes[node]['antibody_max_vacc'] = antibody_max_vacc  # Max antibody level from vaccination
        G.nodes[node]['antibody_max_inf'] = antibody_max_inf    # Max antibody level from infection
        G.nodes[node]['antibody_waning'] = antibody_waning      # Antibody waning per time step
        
        # Calculate individual vaccine probabilities based on confounders
        G.nodes[node]['vaccine_prob'] *= np.exp(G.nodes[node]['U'] + G.nodes[node]['X'] - 1)
        
        # Calculate individual infection probabilities based on confounders
        G.nodes[node]['infection_prob_1'] *= np.exp(G.nodes[node]['X'] - 0.5) 
        G.nodes[node]['infection_prob_2'] *= np.exp(G.nodes[node]['X'] - 0.5) 
        G.nodes[node]['infection_prob_1U'] = np.exp(G.nodes[node]['U'] - 0.5) * G.nodes[node]['infection_prob_1']
        G.nodes[node]['infection_prob_2U'] = np.exp(G.nodes[node]['U'] - 0.5) * G.nodes[node]['infection_prob_2']   

        # Calculate individual testing probabilities based on confounders
        G.nodes[node]['testing_prob_1'] *= np.exp(G.nodes[node]['X'] - 0.5) 
        G.nodes[node]['testing_prob_2'] *= np.exp(G.nodes[node]['X'] - 0.5)
        G.nodes[node]['testing_prob_1U'] = np.exp(G.nodes[node]['U'] - 0.5) * G.nodes[node]['testing_prob_1']
        G.nodes[node]['testing_prob_2U'] = np.exp(G.nodes[node]['U'] - 0.5) * G.nodes[node]['testing_prob_2']

def assign_initial_infected(G, num_initial_infected):
    """Assigns initial infected nodes in the graph."""
    initial_infected = random.sample(G.nodes(), np.sum(num_initial_infected))
    for ind, node in enumerate(initial_infected):
        if ind < num_initial_infected[0]:
            G.nodes[node]['I1'] = True
            G.nodes[node]['S1'] = False
            # Set antibody level for test negative pathogen upon infection
            G.nodes[node]['antibody_level_1'] = G.nodes[node]['antibody_max_inf']
        else:
            G.nodes[node]['I2'] = True
            G.nodes[node]['S2'] = False
            # Set antibody level for test positive pathogen upon infection
            G.nodes[node]['antibody_level_2'] = G.nodes[node]['antibody_max_inf']

def assign_vaccine(G):
    """Assigns a vaccine to nodes in the graph based on a given probability."""
    for node in G.nodes():
        if random.random() < G.nodes[node]['vaccine_prob']:
            G.nodes[node]['vaccinated'] = True
            
            # Set antibody level for test positive pathogen (vaccine only affects pathogen 2)
            # Use uniform distribution between 0 and max vaccine antibody level for variability
            vaccine_antibody = random.uniform(0, G.nodes[node]['antibody_max_vacc'])
            G.nodes[node]['antibody_level_2'] = vaccine_antibody

            # Adjust infection probabilities based on antibody levels (not vaccine efficacy)
            # Protection is proportional to antibody level, so multiply by (1 - antibody_level)
            protection_factor_2 = 1 - G.nodes[node]['antibody_level_2']
            G.nodes[node]['infection_prob_2'] *= protection_factor_2
            G.nodes[node]['infection_prob_2U'] *= protection_factor_2
            
            # Test negative pathogen is not affected by vaccination (antibody_level_1 should be 0)
            # But we'll be consistent and use the antibody level for pathogen 1 as well
            protection_factor_1 = 1 - G.nodes[node]['antibody_level_1'] 
            G.nodes[node]['infection_prob_1'] *= protection_factor_1
            G.nodes[node]['infection_prob_1U'] *= protection_factor_1
        else:
            G.nodes[node]['vaccinated'] = False

def update_antibody_levels(G):
    """Updates antibody levels for all nodes, applying waning."""
    for node in G.nodes():
        # Wane antibody levels, but don't go below 0
        G.nodes[node]['antibody_level_1'] = max(0.0, 
            G.nodes[node]['antibody_level_1'] - G.nodes[node]['antibody_waning'])
        G.nodes[node]['antibody_level_2'] = max(0.0, 
            G.nodes[node]['antibody_level_2'] - G.nodes[node]['antibody_waning'])

def get_protection_factor(antibody_level):
    """Calculate protection factor based on antibody level (0-1 scale where 1 = complete protection)."""
    return antibody_level

def simulate_outbreaks(G, steps, plot=True, print_progress=True):
    """Simulates disease outbreaks in the graph over a number of steps."""
    
    # Initialize tracking arrays (removed recovered tracking)
    infected_1_over_time = []
    infected_2_over_time = []
    tested_1_over_time = []
    tested_2_over_time = []
    tested_1_over_time_vaccinated = []
    tested_2_over_time_vaccinated = []
    tested_1_over_time_unvaccinated = []
    tested_2_over_time_unvaccinated = []
    antibody_1_over_time = []
    antibody_2_over_time = []
    
    for step in range(steps):
        
        # Update antibody levels (waning)
        update_antibody_levels(G)
        
        for node in G.nodes():
            # If infected with test negative pathogen
            if G.nodes[node]['I1']:
                # Recover after a certain number of steps (become susceptible again)
                if random.random() < G.nodes[node]['recovery_prob_1']:
                    G.nodes[node]['S1'] = True  # Become susceptible again
                    G.nodes[node]['I1'] = False
                    # Set antibody level to maximum upon recovery
                    G.nodes[node]['antibody_level_1'] = G.nodes[node]['antibody_max_inf']
                
                # Spread infection to neighbors if they are susceptible
                for neighbor in G.neighbors(node):
                    if G.nodes[neighbor]['S1']:
                        # Apply antibody protection
                        protection_1 = get_protection_factor(G.nodes[neighbor]['antibody_level_1'])
                        effective_prob_1 = G.nodes[neighbor]['infection_prob_1'] * (1 - protection_1)
                        effective_prob_1U = G.nodes[neighbor]['infection_prob_1U'] * (1 - protection_1)
                        
                        if step < 20:  # Only measured confounder in the first 20 steps
                            if random.random() < effective_prob_1:
                                G.nodes[neighbor]['I1'] = True
                                G.nodes[neighbor]['S1'] = False
                                if random.random() < G.nodes[neighbor]['testing_prob_1'] and not G.nodes[neighbor]['T2']:
                                    G.nodes[neighbor]['T1'] = True
                                    G.nodes[neighbor]['time1'] = step
                        else:  # After 20 steps, activate unmeasured confounder
                            if random.random() < effective_prob_1U:
                                G.nodes[neighbor]['I1'] = True
                                G.nodes[neighbor]['S1'] = False
                                if random.random() < G.nodes[neighbor]['testing_prob_1U'] and not G.nodes[neighbor]['T2']:
                                    G.nodes[neighbor]['T1'] = True
                                    G.nodes[neighbor]['time1'] = step

            # If infected with test positive pathogen
            if G.nodes[node]['I2']:
                # Recover after a certain number of steps (become susceptible again)
                if random.random() < G.nodes[node]['recovery_prob_2']:
                    G.nodes[node]['S2'] = True  # Become susceptible again
                    G.nodes[node]['I2'] = False
                    # Set antibody level to maximum upon recovery
                    G.nodes[node]['antibody_level_2'] = G.nodes[node]['antibody_max_inf']
                
                # Spread infection to neighbors if they are susceptible
                for neighbor in G.neighbors(node):
                    if G.nodes[neighbor]['S2']:
                        # Apply antibody protection
                        protection_2 = get_protection_factor(G.nodes[neighbor]['antibody_level_2'])
                        effective_prob_2 = G.nodes[neighbor]['infection_prob_2'] * (1 - protection_2)
                        effective_prob_2U = G.nodes[neighbor]['infection_prob_2U'] * (1 - protection_2)
                        
                        if step < 20:  # Only measured confounder in the first 20 steps
                            if random.random() < effective_prob_2:
                                G.nodes[neighbor]['I2'] = True
                                G.nodes[neighbor]['S2'] = False
                                if random.random() < G.nodes[neighbor]['testing_prob_2'] and not G.nodes[neighbor]['T1']:
                                    G.nodes[neighbor]['T2'] = True
                                    G.nodes[neighbor]['time2'] = step
                        else: # After 20 steps, activate unmeasured confounder
                            if random.random() < effective_prob_2U:
                                G.nodes[neighbor]['I2'] = True
                                G.nodes[neighbor]['S2'] = False
                                if random.random() < G.nodes[neighbor]['testing_prob_2U'] and not G.nodes[neighbor]['T1']:
                                    G.nodes[neighbor]['T2'] = True
                                    G.nodes[neighbor]['time2'] = step
            
            # If untested update follow up times
            if not G.nodes[node]['T1']:
                G.nodes[node]['time1'] += 1
            if not G.nodes[node]['T2']:
                G.nodes[node]['time2'] += 1

        # Track epidemic curves (removed recovered tracking)
        num_infected_1 = sum(1 for n in G.nodes() if G.nodes[n]['I1'])
        num_infected_2 = sum(1 for n in G.nodes() if G.nodes[n]['I2'])
        
        infected_1_over_time.append(num_infected_1)
        infected_2_over_time.append(num_infected_2)

        # Track the number of tested nodes
        num_tested_1 = sum(1 for n in G.nodes() if G.nodes[n]['T1'])
        num_tested_2 = sum(1 for n in G.nodes() if G.nodes[n]['T2'])
        num_tested_1_vaccinated = sum(1 for n in G.nodes() if G.nodes[n]['T1'] and G.nodes[n]['vaccinated'])
        num_tested_2_vaccinated = sum(1 for n in G.nodes() if G.nodes[n]['T2'] and G.nodes[n]['vaccinated'])
        num_tested_1_unvaccinated = sum(1 for n in G.nodes() if G.nodes[n]['T1'] and not G.nodes[n]['vaccinated'])
        num_tested_2_unvaccinated = sum(1 for n in G.nodes() if G.nodes[n]['T2'] and not G.nodes[n]['vaccinated'])
        
        tested_1_over_time.append(num_tested_1)
        tested_2_over_time.append(num_tested_2)
        tested_1_over_time_vaccinated.append(num_tested_1_vaccinated)
        tested_2_over_time_vaccinated.append(num_tested_2_vaccinated)
        tested_1_over_time_unvaccinated.append(num_tested_1_unvaccinated)
        tested_2_over_time_unvaccinated.append(num_tested_2_unvaccinated)

        # Track average antibody levels
        avg_antibody_1 = np.mean([G.nodes[n]['antibody_level_1'] for n in G.nodes()])
        avg_antibody_2 = np.mean([G.nodes[n]['antibody_level_2'] for n in G.nodes()])
        antibody_1_over_time.append(avg_antibody_1)
        antibody_2_over_time.append(avg_antibody_2)

        # Print the number of infected and tested nodes at each step
        if step % 10 == 0 and print_progress:  # Print every 10 steps
            print(f"Step {step}: Infected (Test Neg): {num_infected_1}, Infected (Test Pos): {num_infected_2}")
            print(f"\tTested (Neg): {num_tested_1}, Tested (Pos): {num_tested_2}")
            print(f"\tAvg Antibody (Neg): {avg_antibody_1:.3f}, Avg Antibody (Pos): {avg_antibody_2:.3f}")

    # Plot curves at the end (removed recovered plots)
    if plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot epidemic curve
        ax1.plot(range(steps), infected_1_over_time, label='Test Negative Infected', color='blue')
        ax1.plot(range(steps), infected_2_over_time, label='Test Positive Infected', color='red')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Number of Individuals')
        ax1.set_title('Epidemic Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot testing curve
        ax2.plot(range(steps), tested_1_over_time, label='Test Negative Tested', color='blue')
        ax2.plot(range(steps), tested_2_over_time, label='Test Positive Tested', color='red')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Number of Tested Individuals')
        ax2.set_title('Testing Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot antibody levels over time
        ax3.plot(range(steps), antibody_1_over_time, label='Avg Antibody Level (Test Neg)', color='blue')
        ax3.plot(range(steps), antibody_2_over_time, label='Avg Antibody Level (Test Pos)', color='red')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Average Antibody Level')
        ax3.set_title('Antibody Levels Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot testing curve by vaccination status
        ax4.plot(range(steps), tested_1_over_time_vaccinated, label='Test Neg Vaccinated', color='blue', linestyle='-', alpha=0.7)
        ax4.plot(range(steps), tested_2_over_time_vaccinated, label='Test Pos Vaccinated', color='red', linestyle='-', alpha=0.7)
        ax4.plot(range(steps), tested_1_over_time_unvaccinated, label='Test Neg Unvaccinated', color='blue', linestyle='--', alpha=0.7)
        ax4.plot(range(steps), tested_2_over_time_unvaccinated, label='Test Pos Unvaccinated', color='red', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Number of Tested Individuals')
        ax4.set_title('Testing by Vaccination Status')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    # Return graph and time series data (removed recovered data)
    time_series_data = {
        'infected_1_over_time': infected_1_over_time,
        'infected_2_over_time': infected_2_over_time,
        'tested_1_over_time': tested_1_over_time,
        'tested_2_over_time': tested_2_over_time,
        'tested_1_over_time_vaccinated': tested_1_over_time_vaccinated,
        'tested_2_over_time_vaccinated': tested_2_over_time_vaccinated,
        'tested_1_over_time_unvaccinated': tested_1_over_time_unvaccinated,
        'tested_2_over_time_unvaccinated': tested_2_over_time_unvaccinated,
        'antibody_1_over_time': antibody_1_over_time,
        'antibody_2_over_time': antibody_2_over_time,
    }
    
    return G, time_series_data

def estimate_models(G, print_results=True):
    """
    Estimates models using restricted cubic splines for antibody levels.
    Returns protection functions (1 - RR) for different analytical approaches.
    """
    # Create a DataFrame from the graph's node attributes
    data = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    
    # Create time variable that is the minimum of time1 and time2
    data['time'] = data[['time1', 'time2']].min(axis=1)
    data['event1'] = np.where(data['T1'] & (data['time1'] < data['time2']), 1, 0)
    data['event2'] = np.where(data['T2'] & (data['time2'] < data['time1']), 1, 0)

    # Define antibody level grid for predictions
    antibody_grid = np.linspace(0, 1, 10)  # 10 points from 0 to 1
    
    # Initialize results dictionary
    results = {
        'antibody_grid': antibody_grid,
        'protection_test_negative': None,
        'protection_test_positive': None, 
        'protection_with_confounders': None,
        'protection_difference_in_differences': None,
        'protection_test_negative_design': None,
        'hr_vaccination_test_negative': None,
        'hr_vaccination_test_positive': None,
        'hr_vaccination_with_confounders': None,
        'hr_vaccination_did': None,
        'hr_vaccination_tnd': None
    }
    
    try:
        # 1. Test Negative Pathogen Analysis
        df1 = data[['time', 'event1', 'antibody_level_2', 'X']].copy()
        df1 = df1.dropna()
        
        if len(df1) > 10:  # Need sufficient data
            # Create spline basis for antibody_level_2
            spline_transformer_1 = SplineTransformer(n_knots=4, degree=3, include_bias=False)
            antibody_splines_1 = spline_transformer_1.fit_transform(df1[['antibody_level_2']])
            
            # Add spline columns to dataframe
            for i in range(antibody_splines_1.shape[1]):
                df1[f'spline1_{i}'] = antibody_splines_1[:, i]
            
            # Fit Cox model with splines
            spline_cols_1 = [f'spline1_{i}' for i in range(antibody_splines_1.shape[1])]
            model_cols_1 = ['time', 'event1', 'X'] + spline_cols_1
            model_1 = CoxPHFitter()
            model_1.fit(df1[model_cols_1], duration_col='time', event_col='event1')
            
            # Predict at antibody grid points
            grid_splines_1 = spline_transformer_1.transform(antibody_grid.reshape(-1, 1))
            
            # Calculate linear predictor for each grid point
            linear_pred_1 = np.zeros(len(antibody_grid))
            for i, col in enumerate(spline_cols_1):
                if col in model_1.params_.index:
                    linear_pred_1 += model_1.params_[col] * grid_splines_1[:, i]
            
            # Convert to relative risk and protection
            rr_1 = np.exp(linear_pred_1 - linear_pred_1[0])  # Relative to antibody level 0
            protection_1 = 1 - rr_1
            results['protection_test_negative'] = protection_1
        
        # 2. Test Positive Pathogen Analysis  
        df2 = data[['time', 'event2', 'antibody_level_2', 'X']].copy()
        df2 = df2.dropna()
        
        if len(df2) > 10:
            # Create spline basis for antibody_level_2
            spline_transformer_2 = SplineTransformer(n_knots=4, degree=3, include_bias=False)
            antibody_splines_2 = spline_transformer_2.fit_transform(df2[['antibody_level_2']])
            
            # Add spline columns to dataframe
            for i in range(antibody_splines_2.shape[1]):
                df2[f'spline2_{i}'] = antibody_splines_2[:, i]
            
            # Fit Cox model with splines
            spline_cols_2 = [f'spline2_{i}' for i in range(antibody_splines_2.shape[1])]
            model_cols_2 = ['time', 'event2', 'X'] + spline_cols_2
            model_2 = CoxPHFitter()
            model_2.fit(df2[model_cols_2], duration_col='time', event_col='event2')
            
            # Predict at antibody grid points
            grid_splines_2 = spline_transformer_2.transform(antibody_grid.reshape(-1, 1))
            
            # Calculate linear predictor for each grid point
            linear_pred_2 = np.zeros(len(antibody_grid))
            for i, col in enumerate(spline_cols_2):
                if col in model_2.params_.index:
                    linear_pred_2 += model_2.params_[col] * grid_splines_2[:, i]
            
            # Convert to relative risk and protection
            rr_2 = np.exp(linear_pred_2 - linear_pred_2[0])  # Relative to antibody level 0
            protection_2 = 1 - rr_2
            results['protection_test_positive'] = protection_2
        
        # 3. Analysis with Unmeasured Confounders
        df_conf = data[['time', 'event2', 'antibody_level_2', 'X', 'U']].copy()
        df_conf = df_conf.dropna()
        
        if len(df_conf) > 10:
            # Create spline basis
            spline_transformer_conf = SplineTransformer(n_knots=4, degree=3, include_bias=False)
            antibody_splines_conf = spline_transformer_conf.fit_transform(df_conf[['antibody_level_2']])
            
            # Add spline columns
            for i in range(antibody_splines_conf.shape[1]):
                df_conf[f'spline_conf_{i}'] = antibody_splines_conf[:, i]
            
            # Fit Cox model
            spline_cols_conf = [f'spline_conf_{i}' for i in range(antibody_splines_conf.shape[1])]
            model_cols_conf = ['time', 'event2', 'X', 'U'] + spline_cols_conf
            model_conf = CoxPHFitter()
            model_conf.fit(df_conf[model_cols_conf], duration_col='time', event_col='event2')
            
            # Predict at grid points
            grid_splines_conf = spline_transformer_conf.transform(antibody_grid.reshape(-1, 1))
            
            # Calculate linear predictor
            linear_pred_conf = np.zeros(len(antibody_grid))
            for i, col in enumerate(spline_cols_conf):
                if col in model_conf.params_.index:
                    linear_pred_conf += model_conf.params_[col] * grid_splines_conf[:, i]
            
            # Convert to protection
            rr_conf = np.exp(linear_pred_conf - linear_pred_conf[0])
            protection_conf = 1 - rr_conf
            results['protection_with_confounders'] = protection_conf
        
        # 4. Difference-in-Differences using antibody levels
        if (results['protection_test_negative'] is not None and 
            results['protection_test_positive'] is not None):
            # DiD protection = protection_test_positive / protection_test_negative (ratio scale)
            # Convert protections to relative risks first, then take ratio
            rr_test_negative = 1 - results['protection_test_negative'] 
            rr_test_positive = 1 - results['protection_test_positive']
            # DiD relative risk = RR_test_positive / RR_test_negative  
            rr_did = rr_test_positive / rr_test_negative
            protection_did = 1 - rr_did  # Convert back to protection scale
            results['protection_difference_in_differences'] = protection_did
        
        # 5. Test Negative Design with antibody levels
        tnd_data = data[(data['T1']) | (data['T2'])].copy()
        if len(tnd_data) > 20:  # Need more data for TND
            # Create outcome (1 = test positive, 0 = test negative)
            tnd_data['test_positive'] = tnd_data['T2'].astype(int)
            
            # Use antibody level 2 for test-negative design (the target pathogen)
            tnd_df = tnd_data[['test_positive', 'antibody_level_2', 'X']].copy()
            tnd_df = tnd_df.dropna()
            
            if len(tnd_df) > 20 and tnd_df['antibody_level_2'].var() > 0:
                try:
                    # Create spline basis for TND
                    spline_transformer_tnd = SplineTransformer(n_knots=3, degree=3, include_bias=False)
                    antibody_splines_tnd = spline_transformer_tnd.fit_transform(tnd_df[['antibody_level_2']])
                    
                    # Add splines to dataframe
                    X_tnd = np.column_stack([tnd_df[['X']].values, antibody_splines_tnd])
                    
                    # Fit logistic regression
                    tnd_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
                    tnd_model.fit(X_tnd, tnd_df['test_positive'])
                    
                    # Predict at grid points (using mean X value)
                    grid_splines_tnd = spline_transformer_tnd.transform(antibody_grid.reshape(-1, 1))
                    mean_X = tnd_df['X'].mean()
                    X_grid_tnd = np.column_stack([np.full(len(antibody_grid), mean_X), grid_splines_tnd])
                    
                    # Get predicted probabilities
                    prob_test_pos = tnd_model.predict_proba(X_grid_tnd)[:, 1]
                    
                    # Convert to odds ratios relative to antibody level 0
                    odds_grid = prob_test_pos / (1 - prob_test_pos)
                    or_grid = odds_grid / odds_grid[0]
                    
                    # Protection = 1 - OR
                    protection_tnd = 1 - or_grid
                    results['protection_test_negative_design'] = protection_tnd
                    
                except Exception as e:
                    if print_results:
                        print(f"TND spline fitting failed: {e}")
        
        # Also calculate traditional vaccination-based hazard ratios for comparison
        try:
            # Traditional vaccination-based models (for backward compatibility)
            df1_vacc = data[['time', 'event1', 'vaccinated', 'X']].copy()
            df2_vacc = data[['time', 'event2', 'vaccinated', 'X']].copy()
            df_vacc = data[['time', 'event2', 'vaccinated', 'X', 'U']].copy()
            
            model_1_vacc = CoxPHFitter()
            model_1_vacc.fit(df1_vacc, duration_col='time', event_col='event1')
            results['hr_vaccination_test_negative'] = np.exp(model_1_vacc.params_['vaccinated'])
            
            model_2_vacc = CoxPHFitter()
            model_2_vacc.fit(df2_vacc, duration_col='time', event_col='event2')
            results['hr_vaccination_test_positive'] = np.exp(model_2_vacc.params_['vaccinated'])
            
            model_vacc = CoxPHFitter()
            model_vacc.fit(df_vacc, duration_col='time', event_col='event2')
            results['hr_vaccination_with_confounders'] = np.exp(model_vacc.params_['vaccinated'])
            
            results['hr_vaccination_did'] = (results['hr_vaccination_test_positive'] / 
                                           results['hr_vaccination_test_negative'])
            
            # TND with vaccination
            tnd_vacc = tnd_data[['test_positive', 'vaccinated', 'X']].copy()
            if len(tnd_vacc) > 10 and tnd_vacc['vaccinated'].var() > 0:
                tnd_model_vacc = LogisticRegression(penalty=None)
                tnd_model_vacc.fit(tnd_vacc[['vaccinated', 'X']], tnd_vacc['test_positive'])
                results['hr_vaccination_tnd'] = np.exp(tnd_model_vacc.coef_[0][0])
            
        except Exception as e:
            if print_results:
                print(f"Vaccination-based model fitting failed: {e}")
    
    except Exception as e:
        if print_results:
            print(f"Model fitting failed: {e}")
    
    # Print results
    if print_results:
        print("=== ANTIBODY-BASED PROTECTION ANALYSIS ===")
        print(f"Antibody levels: {antibody_grid}")
        
        if results['protection_test_negative'] is not None:
            print(f"Test Negative Pathogen Protection: {results['protection_test_negative']}")
        
        if results['protection_test_positive'] is not None:
            print(f"Test Positive Pathogen Protection: {results['protection_test_positive']}")
        
        if results['protection_with_confounders'] is not None:
            print(f"With Confounders Protection: {results['protection_with_confounders']}")
        
        if results['protection_difference_in_differences'] is not None:
            print(f"Difference-in-Differences Protection: {results['protection_difference_in_differences']}")
        
        if results['protection_test_negative_design'] is not None:
            print(f"Test Negative Design Protection: {results['protection_test_negative_design']}")
        
        print("\n=== VACCINATION-BASED HAZARD RATIOS (for comparison) ===")
        if results['hr_vaccination_test_negative'] is not None:
            print(f"Test Negative HR: {results['hr_vaccination_test_negative']:.3f}")
        if results['hr_vaccination_test_positive'] is not None:
            print(f"Test Positive HR: {results['hr_vaccination_test_positive']:.3f}")
        if results['hr_vaccination_with_confounders'] is not None:
            print(f"With Confounders HR: {results['hr_vaccination_with_confounders']:.3f}")
        if results['hr_vaccination_did'] is not None:
            print(f"Difference-in-Differences HR: {results['hr_vaccination_did']:.3f}")
        if results['hr_vaccination_tnd'] is not None:
            print(f"Test Negative Design HR: {results['hr_vaccination_tnd']:.3f}")
    
    return results

def run_simulation(num_nodes=1000, edges_per_node=5, infection_prob=(0.02, 0.02), 
                  recovery_prob=(0.1, 0.1), testing_prob=(0.3, 0.3), vaccine_prob=0.5,
                  vaccine_efficacy=(0.0, 0.8), num_initial_infected=(5, 5), steps=100,
                  antibody_max_vacc=0.8, antibody_max_inf=0.9, antibody_waning=0.01,
                  plot=True, print_progress=True):
    """
    Convenience function to run a complete test-negative design simulation.
    
    Parameters:
    - num_nodes: Number of nodes in the graph
    - edges_per_node: Expected number of edges per node
    - infection_prob: Tuple of base infection probabilities (test_neg, test_pos)
    - recovery_prob: Tuple of recovery probabilities (test_neg, test_pos)
    - testing_prob: Tuple of testing probabilities (test_neg, test_pos)
    - vaccine_prob: Base probability of vaccination
    - vaccine_efficacy: Tuple of vaccine efficacies (test_neg, test_pos)
    - num_initial_infected: Tuple of initial infected (test_neg, test_pos)
    - steps: Number of simulation steps
    - antibody_max_vacc: Maximum antibody level from vaccination
    - antibody_max_inf: Maximum antibody level from infection
    - antibody_waning: Antibody waning per time step
    - plot: Whether to generate plots
    - print_progress: Whether to print progress updates
    
    Returns:
    - G: Final graph state
    - results: Dictionary with hazard ratio estimates and time series data
    """
    
    # Create graph and initialize
    G = generate_random_graph(num_nodes, edges_per_node)
    assign_initial_states(G, infection_prob, recovery_prob, testing_prob, vaccine_prob, 
                         vaccine_efficacy, antibody_max_vacc, antibody_max_inf, antibody_waning)
    assign_initial_infected(G, num_initial_infected)
    assign_vaccine(G)
    
    # Run simulation
    G, time_series_data = simulate_outbreaks(G, steps, plot=plot, print_progress=print_progress)
    
    # Estimate models
    model_results = estimate_models(G, print_results=print_progress)
    
    # Combine results
    results = {
        'model_results': model_results,  # Full antibody-based analysis results
        'time_series': time_series_data,
        # Backward compatibility - extract vaccination-based HRs if available
        'hr_test_negative': model_results.get('hr_vaccination_test_negative'),
        'hr_test_positive': model_results.get('hr_vaccination_test_positive'),
        'hr_with_confounders': model_results.get('hr_vaccination_with_confounders'),
        'hr_difference_in_differences': model_results.get('hr_vaccination_did'),
        'hr_test_negative_design': model_results.get('hr_vaccination_tnd')
    }
    
    return G, results
