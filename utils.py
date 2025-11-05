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

def assign_initial_states(G, infection_prob, recovery_prob, testing_prob, vaccine_prob,
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
        G.nodes[node]['recovery_prob_1'] = recovery_prob[0] # Probability of recovery for test negative pathogen
        G.nodes[node]['recovery_prob_2'] = recovery_prob[1] # Probability of recovery for test positive pathogen
        
        # Antibody levels (0 = no protection, 1 = complete protection)
        G.nodes[node]['antibody_level_1'] = 0.0  # Antibody level for test negative pathogen (not affected by vaccination)
        G.nodes[node]['antibody_level_2'] = 0.0  # Antibody level for test positive pathogen (affected by vaccination and infection)
        
        # Antibody levels at time of testing (initially None, will be set when testing occurs)
        G.nodes[node]['antibody_level_1_at_test'] = None  # Antibody level 1 at time of testing
        G.nodes[node]['antibody_level_2_at_test'] = None  # Antibody level 2 at time of testing
        
        # Antibody parameters
        G.nodes[node]['antibody_max_vacc'] = antibody_max_vacc  # Max antibody level from vaccination
        G.nodes[node]['antibody_max_inf'] = antibody_max_inf    # Max antibody level from infection
        G.nodes[node]['antibody_waning'] = antibody_waning      # Antibody waning per time step
        
        # Calculate individual vaccine probabilities based on confounders
        G.nodes[node]['vaccine_prob'] *= np.exp(G.nodes[node]['U'] + G.nodes[node]['X'] - 1)
        
        # Calculate individual infection probabilities based on confounders (including U always)
        G.nodes[node]['infection_prob_1'] = infection_prob[0] * np.exp(
            G.nodes[node]['X'] - 0.5 + G.nodes[node]['U'] - 0.5
        )
        G.nodes[node]['infection_prob_2'] = infection_prob[1] * np.exp(
            G.nodes[node]['X'] - 0.5 + G.nodes[node]['U'] - 0.5
        )

        # Calculate individual testing probabilities based on confounders (including U always)
        G.nodes[node]['testing_prob_1'] = testing_prob[0] * np.exp(
            G.nodes[node]['X'] - 0.5 + G.nodes[node]['U'] - 0.5
        ) 
        G.nodes[node]['testing_prob_2'] = testing_prob[1] * np.exp(
            G.nodes[node]['X'] - 0.5 + G.nodes[node]['U'] - 0.5
        )
        
def assign_initial_infected(G, num_initial_infected):
    """Assigns initial infected nodes in the graph."""
    initial_infected = random.sample(list(G.nodes()), int(np.sum(num_initial_infected)))
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

def get_protection_factor(antibody_level, l = 0.9, a =3, b = -10, scaled_logit = False):
    """Calculate protection factor based on antibody level (0-1 scale where 1 = complete protection)."""
    if scaled_logit:
        return (l / (1 + np.exp(a + b * antibody_level)))
    else:
        return antibody_level

def simulate_outbreaks(G, steps, plot=True, print_progress=True, scaled_logit = False):
    """Simulates disease outbreaks in the graph over a number of steps."""
    
    # Initialize tracking arrays
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
                        protection_1 = get_protection_factor(G.nodes[neighbor]['antibody_level_1'], scaled_logit = scaled_logit)
                        effective_prob_1 = G.nodes[neighbor]['infection_prob_1'] * (1 - protection_1)
                        
                        if random.random() < effective_prob_1:
                            G.nodes[neighbor]['I1'] = True
                            G.nodes[neighbor]['S1'] = False
                            if random.random() < G.nodes[neighbor]['testing_prob_1'] and not G.nodes[neighbor]['T2'] and not G.nodes[neighbor]['T1']:
                                G.nodes[neighbor]['T1'] = True
                                G.nodes[neighbor]['time1'] = step
                                # Capture antibody levels at time of testing
                                G.nodes[neighbor]['antibody_level_1_at_test'] = G.nodes[neighbor]['antibody_level_1']
                                G.nodes[neighbor]['antibody_level_2_at_test'] = G.nodes[neighbor]['antibody_level_2']

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
                        protection_2 = get_protection_factor(G.nodes[neighbor]['antibody_level_2'], scaled_logit = scaled_logit)
                        effective_prob_2 = G.nodes[neighbor]['infection_prob_2'] * (1 - protection_2)
                        
                        if random.random() < effective_prob_2:
                            G.nodes[neighbor]['I2'] = True
                            G.nodes[neighbor]['S2'] = False
                            if random.random() < G.nodes[neighbor]['testing_prob_2'] and not G.nodes[neighbor]['T1'] and not G.nodes[neighbor]['T2']:
                                G.nodes[neighbor]['T2'] = True
                                G.nodes[neighbor]['time2'] = step
                                # Capture antibody levels at time of testing
                                G.nodes[neighbor]['antibody_level_1_at_test'] = G.nodes[neighbor]['antibody_level_1']
                                G.nodes[neighbor]['antibody_level_2_at_test'] = G.nodes[neighbor]['antibody_level_2']
            
            # If untested update follow up times
            if not G.nodes[node]['T1']:
                G.nodes[node]['time1'] += 1
            if not G.nodes[node]['T2']:
                G.nodes[node]['time2'] += 1

        # Track epidemic curves
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

    # Plot curves at the end
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

    # Return graph and time series data
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
    Estimates test-negative design model using antibody levels at time of testing.
    Returns protection function (1 - OR) from logistic regression.
    """
    # Create a DataFrame from the graph's node attributes
    data = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')

    # Define antibody level grid for predictions
    antibody_grid = np.linspace(.1, .9, 9)  # 10 points from 0 to 1
    antibody_grid = np.concatenate((.01, antibody_grid, .99), axis = None)
    
    # Initialize results dictionary
    results = {
        'antibody_grid': antibody_grid,
        'protection_spline': None,
        'hr_vaccination_tnd': None,
        'protection_log': None,
        'antibody_distribution': None
    }
    
    try:
        # Test Negative Design with antibody levels at time of testing
        tnd_data = data[(data['T1']) | (data['T2'])].copy()
        
        if len(tnd_data) > 20:  # Need sufficient data for TND
            # Create outcome (1 = test positive, 0 = test negative)
            tnd_data['test_positive'] = tnd_data['T2'].astype(int)
            
            # Use antibody level 2 at test for test-negative design (the target pathogen)
            tnd_df = tnd_data[['test_positive', 'antibody_level_2_at_test', 'X']].copy()
            tnd_df = tnd_df.dropna()
            bin_edges = (antibody_grid[:-1] + antibody_grid[1:]) / 2
            bin_edges = np.concatenate(([0.0], bin_edges, [1.0]))
            #print(bin_edges) 

            hist_counts, _ = np.histogram(tnd_df['antibody_level_2_at_test'], bins=bin_edges)
            #print(tnd_df) 
            results['antibody_distribution'] = hist_counts
            #print(results['antibody_distribution'])
            #print(hist_counts)


            if len(tnd_df) > 20 and tnd_df['antibody_level_2_at_test'].var() > 0:

                #Spline transformation
                try:
                    # Create spline basis for TND
                    spline_transformer_tnd = SplineTransformer(n_knots=3, degree=3, include_bias=False)
                    antibody_splines_tnd = spline_transformer_tnd.fit_transform(tnd_df[['antibody_level_2_at_test']])
                    
                    # Add splines to dataframe
                    X_tnd = np.column_stack([tnd_df[['X']].values, antibody_splines_tnd])
                    
                    # Fit logistic regression without regularization
                    tnd_model = LogisticRegression(penalty=None, max_iter=1000)
                    tnd_model.fit(X_tnd, tnd_df['test_positive'])
                    
                    # Predict at grid points (using mean X value)
                    grid_splines_tnd = spline_transformer_tnd.transform(antibody_grid.reshape(-1, 1))
                    mean_X = 0.5
                    X_grid_tnd = np.column_stack([np.full(len(antibody_grid), mean_X), grid_splines_tnd])
                    
                    # Get predicted probabilities
                    prob_test_pos = tnd_model.predict_proba(X_grid_tnd)[:, 1]
                    
                    # Convert to odds ratios relative to antibody level 0
                    odds_grid = prob_test_pos / (1 - prob_test_pos)
                    or_grid = odds_grid / odds_grid[0]
                    
                    # Protection = 1 - OR
                    protection_tnd = 1 - or_grid
                    results['protection_spline'] = protection_tnd
                    
                except Exception as e:
                    if print_results:
                        print(f"TND spline fitting failed: {e}")
        

                # Log-transformed antibody model (without spline transformation)
                try:
                    tnd_df2 = tnd_df.copy()
                    tnd_df2['log_antibody'] = np.log(1 - tnd_df2['antibody_level_2_at_test'])

                    if tnd_df2['log_antibody'].var() > 0:
                        X_log = np.column_stack([tnd_df2[['X']].values, tnd_df2[['log_antibody']].values])

                        log_model = LogisticRegression(penalty=None, max_iter=1000)
                        log_model.fit(X_log, tnd_df2['test_positive'])

                        # Predict at grid points
                        log_grid = np.log(1 - antibody_grid)
                        X_grid_log = np.column_stack([np.full(len(log_grid), 0.5), log_grid.reshape(-1, 1)])

                        #Predicted probabilities
                        prob_log = log_model.predict_proba(X_grid_log)[:, 1]

                        odds_log = prob_log / (1 - prob_log)
                        or_log = odds_log / odds_log[0]
                        protection_log = 1 - or_log

                        results['protection_log'] = protection_log

                except Exception as e:
                    if print_results:
                        print(f"Log antibody model failed: {e}")

        # Also calculate traditional vaccination-based TND for comparison
        try:
            tnd_vacc = tnd_data[['test_positive', 'vaccinated', 'X']].copy()
            if len(tnd_vacc) > 10 and tnd_vacc['vaccinated'].var() > 0:
                tnd_model_vacc = LogisticRegression(penalty=None, max_iter=1000)
                tnd_model_vacc.fit(tnd_vacc[['vaccinated', 'X']], tnd_vacc['test_positive'])
                results['hr_vaccination_tnd'] = np.exp(tnd_model_vacc.coef_[0][0])
            
        except Exception as e:
            if print_results:
                print(f"Vaccination-based TND model fitting failed: {e}")

  

    except Exception as e:
        if print_results:
            print(f"Model fitting failed: {e}")
    
    # Print results
    if print_results:
        print("=== TEST-NEGATIVE DESIGN ANALYSIS (antibody levels at time of testing) ===")
        print(f"Antibody levels: {antibody_grid}")
        
        print("=== SPLINE ===")
        if results['protection_spline'] is not None:
            print(f"Antibody-based Protection: {results['protection_spline']}")
        else:
            print("Antibody-based Protection: Could not estimate (insufficient data)")
        
        print("=== LOG-TRANSFORMED ===")
        if results['protection_log'] is not None:
            print(f"Antibody-based Protection: {results['protection_log']}")
        else:
            print("Antibody-based Protection: Could not estimate (insufficient data)")

        print("\n=== VACCINATION-BASED TND (for comparison) ===")
        if results['hr_vaccination_tnd'] is not None:
            print(f"Vaccination TND HR: {results['hr_vaccination_tnd']:.3f}")
            print(f"Vaccination TND VE: {(1 - results['hr_vaccination_tnd']) * 100:.1f}%")
        else:
            print("Vaccination TND: Could not estimate (insufficient data)")
    return results

def run_simulation(num_nodes=1000, edges_per_node=5, infection_prob=(0.02, 0.02), 
                  recovery_prob=(0.1, 0.1), testing_prob=(0.3, 0.3), vaccine_prob=0.5,
                  num_initial_infected=(5, 5), steps=100,
                  antibody_max_vacc=0.8, antibody_max_inf=0.9, antibody_waning=0.01,
                  plot=True, print_progress=True, scaled_logit = False):
    """
    Convenience function to run a complete test-negative design simulation.
    
    Parameters:
    - num_nodes: Number of nodes in the graph
    - edges_per_node: Expected number of edges per node
    - infection_prob: Tuple of base infection probabilities (test_neg, test_pos)
    - recovery_prob: Tuple of recovery probabilities (test_neg, test_pos)
    - testing_prob: Tuple of testing probabilities (test_neg, test_pos)
    - vaccine_prob: Base probability of vaccination
    - num_initial_infected: Tuple of initial infected (test_neg, test_pos)
    - steps: Number of simulation steps
    - antibody_max_vacc: Maximum antibody level from vaccination
    - antibody_max_inf: Maximum antibody level from infection
    - antibody_waning: Antibody waning per time step
    - plot: Whether to generate plots
    - print_progress: Whether to print progress updates
    
    Returns:
    - G: Final graph state
    - results: Dictionary with TND estimates and time series data
    """
    
    # Create graph and initialize
    G = generate_random_graph(num_nodes, edges_per_node)
    assign_initial_states(G, infection_prob, recovery_prob, testing_prob, vaccine_prob, 
                         antibody_max_vacc, antibody_max_inf, antibody_waning)
    assign_initial_infected(G, num_initial_infected)
    assign_vaccine(G)
    
    # Run simulation
    G, time_series_data = simulate_outbreaks(G, steps, plot=plot, print_progress=print_progress, scaled_logit = scaled_logit)
    
    # Estimate models
    model_results = estimate_models(G, print_results=print_progress)
    
    # Combine results
    results = {
        'model_results': model_results,  # Full TND analysis results
        'time_series': time_series_data,
        # Extract key results for backward compatibility
        'hr_test_negative_design': model_results.get('hr_vaccination_tnd'),
        'protection_spline': model_results.get('protection_spline'),
        'protection_log': model_results.get('protection_log'),
        'antibody_grid': model_results.get('antibody_grid'),
        'antibody_distribution': model_results.get('antibody_distribution')
    }
    
    return G, results

def compute_bias_mse(predictions, true, dist):
    diffs = predictions - true[np.newaxis, :]
    bias_vec = np.nanmean(diffs, axis=0) #average for all sims
    mse_vec = np.nanmean(diffs**2, axis=0) #average for all sims

    avg_dist = np.nanmean(dist, axis=0)

    se_bias = np.nanstd(diffs, axis=0, ddof=1) 
    se_mse = np.nanstd(diffs**2, axis=0, ddof=1)

    sim_bias = np.nanmean(diffs, axis=1)  #average for all grid points
    sim_mse = np.nanmean(diffs**2, axis=1) #average for all grid points

    bias_overall = np.nanmean(sim_bias) #avergae over all grid points/all sims
    mse_overall = np.nanmean(sim_mse)  #avergae over all grid points/all sims
    se_bias_overall = np.nanstd(sim_bias, ddof=1) / np.sqrt(np.sum(~np.isnan(sim_bias)))
    se_mse_overall = np.nanstd(sim_mse, ddof=1) / np.sqrt(np.sum(~np.isnan(sim_mse)))

    weighted_bias = np.average(bias_vec, weights=avg_dist)
    weighted_mse = np.average(mse_vec, weights=avg_dist)
    print(avg_dist)

    return {
        'bias_vec': bias_vec,
        'mse_vec': mse_vec,
        'se_bias': se_bias,
        'se_mse': se_mse,
        'bias_overall': bias_overall,
        'mse_overall': mse_overall,
        'se_bias_overall': se_bias_overall,
        'se_mse_overall': se_mse_overall,
        'weighted_bias': weighted_bias, 
        'weighted_mse': weighted_mse
    }

def scaled_logit(x, l, a, b):
    return (l / (1 + np.exp(a + b * x)))