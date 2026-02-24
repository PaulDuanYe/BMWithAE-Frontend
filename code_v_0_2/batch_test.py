"""
Batch Test Script for Config Parameter Testing

This script allows testing different combinations of config.py parameters
and aggregates the results for comparison and analysis.

Usage:
    python batch_test.py
"""

import os
import sys
import json
import time
import copy
import importlib
from datetime import datetime
from itertools import product

import config
from main import run_test


def get_config_params():
    """
    Get all configurable parameters from config module.
    
    Returns:
        dict: Dictionary of parameter names and their current values
    """
    params = {}
    for param_name in dir(config):
        if not param_name.startswith('__') and param_name.isupper():
            params[param_name] = getattr(config, param_name)
    return params


def set_config_params(params_dict):
    """
    Dynamically set config parameters.
    
    Args:
        params_dict (dict): Dictionary of parameter names and values to set
    """
    for param_name, param_value in params_dict.items():
        if hasattr(config, param_name):
            setattr(config, param_name, param_value)
        else:
            print(f"Warning: Parameter {param_name} not found in config")


def generate_param_combinations(param_grid):
    """
    Generate all combinations of parameters from a grid.
    
    Args:
        param_grid (dict): Dictionary where keys are parameter names and values are lists of values to test
    
    Returns:
        list: List of dictionaries, each representing one parameter combination
    """
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    return combinations


def run_single_test(test_id, params, base_config, verbose=False):
    """
    Run a single test with specified parameters.
    
    Args:
        test_id (int): Test identifier
        params (dict): Parameters to use for this test
        base_config (dict): Base configuration to start from
        verbose (bool): Whether to print verbose output
    
    Returns:
        dict: Test results including metrics and configuration
    """
    print(f"\n{'='*60}")
    print(f"Test #{test_id}")
    print(f"{'='*60}")
    
    # Reset to base config first
    set_config_params(base_config)
    
    # Apply test-specific parameters
    set_config_params(params)
    
    # Disable verbose output for batch testing unless specified
    original_verbose = config.VERBOSE
    config.VERBOSE = verbose
    
    # Print test configuration
    print("Test parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Run the test
    start_time = time.time()
    try:
        final_metrics, changed_dict, results_history = run_test()
        success = True
        error_msg = None
    except Exception as e:
        final_metrics = None
        changed_dict = None
        results_history = None
        success = False
        error_msg = str(e)
        print(f"Error during test: {error_msg}")
    
    elapsed_time = time.time() - start_time
    
    # Restore verbose setting
    config.VERBOSE = original_verbose
    
    # Prepare results
    test_result = {
        'test_id': test_id,
        'timestamp': datetime.now().isoformat(),
        'elapsed_time_seconds': round(elapsed_time, 2),
        'success': success,
        'error_message': error_msg,
        'parameters': params,
        'final_metrics': final_metrics,
        'changed_dict': changed_dict,
    }
    
    # Print summary
    if success and final_metrics:
        print(f"\nTest #{test_id} completed in {elapsed_time:.2f}s")
        print("Final metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    else:
        print(f"\nTest #{test_id} failed: {error_msg}")
    
    return test_result


def save_batch_results(results, output_dir='batch_results'):
    """
    Save all batch test results to files.
    
    Args:
        results (list): List of test results
        output_dir (str): Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save full results
    full_results_file = os.path.join(output_dir, f'batch_results_{timestamp}.json')
    with open(full_results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {full_results_file}")
    
    # Save summary CSV
    summary_data = []
    for result in results:
        if result['success'] and result['final_metrics']:
            row = {
                'test_id': result['test_id'],
                'elapsed_time': result['elapsed_time_seconds'],
            }
            row.update(result['parameters'])
            row.update(result['final_metrics'])
            summary_data.append(row)
    
    if summary_data:
        import pandas as pd
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, f'batch_summary_{timestamp}.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary saved to: {summary_file}")
        
        return summary_df
    
    return None


def print_comparison_table(summary_df):
    """
    Print a comparison table of test results.
    
    Args:
        summary_df (pd.DataFrame): Summary dataframe with test results
    """
    if summary_df is None or summary_df.empty:
        print("No successful tests to compare.")
        return
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    # Identify metric columns (numeric columns that are not parameters)
    param_cols = ['test_id', 'elapsed_time']
    metric_cols = [col for col in summary_df.columns 
                   if col not in param_cols and 
                   summary_df[col].dtype in ['float64', 'int64']]
    
    # Print header
    print(f"{'Test ID':<10}", end='')
    for col in metric_cols[:10]:  # Limit to first 10 metrics
        print(f"{col:<15}", end='')
    print()
    print("-"*80)
    
    # Print rows
    for _, row in summary_df.iterrows():
        print(f"{row['test_id']:<10}", end='')
        for col in metric_cols[:10]:
            val = row[col]
            if isinstance(val, (int, float)):
                print(f"{val:<15.4f}", end='')
            else:
                print(f"{str(val):<15}", end='')
        print()


def run_batch_tests(param_grid, verbose=False):
    """
    Run batch tests with all parameter combinations.
    
    Args:
        param_grid (dict): Dictionary of parameter grids
        verbose (bool): Whether to print verbose output for each test
    
    Returns:
        list: List of all test results
    """
    # Get base configuration
    base_config = get_config_params()
    
    # Generate all parameter combinations
    combinations = generate_param_combinations(param_grid)
    total_tests = len(combinations)
    
    print(f"\nStarting batch testing...")
    print(f"Total parameter combinations: {total_tests}")
    print(f"Base configuration: {len(base_config)} parameters")
    
    # Store all results
    all_results = []
    
    # Run each test
    for i, params in enumerate(combinations, 1):
        result = run_single_test(i, params, base_config, verbose=verbose)
        all_results.append(result)
    
    return all_results


def define_test_scenarios():
    """
    Define different test scenarios for batch testing.
    
    Returns:
        dict: Dictionary of scenario names and their parameter grids
    """
    scenarios = {}
    
    # Scenario 1: Test different classifiers
    scenarios['classifier_comparison'] = {
        'PARAMS_MAIN_CLASSIFIER': ['LR', 'DT', 'RF', 'XGBoost', 'LGBM'],
        'USE_BIAS_MITIGATION': [True],
        'USE_ACCURACY_ENHANCEMENT': [False],
    }
    
    # Scenario 2: Test bias mitigation vs accuracy enhancement
    scenarios['mitigation_enhancement_comparison'] = {
        'USE_BIAS_MITIGATION': [True, False],
        'USE_ACCURACY_ENHANCEMENT': [True, False],
        'PARAMS_MAIN_CLASSIFIER': ['LR'],
    }
    
    # Scenario 3: Test different threshold values
    scenarios['threshold_sensitivity'] = {
        'PARAMS_MAIN_THRESHOLD_EPSILON': [0.3, 0.5, 0.7],
        'PARAMS_MAIN_THRESHOLD_PHI': [0.3, 0.5, 0.7],
        'PARAMS_MAIN_CLASSIFIER': ['LR'],
        'USE_BIAS_MITIGATION': [True],
    }
    
    # Scenario 4: Test different iteration limits
    scenarios['iteration_comparison'] = {
        'PARAMS_MAIN_MAX_ITERATION': [3, 5, 10],
        'PARAMS_MAIN_CLASSIFIER': ['LR'],
        'USE_BIAS_MITIGATION': [True],
    }
    
    # Scenario 5: Test different rebin methods
    scenarios['rebin_method_comparison'] = {
        'PARAMS_MAIN_BM_REBIN_METHOD': ['r1', 'r2'],
        'PARAMS_MAIN_CLASSIFIER': ['LR'],
        'USE_BIAS_MITIGATION': [True],
    }
    
    # Scenario 6: Test different normalization methods
    scenarios['normalization_comparison'] = {
        'PARAMS_EVAL_NORM': ['min-max', 'z-score'],
        'PARAMS_MAIN_CLASSIFIER': ['LR'],
    }
    
    # Scenario 7: Comprehensive test (smaller subset)
    scenarios['comprehensive_small'] = {
        'PARAMS_MAIN_CLASSIFIER': ['LR', 'RF'],
        'USE_BIAS_MITIGATION': [True, False],
        'PARAMS_MAIN_MAX_ITERATION': [3, 5],
    }
    
    return scenarios


def run_scenario(scenario_name, param_grid, verbose=False):
    """
    Run a specific test scenario.
    
    Args:
        scenario_name (str): Name of the scenario
        param_grid (dict): Parameter grid for the scenario
        verbose (bool): Whether to print verbose output
    
    Returns:
        tuple: (results list, summary dataframe)
    """
    print(f"\n{'#'*80}")
    print(f"# Running Scenario: {scenario_name}")
    print(f"{'#'*80}")
    
    results = run_batch_tests(param_grid, verbose=verbose)
    summary_df = save_batch_results(results)
    
    if summary_df is not None:
        print_comparison_table(summary_df)
    
    return results, summary_df


def main():
    """
    Main function to run batch tests.
    """
    print("="*80)
    print("BATCH CONFIG PARAMETER TESTING")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define test scenarios
    scenarios = define_test_scenarios()
    
    print("\nAvailable test scenarios:")
    for i, (name, grid) in enumerate(scenarios.items(), 1):
        num_combinations = 1
        for v in grid.values():
            num_combinations *= len(v)
        print(f"  {i}. {name} ({num_combinations} combinations)")
    
    # Select scenario to run
    print("\nOptions:")
    print("  - Enter scenario number (1-7) to run specific scenario")
    print("  - Enter 'all' to run all scenarios")
    print("  - Enter 'custom' to define custom parameters")
    print("  - Press Enter to run default scenario (classifier_comparison)")
    
    user_input = input("\nYour choice: ").strip().lower()
    
    all_scenario_results = {}
    
    if user_input == 'all':
        # Run all scenarios
        for name, grid in scenarios.items():
            results, summary = run_scenario(name, grid)
            all_scenario_results[name] = {
                'results': results,
                'summary': summary
            }
    
    elif user_input == 'custom':
        # Custom parameter testing
        print("\nCustom parameter testing mode")
        print("Example: PARAMS_MAIN_CLASSIFIER=['LR','RF'], USE_BIAS_MITIGATION=[True,False]")
        
        try:
            custom_input = input("Enter parameter grid (Python dict format): ")
            custom_grid = eval(custom_input)
            results, summary = run_scenario('custom', custom_grid)
            all_scenario_results['custom'] = {
                'results': results,
                'summary': summary
            }
        except Exception as e:
            print(f"Error parsing custom parameters: {e}")
            return
    
    elif user_input == '' or user_input == '1':
        # Run default scenario
        name = list(scenarios.keys())[0]
        results, summary = run_scenario(name, scenarios[name])
        all_scenario_results[name] = {
            'results': results,
            'summary': summary
        }
    
    else:
        # Run specific scenario by number
        try:
            scenario_idx = int(user_input) - 1
            scenario_names = list(scenarios.keys())
            if 0 <= scenario_idx < len(scenario_names):
                name = scenario_names[scenario_idx]
                results, summary = run_scenario(name, scenarios[name])
                all_scenario_results[name] = {
                    'results': results,
                    'summary': summary
                }
            else:
                print(f"Invalid scenario number. Please enter 1-{len(scenarios)}")
                return
        except ValueError:
            print("Invalid input. Please enter a number, 'all', or 'custom'.")
            return
    
    # Final summary
    print("\n" + "="*80)
    print("BATCH TESTING COMPLETED")
    print("="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_tests = sum(
        len(data['results']) 
        for data in all_scenario_results.values()
    )
    successful_tests = sum(
        sum(1 for r in data['results'] if r['success'])
        for data in all_scenario_results.values()
    )
    
    print(f"\nTotal tests run: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {total_tests - successful_tests}")
    
    print("\nResults saved in 'batch_results' directory")


if __name__ == "__main__":
    main()
