#!/usr/bin/env python3
"""
Generate LaTeX table from experiment_summary.json
"""

import json
from pathlib import Path


def load_results(json_path):
    """Load experiment results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def parse_victim_name(victim_name):
    """
    Extract defense type and dataset from victim_name.
    
    Examples:
    - none_unnatural_test_gpt_4o_mini -> (None, UNNATURAL)
    - baseline_system_prompt_leakage_gpt_4o_mini -> (Direct, Synthetic System Prompts)
    - psm_unnatural_test_gpt_5_mini -> (PSM, UNNATURAL)
    """
    parts = victim_name.lower()
    
    # Determine defense type
    if parts.startswith('none_'):
        defense = 'No Defense'
    elif parts.startswith('baseline_'):
        defense = 'Direct'
    elif parts.startswith('psm_'):
        defense = 'PSM'
    elif parts.startswith('ngram_filter_'):
        defense = 'N-gram Filter'
    elif parts.startswith('fake_'):
        defense = 'Fake'
    else:
        defense = 'Unknown'
    
    # Determine dataset type
    if 'system_prompt_leakage' in parts:
        dataset = 'Synthetic System Prompts'
    elif 'unnatural_test' in parts:
        dataset = 'UNNATURAL'
    else:
        dataset = 'Unknown'
    
    return defense, dataset


def get_attack_name(attack_file):
    """Convert attack file name to attack name."""
    # Remove .json extension
    attack = attack_file.replace('.json', '')
    
    # Map raccon_language to raccon
    if attack == 'raccon_language':
        return 'raccon_language'
    
    return attack


def normalize_model_name(model_name):
    """Normalize model names to match expected format."""
    # Map various model name formats to standard ones
    mapping = {
        'gpt-4o-mini': 'gpt-4o-mini',
        'gpt-4.1-mini': 'gpt-4.1-mini',
        'gpt-5-mini': 'gpt-5-mini',
    }
    return mapping.get(model_name, model_name)


def organize_data(results):
    """
    Organize results into a nested structure for easy table generation.
    
    Structure:
    {
        dataset: {
            attack: {
                defense: {
                    model: {
                        'am_avg': value,
                        'jm_avg': value
                    }
                }
            }
        }
    }
    """
    organized = {}
    
    for model, attacks in results.items():
        model = normalize_model_name(model)
        
        for attack_file, victims in attacks.items():
            attack = get_attack_name(attack_file)
            
            for victim_name, metrics in victims.items():
                defense, dataset = parse_victim_name(victim_name)
                
                # Initialize nested structure
                if dataset not in organized:
                    organized[dataset] = {}
                if attack not in organized[dataset]:
                    organized[dataset][attack] = {}
                if defense not in organized[dataset][attack]:
                    organized[dataset][attack][defense] = {}
                
                # Store metrics
                organized[dataset][attack][defense][model] = {
                    'am_avg': metrics['am_avg_asr'] * 100,  # Convert to percentage
                    'jm_avg': metrics['jm_avg_asr'] * 100,  # Convert to percentage
                }
    
    return organized


def find_min_values(data, dataset, attack, defenses_to_display):
    """Find minimum ROUNDED values for each metric and model in a dataset-attack combination.
    
    This compares across defenses for a given (dataset, attack, model, metric) combination.
    Only considers defenses that will be displayed in the table.
    
    Args:
        data: The organized data structure
        dataset: The dataset name
        attack: The attack name
        defenses_to_display: List of defense names to include in minimum calculation
    """
    min_values = {
        'gpt-5-mini': {'am_avg': float('inf'), 'jm_avg': float('inf')},
        'gpt-4.1-mini': {'am_avg': float('inf'), 'jm_avg': float('inf')},
        'gpt-4o-mini': {'am_avg': float('inf'), 'jm_avg': float('inf')},
    }
    
    if dataset in data and attack in data[dataset]:
        for defense, models in data[dataset][attack].items():
            # Only consider defenses that will be displayed
            if defense not in defenses_to_display:
                continue
                
            for model, metrics in models.items():
                # Round first, then compare
                am_rounded = round(metrics['am_avg'])
                jm_rounded = round(metrics['jm_avg'])
                
                if am_rounded < min_values[model]['am_avg']:
                    min_values[model]['am_avg'] = am_rounded
                if jm_rounded < min_values[model]['jm_avg']:
                    min_values[model]['jm_avg'] = jm_rounded
    
    return min_values


def format_value(value, is_min=False):
    """Format a value as percentage with optional bold.
    
    Args:
        value: The value to format (should already be rounded if needed for min comparison)
        is_min: Whether this is a minimum value (for bolding)
    """
    # Round to nearest integer for display
    val_str = f"{round(value)}\\%"
    
    if is_min:
        return f"\\textbf{{{val_str}}}"
    return val_str


def generate_latex_table(data):
    """Generate LaTeX table from organized data."""
    
    # Define order
    datasets = ['Synthetic System Prompts', 'UNNATURAL']
    attacks = ['raccon', 'raccon_language']
    defenses = ['No Defense', 'N-gram Filter', 'PSM']
    models = ['gpt-5-mini', 'gpt-4.1-mini', 'gpt-4o-mini']
    
    lines = []
    
    # Table header
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append("")
    lines.append("\\begin{tabular}{|c|c|c|cc|cc|cc|}")
    lines.append("\\hline")
    lines.append("\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Attack} & \\multirow{2}{*}{Defense} &")
    lines.append("\\multicolumn{2}{c|}{gpt-5-mini} & \\multicolumn{2}{c|}{gpt-4.1-mini} & \\multicolumn{2}{c|}{gpt-4o-mini} \\\\")
    lines.append("\\cline{4-9}")
    lines.append("& & & AM Avg & JM Avg & AM Avg & JM Avg & AM Avg & JM Avg \\\\")
    lines.append("\\hline")
    lines.append("")
    
    # Generate data rows
    for dataset in datasets:
        if dataset not in data:
            continue
            
        # Count total rows for this dataset
        num_attacks = len([a for a in attacks if a in data[dataset]])
        num_rows = num_attacks * len(defenses)
        
        lines.append(f"% {dataset}")
        lines.append(f"\\multirow{{{num_rows}}}{{*}}{{{dataset}}}")
        
        for attack_idx, attack in enumerate(attacks):
            if attack not in data[dataset]:
                continue
            
            # Find minimum values for this attack (only among displayed defenses)
            min_values = find_min_values(data, dataset, attack, defenses)
            
            for defense_idx, defense in enumerate(defenses):
                if defense not in data[dataset][attack]:
                    continue
                
                # Build row
                row_parts = []
                
                # Attack column (only on first defense for each attack)
                if defense_idx == 0:
                    row_parts.append(f"\\multirow{{3}}{{*}}{{{attack}}}")
                else:
                    row_parts.append("")
                
                # Defense column
                row_parts.append(defense)
                
                # Metrics for each model
                for model in models:
                    if model in data[dataset][attack][defense]:
                        metrics = data[dataset][attack][defense][model]
                        
                        # Check if ROUNDED values equal the minimum
                        # (min_values already contains rounded values)
                        am_rounded = round(metrics['am_avg'])
                        jm_rounded = round(metrics['jm_avg'])
                        am_is_min = am_rounded == min_values[model]['am_avg']
                        jm_is_min = jm_rounded == min_values[model]['jm_avg']
                        
                        am_str = format_value(metrics['am_avg'], am_is_min)
                        jm_str = format_value(metrics['jm_avg'], jm_is_min)
                        
                        row_parts.append(am_str)
                        row_parts.append(jm_str)
                    else:
                        row_parts.append("--")
                        row_parts.append("--")
                
                # Add row
                lines.append("& " + " & ".join(row_parts) + " \\\\")
            
            # Add \cline after each attack except the last one
            if attack_idx < len([a for a in attacks if a in data[dataset]]) - 1:
                lines.append("\\cline{2-9}")
        
        lines.append("\\hline")
        lines.append("")
    
    # Table footer
    lines.append("\\end{tabular}")
    lines.append("\\caption{Attack metrics (percentages) for UNNATURAL and Synthetic System Prompts datasets across models and defenses, using average metrics (AM Avg, JM Avg). Lowest values in each block are bolded.}")
    lines.append("\\label{tab:gpt_datasets_avg_metrics}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def main():
    # Path to the JSON file
    json_path = Path(__file__).parent.parent / "results" / "experiment_summary.json"
    
    # Load and process results
    print(f"Loading results from: {json_path}")
    results = load_results(json_path)
    
    print("Organizing data...")
    organized_data = organize_data(results)
    
    print("Generating LaTeX table...")
    latex_table = generate_latex_table(organized_data)
    
    # Print to stdout
    print("\n" + "="*80)
    print("GENERATED LATEX TABLE")
    print("="*80 + "\n")
    print(latex_table)
    
    # Optionally save to file
    output_path = Path(__file__).parent.parent / "results" / "experiment_table.tex"
    with open(output_path, 'w') as f:
        f.write(latex_table)
    print(f"\n\nTable saved to: {output_path}")


if __name__ == "__main__":
    main()

