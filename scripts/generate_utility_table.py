#!/usr/bin/env python3
"""
Generate LaTeX table showing average utility preservation across models
for PSM. Reads from data/defense_prompts/ JSONL files.
"""

import json
from pathlib import Path
from collections import defaultdict


def parse_filename(filename):
    """
    Extract model name and dataset from defense prompt filename.
    
    Example:
    - psm_target_gpt_5_mini_dataset_system_prompt_leakage.jsonl 
      -> (gpt-5-mini, Synthetic System Prompts)
    - psm_target_gpt_4o_mini_dataset_unnatural_test.jsonl
      -> (gpt-4o-mini, UNNATURAL)
    """
    # Remove .jsonl extension
    name = filename.replace('.jsonl', '')
    
    # Extract model name
    if 'gpt_5_mini' in name:
        model = 'gpt-5-mini'
    elif 'gpt_5_nano' in name:
        model = 'gpt-5-nano'
    elif 'gpt_4.1_mini' in name:
        model = 'gpt-4.1-mini'
    elif 'gpt_4o_mini' in name:
        model = 'gpt-4o-mini'
    else:
        model = 'unknown'
    
    # Extract dataset name
    if 'system_prompt_leakage' in name:
        dataset = 'Synthetic System Prompts'
    elif 'unnatural_test' in name:
        dataset = 'UNNATURAL'
    else:
        dataset = 'Unknown'
    
    return model, dataset


def load_utility_scores(defense_prompts_dir):
    """
    Load utility scores from all PSM defense prompt files.
    
    Returns:
    {
        dataset: {
            model: [list of utility scores]
        }
    }
    """
    data = defaultdict(lambda: defaultdict(list))
    
    # Find all PSM jsonl files
    psm_files = list(Path(defense_prompts_dir).glob('psm_*.jsonl'))
    
    for file_path in psm_files:
        model, dataset = parse_filename(file_path.name)
        
        # Read JSONL file
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        if 'utility_score' in entry:
                            data[dataset][model].append(entry['utility_score'])
                    except json.JSONDecodeError:
                        continue
    
    return data


def compute_averages(data):
    """
    Compute average utility scores for each dataset-model combination.
    
    Returns:
    {
        dataset: {
            model: avg_utility_score
        }
    }
    """
    averages = {}
    
    for dataset, models in data.items():
        averages[dataset] = {}
        for model, scores in models.items():
            if scores:
                averages[dataset][model] = sum(scores) / len(scores)
            else:
                averages[dataset][model] = 0.0
    
    return averages


def find_max_values(averages, dataset, models_to_display):
    """
    Find maximum utility scores for each dataset.
    Only considers models that will be displayed in the table.
    
    Args:
        averages: The averages data structure
        dataset: The dataset name
        models_to_display: List of model names to include in maximum
            calculation
    """
    max_value = 0.0
    
    if dataset in averages:
        for model in models_to_display:
            if model in averages[dataset]:
                value = averages[dataset][model]
                if value > max_value:
                    max_value = value
    
    return max_value


def format_utility_value(value, is_max=False, as_percentage=True):
    """
    Format a utility value with optional bold for maximum values.
    
    Args:
        value: The utility score (where 1.0 = 100% preservation)
        is_max: Whether this is a maximum value (for bolding)
        as_percentage: Whether to format as percentage
    """
    if as_percentage:
        # Value is in format where 1.0 = 100% preservation
        # Convert to percentage by multiplying by 100
        val_str = f"{value * 100:.2f}\\%"
    else:
        val_str = f"{value:.4f}"
    
    if is_max:
        return f"\\textbf{{{val_str}}}"
    return val_str


def generate_latex_table(averages, include_counts=False, counts=None):
    """
    Generate LaTeX table from utility preservation averages.
    
    Args:
        averages: Dictionary of average utility scores
        include_counts: Whether to include sample counts in the table
        counts: Dictionary of sample counts (if include_counts is True)
    """
    # Define order
    datasets = ['Synthetic System Prompts', 'UNNATURAL']
    models = ['gpt-5-mini', 'gpt-4.1-mini', 'gpt-4o-mini']
    
    lines = []
    
    # Table header
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("")
    
    if include_counts and counts:
        lines.append("\\begin{tabular}{|l|c|c|c|c|}")
        lines.append("\\hline")
        lines.append(
            "\\textbf{Dataset} & \\textbf{gpt-5-mini} & "
            "\\textbf{gpt-4.1-mini} & \\textbf{gpt-4o-mini} \\\\"
        )
        lines.append("\\hline")
        lines.append("")
    else:
        lines.append("\\begin{tabular}{|l|c|c|c|}")
        lines.append("\\hline")
        lines.append(
            "\\textbf{Dataset} & \\textbf{gpt-5-mini} & "
            "\\textbf{gpt-4.1-mini} & \\textbf{gpt-4o-mini} \\\\"
        )
        lines.append("\\hline")
        lines.append("")
    
    # Generate data rows
    for dataset in datasets:
        if dataset not in averages:
            continue
        
        # Find maximum value for this dataset
        max_value = find_max_values(averages, dataset, models)
        
        # Build row
        row_parts = [dataset]
        
        for model in models:
            if model in averages[dataset]:
                value = averages[dataset][model]
                # Float comparison with tolerance
                is_max = abs(value - max_value) < 0.0001
                
                if (include_counts and counts and dataset in counts
                        and model in counts[dataset]):
                    count = counts[dataset][model]
                    val_str = format_utility_value(value, is_max)
                    row_parts.append(f"{val_str} ({count})")
                else:
                    row_parts.append(format_utility_value(value, is_max))
            else:
                row_parts.append("--")
        
        lines.append(" & ".join(row_parts) + " \\\\")
        lines.append("\\hline")
    
    lines.append("")
    
    # Table footer
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Average utility preservation (\\%) across models for "
        "PSM defense on different datasets. Higher values indicate better "
        "preservation of original functionality. Highest values in each "
        "row are bolded.}"
    )
    lines.append("\\label{tab:psm_utility_preservation}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def compute_counts(data):
    """
    Compute sample counts for each dataset-model combination.
    
    Returns:
    {
        dataset: {
            model: count
        }
    }
    """
    counts = {}
    
    for dataset, models in data.items():
        counts[dataset] = {}
        for model, scores in models.items():
            counts[dataset][model] = len(scores)
    
    return counts


def generate_summary_stats(averages, counts):
    """Generate summary statistics for display."""
    lines = []
    lines.append("\n" + "="*80)
    lines.append("UTILITY PRESERVATION SUMMARY")
    lines.append("="*80 + "\n")
    
    datasets = ['Synthetic System Prompts', 'UNNATURAL']
    models = ['gpt-5-mini', 'gpt-4.1-mini', 'gpt-4o-mini']
    
    for dataset in datasets:
        if dataset not in averages:
            continue
        
        lines.append(f"\n{dataset}:")
        lines.append("-" * 60)
        
        for model in models:
            if model in averages[dataset]:
                avg = averages[dataset][model]
                count = (counts[dataset][model] if dataset in counts
                         and model in counts[dataset] else 0)
                lines.append(
                    f"  {model:20s}: {avg*100:6.2f}% (n={count})"
                )
    
    # Overall averages
    lines.append("\n" + "="*60)
    lines.append("Overall Averages by Model:")
    lines.append("-" * 60)
    
    for model in models:
        all_scores = []
        for dataset in datasets:
            if dataset in averages and model in averages[dataset]:
                all_scores.append(averages[dataset][model])
        
        if all_scores:
            overall_avg = sum(all_scores) / len(all_scores)
            lines.append(f"  {model:20s}: {overall_avg*100:6.2f}%")
    
    return "\n".join(lines)


def main():
    # Path to the defense prompts directory
    defense_prompts_dir = (Path(__file__).parent.parent / "data"
                           / "defense_prompts")
    
    # Load utility scores
    print(f"Loading utility scores from: {defense_prompts_dir}")
    data = load_utility_scores(defense_prompts_dir)
    
    # Compute averages and counts
    print("Computing averages...")
    averages = compute_averages(data)
    counts = compute_counts(data)
    
    # Print summary statistics
    print(generate_summary_stats(averages, counts))
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(averages, include_counts=False,
                                       counts=counts)
    
    # Print to stdout
    print("\n" + "="*80)
    print("GENERATED LATEX TABLE")
    print("="*80 + "\n")
    print(latex_table)
    
    # Save to file
    output_path = (Path(__file__).parent.parent / "results"
                   / "utility_preservation_table.tex")
    with open(output_path, 'w') as f:
        f.write(latex_table)
    print(f"\n\nTable saved to: {output_path}")


if __name__ == "__main__":
    main()

