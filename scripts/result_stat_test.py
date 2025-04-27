# uv run scripts/result_stat_test.py 
import numpy as np
from scipy.stats import ttest_ind, fisher_exact
from typing import Dict, Tuple

def generate_trials(success_rate_percent: float, n_trials: int = 15, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    n_successes = int(success_rate_percent * n_trials / 100)
    n_failures = n_trials - n_successes
    trials = np.array([1]*n_successes + [0]*n_failures)
    np.random.shuffle(trials)
    return trials

def perform_tests(r2r2r_trials: np.ndarray, teleop_trials: np.ndarray) -> Dict[str, float]:
    # Welch's t-test (one-sided)
    t_stat, p_val = ttest_ind(teleop_trials, r2r2r_trials, equal_var=False)
    ttest_pval_one_sided = p_val / 2  # one-sided test

    # Fisher's Exact Test
    contingency_table = np.array([
        [teleop_trials.sum(), len(teleop_trials) - teleop_trials.sum()],
        [r2r2r_trials.sum(), len(r2r2r_trials) - r2r2r_trials.sum()]
    ])
    _, fisher_pval = fisher_exact(contingency_table, alternative='greater')

    return {
        "Welch_ttest_one_sided_pval": ttest_pval_one_sided,
        "Fisher_exact_one_sided_pval": fisher_pval
    }

def run_all_tests(task_data: Dict[str, Dict[str, Tuple[float, float]]], n_trials: int = 15, seed: int = 42):
    results = {}
    all_r2r2r_trials = []
    all_teleop_trials = []

    for task_name, policies in task_data.items():
        print(f"\n=== Task: {task_name} ===")
        results[task_name] = {}
        for policy_name, (r2r2r_success_rate, teleop_success_rate) in policies.items():
            print(f"Policy: {policy_name}")

            # Generate synthetic trial data
            r2r2r_trials = generate_trials(r2r2r_success_rate, n_trials=n_trials, seed=seed)
            teleop_trials = generate_trials(teleop_success_rate, n_trials=n_trials, seed=seed+1)  # different seed

            # Collect for overall test
            all_r2r2r_trials.append(r2r2r_trials)
            all_teleop_trials.append(teleop_trials)

            # Perform per-task test
            pvals = perform_tests(r2r2r_trials, teleop_trials)

            results[task_name][policy_name] = pvals

            print(f"  Welch’s t-test p-value (one-sided): {pvals['Welch_ttest_one_sided_pval']:.4f}")
            print(f"  Fisher’s exact test p-value (one-sided): {pvals['Fisher_exact_one_sided_pval']:.4f}")

    # === Global test across all tasks ===
    all_r2r2r_trials = np.concatenate(all_r2r2r_trials)
    all_teleop_trials = np.concatenate(all_teleop_trials)

    print(f"\n=== Overall Global Test Across All Tasks ===")
    global_pvals = perform_tests(all_r2r2r_trials, all_teleop_trials)
    print(f"  Overall Welch’s t-test p-value (one-sided): {global_pvals['Welch_ttest_one_sided_pval']:.4f}")
    print(f"  Overall Fisher’s exact test p-value (one-sided): {global_pvals['Fisher_exact_one_sided_pval']:.4f}")

    results["Global"] = global_pvals

    return results

# ======================
# Task data input
# Format: {task_name: {policy_name: (r2r2r_1000_success_rate, teleop_150_success_rate)}}
# ======================
task_data = {
    "Put the mug on the coffee maker": {
        "Diffusion Policy": (53.3, 40.0),
        "PI0-FAST (Finetuned)": (80.0, 73.3),
    },
    "Turn the faucet off": {
        "Diffusion Policy": (80.0, 66.6),
        "PI0-FAST (Finetuned)": (80.0, 80.0),
    },
    "Open the Drawer": {
        "Diffusion Policy": (66.6, 66.6),
        "PI0-FAST (Finetuned)": (86.6, 60.0),
    },
    "Pick up the package with both hands": {
        "Diffusion Policy": (46.6, 80.0),
        "PI0-FAST (Finetuned)": (66.6, 60.0),
    },
    # "Pick up the tiger": {
    #     "Diffusion Policy": (0.0, 73.3),  # No R2R2R results available, assumed 0%
    #     "PI0-FAST (Finetuned)": (0.0, 73.3),
    # }
}

# ======================
# Run!
# ======================
if __name__ == "__main__":
    results = run_all_tests(task_data)
