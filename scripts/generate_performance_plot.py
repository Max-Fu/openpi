"""Generates standalone plots comparing policy performance on real vs. synthetic data,
plotting against both time and number of demonstrations.

Generates:
1. A figure with subplots, each showing avg performance for a single comparable task.
2. A figure showing the overall average performance across comparable tasks.
Both figure types are generated for x-axis as Time and as Demonstrations.

uv run scripts/generate_performance_plot.py --base-output-dir output_plots
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tyro
import pathlib
import collections

# --- Data Generation Rates ---
# Calculate average time for the tasks with specific times
avg_real_time_min = (86.0 + 71.0 + 104.0) / 3.0 # Avg time for 150 demos
DEFAULT_REAL_TRAJ_PER_HOUR = (150.0 * 60.0) / avg_real_time_min # ~103.45 traj/hr
TASK_SPECIFIC_REAL_RATES = {
    "Put the mug on the coffee maker": (150.0 * 60.0) / 86.0,  # 1hr 26min
    "Open the Drawer": (150.0 * 60.0) / 71.0,                 # 1hr 11min
    "Turn the faucet off": (150.0 * 60.0) / 104.0,             # 1hr 44min
}
SYNTH_TRAJ_PER_HOUR = (1000.0 * 60.0) / 35.0 # ~1714.29 traj/hr (Using 1000 as reference count)
SYNTH_SETUP_TIME_MINUTES = 5.0

# --- Plot Styling ---
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,       
    "axes.titlesize": 14,  
    "axes.labelsize": 13,  
    "xtick.labelsize": 11, # Adjusted for potential subplot density
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 2.5,
    "lines.markersize": 6 # Slightly smaller markers
})

    # --- Performance Data ---
performance_data = collections.defaultdict(lambda: collections.defaultdict(lambda: {"demos": np.array([]), "pi0": np.array([]), "dp": np.array([])}))

# Task: Put the mug on the coffee maker
mug_synth = performance_data["Put the mug on the coffee maker"]["synthetic"]
mug_synth["demos"] = np.array([50, 100, 150, 1000])
mug_synth["pi0"] = np.array([0.0, 0.0, 0.333, 0.800])
mug_synth["dp"] = np.array([0.133, 0.133, 0.333, 0.533])
mug_real = performance_data["Put the mug on the coffee maker"]["real"]
mug_real["demos"] = np.array([50, 100, 150])
mug_real["pi0"] = np.array([0.066, 0.133, 0.733])
mug_real["dp"] = np.array([0.133, 0.333, 0.400])

# Task: Turn the faucet off
faucet_synth = performance_data["Turn the faucet off"]["synthetic"]
faucet_synth["demos"] = np.array([50, 100, 150, 1000])
faucet_synth["pi0"] = np.array([0.0, 0.133, 0.133, 0.800])
faucet_synth["dp"] = np.array([0.200, 0.333, 0.533, 0.800])
faucet_real = performance_data["Turn the faucet off"]["real"]
faucet_real["demos"] = np.array([50, 100, 150])
faucet_real["pi0"] = np.array([0.353, 0.600, 0.800])
faucet_real["dp"] = np.array([0.200, 0.467, 0.667])

# Task: Open the Drawer
drawer_synth = performance_data["Open the Drawer"]["synthetic"]
drawer_synth["demos"] = np.array([50, 100, 150, 1000])
drawer_synth["pi0"] = np.array([0.0, 0.200, 0.133, 0.866])
drawer_synth["dp"] = np.array([0.133, 0.333, 0.467, 0.666])
drawer_real = performance_data["Open the Drawer"]["real"]
drawer_real["demos"] = np.array([50, 100, 150])
drawer_real["pi0"] = np.array([0.0, 0.400, 0.600])
drawer_real["dp"] = np.array([0.200, 0.600, 0.667])

# Task: Pick up the package with both hands
package_synth = performance_data["Pick up the package with both hands"]["synthetic"]
package_synth["demos"] = np.array([50, 100, 150, 1000])
package_synth["pi0"] = np.array([0.066, 0.133, 0.066, 0.666])
package_synth["dp"] = np.array([0.200, 0.333, 0.200, 0.733])
package_real = performance_data["Pick up the package with both hands"]["real"]
package_real["demos"] = np.array([50, 100, 150])
package_real["pi0"] = np.array([0.067, 0.467, 0.600])
package_real["dp"] = np.array([0.667, 0.667, 0.800])

# Task: Pick up the tiger
tiger_real = performance_data["Pick up the tiger"]["real"]
tiger_real["demos"] = np.array([150])
tiger_real["pi0"] = np.array([0.733])
tiger_real["dp"] = np.array([0.733])

# --- Helper Functions ---
def _convert_demos_to_time(demos: np.ndarray, data_type: str, task_name: str | None = None) -> np.ndarray:
    """Converts demonstration counts to generation time in hours, including setup time.
       Uses task-specific rates for real data if available, otherwise uses default."""
    if data_type == "synthetic":
        rate = SYNTH_TRAJ_PER_HOUR
        setup_time_hours = SYNTH_SETUP_TIME_MINUTES / 60.0
        generation_time = np.maximum(0, demos) / rate
        return setup_time_hours + generation_time
    elif data_type == "real":
        # Use task-specific rate if provided, otherwise default
        rate = TASK_SPECIFIC_REAL_RATES.get(task_name, DEFAULT_REAL_TRAJ_PER_HOUR)
        return np.maximum(0, demos) / rate
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

def _get_data_at_demos(task_data: dict, demos_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts pi0 and dp performance at specific demonstration counts."""
    pi0_vals = np.full(demos_target.shape, np.nan)
    dp_vals = np.full(demos_target.shape, np.nan)
    # Ensure task_data["demos"] exists and is not empty before using np.isin
    if "demos" in task_data and len(task_data["demos"]) > 0:
        valid_indices_target = np.where(np.isin(demos_target, task_data["demos"]))[0]
        valid_indices_data = np.where(np.isin(task_data["demos"], demos_target))[0]
        if len(valid_indices_data) > 0: # Check if any data exists for the target demos
            pi0_vals[valid_indices_target] = task_data["pi0"][valid_indices_data]
            dp_vals[valid_indices_target] = task_data["dp"][valid_indices_data]
    return demos_target, pi0_vals, dp_vals

# ==============================================================================
# Core Plotting Logic for Average Performance (Plots on existing Axes)
# ==============================================================================

def _plot_average_perf_on_ax(ax: plt.Axes, task_name: str, data: dict, plot_vs: str):
    """Helper to plot average performance for a single task onto a given Axes object."""
    task_data_synth = data.get("synthetic")
    task_data_real = data.get("real")

    if not task_data_synth or not task_data_real or len(task_data_synth["demos"]) == 0 or len(task_data_real["demos"]) == 0:
        ax.text(0.5, 0.5, 'No comparable data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(task_name)
        return # Cannot plot average if one dataset is missing or empty

    # --- Data Preparation ---
    demos_synth = task_data_synth["demos"]
    pi0_synth = task_data_synth["pi0"]
    dp_synth = task_data_synth["dp"]
    avg_synth = (pi0_synth + dp_synth) / 2.0

    # For average real plot, only use demos common with synth for a fair line comparison
    common_demos = np.intersect1d(demos_synth, task_data_real["demos"])
    if len(common_demos) == 0:
         ax.text(0.5, 0.5, 'No common demos', ha='center', va='center', transform=ax.transAxes)
         ax.set_title(task_name)
         return

    real_indices = np.where(np.isin(task_data_real["demos"], common_demos))[0]
    demos_real_common = task_data_real["demos"][real_indices]
    pi0_real_common = task_data_real["pi0"][real_indices]
    dp_real_common = task_data_real["dp"][real_indices]
    avg_real = (pi0_real_common + dp_real_common) / 2.0

    # Determine X axis values
    if plot_vs == 'time':
        # Pass task_name for potentially specific real rate
        x_synth = _convert_demos_to_time(demos_synth, "synthetic", task_name)
        x_real = _convert_demos_to_time(demos_real_common, "real", task_name)
        x_real_individual = _convert_demos_to_time(task_data_real["demos"], "real", task_name)
        x_label = "Data Generation Time (Hours)"
        x_scale = 'linear'
    elif plot_vs == 'demos':
        x_synth = demos_synth
        x_real = demos_real_common
        x_real_individual = task_data_real["demos"]
        x_label = "Number of Demonstrations (Log Scale)"
        x_scale = 'log'
    else:
        raise ValueError("plot_vs must be 'time' or 'demos'")

    # --- Plotting ---
    # Plot individual lines WITHOUT labels (legend handled by caller)
    ax.plot(x_real_individual, task_data_real["pi0"], marker='o', linestyle='--', color='lightsalmon', linewidth=2.0, alpha=0.9)
    ax.plot(x_real_individual, task_data_real["dp"], marker='X', linestyle=':', color='lightsalmon', linewidth=2.0, alpha=0.9)
    ax.plot(x_synth, pi0_synth, marker='o', linestyle='--', color='lightblue', linewidth=2.0, alpha=0.9)
    ax.plot(x_synth, dp_synth, marker='X', linestyle=':', color='lightblue', linewidth=2.0, alpha=0.9)

    # Average lines last (with specific labels for figure legend)
    ax.plot(x_real, avg_real, marker='s', linestyle='-', label='Human Teleoperation', color='coral', linewidth=3.0, zorder=10)
    ax.plot(x_synth, avg_synth, marker='s', linestyle='-', label='Real2Render2Real', color='royalblue', linewidth=3.0, zorder=10)

    # --- Add Annotations ---
    annotation_fontsize = 9 # Adjusted for subplots
    text_bbox = dict(facecolor='white', alpha=0.7, pad=0.1, edgecolor='none')
    # Annotate Real average line
    for i, txt in enumerate(demos_real_common):
        label = f'{int(txt)}'
        va_offset = 'bottom' if txt != 100 else 'top'
        y_pos = avg_real[i] * (1.04 if va_offset == 'bottom' else 0.96)
        x_pos_multiplier = 1.1 if plot_vs == 'demos' and txt == 50 else 1.05
        ax.text(x_real[i]*x_pos_multiplier, y_pos, label, fontsize=annotation_fontsize, color='coral', ha='left', va=va_offset, zorder=11, bbox=text_bbox)
    # Annotate Synthetic average line
    for i, txt in enumerate(demos_synth):
        label = f'{int(txt)}'
        va_offset = 'bottom' if txt != 100 else 'top'
        y_pos = avg_synth[i] * (1.04 if va_offset == 'bottom' else 0.96)
        x_pos = x_synth[i]
        ha_val = 'left'
        x_pos_multiplier = 1.0
        if txt == 1000:
            x_pos_multiplier = 0.98 if plot_vs == 'time' else 0.95
            ha_val = 'right'
            va_offset = 'bottom'
            y_pos = avg_synth[i] * 1.04
        elif plot_vs == 'demos' and txt == 50: # Adjustments for log scale closeness
             x_pos_multiplier = 0.95
             ha_val = 'right'
             va_offset = 'bottom'
             y_pos = avg_synth[i] * 1.04
        elif plot_vs == 'demos' and txt == 100:
             x_pos_multiplier = 1.05
             ha_val = 'left'
             va_offset = 'top'
             y_pos = avg_synth[i] * 0.96
        ax.text(x_pos*x_pos_multiplier, y_pos, label, fontsize=annotation_fontsize, color='royalblue', ha=ha_val, va=va_offset, zorder=11, bbox=text_bbox)

    # --- Scales and Limits ---
    ax.set_xscale(x_scale)
    ax.set_yscale('linear')
    # Guard against empty arrays before concatenating
    all_x_real_individual = x_real_individual if len(x_real_individual) > 0 else np.array([])
    all_x_synth = x_synth if len(x_synth) > 0 else np.array([])
    all_x_plot = np.concatenate((all_x_synth, all_x_real_individual))

    if len(all_x_plot) == 0:
         # No data plotted, set default limits
         min_x, max_x = (0, 1) if plot_vs == 'time' else (10, 1100)
         x_buffer_low, x_buffer_high = (0,1) # Avoid calculations on default
    else:
        min_x = 0 if plot_vs == 'time' else min(all_x_plot)
        max_x = max(all_x_plot)
        x_buffer_low = 0.05 * max_x if plot_vs == 'time' else 0.8
        x_buffer_high = 1.05 if plot_vs == 'time' else 1.2

    ax.set_xlim(min_x - x_buffer_low if plot_vs == 'time' else min_x * x_buffer_low,
                max_x * x_buffer_high)
    ax.set_ylim(-0.05, 1.05)

    # --- Spines and Ticks ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', direction='in')
    if plot_vs == 'time':
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='both'))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.2f}'))
    else:
        if len(all_x_plot) > 0:
            tick_locs = sorted(list(set(all_x_plot)))
            # Filter ticks for log scale clarity if too many
            if len(tick_locs) > 6:
                 tick_locs = [loc for loc in tick_locs if loc in [50, 100, 150, 500, 1000]] # Common sensible points
            ax.xaxis.set_major_locator(ticker.FixedLocator(tick_locs))
            ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(int(loc)) for loc in tick_locs]))
            ax.xaxis.set_minor_locator(ticker.NullLocator())
        # else: keep default ticks if no data
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # --- Labels, Title, Grid ---
    ax.set_xlabel(x_label)
    ax.set_ylabel("Success Rate")
    ax.set_title(task_name)
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(True, which='major', axis='x', linestyle=':', linewidth=0.5, alpha=0.5)

# ==============================================================================
# Overall Average Plot Generation (Standalone Function)
# ==============================================================================
def generate_overall_average_plot(tasks_to_average: list[str], data: dict, plot_vs: str, output_dir: pathlib.Path):
    """Generates a plot showing the average performance across multiple tasks, including error bars."""
    fig, ax = plt.subplots(figsize=(7, 5))
    print(f"[{plot_vs.capitalize()} Overall Avg] Averaging tasks: {tasks_to_average}")

    common_demos_real = np.array([50, 100, 150])
    synth_avg_demos = np.array([50, 100, 150, 1000])

    # Store lists of task averages and N for SE calculation
    task_averages_real_allpts = []
    n_real_allpts = []
    task_averages_synth_allpts = []
    n_synth_allpts = []

    # Calculate Real Overall Average & collect task avgs
    for demo_count in common_demos_real:
        task_averages_real = []
        for task_name in tasks_to_average:
            real_data = data.get(task_name, {}).get("real", {})
            _, pi0_r, dp_r = _get_data_at_demos(real_data, np.array([demo_count]))
            if not np.isnan(pi0_r[0]) and not np.isnan(dp_r[0]):
                task_averages_real.append((pi0_r[0] + dp_r[0]) / 2.0)
        task_averages_real_allpts.append(task_averages_real)
        n_real_allpts.append(len(task_averages_real)) # N = num_tasks_with_data
        # n_real_allpts.append(15 * len(task_averages_real)) # N = 15 * num_tasks_with_data

    # Calculate Synth Overall Average & collect task avgs
    for demo_count in synth_avg_demos:
        task_averages_synth = []
        for task_name in tasks_to_average:
            synth_data = data.get(task_name, {}).get("synthetic", {})
            _, pi0_s, dp_s = _get_data_at_demos(synth_data, np.array([demo_count]))
            if not np.isnan(pi0_s[0]) and not np.isnan(dp_s[0]):
                task_averages_synth.append((pi0_s[0] + dp_s[0]) / 2.0)
        task_averages_synth_allpts.append(task_averages_synth)
        # n_synth_allpts.append(15 * len(task_averages_synth)) # N = 15 * num_tasks_with_data
        n_synth_allpts.append(len(task_averages_synth)) # N = num_tasks_with_data

    # Calculate overall mean and standard error
    overall_avg_real = np.array([np.mean(avgs) if avgs else np.nan for avgs in task_averages_real_allpts])
    std_dev_real = np.array([np.std(avgs) if avgs else np.nan for avgs in task_averages_real_allpts])
    n_real_allpts = np.array(n_real_allpts)
    se_real = np.divide(std_dev_real, np.sqrt(n_real_allpts), out=np.full_like(std_dev_real, np.nan), where=n_real_allpts>0)

    overall_avg_synth = np.array([np.mean(avgs) if avgs else np.nan for avgs in task_averages_synth_allpts])
    std_dev_synth = np.array([np.std(avgs) if avgs else np.nan for avgs in task_averages_synth_allpts])
    n_synth_allpts = np.array(n_synth_allpts)
    se_synth = np.divide(std_dev_synth, np.sqrt(n_synth_allpts), out=np.full_like(std_dev_synth, np.nan), where=n_synth_allpts>0)

    # Filter out NaN results independently
    valid_indices_synth = ~np.isnan(overall_avg_synth) & ~np.isnan(se_synth)
    plot_demos_synth = synth_avg_demos[valid_indices_synth]
    plot_avg_synth = overall_avg_synth[valid_indices_synth]
    plot_se_synth = se_synth[valid_indices_synth]

    valid_indices_real = ~np.isnan(overall_avg_real) & ~np.isnan(se_real)
    plot_demos_real = common_demos_real[valid_indices_real]
    plot_avg_real = overall_avg_real[valid_indices_real]
    plot_se_real = se_real[valid_indices_real]

    # Print calculated SE values
    print(f"--- [{plot_vs.capitalize()} Overall Avg] Calculated Standard Errors ---")
    print(f"Real Data Demos: {plot_demos_real}")
    print(f"Real Data SE:    {np.round(plot_se_real, 3)}")
    print(f"Synth Data Demos: {plot_demos_synth}")
    print(f"Synth Data SE:   {np.round(plot_se_synth, 3)}")
    print("----------------------------------------------------")

    if len(plot_demos_synth) == 0 and len(plot_demos_real) == 0:
        print(f"[{plot_vs.capitalize()} Overall Avg] No valid points found for overall average plot.")
        return

    # Determine X axis
    if plot_vs == 'time':
        x_synth = _convert_demos_to_time(plot_demos_synth, "synthetic")
        x_real = _convert_demos_to_time(plot_demos_real, "real")
        x_label = "Data Generation Time (Hours)"
        x_scale = 'linear'
    else:
        x_synth = plot_demos_synth
        x_real = plot_demos_real
        x_label = "Number of Demonstrations (Log Scale)"
        x_scale = 'log'

    # --- Plotting ---
    # Plot main average lines
    ax.plot(x_real, plot_avg_real, marker='s', linestyle='-', label='Human Teleoperation', color='coral', linewidth=3.0, zorder=10)
    ax.plot(x_synth, plot_avg_synth, marker='s', linestyle='-', label='Real2Render2Real', color='royalblue', linewidth=3.0, zorder=10)

    # Calculate bounds for shaded region
    lower_bound_real = plot_avg_real - plot_se_real
    upper_bound_real = plot_avg_real + plot_se_real
    lower_bound_synth = plot_avg_synth - plot_se_synth
    upper_bound_synth = plot_avg_synth + plot_se_synth

    # Add shaded regions for standard error
    ax.fill_between(x_real, lower_bound_real, upper_bound_real, color='coral', alpha=0.2, zorder=5)
    ax.fill_between(x_synth, lower_bound_synth, upper_bound_synth, color='royalblue', alpha=0.2, zorder=5)

    # --- Add Annotations ---
    annotation_fontsize = 10
    text_bbox = dict(facecolor='white', alpha=0.7, pad=0.1, edgecolor='none')
    # Annotate Real line
    for i, txt in enumerate(plot_demos_real):
        label = f'{int(txt)}'
        va_offset_r = 'bottom' if txt != 100 else 'top'
        # Add a small constant vertical offset from the mean, independent of error bar
        y_pos_r = plot_avg_real[i] + (0.02 if va_offset_r == 'bottom' else -0.02)
        x_pos_r = x_real[i]
        x_pos_multiplier_r = 1.1 if plot_vs == 'demos' and txt == 50 else 1.05
        ax.text(x_pos_r * x_pos_multiplier_r, y_pos_r, label, fontsize=annotation_fontsize, color='coral', ha='left', va=va_offset_r, zorder=11, bbox=text_bbox)
    # Annotate Synthetic line
    for i, txt in enumerate(plot_demos_synth):
        label = f'{int(txt)}'
        va_offset_s = 'bottom' if txt != 100 else 'top'
        # Add a small constant vertical offset from the mean, independent of error bar
        y_pos_s = plot_avg_synth[i] + (0.02 if va_offset_s == 'bottom' else -0.02)
        x_pos_s = x_synth[i]
        x_pos_multiplier_s = 1.0
        ha_val_s = 'left'
        if txt == 1000:
            x_pos_multiplier_s = 0.98 if plot_vs == 'time' else 0.95
            ha_val_s = 'right'
            va_offset_s = 'bottom'
            y_pos_s = plot_avg_synth[i] + 0.02
        elif plot_vs == 'demos' and txt == 50:
            x_pos_multiplier_s = 0.95
            ha_val_s = 'right'
            va_offset_s = 'bottom'
            y_pos_s = plot_avg_synth[i] + 0.02
        elif plot_vs == 'demos' and txt == 100:
             x_pos_multiplier_s = 1.05
             ha_val_s = 'left'
             va_offset_s = 'top'
             y_pos_s = plot_avg_synth[i] - 0.02

        ax.text(x_pos_s * x_pos_multiplier_s, y_pos_s, label, fontsize=annotation_fontsize, color='royalblue', ha=ha_val_s, va=va_offset_s, zorder=11, bbox=text_bbox)

    # --- Scales and Limits ---
    ax.set_xscale(x_scale)
    ax.set_yscale('linear')
    all_x_plot = np.concatenate((x_synth if len(x_synth)>0 else [], x_real if len(x_real)>0 else []))
    if len(all_x_plot) == 0:
        min_x, max_x = (0, 1) if plot_vs == 'time' else (10, 1100)
        x_buffer_low, x_buffer_high = (0,1)
    else:
        min_x = 0 if plot_vs == 'time' else min(all_x_plot)
        max_x = max(all_x_plot)
        x_buffer_low = 0.05 * max_x if plot_vs == 'time' else 0.8
        x_buffer_high = 1.05 if plot_vs == 'time' else 1.2
    ax.set_xlim(min_x - x_buffer_low if plot_vs == 'time' else min_x * x_buffer_low,
                max_x * x_buffer_high)
    ax.set_ylim(-0.05, 1.05)

    # --- Spines and Ticks ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', direction='in')
    if plot_vs == 'time':
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='both'))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.2f}'))
    else:
        tick_locs = sorted(list(set(np.concatenate((plot_demos_synth, plot_demos_real)))))
        ax.xaxis.set_major_locator(ticker.FixedLocator(tick_locs))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(int(loc)) for loc in tick_locs]))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # --- Labels, Title, Grid, Legend ---
    ax.set_xlabel(x_label)
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Overall Average Performance Across Tasks vs. {plot_vs.capitalize()}")
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(True, which='major', axis='x', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.legend(loc='lower right')

    plt.tight_layout(pad=1.0) 
    output_path = output_dir / f"overall_avg_perf_vs_{plot_vs}.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"[{plot_vs.capitalize()} Overall Avg] Plot saved to {output_path}")
    plt.close(fig)

# ==============================================================================
# Subplot Figure Generation
# ==============================================================================
def generate_task_subplot_figure(subplot_tasks: list[str], data: dict, plot_vs: str, output_dir: pathlib.Path):
    """Generates a 2x2 figure with subplots for the specified tasks."""
    num_tasks = len(subplot_tasks)
    if num_tasks != 4:
        print(f"Warning: Expected 4 subplot tasks for 2x2 grid, but got {num_tasks}. Adjusting grid...")
        ncols = min(num_tasks, 2)
    else:
        ncols = 2
    nrows = (num_tasks + ncols - 1) // ncols

    fig_width = ncols * 5.5
    fig_height = nrows * 4.5
    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    axs_flat = axs.flatten()

    print(f"[{plot_vs.capitalize()} Subplots] Generating 2x2 subplots for tasks: {subplot_tasks}")

    # Plot dummy lines on the first axis for the legend
    axs_flat[0].plot([], [], marker='o', linestyle='--', label='PI0-FAST', color='gray', linewidth=2.0)
    axs_flat[0].plot([], [], marker='X', linestyle=':', label='Diffusion Policy', color='gray', linewidth=2.0)
    axs_flat[0].plot([], [], marker='s', linestyle='-', label='Human Teleoperation', color='coral', linewidth=3.0)
    axs_flat[0].plot([], [], marker='s', linestyle='-', label='Real2Render2Real', color='royalblue', linewidth=3.0)

    # Plot per-task subplots
    for i, task_name in enumerate(subplot_tasks):
        ax = axs_flat[i]
        if task_name in data:
            _plot_average_perf_on_ax(ax, task_name, data[task_name], plot_vs)
        else:
            ax.text(0.5, 0.5, 'Data missing', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(task_name)

    # Create figure legend
    handles, labels = axs_flat[0].get_legend_handles_labels()
    order = ['Human Teleoperation', 'Real2Render2Real', 'PI0-FAST', 'Diffusion Policy']
    ordered_handles = []
    ordered_labels = []
    label_to_handle = dict(zip(labels, handles))
    for label in order:
        if label in label_to_handle:
            ordered_handles.append(label_to_handle[label])
            ordered_labels.append(label)

    # Move legend closer to plots (adjust y in bbox_to_anchor)
    fig.legend(ordered_handles, ordered_labels, loc='lower center', bbox_to_anchor=(0.5, 0.03), ncol=4, fontsize=13)

    # Adjust layout for legend - increase bottom slightly
    plt.tight_layout(rect=[0, 0.08, 1, 1.0]) # Adjusted rect top and bottom

    output_path = output_dir / f"subplot_perf_vs_{plot_vs}.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"[{plot_vs.capitalize()} Subplots] Figure saved to {output_path}")
    plt.close(fig)

# ==============================================================================
# Main Execution
# ==============================================================================
def main(base_output_dir: pathlib.Path = pathlib.Path("./outputs")) -> None:
    """Generates performance scaling plots vs. time and vs. demos."""
    plots_vs_time_dir = base_output_dir / "plots_vs_time"
    plots_vs_demos_dir = base_output_dir / "plots_vs_demos"
    plots_vs_time_dir.mkdir(parents=True, exist_ok=True)
    plots_vs_demos_dir.mkdir(parents=True, exist_ok=True)

    # Define tasks suitable for 2x2 subplot figure
    subplot_tasks = [
        "Put the mug on the coffee maker",
        "Turn the faucet off",
        "Open the Drawer",
        "Pick up the package with both hands"
    ]
    if len(subplot_tasks) != 4:
        print("Error: Script currently assumes exactly 4 tasks for the 2x2 subplot layout.")
        return

    # Define tasks for OVERALL average (must have 50, 100, 150 demo points for BOTH real/synth)
    overall_avg_tasks = [
        "Put the mug on the coffee maker",
        "Turn the faucet off",
        "Open the Drawer"
    ]

    # --- Generate 2x2 Subplot Figures --- 
    print("\n--- Generating 2x2 Subplot Figures ---")
    generate_task_subplot_figure(subplot_tasks, performance_data, 'time', plots_vs_time_dir)
    generate_task_subplot_figure(subplot_tasks, performance_data, 'demos', plots_vs_demos_dir)

    # --- Generate Separate Overall Average Figures --- 
    print("\n--- Generating Overall Average Figures ---")
    generate_overall_average_plot(overall_avg_tasks, performance_data, 'time', plots_vs_time_dir)
    generate_overall_average_plot(overall_avg_tasks, performance_data, 'demos', plots_vs_demos_dir)

    print("\nPlot generation complete.")

if __name__ == "__main__":
    tyro.cli(main) 