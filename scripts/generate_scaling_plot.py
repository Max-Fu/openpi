"""
Generates a dual plot:
1. Left: Log-log scaling comparison between different numbers of human teleoperators and R2R2R for data generation
2. Right: Average performance comparison between policies trained on real vs. synthetic data

uv run scripts/generate_scaling_plot.py --output-dir output_plots
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tyro
import pathlib
import math
import collections

# --- Data Generation Rates ---
# Calculate average time for the tasks with specific times
avg_real_time_min = (86.0 + 71.0 + 104.0 + 90.0) / 4.0 # Avg time for 150 demos (including bimanual box)
DEFAULT_REAL_TRAJ_PER_HOUR = (150.0 * 60.0) / avg_real_time_min # ~103.45 traj/hr
TIME_TELEOP_PER_DEMO_S = 3600 / DEFAULT_REAL_TRAJ_PER_HOUR # Convert traj/hr to seconds per demo
SYNTH_TRAJ_PER_HOUR = (1000.0 * 60.0) / 35.0 # ~1714.29 traj/hr (Using 1000 as reference count)
TIME_R2R2R_PER_DEMO_S = 3600 / SYNTH_TRAJ_PER_HOUR # Convert traj/hr to seconds per demo
SYNTH_SETUP_TIME_MINUTES = 5.0
TASK_SPECIFIC_REAL_RATES = {
    "Put the mug on the coffee maker": (150.0 * 60.0) / 86.0,  # 1hr 26min
    "Open the Drawer": (150.0 * 60.0) / 71.0,                 # 1hr 11min
    "Turn the faucet off": (150.0 * 60.0) / 104.0,             # 1hr 44min
    "Pick up the package with both hands": (150.0 * 60.0) / 90.0,  # 1hr 30min
}

# --- Plot Styling ---
plt.style.use('seaborn-v0_8-paper')
font_offsize = 3
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13 + font_offsize,       # Increased base font size
    "axes.titlesize": 14 + font_offsize,  # Increased title font size
    "axes.labelsize": 13 + font_offsize,  # Increased axis label font size
    "xtick.labelsize": 12 + font_offsize, # Increased X tick label font size
    "ytick.labelsize": 12 + font_offsize, # Increased Y tick label font size
    "legend.fontsize": 11 + font_offsize, # Increased legend font size
    # figsize is set in subplots call below
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

def _plot_performance_vs_time(ax: plt.Axes, tasks_to_average: list[str], data: dict, show_individual_tasks: bool = False):
    """Plots the overall average performance across multiple tasks vs. time, including error bars.
    Also plots individual task performances as faint background lines (GPT-3 paper style) if show_individual_tasks is True."""
    common_demos_real = np.array([50, 100, 150])
    synth_avg_demos = np.array([50, 100, 150, 1000])
    
    # First, plot individual task performances as faint lines if flag is set
    if show_individual_tasks:
        # Use lighter versions of the main colors
        color_synth_light = 'lightblue'
        color_real_light = 'lightsalmon'
        
        print(f"Plotting individual task performances for: {tasks_to_average}")
        
        # Plot each task's performance
        for task_name in tasks_to_average:
            # Get real data for this task
            real_data = data.get(task_name, {}).get("real", {})
            if "demos" in real_data and len(real_data["demos"]) > 0:
                real_demos = real_data["demos"]
                real_pi0 = real_data["pi0"]
                real_dp = real_data["dp"]
                real_avg = (real_pi0 + real_dp) / 2.0
                x_real = _convert_demos_to_time(real_demos, "real", task_name)
                # Plot real data with thin, faint line
                ax.plot(x_real, real_avg, marker='o', markersize=4, alpha=0.3, 
                        linestyle='-', linewidth=1.2, color=color_real_light, zorder=1)
            
            # Get synthetic data for this task
            synth_data = data.get(task_name, {}).get("synthetic", {})
            if "demos" in synth_data and len(synth_data["demos"]) > 0:
                synth_demos = synth_data["demos"]
                synth_pi0 = synth_data["pi0"]
                synth_dp = synth_data["dp"]
                synth_avg = (synth_pi0 + synth_dp) / 2.0
                x_synth = _convert_demos_to_time(synth_demos, "synthetic", task_name)
                # Plot synthetic data with thin, faint line
                ax.plot(x_synth, synth_avg, marker='o', markersize=4, alpha=0.3, 
                        linestyle='-', linewidth=1.2, color=color_synth_light, zorder=1)
    
    # Now compute and plot the average performance (highlighted foreground)
    
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

    # Calculate Synth Overall Average & collect task avgs
    for demo_count in synth_avg_demos:
        task_averages_synth = []
        for task_name in tasks_to_average:
            synth_data = data.get(task_name, {}).get("synthetic", {})
            _, pi0_s, dp_s = _get_data_at_demos(synth_data, np.array([demo_count]))
            if not np.isnan(pi0_s[0]) and not np.isnan(dp_s[0]):
                task_averages_synth.append((pi0_s[0] + dp_s[0]) / 2.0)
        task_averages_synth_allpts.append(task_averages_synth)
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
    print(f"--- Performance vs Time - Calculated Standard Errors ---")
    print(f"Real Data Demos: {plot_demos_real}")
    print(f"Real Data SE:    {np.round(plot_se_real, 3)}")
    print(f"Synth Data Demos: {plot_demos_synth}")
    print(f"Synth Data SE:   {np.round(plot_se_synth, 3)}")
    print("----------------------------------------------------")

    if len(plot_demos_synth) == 0 and len(plot_demos_real) == 0:
        print(f"No valid points found for overall average plot.")
        return

    # Determine X axis values (time for both)
    x_synth = _convert_demos_to_time(plot_demos_synth, "synthetic")
    x_real = _convert_demos_to_time(plot_demos_real, "real")

    # --- Plotting the average lines (highlighted foreground) ---
    # Adjust emphasis based on whether individual tasks are shown
    line_width = 4.0 if show_individual_tasks else 3.0
    marker_size = 8 if show_individual_tasks else 6
    
    # Labels change based on whether individual tasks are shown
    real_label = 'Human Teleoperation (avg)' if show_individual_tasks else 'Human Teleoperation'
    synth_label = 'Real2Render2Real (avg)' if show_individual_tasks else 'Real2Render2Real'
    
    # Plot main average lines with appropriate emphasis
    ax.plot(x_real, plot_avg_real, marker='s', markersize=marker_size, linestyle='-', 
            label=real_label, color='coral', linewidth=line_width, zorder=20)
    ax.plot(x_synth, plot_avg_synth, marker='s', markersize=marker_size, linestyle='-', 
            label=synth_label, color='royalblue', linewidth=line_width, zorder=20)

    # Calculate bounds for shaded region
    lower_bound_real = plot_avg_real - plot_se_real
    upper_bound_real = plot_avg_real + plot_se_real
    lower_bound_synth = plot_avg_synth - plot_se_synth
    upper_bound_synth = plot_avg_synth + plot_se_synth

    # Add shaded regions for standard error (slightly more transparent if showing individual tasks)
    alpha_value = 0.15 if show_individual_tasks else 0.2
    ax.fill_between(x_real, lower_bound_real, upper_bound_real, color='coral', alpha=alpha_value, zorder=15)
    ax.fill_between(x_synth, lower_bound_synth, upper_bound_synth, color='royalblue', alpha=alpha_value, zorder=15)

    # --- Add Annotations ---
    annotation_fontsize = 10
    text_bbox = dict(facecolor='white', alpha=0.7, pad=0.1, edgecolor='none')
    
    # Annotate Real line
    for i, txt in enumerate(plot_demos_real):
        label = f'{int(txt)}'
        va_offset_r = 'bottom' if txt != 100 else 'top'
        y_pos_r = plot_avg_real[i] + (0.02 if va_offset_r == 'bottom' else -0.02)
        x_pos_r = x_real[i]
        x_pos_multiplier_r = 1.05
        ha_position = 'left'
        
        # Adjust position for linear scale - ensure clear readability
        if i > 0 and x_real[i] - x_real[i-1] < 0.2:  # If points are close together
            if i % 2 == 0:  # Alternate positioning
                ha_position = 'right'
                x_pos_multiplier_r = 0.95
            
        ax.text(x_pos_r * x_pos_multiplier_r, y_pos_r, label, fontsize=annotation_fontsize, 
                color='coral', ha=ha_position, va=va_offset_r, zorder=25, bbox=text_bbox)
    
    # Annotate Synthetic line
    for i, txt in enumerate(plot_demos_synth):
        label = f'{int(txt)}'
        va_offset_s = 'bottom' if txt != 100 else 'top'
        y_pos_s = plot_avg_synth[i] + (0.02 if va_offset_s == 'bottom' else -0.02)
        x_pos_s = x_synth[i]
        x_pos_multiplier_s = 1.0
        ha_val_s = 'left'
        
        # Special handling for 1000 demos point
        if txt == 1000:
            x_pos_multiplier_s = 0.98
            ha_val_s = 'right'
            va_offset_s = 'bottom'
            y_pos_s = plot_avg_synth[i] + 0.02
            
        # Adjust position for linear scale - ensure clear readability
        if i > 0 and x_synth[i] - x_synth[i-1] < 0.2:  # If points are close together
            if i % 2 == 0:  # Alternate positioning
                ha_val_s = 'right' if ha_val_s == 'left' else 'left'
                x_pos_multiplier_s = 0.95 if ha_val_s == 'right' else 1.05

        ax.text(x_pos_s * x_pos_multiplier_s, y_pos_s, label, fontsize=annotation_fontsize, 
                color='royalblue', ha=ha_val_s, va=va_offset_s, zorder=25, bbox=text_bbox)

    # --- Axis Setup ---
    ax.set_xscale('linear')  # Linear scale for time
    ax.set_yscale('linear')
    ax.set_xlabel("Data Generation Time (Hours)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Average Performance vs. Data Generation Time")
    
    # Set y-axis to percentage format
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    
    # Set grid
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(True, which='major', axis='x', linestyle=':', linewidth=0.5, alpha=0.5)
    
    # Set limits
    ax.set_ylim(-0.05, 1.05)
    all_x_vals = np.concatenate((x_synth, x_real))
    min_x = 0  # Start from 0 for linear scale
    max_x = max(all_x_vals) * 1.1
    ax.set_xlim(min_x, max_x)
    
    # Format x-axis with appropriate hour markers
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.1f}'))
    
    # Create legend based on whether individual tasks are shown
    if show_individual_tasks:
        # Add legend with the individual task indication
        from matplotlib.lines import Line2D
        
        # Create custom legend entries
        legend_elements = [
            Line2D([0], [0], color='coral', linewidth=line_width, label=real_label),
            Line2D([0], [0], color='royalblue', linewidth=line_width, label=synth_label),
            Line2D([0], [0], color='lightsalmon', linewidth=1.2, alpha=0.3, label='Individual Tasks'),
        ]
        
        ax.legend(handles=legend_elements, loc='lower right')
    else:
        # Simple legend with just the average lines
        ax.legend(loc='lower right')

def main(output_dir: pathlib.Path = pathlib.Path("./outputs"), show_individual_tasks: bool = False) -> None:
    """Generates a dual plot: scaling comparison and performance comparison.
    
    Args:
        output_dir: Directory to save the generated plot.
        show_individual_tasks: Whether to show individual task performances as faint lines in the
                              performance plot (GPT-3 paper style).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scaling_and_performance.pdf"

    # --- Constants ---
    UPFRONT_TIME_R2R2R_S = SYNTH_SETUP_TIME_MINUTES * 60.0
    MAX_TIME_S = 12 * 3600  # 12 hours

    # --- Calculations ---
    # Speed (demos per second)
    speed_teleop = 1.0 / TIME_TELEOP_PER_DEMO_S
    speed_r2r2r = 1.0 / TIME_R2R2R_PER_DEMO_S

    # Calculate potential speedup with parallel GPUs
    TARGET_SPEEDUP_X = 100_000
    gpus_needed = math.ceil(TARGET_SPEEDUP_X * TIME_R2R2R_PER_DEMO_S / TIME_TELEOP_PER_DEMO_S)
    speed_r2r2r_parallel = gpus_needed * speed_r2r2r

    print(f"Teleoperation speed: {speed_teleop:.4f} demos/sec ({DEFAULT_REAL_TRAJ_PER_HOUR:.2f} demos/hour)")
    print(f"Real2Render2Real speed: {speed_r2r2r:.4f} demos/sec ({SYNTH_TRAJ_PER_HOUR:.2f} demos/hour)")
    print(f"Real2Render2Real upfront time: {UPFRONT_TIME_R2R2R_S:.1f} s ({SYNTH_SETUP_TIME_MINUTES:.1f} min)")
    print(f"Speedup (R2R2R vs Teleop): {speed_r2r2r / speed_teleop:.2f}x")
    print(f"GPUs needed for {TARGET_SPEEDUP_X:,}x speedup: {gpus_needed:,}")

    # --- Plotting Data ---
    # Number of operators for teleop comparison
    operators_teleop = [1, 10, 100]

    # Define a function to calculate demos considering upfront time
    def calculate_demos(time_array, speed, upfront_time):
        effective_time = np.maximum(0, time_array - upfront_time)
        return speed * effective_time

    time_s = np.logspace(np.log10(10), np.log10(MAX_TIME_S), 130) # Time from 10s to 12 hours

    # Calculate demos for different numbers of teleop operators
    demos_teleop_dict = {}
    for n_ops in operators_teleop:
        demos_teleop_dict[n_ops] = calculate_demos(time_s, n_ops * speed_teleop, 0)

    # Real2Render2Real methods have upfront time
    demos_r2r2r_1gpu = calculate_demos(time_s, speed_r2r2r, UPFRONT_TIME_R2R2R_S)
    demos_r2r2r_parallel = calculate_demos(time_s, speed_r2r2r_parallel, UPFRONT_TIME_R2R2R_S)

    # Create a modified time vector for ax0 R2R2R plotting to show start near y=1
    time_s_r2r2r_ax0 = np.sort(np.concatenate([
        time_s,
        np.array([UPFRONT_TIME_R2R2R_S + 0.1]) # Add point just after start
    ]))
    demos_r2r2r_1gpu_ax0 = calculate_demos(time_s_r2r2r_ax0, speed_r2r2r, UPFRONT_TIME_R2R2R_S)

    # Add lines for intermediate parallelism levels
    gpus_intermediate = [100] # Reduced for clarity
    demos_r2r2r_intermediate = {}
    for n_gpu in gpus_intermediate:
        speed_intermediate = n_gpu * speed_r2r2r
        demos_r2r2r_intermediate[n_gpu] = calculate_demos(time_s, speed_intermediate, UPFRONT_TIME_R2R2R_S)

    # --- Define Base Colors --- 
    color_r2r2r = 'royalblue' # Slightly lighter blue
    color_teleop = 'firebrick' # Darker red/orange

    # --- Create Figure with Two Subplots ---
    # Use specific figsize, ignore rcParams for this adjustment
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False) 
    ax0, ax1 = axes # Left, Right axes

    # === Plot 1 (Left): Log-Log Scale ===
    ax = ax0 # Use ax0 for the first plot
    
    # Plot teleop lines
    teleop_styles = ['--', ':', '-.']
    for i, n_ops in enumerate(operators_teleop):
        # Ensure data is positive for log plot
        mask = demos_teleop_dict[n_ops] > 0
        if np.any(mask):
            ax.loglog(time_s[mask], demos_teleop_dict[n_ops][mask], 
                      label=f'Teleop ({n_ops} Op.)', 
                      color=color_teleop, # Use base color 
                      linestyle=teleop_styles[i % len(teleop_styles)], 
                      linewidth=3.0) # Increased linewidth

    # Plot only R2R2R (1 GPU) on the left plot
    mask_1gpu_ax0 = demos_r2r2r_1gpu_ax0 > 0
    if np.any(mask_1gpu_ax0):
        ax.loglog(time_s_r2r2r_ax0[mask_1gpu_ax0], demos_r2r2r_1gpu_ax0[mask_1gpu_ax0], 
                  label='R2R2R (1 GPU)', 
                  color=color_r2r2r, # Use base R2R2R color
                  linestyle='-', 
                  linewidth=3.0) # Increased linewidth

    # --- Annotations and Labels for Plot 1 ---
    # Add time markers for 10 min, 1 hour, and 12 hours
    ax.axvline(600, color='grey', linestyle=':', linewidth=1, alpha=0.7) # 10 min = 600s
    ax.axvline(3600, color='grey', linestyle=':', linewidth=1, alpha=0.7) # 1 hour = 3600s
    ax.axvline(MAX_TIME_S, color='grey', linestyle=':', linewidth=1, alpha=0.7) # 12 hours

    # Get current y-limits *before* adding text to avoid potential expansion
    current_ymin, current_ymax = ax.get_ylim() 

    # Position time markers near the bottom
    ymin_ax1, ymax_ax1 = ax.get_ylim()
    text_y_pos = ymin_ax1 + (ymax_ax1 - ymin_ax1) * 0.001 # Position near bottom

    ax.text(600, text_y_pos, '10 Min', ha='right', va='bottom', color='grey', alpha=0.9, rotation=90)
    ax.text(3600, text_y_pos, '1 Hour', ha='right', va='bottom', color='grey', alpha=0.9, rotation=90)
    ax.text(MAX_TIME_S, text_y_pos, '12 Hours', ha='right', va='bottom', color='grey', alpha=0.9, rotation=90)

    # --- Parameter Summary Text Box for Plot 1 (Restored) ---
    param_text = (f"Parameters:\n"
                  f"Teleop: {TIME_TELEOP_PER_DEMO_S:.1f} s/demo\n"
                  f"R2R2R: {TIME_R2R2R_PER_DEMO_S:.1f} s/demo (1 GPU)\n"
                  f"R2R2R Upfront: {UPFRONT_TIME_R2R2R_S/60:.0f} min")
    # Place text box in bottom right area of ax0
    ax.text(0.95, 0.05, param_text, transform=ax.transAxes, fontsize=10, # Increased fontsize 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.7))

    # --- Axis Setup for Plot 1 ---
    ax.set_xscale('log') # Set log scale for x-axis
    ax.set_yscale('log') # Set log scale for y-axis
    ax.set_xlabel("Wall-clock Time (Log-Scale Seconds)")
    ax.set_ylabel("Log-Scale Number of Demonstrations Generated")
    ax.set_title("Log-Log Scaling Comparison")
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.6) # Slightly increased alpha
    ax.legend(loc='upper left') # Consistent legend placement
    ax.set_xlim(90, MAX_TIME_S * 1.1) # Start from 90s, end slightly past 12 hours
    
    # Recalculate max_demos including all teleop lines
    all_max_demos = [d.max() for d in demos_teleop_dict.values()] + [demos_r2r2r_1gpu_ax0.max()]
    max_demos_ax0 = max(all_max_demos) if all_max_demos else 10

    # Set fixed lower y-limit to 1 for log-log plot to ensure lines start from bottom edge
    ax.set_ylim(1, max_demos_ax0 * 5) 

    # === Plot 2 (Right): Performance vs. Time ===
    # Define tasks for overall average 
    overall_avg_tasks = [
        "Put the mug on the coffee maker",
        "Turn the faucet off",
        "Open the Drawer"
    ]
    
    _plot_performance_vs_time(ax1, overall_avg_tasks, performance_data, show_individual_tasks)

    # --- Final Layout ---
    plt.tight_layout(pad=1.0, w_pad=2.0)

    # --- Save Plot ---
    file_suffix = "_with_individual_tasks" if show_individual_tasks else ""
    output_path = output_dir / f"scaling_and_performance{file_suffix}.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    tyro.cli(main)