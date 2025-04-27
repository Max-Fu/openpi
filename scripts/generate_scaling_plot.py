"""
I want to generate a plot that is

Vision and Language Models have achieved impressive performance by training on more than 1T tokens, tens of thousands of times data more than robotics. The prevailing paradigm—human teleoperation—is fundamentally constrained to mitigate this data scale gap: robots and human operators are expensive, demonstrations are slow, and each new robot embodiment requires additional data collection. 
% Consequently, building robot learning datasets at the scale of modern image and language corpora remains infeasible. 
In this work, we introduce a new data collection pipeline, Real2Render2Real (\algabbr), that doesn't require physics simulation or robot hardwares. With a smartphone, anyone can scan a scene and record one human demonstration of the task from anywhere. Real2Render2Real then reconstructs object appearance and 3D geometry from the scan, extracts object 6-DoF motion trajectory from human demonstration video, and synthesizes thousands of diverse, robot-agnostic trajectories by randomizing object positions and surrounding environments. For each generated trajectory, we compute feasible robot joint configurations via inverse kinematics to render realistic robot training data. \algabbr integrates seamlessly with existing vision-language-action (VLA) and diffusion-based imitation policies, producing training data on novel objects and environments, that enables policies to match the performance of those trained on human teleoperation—while requiring only a fraction of the time and cost. Crucially, experiments suggest policies trained using \algabbr generalize directly to real-world execution without any in-domain teleoperation data, offering a practical path toward large-scale robot learning.
Scaling curve on time
Compute (x axis: time) vs Unique trajectories (y: axis), where there are a few lines, both x and y axis should be log scale. 
I want to create a certain setup where our method can generate at 100,000x real time speed. You need to calculate how many gpus we need for that 
1. teleop: to collect 1 demonstration, it takes roughly 20 seconds. The line is linear and in the x axis I want to stop at 1 hour mark
2. our method (single 4090 gpu). To generate 1000 feasible demonstrations it takes roughly 1 hour. (3.6 sec/demonstration)
3. create a projected line where we have gpu parallelism. I want to measure where I can generate at 100,000x realtime. 

Think about how to best present the plot and the easiest to convey the information. You can add more lines and make more labels next to the line. Pick good colors and gradients to show the point. I am submitting the paper to CoRL so make it good! 
"""

import matplotlib.pyplot as plt
import numpy as np
import tyro
import pathlib
import math
import matplotlib.ticker as ticker

# Set plot style and parameters directly
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

def main(output_dir: pathlib.Path = pathlib.Path("./outputs")) -> None:
    """Generates a scaling plot comparing teleoperation and Real2Render2Real data generation.

    Args:
        output_dir: Directory to save the generated plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scaling_plot_time.pdf"

    # --- Constants ---
    TIME_TELEOP_PER_DEMO_S = 37.25*2*60/107
    # TIME_TELEOP_PER_DEMO_S = 30*60/8
    TIME_Real2Render2Real_PER_DEMO_S_1GPU = 35*60/1019  # 1 hour / 2000 demos = 3600s / 2000 = 3.6s
    UPFRONT_TIME_Real2Render2Real_S = 300.0 # 5 minutes for scanning + NeRF/GS training
    MAX_TIME_S = 12 * 3600  # 12 hours
    TARGET_SPEEDUP_X = 100_000

    # --- Calculations ---
    # Speed (demos per second)
    speed_teleop = 1.0 / TIME_TELEOP_PER_DEMO_S
    speed_Real2Render2Real_1gpu = 1.0 / TIME_Real2Render2Real_PER_DEMO_S_1GPU

    # Calculate required GPUs for target speedup
    # N * speed_Real2Render2Real_1gpu = TARGET_SPEEDUP_X * speed_teleop
    # N = TARGET_SPEEDUP_X * speed_teleop / speed_Real2Render2Real_1gpu
    # N = TARGET_SPEEDUP_X * (1 / TIME_TELEOP_PER_DEMO_S) / (1 / TIME_Real2Render2Real_PER_DEMO_S_1GPU)
    # N = TARGET_SPEEDUP_X * TIME_Real2Render2Real_PER_DEMO_S_1GPU / TIME_TELEOP_PER_DEMO_S
    gpus_needed = math.ceil(TARGET_SPEEDUP_X * TIME_Real2Render2Real_PER_DEMO_S_1GPU / TIME_TELEOP_PER_DEMO_S)
    speed_Real2Render2Real_parallel = gpus_needed * speed_Real2Render2Real_1gpu

    print(f"Teleoperation speed: {speed_teleop:.2f} demos/sec")
    print(f"Real2Render2Real (1 GPU) speed: {speed_Real2Render2Real_1gpu:.2f} demos/sec")
    print(f"Real2Render2Real upfront time: {UPFRONT_TIME_Real2Render2Real_S:.1f} s ({UPFRONT_TIME_Real2Render2Real_S/60:.1f} min)")
    print(f"Speedup (1 GPU vs Teleop): {speed_Real2Render2Real_1gpu / speed_teleop:.2f}x")
    print(f"GPUs needed for {TARGET_SPEEDUP_X:,}x speedup: {gpus_needed:,}")
    print(f"Projected Real2Render2Real ({gpus_needed:,} GPUs) speed: {speed_Real2Render2Real_parallel:,.2f} demos/sec")

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
    demos_Real2Render2Real_1gpu = calculate_demos(time_s, speed_Real2Render2Real_1gpu, UPFRONT_TIME_Real2Render2Real_S)
    demos_Real2Render2Real_parallel = calculate_demos(time_s, speed_Real2Render2Real_parallel, UPFRONT_TIME_Real2Render2Real_S)

    # Create a modified time vector for ax0 R2R2R plotting to show start near y=1
    time_s_r2r2r_ax0 = np.sort(np.concatenate([
        time_s,
        np.array([UPFRONT_TIME_Real2Render2Real_S + 0.1]) # Add point just after start
    ]))
    demos_Real2Render2Real_1gpu_ax0 = calculate_demos(time_s_r2r2r_ax0, speed_Real2Render2Real_1gpu, UPFRONT_TIME_Real2Render2Real_S)

    # Add lines for intermediate parallelism levels
    gpus_intermediate = [100] # Reduced for clarity
    demos_Real2Render2Real_intermediate = {}
    for n_gpu in gpus_intermediate:
        speed_intermediate = n_gpu * speed_Real2Render2Real_1gpu
        demos_Real2Render2Real_intermediate[n_gpu] = calculate_demos(time_s, speed_intermediate, UPFRONT_TIME_Real2Render2Real_S)

    # --- Define Base Colors --- 
    color_r2r2r = 'royalblue' # Slightly lighter blue
    color_teleop = 'firebrick' # Darker red/orange

    # --- Create Figure with Two Subplots ---
    # Use specific figsize, ignore rcParams for this adjustment
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False) 
    ax0, ax1 = axes # Left, Middle axes

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
    mask_1gpu_ax0 = demos_Real2Render2Real_1gpu_ax0 > 0
    if np.any(mask_1gpu_ax0):
        ax.loglog(time_s_r2r2r_ax0[mask_1gpu_ax0], demos_Real2Render2Real_1gpu_ax0[mask_1gpu_ax0], 
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
    # text_y_pos = current_ymin * 1.2 # Based on min y-limit

    ymin_ax1, ymax_ax1 = ax.get_ylim()
    # text_y_pos = ymin_ax1 + (ymax_ax1 - ymin_ax1) * 0.02 # Position near bottom
    text_y_pos = ymin_ax1 + (ymax_ax1 - ymin_ax1) * 0.001 # Position near bottom

    ax.text(600, text_y_pos, '10 Min', ha='right', va='bottom', color='grey', alpha=0.9, rotation=90)
    ax.text(3600, text_y_pos, '1 Hour', ha='right', va='bottom', color='grey', alpha=0.9, rotation=90)
    ax.text(MAX_TIME_S, text_y_pos, '12 Hours', ha='right', va='bottom', color='grey', alpha=0.9, rotation=90)

    # --- Parameter Summary Text Box for Plot 1 (Restored) ---
    param_text = (f"Parameters:\n"
                  f"Teleop: {TIME_TELEOP_PER_DEMO_S:.1f} s/demo\n"
                  f"R2R2R: {TIME_Real2Render2Real_PER_DEMO_S_1GPU:.1f} s/demo (1 GPU)\n"
                  f"R2R2R Upfront: {UPFRONT_TIME_Real2Render2Real_S/60:.0f} min")
                #   f"{gpus_needed:,} GPUs for {TARGET_SPEEDUP_X:,}x speedup")
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
    # ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3) # Add minor gridlines
    ax.legend(loc='upper left') # Consistent legend placement
    ax.set_xlim(90, MAX_TIME_S * 1.1) # Start from 90s, end slightly past 12 hours
    # Recalculate max_demos including all teleop lines
    all_max_demos = [d.max() for d in demos_teleop_dict.values()] + \
                    [demos_Real2Render2Real_1gpu_ax0.max()]
    max_demos_ax0 = max(all_max_demos) if all_max_demos else 10

    # Set fixed lower y-limit to 1 for log-log plot to ensure lines start from bottom edge
    ax.set_ylim(1, max_demos_ax0 * 5) 

    # === Plot 2 (Middle): R2R2R Scaling Comparison (Semi-Log) ===
    ax = ax1 # Use ax1 for the second plot

    # Plot selected Teleop lines
    selected_ops = [1, 100]
    for n_ops in selected_ops:
        if n_ops in demos_teleop_dict:
            try:
                op_idx = operators_teleop.index(n_ops)
                op_style = teleop_styles[op_idx % len(teleop_styles)]
                ax.plot(time_s, demos_teleop_dict[n_ops], 
                        label=f'Teleop ({n_ops} Op.)', 
                        color=color_teleop, 
                        linestyle=op_style, 
                        linewidth=3.0) # Increased linewidth
            except ValueError:
                 print(f"Warning: Operator count {n_ops} style index not found.")
        else:
             print(f"Warning: Data for {n_ops} operators not found for plot 2.")

    # Plot R2R2R (100 GPUs - intermediate)
    try: # Plot R2R2R 100 GPU line
        r2r2r_idx_100 = gpus_intermediate.index(100)
        if 100 in demos_Real2Render2Real_intermediate:
            ax.plot(time_s, demos_Real2Render2Real_intermediate[100], 
                    label=f'R2R2R ({100} GPUs)', 
                    color=color_r2r2r, 
                    linestyle='-', # Solid line for R2R2R
                    linewidth=3.0) # Increased linewidth
    except ValueError:
         print("Warning: 100 GPUs not found in gpus_intermediate list for plot 2.")

    # Plot R2R2R (Parallel - {gpus_needed} GPUs)
    mask_parallel = demos_Real2Render2Real_parallel > 0 # Still needed for y-limit calc
    ax.plot(time_s, demos_Real2Render2Real_parallel, 
            label=f'R2R2R ({gpus_needed:,} GPUs)', 
            color='darkblue', # Keep darkest blue
            linestyle='-', 
            linewidth=2.5) # Increased linewidth

    # --- Axis Setup for Plot 2 ---
    ax.set_xscale('log')   # Set log scale for x-axis
    ax.set_yscale('linear') # Keep linear scale for y-axis
    ax.set_xlim(90, MAX_TIME_S * 1.1) # Log time axis from 90s to 12 hours
    
    # Calculate Y limits for the second plot (linear scale)
    min_y_ax1_data = []
    max_y_ax1_data = []
    for n_ops in selected_ops:
        if n_ops in demos_teleop_dict:
            min_y_ax1_data.append(demos_teleop_dict[n_ops].min())
            max_y_ax1_data.append(demos_teleop_dict[n_ops].max())
    if 100 in demos_Real2Render2Real_intermediate:
        min_y_ax1_data.append(demos_Real2Render2Real_intermediate[100].min())
        max_y_ax1_data.append(demos_Real2Render2Real_intermediate[100].max())
    min_y_ax1_data.append(demos_Real2Render2Real_parallel.min())
    max_y_ax1_data.append(demos_Real2Render2Real_parallel.max())
    
    min_y_ax1 = min(min_y_ax1_data) if min_y_ax1_data else 0
    max_y_ax1 = max(max_y_ax1_data) if max_y_ax1_data else 10
    ax.set_ylim(min_y_ax1 - (max_y_ax1 - min_y_ax1)*0.05 , max_y_ax1 * 1.1) # Linear scale padding

    # Labels, Title, Grid, Legend for Plot 2
    ax.set_xlabel("Wall-clock Time (Log-Scale Seconds)")
    
    # Format Y-axis ticks using scientific notation (e.g., 1x10^8)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 5)) # Trigger scientific notation for large/small numbers
    ax.yaxis.set_major_formatter(formatter)
    
    ax.set_ylabel("Number of Demonstrations Generated")
    ax.set_title("R2R2R Parallel Scaling")
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.5, alpha=0.6)
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5, alpha=0.4)
    # ax.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.3) # Add minor x gridlines for log scale

    # Add time markers to Plot 2
    ymin_ax1, ymax_ax1 = ax.get_ylim()
    text_y_pos_ax1 = ymin_ax1 + (ymax_ax1 - ymin_ax1) * 0.1 # Position near bottom
    ax.axvline(600, color='grey', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(600, text_y_pos_ax1, '10 Min', ha='right', va='bottom', color='grey', alpha=0.9, rotation=90)
    ax.axvline(3600, color='grey', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(3600, text_y_pos_ax1, '1 Hour', ha='right', va='bottom', color='grey', alpha=0.9, rotation=90)
    ax.axvline(MAX_TIME_S, color='grey', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(MAX_TIME_S, text_y_pos_ax1, '12 Hours', ha='right', va='bottom', color='grey', alpha=0.9, rotation=90)

    ax.legend(loc='upper left')

    # Annotations for Plot 2 (Values at 12 Hours)
    demos_at_12hr_parallel = calculate_demos(np.array([MAX_TIME_S]), speed_Real2Render2Real_parallel, UPFRONT_TIME_Real2Render2Real_S)[0]
    
    ax.annotate(f'{TARGET_SPEEDUP_X:,}x Real-time Speed\n({gpus_needed:,} GPUs needed)', 
                xy=(MAX_TIME_S, demos_at_12hr_parallel),
                xytext=(MAX_TIME_S * 0.2, demos_at_12hr_parallel * 0.5), # Adjust text pos relative to 12hr value
                arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                fontsize=10 + font_offsize, ha='right') # Increased fontsize
                
    # Value annotation for Teleop (1 Op) at 12 Hours
    demos_teleop_1_12hr = calculate_demos(np.array([MAX_TIME_S]), speed_teleop, 0)[0]
    if demos_teleop_1_12hr > 0:
        # Place slightly below the line end
        ax.text(MAX_TIME_S * 0.95, demos_teleop_1_12hr, f'{demos_teleop_1_12hr:,.0f}', 
                ha='right', va='top', color=color_teleop, fontsize=10 + font_offsize, alpha=0.9) # Increased fontsize
                
    # Value annotation for Teleop (100 Op) at 12 Hours
    demos_teleop_100_12hr = calculate_demos(np.array([MAX_TIME_S]), 100 * speed_teleop, 0)[0]
    if demos_teleop_100_12hr > 0:
         # Place slightly above the line end
         ax.text(MAX_TIME_S * 0.95, demos_teleop_100_12hr, f'{demos_teleop_100_12hr:,.0f}', 
                 ha='right', va='bottom', color=color_teleop, fontsize=10 + font_offsize, alpha=0.9) # Increased fontsize

    if demos_at_12hr_parallel > 0:
         ax.text(MAX_TIME_S * 0.95, demos_at_12hr_parallel, f'{demos_at_12hr_parallel:,.0f} demos', 
                 ha='right', va='bottom', color='darkblue', fontsize=10 + font_offsize, alpha=0.9) # Increased fontsize, Place slightly below

    # --- Final Touches ---
    plt.tight_layout(pad=1.0, w_pad=2.0) # Adjusted w_pad for 2 plots

    # --- Save and Show ---
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show() # Uncomment to display the plot interactively

if __name__ == "__main__":
    tyro.cli(main)