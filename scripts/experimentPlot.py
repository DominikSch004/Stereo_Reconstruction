import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_benchmark_plots(csv_file, x_column, x_label, title_prefix, output_dir, sep=','):
    """
    Reads a benchmark CSV, groups by the X variable, and calculates the 
    median and standard deviation across sampling iterations.
    Generates line graphs with shaded standard deviation regions.
    """
    try:
        df = pd.read_csv(csv_file, sep=sep)
        
        # mean and std
        grouped = df.groupby(x_column)[['rot_error_deg', 'trans_error_deg', 'epipolar_error']].agg(['median', 'std']).reset_index()
        
        # flatten nested strings
        grouped.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped.columns]
        
        grouped = grouped.sort_values(by=x_column)
        x_data = grouped[x_column]
        
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        
        # plot the median line and shaded standard deviation area
        def plot_with_fill(ax, metric_name, color, label, y_label):
            median = grouped[f'{metric_name}_median']
            std = grouped[f'{metric_name}_std'].fillna(0) 
            
            ax.plot(x_data, median, marker='o', linestyle='-', color=color, label=f'Median {label}')
            
            # Create the shaded standard deviation region, ensure error > 0 
            ax.fill_between(x_data, np.clip(median - std, 0, None), median + std, color=color, alpha=0.2, label='±1 Std Dev')
            
            ax.set_title(f'{title_prefix}: {label}')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.7)

        # --- Generate the 3 Graphs ---
        plot_with_fill(axes[0], 'rot_error_deg', '#1f77b4', 'Rotation Error', 'Error (Degrees)')
        plot_with_fill(axes[1], 'trans_error_deg', '#ff7f0e', 'Translation Error', 'Error (Degrees)')
        plot_with_fill(axes[2], 'epipolar_error', 'purple', 'Epipolar Error', 'Epipolar Error')
        
        plt.tight_layout()
        
        # Save the plot
        save_filename = f"{csv_file.stem}_plot.png"
        save_path = output_dir / save_filename
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        
        print(f"Successfully grouped and saved: {save_path}")
        
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
    except KeyError as e:
        print(f"Error: Missing expected column in '{csv_file}'. Make sure your CSV contains: {e}")


# Path Setup
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent

experiments_dir = parent_dir / "experiments"
experiments_dir.mkdir(parents=True, exist_ok=True)

ratio_csv = experiments_dir / "svd_ratio_benchmark.csv"
magnitude_csv = experiments_dir / "svd_magnitude_benchmark.csv"
inlier_csv = experiments_dir / "svd_inliers_benchmark.csv"

generate_benchmark_plots(
    csv_file=ratio_csv,
    x_column="outlier_ratio",
    x_label="Outlier Ratio (%)",
    title_prefix="Ratio Analysis",
    output_dir=experiments_dir,
    sep="," 
)

generate_benchmark_plots(
    csv_file=magnitude_csv,
    x_column="outlier_magnitude",
    x_label="Outlier Magnitude (in px)",
    title_prefix="Magnitude Analysis",
    output_dir=experiments_dir,
    sep="," 
)

generate_benchmark_plots(
    csv_file=inlier_csv,
    x_column="num_inliers",
    x_label="Num. of Inliers",
    title_prefix="Increase Inlier Analysis",
    output_dir=experiments_dir,
    sep="," 
)