import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_benchmark_plots(csv_file, x_column, x_label, title_prefix, output_dir, sep=','):
    """
    Reads a benchmark CSV, groups by the X variable, averages the metrics 
    across multiple image pairs, and generates three line graphs:
    1. Mean Rotation Error vs. X
    2. Mean Translation Error vs. X
    3. Mean Epipolar Error vs. X
    
    Saves the resulting figure to the specified output directory.
    """
    try:
        # Read the data, passing the correct separator
        df = pd.read_csv(csv_file, sep=sep)
        
        # --- NEW DATA AGGREGATION ---
        # Group by the x-axis value (e.g., outlier_magnitude) and calculate the mean
        # This condenses multiple image pairs into a single, average point per X value
        df = df.groupby(x_column)[['rot_error_deg', 'trans_error_deg', 'epipolar_error']].mean().reset_index()
        
        # Sort by the x-axis to ensure the line graph connects properly
        df = df.sort_values(by=x_column)
        
        # Create a figure with three subplots side-by-side
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        
        # --- Graph 1: Rotation Error ---
        axes[0].plot(df[x_column], df['rot_error_deg'], marker='o', linestyle='-', color='#1f77b4', label='Mean Rotation Error')
        axes[0].set_title(f'{title_prefix}: Mean Rotation Error')
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel('Error (Degrees)')
        axes[0].legend()
        axes[0].grid(True, linestyle=':', alpha=0.7)
        
        # --- Graph 2: Translation Error ---
        axes[1].plot(df[x_column], df['trans_error_deg'], marker='s', linestyle='-', color='#ff7f0e', label='Mean Translation Error')
        axes[1].set_title(f'{title_prefix}: Mean Translation Error')
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel('Error (Degrees)')
        axes[1].legend()
        axes[1].grid(True, linestyle=':', alpha=0.7)
        
        # --- Graph 3: Epipolar Error ---
        axes[2].plot(df[x_column], df['epipolar_error'], marker='^', linestyle='-', color='purple', label='Mean Epipolar Error')
        axes[2].set_title(f'{title_prefix}: Mean Epipolar Error')
        axes[2].set_xlabel(x_label)
        axes[2].set_ylabel('Epipolar Error')
        axes[2].legend()
        axes[2].grid(True, linestyle=':', alpha=0.7)
        
        plt.tight_layout()
        
        # Determine the save path based on the CSV filename
        save_filename = f"{csv_file.stem}_plot.png"
        save_path = output_dir / save_filename
        
        # Save the plot and close the figure to free up memory
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        
        print(f"✅ Successfully grouped and saved: {save_path}")
        
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
    except KeyError as e:
        print(f"Error: Missing expected column in '{csv_file}'. Make sure your CSV contains: {e}")
        if 'df' in locals():
            print(f"Columns actually found: {df.columns.tolist()}")


# Path Setup
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent

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