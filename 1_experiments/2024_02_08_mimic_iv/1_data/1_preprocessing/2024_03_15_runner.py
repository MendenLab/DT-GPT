import subprocess
import sys


if __name__ == '__main__':
    
    # Run stats generation
    subprocess.run([sys.executable, '/home/makaron1/uc2_nsclc/2_experiments/2024_02_08_mimic_iv/1_data/1_preprocessing/2024_02_01_overall_stats_generation.py'])

    # Run actual filtering and processing
    subprocess.run([sys.executable, '/home/makaron1/uc2_nsclc/2_experiments/2024_02_08_mimic_iv/1_data/1_preprocessing/2024_03_13_filter_columns.py'])





