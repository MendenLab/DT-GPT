# Manage the setup so we can import pipeline as if they were packages
import sys
import os
if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))
sys.path.append("..")
sys.path.append("../..")
base_path = os.path.dirname(__file__).split("/uc2_nsclc")[0] + "/uc2_nsclc/"  # Hacky way to get the base path
sys.path.append(base_path + "1_pipeline")
sys.path.append(base_path + "1_pipeline/data_generators")
sys.path.append(base_path + "1_pipeline/data_processors")
sys.path.append(base_path + "2_experiments/2023_11_07_neutrophils/6_dt_gpt_meditron/scripts")