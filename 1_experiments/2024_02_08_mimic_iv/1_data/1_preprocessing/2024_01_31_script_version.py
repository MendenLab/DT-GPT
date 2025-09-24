import ipywidgets as widgets
import sys
from pathlib import Path
import os
import importlib
import subprocess


# This whole script is based on the pipeline in: https://github.com/healthylaife/MIMIC-IV-Data-Pipeline
# WARNING: this is a very hacky way of doing things, but it works for now.
# NOTE: this script then generates the data in the pipeline's directory, you need to copy over the results into your own directory (see README.md).


PATH_TO_MIMIC_IV_DATA_PIPELINE = "/home/makaron1/scratch/mimic-iv/"


os.chdir(PATH_TO_MIMIC_IV_DATA_PIPELINE + 'MIMIC-IV-Data-Pipeline')


module_path=PATH_TO_MIMIC_IV_DATA_PIPELINE + 'MIMIC-IV-Data-Pipeline'
if module_path not in sys.path:
    sys.path.append(module_path)

module_path='preprocessing/day_intervals_preproc'
if module_path not in sys.path:
    sys.path.append(module_path)

module_path=PATH_TO_MIMIC_IV_DATA_PIPELINE + 'MIMIC-IV-Data-Pipeline/utils'
if module_path not in sys.path:
    sys.path.append(module_path)
    
module_path='preprocessing/hosp_module_preproc'
if module_path not in sys.path:
    sys.path.append(module_path)
    
module_path='model'
if module_path not in sys.path:
    sys.path.append(module_path)
#print(sys.path)
root_dir = PATH_TO_MIMIC_IV_DATA_PIPELINE + "/MIMIC-IV-Data-Pipeline"
import day_intervals_cohort
from day_intervals_cohort import *

import day_intervals_cohort_v2
from day_intervals_cohort_v2 import *

import data_generation_icu

import data_generation
import evaluation



import feature_selection_hosp
from feature_selection_hosp import *

# import train
# from train import *


#import ml_models
#from ml_models import *

#import dl_train
#from dl_train import *

import tokenization
from tokenization import *


#import behrt_train
#from behrt_train import *

import feature_selection_icu
from feature_selection_icu import *
import fairness
import callibrate_output



importlib.reload(day_intervals_cohort)
import day_intervals_cohort
from day_intervals_cohort import *

importlib.reload(day_intervals_cohort_v2)
import day_intervals_cohort_v2
from day_intervals_cohort_v2 import *

importlib.reload(data_generation_icu)
import data_generation_icu
importlib.reload(data_generation)
import data_generation

importlib.reload(feature_selection_hosp)
import feature_selection_hosp
from feature_selection_hosp import *

importlib.reload(feature_selection_icu)
import feature_selection_icu
from feature_selection_icu import *

importlib.reload(tokenization)
import tokenization
from tokenization import *



import wandb

debug = False

def main():

    #: setup wandb
    if debug:
        wandb.init(mode="disabled")
    else:
        wandb.init(project='UC - MIMIC-IV', group="Data Processing")



    print("Please select the approriate version of MIMIC-IV for which you have downloaded data ?")
    version = 'Version 2'

    print("Please select what prediction task you want to perform ?")
    radio_input4 = 'Length of Stay'
    text1 = 3

    
    print("Extract Data")
    print("Please select below if you want to work with ICU or Non-ICU data ?")
    radio_input1 = 'ICU'

    print("Please select if you want to perform choosen prediction task for a specific disease.")
    radio_input3 = 'No Disease Filter'
    
    disease_label=""
    time=0
    label=radio_input4
    time=text1

    data_icu=radio_input1=="ICU"
    data_mort=label=="Mortality"
    data_admn=label=='Readmission'
    data_los=label=='Length of Stay'
    icd_code='No Disease Filter'
    version_path="mimiciv/2.0"
    cohort_output = day_intervals_cohort_v2.extract_data(radio_input1,label,time,icd_code, root_dir,disease_label)


    print("Which Features you want to include for cohort?")
    check_input1 = True # 'Diagnosis')
    check_input2 = True # 'Output Events')
    check_input3 = True # 'Chart Events(Labs and Vitals)')
    check_input4 = True # 'Procedures')
    check_input5 = True # 'Medications')
    diag_flag=check_input1
    out_flag=check_input2
    chart_flag=check_input3
    proc_flag=check_input4
    med_flag=check_input5

    feature_icu(cohort_output, version_path,diag_flag,out_flag,chart_flag,proc_flag,med_flag)


    print("Do you want to group ICD 10 DIAG codes ?")
    radio_input4 = 'Convert ICD-9 to ICD-10 and group ICD-10 codes'
    group_diag=False
    group_med=False
    group_proc=False
    group_diag=radio_input4
    preprocess_features_icu(cohort_output, diag_flag, group_diag,False,False,False,0,0)

    generate_summary_icu(diag_flag,proc_flag,med_flag,out_flag,chart_flag)

    select_diag=False
    select_med=False
    select_proc=False
    select_lab=False
    select_out=False
    select_chart=False

    if data_icu:
        features_selection_icu(cohort_output, diag_flag,proc_flag,med_flag,out_flag, chart_flag,select_diag,select_med,select_proc,select_out,select_chart)



    print("Outlier removal in values of chart events ?")

    radio_input5 = 'Impute Outlier (default:98)'    
    # Both sides impute over/below 2%
    outlier = 98
    left_outlier = 2

    thresh=0
    clean_chart=radio_input5!='No outlier detection'
    impute_outlier_chart=radio_input5=='Impute Outlier (default:98)'
    thresh=outlier
    left_thresh=left_outlier
    preprocess_features_icu(cohort_output, False, False,chart_flag,clean_chart,impute_outlier_chart,thresh,left_thresh)

    print("=======Time-series Data Represenation=======")

    print("Length of data to be included for time-series prediction ?")
    radio_input8 = 'Custom'
    text2 = 48



    print("What time bucket size you want to choose ?")
    radio_input7 = '1 hour'
    
    print("Do you want to forward fill and mean or median impute lab/chart values to form continuous data signal?")
    radio_impute = 'No Imputation'
    
    predW = 0
    if (radio_input7=='Custom'):
        bucket=int(text1)
    else:
        bucket=int(radio_input7[0].strip())
    if (radio_input8=='Custom'):
        include=int(text2)
    else:
        include=int(radio_input8.split()[1])
    if (radio_impute=='forward fill and mean'):
        impute='Mean'
    elif (radio_impute=='forward fill and median'):
        impute='Median'
    else:
        impute=False

    gen=data_generation_icu.Generator(cohort_output,data_mort,data_admn,data_los,diag_flag,proc_flag,out_flag,chart_flag,med_flag,impute,include,bucket,predW)
        







if __name__ == "__main__":
    main()





