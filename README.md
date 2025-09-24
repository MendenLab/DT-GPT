
# DT-GPT: Large Language Models for Patient Health Trajectory Forecasting

Code to run the experiments for the paper ["Large Language Models forecast Patient Health Trajectories enabling Digital Twins"](https://www.medrxiv.org/content/10.1101/2024.07.05.24309957v2).

> ❗**_NOTE:_** We are working on a Python library that will enable easier conversions, that is both indication agnostic and generally better engineered. This repo is more towards exploratory experiments, often with dead ends and potentially redundant code. Please raise an issue if something is missing or unclear.


## Installation


```python
# First install a fresh environment
cd DT-GPT
pip install -r requirements.txt
pip install -e .
```


## Overview

This repository contains the implementation of DT-GPT (Digital Twin GPT), a framework for using Large Language Models to forecast patient health trajectories. The codebase is organized into three main components: pipeline infrastructure, experimental runs, and exploratory analyses.

Developed as a collaboration project by Roche, Helmholtz Munich & LMU.

In case of any questions, please reach out to [Nikita Makarov](nikita.makarov@roche.com) or [Michael Menden](michael.menden@unimelb.edu.au).

## Repository Structure

### 📁 `pipeline/` - Core Infrastructure
Contains the main pipeline components and utilities for running experiments:

- **`Experiment.py`** - Main experiment orchestration class
- **`EvaluationManager.py`** - Handles model evaluation and metrics computation
- **`MetricManager.py`** - Computes various forecasting metrics
- **`Splitters.py`** - Data splitting strategies for time series forecasting
- **`BaselineHelpers.py`** - Baseline model implementations
- **`DartsHelpers.py`** - Integration with Darts time series forecasting library
- **`LLMTimeHelpers.py`** - Utilities for LLMTime model implementation
- **`NeuralForecastHelpers.py`** - Neural forecasting model helpers
- **`PlottingHelpers.py`** - Visualization utilities
- **`NormalizationFilterManager.py`** - Data preprocessing and filtering
- **`MatchingManagers.py`** - Patient matching strategies

#### 📁 `data_generators/`
- **`DataFrameConverters.py`** - Base classes for data conversion
- **`DataFrameConvertTDBDMIMIC.py`** - MIMIC-IV dataset conversion utilities
- **`DataFrameConvertTemplateTextBasicDescription.py`** - Text template generation for LLM input

#### 📁 `data_processors/`
- **`DataProcessorBiomistral.py`** - BioMistral model-specific data processing

### 📁 `1_experiments/` - Main Experimental Results

#### 📁 `2024_02_05_critical_vars/` - Critical Variables Dataset Experiments
Experiments on the critical care variables dataset:
- **`2_copy_forward_baseline/`** - Copy-forward baseline implementations
- **`3_1_time_llm/`** - Time-LLM model experiments
- **`3_2_llmtime/`** - LLMTime model experiments  
- **`3_3_general_llms/`** - General LLM experiments (GPT, BioMistral, etc.)
- **`3_dart_models_combined/`** - Traditional time series models using Darts library
- **`4_dt_gpt_instruction/`** - DT-GPT instruction-based forecasting experiments

#### 📁 `2024_02_08_mimic_iv/` - MIMIC-IV Dataset Experiments
Same experimental structure as above but applied to MIMIC-IV dataset:
- Similar folder structure with baseline, LLM, and traditional model experiments

#### 📁 `2025_02_03_adni/` - ADNI Dataset Experiments
Alzheimer's Disease Neuroimaging Initiative dataset experiments:
- **`2_copy_forward/`** - Baseline experiments
- **`3_dt_gpt/`** - DT-GPT experiments
- **`4_darts_models/`** - Traditional forecasting models
- **`5_time_llm/`** - Time-LLM experiments
- **`6_llmtime/`** - LLMTime experiments
- **`7_general_llms/`** - General LLM experiments

### 📁 `2_various_explorations/` - Exploratory Analyses

#### 📁 `zero_shot/`
Zero-shot forecasting experiments:
- **`2025_02_07_run_on_all_prompts.py`** - Zero-shot evaluation script
- **`2025_02_10_post_process_200_patient_run.ipynb`** - Post-processing of zero-shot results

#### 📁 `zero_shot_interpretation/`
- **`2025_07_28_run_zero_shot_interpretation.py`** - Zero-shot interpretation analysis

## Key Experiment Folders

### Main DT-GPT Experiments
The primary DT-GPT experiments can be found in the following folders:

1. **Critical Variables Dataset** (`2024_02_05_critical_vars/`):
   - Main DT-GPT experiments: `4_dt_gpt_instruction/`
   - Baseline experiments: `2_copy_forward_baseline/`
   - Comparison LLM experiments: `3_3_general_llms/`
   - Traditional model experiments: `3_dart_models_combined/`

2. **MIMIC-IV Dataset** (`2024_02_08_mimic_iv/`):
   - Main DT-GPT experiments: `4_dt_gpt_instruction/`
   - Similar structure with baseline and comparison model folders

3. **ADNI Dataset** (`2025_02_03_adni/`):
   - Main DT-GPT experiments: `3_dt_gpt/`
   - Baseline experiments: `2_copy_forward/`
   - Comparison model experiments: `4_darts_models/`, `5_time_llm/`, `6_llmtime/`, `7_general_llms/`

### Post-processing and Analysis
- Zero-shot experiment analysis: `3_various_explorations/zero_shot/2025_02_10_post_process_200_patient_run.ipynb`
- Model interpretation results: `3_various_explorations/zero_shot_interpretation/`

## Methodology

The framework implements several forecasting approaches:

1. **Baseline Models**: Copy-forward and simple statistical baselines
2. **Traditional Time Series Models**: Using Darts library (Linear model, LSTM, TiDE, ...)
3. **Specialized Time Series LLMs**: Time-LLM and LLMTime
4. **General Purpose LLMs**: BioMistral
5. **DT-GPT**: Our instruction-based approach for patient trajectory forecasting

## Dataset Support

The codebase supports multiple healthcare datasets:
- **Critical Care Variables**: ICU patient monitoring data
- **MIMIC-IV**: Publicly available critical care database
- **ADNI**: Alzheimer's disease progression data

## Data

Due to license restrictions, each dataset needs to be accessed from the respective organisation. In case of any specific data preprocessing questions that are not clear from the code, please reach out to [nikita.makarov@roche.com](nikita.makarov@roche.com) or [michael.menden@unimelb.edu.au](michael.menden@unimelb.edu.au). Additionally, we use the custom R library `FlatironTimeSeries` for preprocessing the Flatiron data, please reach out to us in case you need access to it.

The data preprocessing scripts are located in their respective dataset folders under `1_data`, with each indication having its own README. Please note, that you will need to adjust the paths in the scripts to your specific system setup.



### Example processed input and output on synthetic patient

This is the final processed text version for DT-GPT on a synthetic patient, in the NSCLC set up. Note, if you run this in vLLM, you need to use the completions enpoint (i.e. not using chat completions or chat templates). The ` <patient_prediction>` part is an essential input for the model.

Input:
```
First, patient chronological patient history up until the current day. Patient visits for the first time, with the following values: Hyperlipidemia, unspecified is diagnosed, Essential (primary) hypertension is diagnosed, ECOG is 1, <...>.
135 days after previous visit, patient visits again, with the following values: ECOG is 1, neutrophils.segmented is 2.8, <...>.
17 days after previous visit, patient visits again, with the following values: ECOG is 1, alanine aminotransferase is 6, <...>.
7 days after previous visit, <...>.

<...>

Next, the baseline data for the patient:  birth year is 1953, gender is M, ses index is 3, is cancer advanced is True, histology is Non-squamous cell carcinoma, cancer stage is Stage IV, smoking status is History of smoking, ethnicity is Not Hispanic or Latino, Current line of therapy is Pembrolizumab, Current line number is 2.
Finally, the variables which you should predict, and for which days in the future from the current day: {"hemoglobin": [21, 42, 63, 70, 77, 84]}
Now, your task is as follows: Given the non small cell lung cancer patient's history, please predict for this patient the previously noted down variables and future days, in the same JSON format. <patient_prediction>
```

Output:
```
{"hemoglobin": ["10.8", "11.3", "11.2", "11.6", "11.7", "11.7"]}
```



## Citation

If you use this code, please cite our paper:
```bibtex
@article{makarov2024large,
  title={Large language models forecast patient health trajectories enabling digital twins},
  author={Makarov, Nikita and Bordukova, Maria and Rodriguez-Esteban, Raul and Schmich, Fabian and Menden, Michael P},
  journal={medRxiv},
  pages={2024--07},
  year={2024},
  publisher={Cold Spring Harbor Laboratory Press}
}
```

## License

See `License.txt` for licensing information.