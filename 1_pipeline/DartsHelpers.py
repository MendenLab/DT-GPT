from darts import TimeSeries
from sklearn.preprocessing import OneHotEncoder
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import StaticCovariatesTransformer
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.set_loglevel("error")
from darts.utils.missing_values import missing_values_ratio
import wandb
from sklearn.preprocessing import StandardScaler
from typing import Sequence
from darts.dataprocessing.transformers import BaseDataTransformer



# Custom transformer that clips values to a specified range
class ClipTransformer(BaseDataTransformer):
    def __init__(self, min_value=-3, max_value=3):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__()

    @staticmethod
    def ts_transform(series, params, **kwargs):
        min_value = params['fixed']['min_value']
        max_value = params['fixed']['max_value']

        df = series.pd_dataframe()

        # Perform the clipping operation on the DataFrame
        clipped_df = df.clip(lower=min_value, upper=max_value)

        # If the TimeSeries has static covariates, extract them
        if series.has_static_covariates:
            static_covariates = series.static_covariates

        # Convert the clipped DataFrame back to a TimeSeries object
        clipped_series = TimeSeries.from_dataframe(clipped_df, 
                                                   freq=series.freq_str, 
                                                   fill_missing_dates=False)

        # If static covariates were present, add them back to the new TimeSeries
        if series.has_static_covariates:
            clipped_series = clipped_series.with_static_covariates(static_covariates)
        
        return clipped_series
    



class ConstantFiller(BaseDataTransformer):
    def __init__(self, fill_values, name="ConstantFiller"):
        self.fill_values = fill_values
        super().__init__()

    @staticmethod
    def ts_transform(series, params, **kwargs):
        fill_values = params['fixed']['fill_values']
        filled_df = series.pd_dataframe().fillna(fill_values)

        if series.has_static_covariates:
            static_covariates = series.static_covariates

        filled_series = TimeSeries.from_dataframe(filled_df, 
                                                    freq=series.freq_str, 
                                                    fill_missing_dates=False)
        if series.has_static_covariates:
            filled_series = filled_series.with_static_covariates(static_covariates)
        
        return filled_series



class ForwardFillNAsUntilSplitDateThenLinear(BaseDataTransformer):

    def __init__(self):
        super().__init__()

    @staticmethod
    def ts_transform(series, params, **kwargs):
        
        # Preprocess
        if series.has_static_covariates:
            static_covariates = series.static_covariates
        df = series.pd_dataframe()
        
        # Get last row from input side
        split_date_raw = series.static_covariates["split_date"].values[0]
        mask = df.index > split_date_raw
        filtered_df = df[mask]
        first_valid_index = filtered_df.apply(pd.Series.first_valid_index)
        split_date = first_valid_index.min()
        
        # Actual filling
        df_before_rows = df.index < split_date
        df_after_rows = df.index >= split_date
        df.loc[df_before_rows] = df.loc[df_before_rows].fillna(method='ffill')
        df.loc[df_after_rows] = df.loc[df_after_rows].interpolate(method='linear', limit_direction='both')
    
        # Post process
        filled_series = TimeSeries.from_dataframe(df, 
                                                freq=series.freq_str, 
                                                fill_missing_dates=False)

        if series.has_static_covariates:
            filled_series = filled_series.with_static_covariates(static_covariates)
        
        return filled_series





def convert_to_darts_dataset(data_to_use, meta_data, forecast_horizon, max_look_back_window, constant_row_columns,
                             eval_manager, statistics_dic,
                             pipeline_targets=None, pipeline_past_covariates=None, pipeline_static_covariates=None,
                             past_covariate_encoder=None, n_jobs=-1, save_target_dfs=False,
                             drop_static_covariates_ids=True):

    # Setup
    inputs_cols, future_known_inputs_cols, target_cols = eval_manager.get_column_usage()
    original_target_cols = target_cols.copy()
    target_cols = target_cols.copy()
    inputs_cols = inputs_cols.copy()
    original_input_cols = inputs_cols.copy()
    target_cols.extend(["date", "patientid", "patient_sample_index"])
    inputs_cols.extend(["patient_sample_index"])
    constant_row_columns = constant_row_columns.copy()
    constant_row_columns.extend(["patientid"])
    original_input_cols = [col for col in original_input_cols if col not in ["patientid", "patient_sample_index", "date"]]
    missing_metadata = {}
    min_value = -3
    max_value = 3

    logging.info("Starting to convert to Darts dataset")
    logging.info("For missing data, using linear imputation, with fallback to forward/backward fill, and then 0. Drugs and diagnosis missing are handled directly as 0.")
    logging.info("For scaling, using Standard scaling, based on cleaned (<= 3 std) training data, to scale all values into z scores")
    logging.info("We clip all values to be between " + str(min_value) + " and " + str(max_value) + ", to avoid outliers messing up the model")

    ########################################################### SETUP TARGETS DF ###########################################################

    logging.info("Setting up target DF")

    #: extract target DF containing the values to predict, but drop full nan rows first
    target_df = [x[3].copy().reset_index(drop=True) for x in data_to_use]
    target_df = [target_df[idx][(meta_data[idx]["split_date"] - target_df[idx]["date"]).dt.days <= max_look_back_window] for idx in range(len(target_df))]  # Filter out too early dates to not mess up predictions
    
    # Add also in the targets which was in the input dataframes
    target_df_from_input = [x[1].loc[:, target_cols].copy().reset_index(drop=True) for x in data_to_use]
    target_df_from_input = [target_df_from_input[idx][(meta_data[idx]["split_date"] - target_df_from_input[idx]["date"]).dt.days <= max_look_back_window] for idx in range(len(target_df_from_input))]  # Filter out too early dates to not mess up predictions
    
    #: we also need to look into the past max_look_back_window to make sure that all series match up together
    empty_forecast_past_date_list = [{"patientid" : meta_data[idx]["patientid"],
                                    "patient_sample_index" : meta_data[idx]["patient_sample_index"],
                                    **{col : np.nan for col in original_target_cols},
                                    "date" : meta_data[idx]["split_date"] - pd.Timedelta(days=max_look_back_window)} 
                                    for idx in range(len(target_df))]
    empty_forecast_past_date_list = [pd.DataFrame(empty_forecast_past_date_list[idx], index=[0]) for idx in range(len(empty_forecast_past_date_list)) if empty_forecast_past_date_list[idx]["date"] not in target_df_from_input[idx]["date"].tolist()]
    target_df_from_input.extend(empty_forecast_past_date_list)

    #: add in to predict up to 91 days after LoT start date to ensure further correct processing - use: lot_start_date + (MIN_NR_DAYS_FORECAST - (split_date - lot_start_date))
    empty_forecast_date_list = [{"patientid" : meta_data[idx]["patientid"],
                                 "patient_sample_index" : meta_data[idx]["patient_sample_index"],
                                 **{col : np.nan for col in original_target_cols},
                                 "date" : meta_data[idx]["split_date"] + pd.Timedelta(days=forecast_horizon)}
                                 for idx in range(len(target_df))]
    empty_forecast_date_list = [pd.DataFrame(empty_forecast_date_list[idx], index=[0]) for idx in range(len(empty_forecast_date_list)) if empty_forecast_date_list[idx]["date"] not in target_df[idx]["date"].tolist()]
    target_df_from_input.extend(empty_forecast_date_list)

    # Create final dataframe
    target_df_from_input.extend(target_df)
    target_df = pd.concat(target_df_from_input, axis=0, ignore_index=True)

    # Save for use in future covariates
    target_df_from_input_original = target_df.copy()


    ########################################################### SETUP PAST COVARIATES DF + TS ###########################################################
    
    logging.info("Setting up past covariates")

    #: convert into input covariates DF
    past_covariate_df = [x[1][inputs_cols].copy() for x in data_to_use]
    past_covariate_df = [past_covariate_df[idx][(meta_data[idx]["split_date"] - past_covariate_df[idx]["date"]).dt.days <= max_look_back_window] for idx in range(len(past_covariate_df))]  # Filter out too early dates to not mess up predictions
    
    # we need to make past covariates go at least as long as needed by max lookback - so add in empty dates here as well
    empty_past_date_list = [{"patientid" : meta_data[idx]["patientid"],
                                "patient_sample_index" : meta_data[idx]["patient_sample_index"],
                                **{col : np.nan for col in original_input_cols},
                                "date" : meta_data[idx]["split_date"] - pd.Timedelta(days=max_look_back_window)} 
                                for idx in range(len(past_covariate_df))]
    
    empty_past_date_list = [pd.DataFrame(empty_past_date_list[idx], index=[0]) for idx in range(len(empty_past_date_list)) if empty_past_date_list[idx]["date"] not in past_covariate_df[idx]["date"].tolist()]
    past_covariate_df.extend(empty_past_date_list)
    
    # Merge into dataframe
    past_covariate_df = pd.concat(past_covariate_df, axis=0, ignore_index=True)
    past_covariate_df = past_covariate_df.drop(original_target_cols, axis=1)  # Remove target from past covariates since it is calculated from the target_df
    
    #: some drugs have "administered" --> convert that to the mean of that respective column
    drug_cols = [col for col in past_covariate_df.columns if col.startswith('drug_')]
    original_input_drug_df = past_covariate_df[[col for col in past_covariate_df.columns if col.startswith('drug_') or col in ["patientid", "patient_sample_index"]]].copy()
    mean_vals = past_covariate_df[drug_cols].apply(lambda col: pd.to_numeric(col, errors='coerce')).mean().fillna(1)  # Mean - default to 1 if only administrations in column
    past_covariate_df[drug_cols] = past_covariate_df[drug_cols].fillna(0)                # Fill missing values with 0
    past_covariate_df[drug_cols] = past_covariate_df[drug_cols].apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(mean_vals)
    

    #: convert all diagnosis columns with value to 1, and all missing to 0
    diagnosis_cols = [col for col in past_covariate_df.columns if col.startswith('diagnosis_')]
    past_covariate_df[diagnosis_cols] = past_covariate_df[diagnosis_cols].fillna(0)
    past_covariate_df[diagnosis_cols] = past_covariate_df[diagnosis_cols].replace('diagnosed', 1.0) 

    # Convert all metastases columns to objects
    metastases_cols = [col for col in past_covariate_df.columns if col.startswith('metastasis_')]
    past_covariate_df[metastases_cols] = past_covariate_df[metastases_cols].astype('object')

    #: drop death from past
    past_covariate_df = past_covariate_df.drop(["death_death"], axis=1)

    # convert all object columns to indicator columns, including NaN values
    if past_covariate_encoder is None:
        object_cols = past_covariate_df.select_dtypes(include=['object']).columns
        object_cols = object_cols.drop(['patientid', 'patient_sample_index'])
        past_covariate_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        past_covariate_encoder.fit(past_covariate_df[object_cols])
    else:
        object_cols = past_covariate_encoder.feature_names_in_.tolist()

    past_covariate_obj_cols = past_covariate_df[object_cols].astype('object')
    past_covariate_df_encoded = pd.DataFrame(past_covariate_encoder.transform(past_covariate_obj_cols), columns=past_covariate_encoder.get_feature_names_out())
    past_covariate_df.drop(columns=object_cols, inplace=True)
    past_covariate_df = pd.concat([past_covariate_df, past_covariate_df_encoded], axis=1)

    #: convert past_covariates to TS
    logging.info("Generating time series")
    input_cols_without_target = past_covariate_df.columns.copy().tolist()
    input_cols_without_target = [x for x in input_cols_without_target if x not in original_target_cols and x not in ["patientid", "patient_sample_index", "date"]]
    past_covariate_ts = TimeSeries.from_group_dataframe(past_covariate_df, group_cols=["patientid", "patient_sample_index"],
                                                        time_col="date", value_cols=input_cols_without_target,
                                                        fill_missing_dates=True, freq="7D")
    
    #: get missing data statistics from time series
    missing_metadata["past_covariate_ts_missing_ratio"] = pd.concat([ts.pd_dataframe().isnull().mean() for ts in past_covariate_ts], axis=1).mean(axis=1).to_dict()

    #: make past covariates pipeline  - impute missing values (fallback to 0), and scale globally target values
    logging.info("Processing pipeline")
    if pipeline_past_covariates is None:
        past_covariates_interpolater = MissingValuesFiller(fill="auto", n_jobs=n_jobs)
        past_covariates_filler_0 = MissingValuesFiller(fill=0.0, n_jobs=n_jobs)
        past_covariates_scaler = Scaler(scaler=StandardScaler(), n_jobs=n_jobs, global_fit=True)
        pipeline_past_covariates = Pipeline([past_covariates_interpolater, past_covariates_filler_0, past_covariates_scaler])
        pipeline_past_covariates.fit(past_covariate_ts)
    
    #: run pipeline
    past_covariate_ts = pipeline_past_covariates.transform(past_covariate_ts)


    ########################################################### SETUP STATIC COVARIATES + TARGET TS ###########################################################

    logging.info("Setting up static covariates and target TS")

    #: create static covariate based on drug administation, indicating with 1 any drug that was administered at least once during the current sequence, else fill with 0
    static_covariates_df = original_input_drug_df.groupby(['patientid', 'patient_sample_index']).apply(lambda x: (x.notnull().any().astype(int))).drop(["patientid", "patient_sample_index"], axis=1).reset_index()
 
    #: extract from master.constant row and append to static_covariates_df
    static_constant = [x[0][constant_row_columns].copy().reset_index(drop=True) for x in data_to_use]
    static_constant = pd.concat(static_constant, axis=0, ignore_index=True)
    static_constant = static_constant.drop_duplicates(keep="first")
    static_covariates_df = static_constant.merge(static_covariates_df, on=["patientid"], how="left")

    #: add here line number info, via patient_sample_index (since we split on lot)
    static_covariates_df["line_number"] = static_covariates_df["patient_sample_index"].apply(lambda x: int(x.split("_")[1]) + 1)

    #: merge with targets df
    static_covariates_columns = static_covariates_df.columns.tolist()
    target_df = target_df.merge(static_covariates_df, on=["patientid", "patient_sample_index"], how="left")

    #: convert target to timeseries - check fill_missing_dates, freq
    logging.info("Generating time series")
    target_ts = TimeSeries.from_group_dataframe(target_df, group_cols=["patientid", "patient_sample_index"],
                                                time_col="date", value_cols=original_target_cols, 
                                                static_cols=static_covariates_columns,
                                                fill_missing_dates=True, freq="7D")
    
    #: add split date as static covariate to target_df
    logging.info("Pre-processing split date")
    t1_static = [ts.static_covariates for ts in target_ts]
    static_order_dic = {}
    for idx, x in enumerate(data_to_use):
        curr_pid = x[2]["patientid"].iloc[0]
        curr_psi = x[2]["patient_sample_index"].iloc[0]
        
        if curr_pid not in static_order_dic:
            static_order_dic[curr_pid] = {}
        static_order_dic[curr_pid][curr_psi] = idx

    logging.info("Adding split date to static covariates")
    for curr_idx in range(len(target_ts)):
        #: get original index
        original_index = static_order_dic[t1_static[curr_idx]["patientid"].iloc[0, 0]][t1_static[curr_idx]["patient_sample_index"].iloc[0, 0]]
        #: apply to static covariate
        target_ts[curr_idx].static_covariates["split_date"] = meta_data[original_index]["split_date"]

    
    #: get missing data statistics from time series
    missing_metadata["target_ts_missing_ratio"] = pd.concat([ts.pd_dataframe().isnull().mean() for ts in target_ts], axis=1).mean(axis=1).to_dict()
    
    #: make targets pipeline - impute missing values, and scale globally target values
    logging.info("Processing pipeline")
    target_col_order = None
    if pipeline_targets is None:
        
        fill_value = {x  : statistics_dic[x]["mean_3_sigma_filtered"] for x in target_ts[0].pd_dataframe().columns if x in statistics_dic and statistics_dic[x]["type"] == "numeric"}
        targets_filler = ForwardFillNAsUntilSplitDateThenLinear()
        targets_backup_filler = ConstantFiller(fill_values=fill_value)
        targets_scaler = Scaler(scaler=StandardScaler(), n_jobs=n_jobs, global_fit=True)  # Using MixMax scaling by deault
        clip_transformer = ClipTransformer(min_value=min_value, max_value=max_value)
        pipeline_targets = Pipeline([targets_filler, targets_backup_filler, targets_scaler, clip_transformer])
        pipeline_targets.fit(target_ts)

        #: get clean means and vars in correct list, then numpy array
        logging.info("Using for target scaling the clean mean and std from the training data, i.e. data without outliers over 3 std")
        clean_means = np.array([statistics_dic[col]["mean_3_sigma_filtered"] for col in target_df_from_input_original.columns if col in statistics_dic and statistics_dic[col]["type"] == "numeric"])
        clean_stds = np.array([statistics_dic[col]["std_3_sigma_filtered"] for col in target_df_from_input_original.columns if col in statistics_dic and statistics_dic[col]["type"] == "numeric"])
        target_col_order = [col for col in target_df_from_input_original.columns if col in statistics_dic and statistics_dic[col]["type"] == "numeric"]
        pipeline_targets._transformers[2]._fitted_params[0].mean_ = clean_means
        pipeline_targets._transformers[2]._fitted_params[0].var_ = clean_stds**2
        pipeline_targets._transformers[2]._fitted_params[0].scale_ = clean_stds

    # print some stats
    logging_scaler = pipeline_targets[2]._transformers[0]._fitted_params[0]
    logging.info("Target scaler Mean:" + str(logging_scaler.mean_) + " Std: " + str(np.sqrt(logging_scaler.var_)))
    
    #: run pipeline
    target_ts = pipeline_targets.transform(target_ts)

    # Print some more statistics
    original_df = [ts.pd_dataframe() for ts in target_ts]
    num_clipped_values = sum([((ts_df == min_value) | (ts_df == max_value)).sum().sum() for ts_df in original_df])
    num_total_elements = sum([ts_df.count().sum() for ts_df in original_df])
    logging.info("Number of clipped values: " + str(num_clipped_values) + " out of total elements: " + str(num_total_elements))

    #: remove patientid and patient_sample_index and split_date from static covariates from target_ts (have to do it before the static covariate one hot encoder)
    ids = []
    for ts in target_ts:
        ids.append(ts.static_covariates[["patientid", "patient_sample_index"]].values[0, [0, 2]].tolist())
        
        if drop_static_covariates_ids:
            ts.static_covariates.drop(["patientid", "patient_sample_index", "split_date"], axis=1, inplace=True)
            assert "patientid" not in ts.static_covariates.columns.tolist()

    
    #: apply scaling & categorical value processing to static covariate (min/max & dummy variable)
    logging.info("Scaling static covariates")
    if pipeline_static_covariates is None:
        pipeline_static_covariates = StaticCovariatesTransformer(transformer_cat=OneHotEncoder(sparse=False, handle_unknown='ignore'))
        pipeline_static_covariates.fit(target_ts)
    target_ts = pipeline_static_covariates.transform(target_ts)
    

    ########################################################### SETUP FUTURE COVARIATE DF + TS ###########################################################

    logging.info("Setting up future covariates DF TS")
    
    #: create future covariates DF, with those days having a needed prediction getting 1, the rest 0 (to give the model the same info as DT-GPT)
    future_covariates_df = target_df_from_input_original
    future_covariates_df[original_target_cols] = future_covariates_df[original_target_cols].notnull().astype(int)
    rename_dict = {col: 'to_predict_' + col for col in original_target_cols}
    future_covariates_df = future_covariates_df.rename(columns=rename_dict)
    future_covariates_columns = list(rename_dict.values())

    #: generate future covarates TS
    future_covariates_ts = TimeSeries.from_group_dataframe(future_covariates_df, group_cols=["patientid", "patient_sample_index"],
                                                            time_col="date", value_cols=future_covariates_columns,
                                                            fill_missing_dates=True, freq="7D", fillna_value=0.0)

    #: get missing data statistics from time series - should be 0
    missing_metadata["future_covariate_ts_missing_ratio"] = pd.concat([ts.pd_dataframe().isnull().mean() for ts in future_covariates_ts], axis=1).mean(axis=1).to_dict()

    ########################################################### SETUP EVAL PARTS ###########################################################
    
    logging.info("Setting up evaluation parts")

    # Copy over original target dfs for use in evaluation, if needed, and reorder correctly to be in the same order as the target_ts (and thus also the other ts)
    if save_target_dfs:

        original_input_dfs = [x[1].copy() for x in data_to_use]
        df_dict2 = {(df['patientid'].iloc[0], df['patient_sample_index'].iloc[0]): df for df in original_input_dfs}
        original_input_dfs = [df_dict2[tuple(id)] for id in ids]

        original_target_dfs = [x[3].copy() for x in data_to_use]
        df_dict = {(df['patientid'].iloc[0], df['patient_sample_index'].iloc[0]): df for df in original_target_dfs}
        original_target_dfs = [df_dict[tuple(id)] for id in ids]


    else:
        original_target_dfs = None
        original_input_dfs = None


    ########################################################### FINAL PROCESSING ###########################################################
    logging.info("Final processing steps")

    #: remove patientid and patient_sample_index from static covariates from past_covariate_ts &future_covarate_ts
    if drop_static_covariates_ids:
        for ts in past_covariate_ts:
            ts.static_covariates.drop(["patientid", "patient_sample_index"], axis=1, inplace=True)
            assert "patientid" not in ts.static_covariates.columns.tolist()

        for ts in future_covariates_ts:
            ts.static_covariates.drop(["patientid", "patient_sample_index"], axis=1, inplace=True)
            assert "patientid" not in ts.static_covariates.columns.tolist()

    # Set to 32 bit
    target_ts = [x.astype(np.float32) for x in target_ts]
    past_covariate_ts = [x.astype(np.float32) for x in past_covariate_ts]
    future_covariates_ts = [x.astype(np.float32) for x in future_covariates_ts]

    #: return all data and pipelines
    ret_dic = {
        "target_ts" : target_ts,
        "past_covariate_ts" : past_covariate_ts,
        "future_covariates_ts" : future_covariates_ts,
        "pipeline_targets" : pipeline_targets,
        "target_col_order": target_col_order,
        "pipeline_past_covariates" : pipeline_past_covariates,
        "pipeline_static_covariates" : pipeline_static_covariates,
        "past_covariate_encoder" : past_covariate_encoder,
        "target_original_dfs": original_target_dfs,
        "input_original_dfs": original_input_dfs,
        "patientids_and_patient_sample_index": ids,
        "missing_data_statistics": missing_metadata,
    }

    return ret_dic





def get_output_for_darts_torch_model(model, eval_manager, eval_dataset_dictionary, forecast_horizon_chunks, plot_index_demo=None):
    

    # Init eval manager for streaming
    eval_manager.evaluate_split_stream_start()

    # Setup cols - using all
    inputs_cols, future_known_inputs_cols, target_cols = eval_manager.get_column_usage()
    target_cols = target_cols.copy()
    target_cols.extend(["date", "patientid", "patient_sample_index"])

    #: set up target_series to be same length as past_covariates_ts
    curr_target_series = [eval_dataset_dictionary["target_ts"][idx][0:len(eval_dataset_dictionary["past_covariate_ts"][idx])] for idx in range(len(eval_dataset_dictionary["target_ts"]))]

    #: run batched prediction
    #predictions_ts = model.predict(n=forecast_horizon_chunks,
    #                                series=curr_target_series, 
    #                                future_covariates=eval_dataset_dictionary["future_covariates_ts"], 
    #                                past_covariates=eval_dataset_dictionary["past_covariate_ts"], 
    #                                num_samples=1)
    predict_kwargs = {
        "n": forecast_horizon_chunks,
        "series": curr_target_series,
        "num_samples": 1
    }

    # Add future_covariates if the model supports them
    if model.supports_future_covariates:
        predict_kwargs["future_covariates"] = eval_dataset_dictionary.get("future_covariates_ts")

    # Add past_covariates if the model supports them
    if model.supports_past_covariates:
        predict_kwargs["past_covariates"] = eval_dataset_dictionary.get("past_covariate_ts")

    # Make the prediction with the appropriate arguments
    predictions_ts = model.predict(**predict_kwargs)
    
    #: transform back to original scale using pipeline from before
    predictions_ts = eval_dataset_dictionary["pipeline_targets"].inverse_transform(predictions_ts, partial=True)

    # Plot
    if plot_index_demo is not None:
        predictions_ts[plot_index_demo].plot()
        plt.show()

    logging.info("First three full predictions:")
    logging.info(predictions_ts[0].pd_dataframe().head(10))
    logging.info(predictions_ts[2].pd_dataframe().head(10))
    logging.info(predictions_ts[5].pd_dataframe().head(10))


    #: each pred a TS --> convert to dataframe
    predicted_df = [x.pd_dataframe().reset_index() for x in predictions_ts]

    #: extract only those dates which are in the targets
    predicted_df = [predicted_df[idx][predicted_df[idx]["date"].isin(eval_dataset_dictionary["target_original_dfs"][idx]["date"].tolist())] for idx in range(len(predicted_df))]

    #: send to eval manager
    for idx, curr_prediction in enumerate(predicted_df):
        
        patientid = eval_dataset_dictionary["patientids_and_patient_sample_index"][idx][0]
        patient_sample_index = eval_dataset_dictionary["patientids_and_patient_sample_index"][idx][1]
        target_df = eval_dataset_dictionary["target_original_dfs"][idx]
        
        # Setup prediction DF
        curr_prediction["patientid"] = patientid
        curr_prediction["patient_sample_index"] = patient_sample_index

        eval_manager.evaluate_split_stream_prediction(curr_prediction, target_df, patientid, patient_sample_index)

    #: post process predictions 
    logging.info("Finished generating samples")

    #: do full eval
    full_df_targets, full_df_prediction = eval_manager.concat_eval()

    # Return
    return full_df_targets, full_df_prediction


def log_to_wandb_missing_data_statistics(dataset, dataset_name):

    missing_data_statistics = dataset["missing_data_statistics"]
    wandb.config.update({"missing_statistics_" + dataset_name : missing_data_statistics}, allow_val_change=True)



def turn_all_over_3_sigma_predictions_to_mean(prediction_df, statistics_dic, to_print=False):
    # NOTE: this function actually clips the values, not mean!
    #: load column statistics from file
    statistics = statistics_dic
    logging.info("Filtering with 3 Sigma Rule")
    
    #: go through every target column
    for col in prediction_df.columns:
        
        if col in statistics.keys():                
            if statistics[col]["type"] == "numeric":
                
                #: calculate 3 sigma
                lower_bound = statistics[col]["mean_3_sigma_filtered"] - (3 * statistics[col]["std_3_sigma_filtered"])
                upper_bound = statistics[col]["mean_3_sigma_filtered"] + (3 * statistics[col]["std_3_sigma_filtered"])

                #: clip values to the bounds
                prediction_df[col] = prediction_df[col].clip(lower=lower_bound, upper=upper_bound)

                #: turn all values out of bounds to mean
                rows_clipped = ((prediction_df[col] == lower_bound) | (prediction_df[col] == upper_bound)).sum()
                logging.info("For column: " + str(col) + " clipping # bad predictions: " + str(rows_clipped))
                if to_print:
                    print("For column: " + str(col) + " clipping # bad predictions: " + str(rows_clipped))

            # keep categorical as is
    
    return prediction_df







def convert_to_darts_dataset_MIMIC(data_to_use, meta_data, forecast_horizon, max_look_back_window, constant_row_columns,
                             eval_manager, statistics_dic,
                             pipeline_targets=None, pipeline_past_covariates=None, pipeline_static_covariates=None,
                             past_covariate_encoder=None, n_jobs=-1, save_target_dfs=False,
                             drop_static_covariates_ids=True, freq="1H"):

    # Setup
    inputs_cols, future_known_inputs_cols, target_cols = eval_manager.get_column_usage()
    original_target_cols = target_cols.copy()
    target_cols = target_cols.copy()
    inputs_cols = inputs_cols.copy()
    original_input_cols = inputs_cols.copy()
    target_cols.extend(["date", "patientid", "patient_sample_index"])
    inputs_cols.extend(["patient_sample_index"])
    constant_row_columns = constant_row_columns.copy()
    constant_row_columns.extend(["patientid"])
    original_input_cols = [col for col in original_input_cols if col not in ["patientid", "patient_sample_index", "date"]]
    missing_metadata = {}
    min_value = -3
    max_value = 3

    logging.info("Starting to convert to Darts dataset")
    logging.info("For missing data, using linear imputation, with fallback to forward/backward fill, and then 0. Drugs and diagnosis missing are handled directly as 0.")
    logging.info("For scaling, using Standard scaling, based on cleaned (<= 3 std) training data, to scale all values into z scores")
    logging.info("We clip all values to be between " + str(min_value) + " and " + str(max_value) + ", to avoid outliers messing up the model")

    ########################################################### SETUP TARGETS DF ###########################################################

    logging.info("Setting up target DF")

    #: extract target DF containing the values to predict
    target_df = [x[3].copy().reset_index(drop=True) for x in data_to_use]
    target_df = [target_df[idx][(meta_data[idx]["split_date"] - target_df[idx]["date"]).dt.days <= max_look_back_window] for idx in range(len(target_df))]  # Filter out too early dates to not mess up predictions
    
    # Add also in the targets which was in the input dataframes
    target_df_from_input = [x[1].loc[:, target_cols].copy().reset_index(drop=True) for x in data_to_use]

    # Create final dataframe
    target_df_from_input.extend(target_df)
    target_df = pd.concat(target_df_from_input, axis=0, ignore_index=True)

    # Save for use in future covariates
    target_df_from_input_original = target_df.copy()


    ########################################################### SETUP PAST COVARIATES DF + TS ###########################################################
    
    logging.info("Setting up past covariates")

    #: convert into input covariates DF
    past_covariate_df = [x[1][inputs_cols].copy() for x in data_to_use]
    past_covariate_df = [past_covariate_df[idx][(meta_data[idx]["split_date"] - past_covariate_df[idx]["date"]).dt.days <= max_look_back_window] for idx in range(len(past_covariate_df))]  # Filter out too early dates to not mess up predictions
    
    # Merge into dataframe
    past_covariate_df = pd.concat(past_covariate_df, axis=0, ignore_index=True)
    past_covariate_df = past_covariate_df.drop(original_target_cols, axis=1)  # Remove target from past covariates since it is calculated from the target_df
    

    # convert all object columns to indicator columns, including NaN values
    if past_covariate_encoder is None:
        object_cols = past_covariate_df.select_dtypes(include=['object']).columns

        if "patientid" in object_cols:
            object_cols = object_cols.drop(['patient_sample_index'])
        if "patient_sample_index" in object_cols:
            object_cols = object_cols.drop(['patient_sample_index'])

        past_covariate_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        past_covariate_encoder.fit(past_covariate_df[object_cols])
    else:

        #: skip if empty object cols
        if not hasattr(past_covariate_encoder, "feature_names_in_"):
            object_cols = []
        else:
            object_cols = past_covariate_encoder.feature_names_in_.tolist()

    past_covariate_obj_cols = past_covariate_df[object_cols].astype('object')
    past_covariate_df_encoded = pd.DataFrame(past_covariate_encoder.transform(past_covariate_obj_cols), columns=past_covariate_encoder.get_feature_names_out())
    past_covariate_df.drop(columns=object_cols, inplace=True)
    past_covariate_df = pd.concat([past_covariate_df, past_covariate_df_encoded], axis=1)

    #: convert past_covariates to TS
    logging.info("Generating time series")
    input_cols_without_target = past_covariate_df.columns.copy().tolist()
    input_cols_without_target = [x for x in input_cols_without_target if x not in original_target_cols and x not in ["patientid", "patient_sample_index", "date"]]
    past_covariate_ts = TimeSeries.from_group_dataframe(past_covariate_df, group_cols=["patientid", "patient_sample_index"],
                                                        time_col="date", value_cols=input_cols_without_target,
                                                        fill_missing_dates=False, freq=freq)
    
    #: get missing data statistics from time series
    missing_metadata["past_covariate_ts_missing_ratio"] = pd.concat([ts.pd_dataframe().isnull().mean() for ts in past_covariate_ts], axis=1).mean(axis=1).to_dict()

    #: make past covariates pipeline  - impute missing values (fallback to 0), and scale globally target values
    logging.info("Processing pipeline")
    if pipeline_past_covariates is None:
        past_covariates_interpolater = MissingValuesFiller(fill="auto", n_jobs=n_jobs)
        past_covariates_filler_0 = MissingValuesFiller(fill=0.0, n_jobs=n_jobs)
        past_covariates_scaler = Scaler(scaler=StandardScaler(), n_jobs=n_jobs, global_fit=True)
        pipeline_past_covariates = Pipeline([past_covariates_interpolater, past_covariates_filler_0, past_covariates_scaler])
        pipeline_past_covariates.fit(past_covariate_ts)
    
    #: run pipeline
    past_covariate_ts = pipeline_past_covariates.transform(past_covariate_ts)


    ########################################################### SETUP STATIC COVARIATES + TARGET TS ###########################################################

    logging.info("Setting up static covariates and target TS")

    #: extract from master.constant row and append to static_covariates_df
    static_constant = [x[0][constant_row_columns].copy().reset_index(drop=True) for x in data_to_use]
    static_constant = pd.concat(static_constant, axis=0, ignore_index=True)
    static_constant = static_constant.drop_duplicates(keep="first")
    static_covariates_df = static_constant
    

    #: merge with targets df
    static_covariates_columns = static_covariates_df.columns.tolist()
    target_df = target_df.merge(static_covariates_df, on=["patientid"], how="left")

    #: convert target to timeseries - check fill_missing_dates, freq
    logging.info("Generating time series")
    target_ts = TimeSeries.from_group_dataframe(target_df, group_cols=["patientid", "patient_sample_index"],
                                                time_col="date", value_cols=original_target_cols, 
                                                static_cols=static_covariates_columns,
                                                fill_missing_dates=False, freq=freq)
    
    #: add split date as static covariate to target_df
    logging.info("Pre-processing split date")
    t1_static = [ts.static_covariates for ts in target_ts]
    static_order_dic = {}
    for idx, x in enumerate(data_to_use):
        curr_pid = str(x[2]["patientid"].iloc[0])
        curr_psi = x[2]["patient_sample_index"].iloc[0]
        
        if curr_pid not in static_order_dic:
            static_order_dic[curr_pid] = {}
        static_order_dic[curr_pid][curr_psi] = idx

    logging.info("Adding split date to static covariates")
    for curr_idx in range(len(target_ts)):
        #: get original index
        original_index = static_order_dic[str(int(t1_static[curr_idx]["patientid"].iloc[0, 0]))][t1_static[curr_idx]["patient_sample_index"].iloc[0]]
        #: apply to static covariate
        target_ts[curr_idx].static_covariates["split_date"] = meta_data[original_index]["split_date"]

    
    #: get missing data statistics from time series
    missing_metadata["target_ts_missing_ratio"] = pd.concat([ts.pd_dataframe().isnull().mean() for ts in target_ts], axis=1).mean(axis=1).to_dict()
    
    #: make targets pipeline - impute missing values, and scale globally target values
    logging.info("Processing pipeline")
    target_col_order = None
    if pipeline_targets is None:
        
        fill_value = {x  : statistics_dic[x]["mean_3_sigma_filtered"] for x in target_ts[0].pd_dataframe().columns if x in statistics_dic and statistics_dic[x]["type"] == "numeric"}
        targets_filler = ForwardFillNAsUntilSplitDateThenLinear()
        targets_backup_filler = ConstantFiller(fill_values=fill_value)
        targets_scaler = Scaler(scaler=StandardScaler(), n_jobs=n_jobs, global_fit=True)  # Using MixMax scaling by deault
        clip_transformer = ClipTransformer(min_value=min_value, max_value=max_value)
        pipeline_targets = Pipeline([targets_filler, targets_backup_filler, targets_scaler, clip_transformer])
        pipeline_targets.fit(target_ts)

        #: get clean means and vars in correct list, then numpy array
        logging.info("Using for target scaling the clean mean and std from the training data, i.e. data without outliers over 3 std")
        clean_means = np.array([statistics_dic[col]["mean_3_sigma_filtered"] for col in target_df_from_input_original.columns if col in statistics_dic and statistics_dic[col]["type"] == "numeric"])
        clean_stds = np.array([statistics_dic[col]["std_3_sigma_filtered"] for col in target_df_from_input_original.columns if col in statistics_dic and statistics_dic[col]["type"] == "numeric"])
        target_col_order = [col for col in target_df_from_input_original.columns if col in statistics_dic and statistics_dic[col]["type"] == "numeric"]
        pipeline_targets._transformers[2]._fitted_params[0].mean_ = clean_means
        pipeline_targets._transformers[2]._fitted_params[0].var_ = clean_stds**2
        pipeline_targets._transformers[2]._fitted_params[0].scale_ = clean_stds

    # print some stats
    logging_scaler = pipeline_targets[2]._transformers[0]._fitted_params[0]
    logging.info("Target scaler Mean:" + str(logging_scaler.mean_) + " Std: " + str(np.sqrt(logging_scaler.var_)))
    
    #: run pipeline
    target_ts = pipeline_targets.transform(target_ts)

    # Print some more statistics
    original_df = [ts.pd_dataframe() for ts in target_ts]
    num_clipped_values = sum([((ts_df == min_value) | (ts_df == max_value)).sum().sum() for ts_df in original_df])
    num_total_elements = sum([ts_df.count().sum() for ts_df in original_df])
    logging.info("Number of clipped values: " + str(num_clipped_values) + " out of total elements: " + str(num_total_elements))

    #: remove patientid and patient_sample_index and split_date from static covariates from target_ts (have to do it before the static covariate one hot encoder)
    ids = []
    for ts in target_ts:
        ids.append(ts.static_covariates[["patientid", "patient_sample_index"]].values[0, [0, 2]].tolist())
        
        if drop_static_covariates_ids:
            ts.static_covariates.drop(["patientid", "patient_sample_index", "split_date"], axis=1, inplace=True)
            assert "patientid" not in ts.static_covariates.columns.tolist()


    #: apply scaling & categorical value processing to static covariate (min/max & dummy variable)
    logging.info("Scaling static covariates")
    if pipeline_static_covariates is None:
        pipeline_static_covariates = StaticCovariatesTransformer(transformer_cat=OneHotEncoder(sparse=False, handle_unknown='ignore'))
        pipeline_static_covariates.fit(target_ts)
    target_ts = pipeline_static_covariates.transform(target_ts)
    

    ########################################################### SETUP FUTURE COVARIATE DF + TS ###########################################################

    logging.info("Setting up future covariates DF TS")
    
    #: create future covariates DF, with those days having a needed prediction getting 1, the rest 0 (to give the model the same info as DT-GPT)
    future_covariates_df = target_df_from_input_original
    future_covariates_df[original_target_cols] = future_covariates_df[original_target_cols].notnull().astype(int)
    rename_dict = {col: 'to_predict_' + col for col in original_target_cols}
    future_covariates_df = future_covariates_df.rename(columns=rename_dict)
    future_covariates_columns = list(rename_dict.values())

    #: generate future covarates TS
    future_covariates_ts = TimeSeries.from_group_dataframe(future_covariates_df, group_cols=["patientid", "patient_sample_index"],
                                                            time_col="date", value_cols=future_covariates_columns,
                                                            fill_missing_dates=False, freq=freq, fillna_value=0.0)

    #: get missing data statistics from time series - should be 0
    missing_metadata["future_covariate_ts_missing_ratio"] = pd.concat([ts.pd_dataframe().isnull().mean() for ts in future_covariates_ts], axis=1).mean(axis=1).to_dict()

    ########################################################### SETUP EVAL PARTS ###########################################################
    
    logging.info("Setting up evaluation parts")

    # Copy over original target dfs for use in evaluation, if needed, and reorder correctly to be in the same order as the target_ts (and thus also the other ts)
    if save_target_dfs:

        original_input_dfs = [x[1].copy() for x in data_to_use]
        df_dict2 = {(df['patientid'].iloc[0], df['patient_sample_index'].iloc[0]): df for df in original_input_dfs}
        original_input_dfs = [df_dict2[tuple(id)] for id in ids]

        original_target_dfs = [x[3].copy() for x in data_to_use]
        df_dict = {(df['patientid'].iloc[0], df['patient_sample_index'].iloc[0]): df for df in original_target_dfs}
        original_target_dfs = [df_dict[tuple(id)] for id in ids]
        original_target_dfs = [x.dropna(axis=0, how='all', subset=x.columns.difference(["patientid", "patient_sample_index", "date"])) for x in original_target_dfs]


    else:
        original_target_dfs = None
        original_input_dfs = None


    ########################################################### FINAL PROCESSING ###########################################################
    logging.info("Final processing steps")

    #: remove patientid and patient_sample_index from static covariates from past_covariate_ts &future_covarate_ts
    if drop_static_covariates_ids:
        for ts in past_covariate_ts:
            ts.static_covariates.drop(["patientid", "patient_sample_index"], axis=1, inplace=True)
            assert "patientid" not in ts.static_covariates.columns.tolist()

        for ts in future_covariates_ts:
            ts.static_covariates.drop(["patientid", "patient_sample_index"], axis=1, inplace=True)
            assert "patientid" not in ts.static_covariates.columns.tolist()

    # Set to 32 bit
    target_ts = [x.astype(np.float32) for x in target_ts]
    past_covariate_ts = [x.astype(np.float32) for x in past_covariate_ts]
    future_covariates_ts = [x.astype(np.float32) for x in future_covariates_ts]

    # Fix patientid issue being seen as float
    ids = [(str(int(x[0])), x[1]) for x in ids]

    #: return all data and pipelines
    ret_dic = {
        "target_ts" : target_ts,
        "past_covariate_ts" : past_covariate_ts,
        "future_covariates_ts" : future_covariates_ts,
        "pipeline_targets" : pipeline_targets,
        "target_col_order": target_col_order,
        "pipeline_past_covariates" : pipeline_past_covariates,
        "pipeline_static_covariates" : pipeline_static_covariates,
        "past_covariate_encoder" : past_covariate_encoder,
        "target_original_dfs": original_target_dfs,
        "input_original_dfs": original_input_dfs,
        "patientids_and_patient_sample_index": ids,
        "missing_data_statistics": missing_metadata,
    }

    return ret_dic








def convert_to_darts_dataset_ADNI(data_to_use, meta_data, forecast_horizon, max_look_back_window, constant_row_columns,
                             eval_manager, statistics_dic,
                             pipeline_targets=None, pipeline_past_covariates=None, pipeline_static_covariates=None,
                             past_covariate_encoder=None, n_jobs=-1, save_target_dfs=False,
                             drop_static_covariates_ids=True, freq="1H"):

    # Setup
    inputs_cols, future_known_inputs_cols, target_cols = eval_manager.get_column_usage()
    original_target_cols = target_cols.copy()
    target_cols = target_cols.copy()
    inputs_cols = inputs_cols.copy()
    original_input_cols = inputs_cols.copy()
    target_cols.extend(["date", "patientid", "patient_sample_index"])
    inputs_cols.extend(["patient_sample_index"])
    constant_row_columns = constant_row_columns.copy()
    constant_row_columns.extend(["patientid"])
    original_input_cols = [col for col in original_input_cols if col not in ["patientid", "patient_sample_index", "date"]]
    missing_metadata = {}
    min_value = -3
    max_value = 3

    logging.info("Starting to convert to Darts dataset")
    logging.info("For missing data, using linear imputation, with fallback to forward/backward fill, and then 0. Drugs and diagnosis missing are handled directly as 0.")
    logging.info("For scaling, using Standard scaling, based on cleaned (<= 3 std) training data, to scale all values into z scores")
    logging.info("We clip all values to be between " + str(min_value) + " and " + str(max_value) + ", to avoid outliers messing up the model")

    ########################################################### SETUP TARGETS DF ###########################################################

    logging.info("Setting up target DF")

    #: extract target DF containing the values to predict
    target_df = [x[3].copy().reset_index(drop=True) for x in data_to_use]
    target_df = [target_df[idx][(meta_data[idx]["split_date"] - target_df[idx]["date"]).dt.days <= max_look_back_window] for idx in range(len(target_df))]  # Filter out too early dates to not mess up predictions
    
    # Add also in the targets which was in the input dataframes
    target_df_from_input = [x[1].loc[:, target_cols].copy().reset_index(drop=True) for x in data_to_use]

    # Create final dataframe
    target_df_from_input.extend(target_df)
    target_df = pd.concat(target_df_from_input, axis=0, ignore_index=True)

    # Save for use in future covariates
    target_df_from_input_original = target_df.copy()


    ########################################################### SETUP PAST COVARIATES DF + TS ###########################################################
    
    logging.info("Setting up past covariates")

    #: convert into input covariates DF
    past_covariate_df = [x[1][inputs_cols].copy() for x in data_to_use]
    past_covariate_df = [past_covariate_df[idx][(meta_data[idx]["split_date"] - past_covariate_df[idx]["date"]).dt.days <= max_look_back_window] for idx in range(len(past_covariate_df))]  # Filter out too early dates to not mess up predictions
    
    # Merge into dataframe
    past_covariate_df = pd.concat(past_covariate_df, axis=0, ignore_index=True)
    past_covariate_df = past_covariate_df.drop(original_target_cols, axis=1)  # Remove target from past covariates since it is calculated from the target_df
    

    # convert all object columns to indicator columns, including NaN values
    if past_covariate_encoder is None:
        object_cols = past_covariate_df.select_dtypes(include=['object']).columns

        if "patientid" in object_cols:
            object_cols = object_cols.drop(['patient_sample_index'])
        if "patient_sample_index" in object_cols:
            object_cols = object_cols.drop(['patient_sample_index'])

        past_covariate_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        past_covariate_encoder.fit(past_covariate_df[object_cols])
    else:

        #: skip if empty object cols
        if not hasattr(past_covariate_encoder, "feature_names_in_"):
            object_cols = []
        else:
            object_cols = past_covariate_encoder.feature_names_in_.tolist()

    past_covariate_obj_cols = past_covariate_df[object_cols].astype('object')
    past_covariate_df_encoded = pd.DataFrame(past_covariate_encoder.transform(past_covariate_obj_cols), columns=past_covariate_encoder.get_feature_names_out())
    past_covariate_df.drop(columns=object_cols, inplace=True)
    past_covariate_df = pd.concat([past_covariate_df, past_covariate_df_encoded], axis=1)

    #: convert past_covariates to TS
    logging.info("Generating time series")
    input_cols_without_target = past_covariate_df.columns.copy().tolist()
    input_cols_without_target = [x for x in input_cols_without_target if x not in original_target_cols and x not in ["patientid", "patient_sample_index", "date"]]
    past_covariate_ts = TimeSeries.from_group_dataframe(past_covariate_df, group_cols=["patientid", "patient_sample_index"],
                                                        time_col="date", value_cols=input_cols_without_target,
                                                        fill_missing_dates=True, freq=freq)  # TODO: fill_missing_dates=False
    
    #: get missing data statistics from time series
    missing_metadata["past_covariate_ts_missing_ratio"] = pd.concat([ts.pd_dataframe().isnull().mean() for ts in past_covariate_ts], axis=1).mean(axis=1).to_dict()

    #: make past covariates pipeline  - impute missing values (fallback to 0), and scale globally target values
    logging.info("Processing pipeline")
    if pipeline_past_covariates is None:
        past_covariates_interpolater = MissingValuesFiller(fill="auto", n_jobs=n_jobs)
        past_covariates_filler_0 = MissingValuesFiller(fill=0.0, n_jobs=n_jobs)
        past_covariates_scaler = Scaler(scaler=StandardScaler(), n_jobs=n_jobs, global_fit=True)
        pipeline_past_covariates = Pipeline([past_covariates_interpolater, past_covariates_filler_0, past_covariates_scaler])
        pipeline_past_covariates.fit(past_covariate_ts)
    
    #: run pipeline
    past_covariate_ts = pipeline_past_covariates.transform(past_covariate_ts)


    ########################################################### SETUP STATIC COVARIATES + TARGET TS ###########################################################

    logging.info("Setting up static covariates and target TS")

    #: extract from master.constant row and append to static_covariates_df
    static_constant = [x[0][constant_row_columns].copy().reset_index(drop=True) for x in data_to_use]
    static_constant = pd.concat(static_constant, axis=0, ignore_index=True)
    static_constant = static_constant.drop_duplicates(keep="first")
    static_covariates_df = static_constant
    

    #: merge with targets df
    static_covariates_columns = static_covariates_df.columns.tolist()
    target_df = target_df.merge(static_covariates_df, on=["patientid"], how="left")

    #: convert target to timeseries - check fill_missing_dates, freq
    logging.info("Generating time series")
    target_ts = TimeSeries.from_group_dataframe(target_df, group_cols=["patientid", "patient_sample_index"],
                                                time_col="date", value_cols=original_target_cols, 
                                                static_cols=static_covariates_columns,
                                                fill_missing_dates=False, freq=freq)
    
    #: add split date as static covariate to target_df
    logging.info("Pre-processing split date")
    t1_static = [ts.static_covariates for ts in target_ts]
    static_order_dic = {}
    for idx, x in enumerate(data_to_use):
        curr_pid = str(x[2]["patientid"].iloc[0])
        curr_psi = x[2]["patient_sample_index"].iloc[0]
        
        if curr_pid not in static_order_dic:
            static_order_dic[curr_pid] = {}
        static_order_dic[curr_pid][curr_psi] = idx

    logging.info("Adding split date to static covariates")
    for curr_idx in range(len(target_ts)):
        #: get original index
        original_index = static_order_dic[str(int(t1_static[curr_idx]["patientid"].iloc[0, 0]))][t1_static[curr_idx]["patient_sample_index"].iloc[0]]
        #: apply to static covariate
        target_ts[curr_idx].static_covariates["split_date"] = meta_data[original_index]["split_date"]

    
    #: get missing data statistics from time series
    missing_metadata["target_ts_missing_ratio"] = pd.concat([ts.pd_dataframe().isnull().mean() for ts in target_ts], axis=1).mean(axis=1).to_dict()
    
    #: make targets pipeline - impute missing values, and scale globally target values
    logging.info("Processing pipeline")
    target_col_order = None
    if pipeline_targets is None:
        
        fill_value = {x  : statistics_dic[x]["mean_3_sigma_filtered"] for x in target_ts[0].pd_dataframe().columns if x in statistics_dic and statistics_dic[x]["type"] == "numeric"}
        targets_filler = ForwardFillNAsUntilSplitDateThenLinear()
        targets_backup_filler = ConstantFiller(fill_values=fill_value)
        targets_scaler = Scaler(scaler=StandardScaler(), n_jobs=n_jobs, global_fit=True)  # Using MixMax scaling by deault
        clip_transformer = ClipTransformer(min_value=min_value, max_value=max_value)
        pipeline_targets = Pipeline([targets_filler, targets_backup_filler, targets_scaler, clip_transformer])
        pipeline_targets.fit(target_ts)

        #: get clean means and vars in correct list, then numpy array
        logging.info("Using for target scaling the clean mean and std from the training data, i.e. data without outliers over 3 std")
        clean_means = np.array([statistics_dic[col]["mean_3_sigma_filtered"] for col in target_df_from_input_original.columns if col in statistics_dic and statistics_dic[col]["type"] == "numeric"])
        clean_stds = np.array([statistics_dic[col]["std_3_sigma_filtered"] for col in target_df_from_input_original.columns if col in statistics_dic and statistics_dic[col]["type"] == "numeric"])
        target_col_order = [col for col in target_df_from_input_original.columns if col in statistics_dic and statistics_dic[col]["type"] == "numeric"]
        pipeline_targets._transformers[2]._fitted_params[0].mean_ = clean_means
        pipeline_targets._transformers[2]._fitted_params[0].var_ = clean_stds**2
        pipeline_targets._transformers[2]._fitted_params[0].scale_ = clean_stds

    # print some stats
    logging_scaler = pipeline_targets[2]._transformers[0]._fitted_params[0]
    logging.info("Target scaler Mean:" + str(logging_scaler.mean_) + " Std: " + str(np.sqrt(logging_scaler.var_)))
    
    #: run pipeline
    target_ts = pipeline_targets.transform(target_ts)

    # Print some more statistics
    original_df = [ts.pd_dataframe() for ts in target_ts]
    num_clipped_values = sum([((ts_df == min_value) | (ts_df == max_value)).sum().sum() for ts_df in original_df])
    num_total_elements = sum([ts_df.count().sum() for ts_df in original_df])
    logging.info("Number of clipped values: " + str(num_clipped_values) + " out of total elements: " + str(num_total_elements))

    #: remove patientid and patient_sample_index and split_date from static covariates from target_ts (have to do it before the static covariate one hot encoder)
    ids = []
    for ts in target_ts:
        ids.append(ts.static_covariates[["patientid", "patient_sample_index"]].values[0, [0, 2]].tolist())
        
        if drop_static_covariates_ids:
            ts.static_covariates.drop(["patientid", "patient_sample_index", "split_date"], axis=1, inplace=True)
            assert "patientid" not in ts.static_covariates.columns.tolist()


    #: apply scaling & categorical value processing to static covariate (min/max & dummy variable)
    logging.info("Scaling static covariates")
    if pipeline_static_covariates is None:
        pipeline_static_covariates = StaticCovariatesTransformer(transformer_cat=OneHotEncoder(sparse=False, handle_unknown='ignore'))
        pipeline_static_covariates.fit(target_ts)
    target_ts = pipeline_static_covariates.transform(target_ts)
    

    ########################################################### SETUP FUTURE COVARIATE DF + TS ###########################################################

    logging.info("Setting up future covariates DF TS")
    
    #: create future covariates DF, with those days having a needed prediction getting 1, the rest 0 (to give the model the same info as DT-GPT)
    future_covariates_df = target_df_from_input_original
    future_covariates_df[original_target_cols] = future_covariates_df[original_target_cols].notnull().astype(int)
    rename_dict = {col: 'to_predict_' + col for col in original_target_cols}
    future_covariates_df = future_covariates_df.rename(columns=rename_dict)
    future_covariates_columns = list(rename_dict.values())

    #: generate future covarates TS
    future_covariates_ts = TimeSeries.from_group_dataframe(future_covariates_df, group_cols=["patientid", "patient_sample_index"],
                                                            time_col="date", value_cols=future_covariates_columns,
                                                            fill_missing_dates=False, freq=freq, fillna_value=0.0)

    #: get missing data statistics from time series - should be 0
    missing_metadata["future_covariate_ts_missing_ratio"] = pd.concat([ts.pd_dataframe().isnull().mean() for ts in future_covariates_ts], axis=1).mean(axis=1).to_dict()

    ########################################################### SETUP EVAL PARTS ###########################################################
    
    logging.info("Setting up evaluation parts")

    # Copy over original target dfs for use in evaluation, if needed, and reorder correctly to be in the same order as the target_ts (and thus also the other ts)
    if save_target_dfs:

        original_input_dfs = [x[1].copy() for x in data_to_use]
        df_dict2 = {(df['patientid'].iloc[0], df['patient_sample_index'].iloc[0]): df for df in original_input_dfs}
        original_input_dfs = [df_dict2[tuple(id)] for id in ids]

        original_target_dfs = [x[3].copy() for x in data_to_use]
        df_dict = {(df['patientid'].iloc[0], df['patient_sample_index'].iloc[0]): df for df in original_target_dfs}
        original_target_dfs = [df_dict[tuple(id)] for id in ids]
        original_target_dfs = [x.dropna(axis=0, how='all', subset=x.columns.difference(["patientid", "patient_sample_index", "date"])) for x in original_target_dfs]


    else:
        original_target_dfs = None
        original_input_dfs = None


    ########################################################### FINAL PROCESSING ###########################################################
    logging.info("Final processing steps")

    #: remove patientid and patient_sample_index from static covariates from past_covariate_ts &future_covarate_ts
    if drop_static_covariates_ids:
        for ts in past_covariate_ts:
            ts.static_covariates.drop(["patientid", "patient_sample_index"], axis=1, inplace=True)
            assert "patientid" not in ts.static_covariates.columns.tolist()

        for ts in future_covariates_ts:
            ts.static_covariates.drop(["patientid", "patient_sample_index"], axis=1, inplace=True)
            assert "patientid" not in ts.static_covariates.columns.tolist()

    # Set to 32 bit
    target_ts = [x.astype(np.float32) for x in target_ts]
    past_covariate_ts = [x.astype(np.float32) for x in past_covariate_ts]
    future_covariates_ts = [x.astype(np.float32) for x in future_covariates_ts]

    # Fix patientid issue being seen as float
    ids = [(str(int(x[0])), x[1]) for x in ids]

    #: return all data and pipelines
    ret_dic = {
        "target_ts" : target_ts,
        "past_covariate_ts" : past_covariate_ts,
        "future_covariates_ts" : future_covariates_ts,
        "pipeline_targets" : pipeline_targets,
        "target_col_order": target_col_order,
        "pipeline_past_covariates" : pipeline_past_covariates,
        "pipeline_static_covariates" : pipeline_static_covariates,
        "past_covariate_encoder" : past_covariate_encoder,
        "target_original_dfs": original_target_dfs,
        "input_original_dfs": original_input_dfs,
        "patientids_and_patient_sample_index": ids,
        "missing_data_statistics": missing_metadata,
    }

    return ret_dic



