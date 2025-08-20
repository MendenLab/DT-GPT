from neuralforecast import NeuralForecast
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from darts import TimeSeries



def get_output_for_neuralforecast_model(model, eval_manager, eval_dataset_dictionary, forecast_horizon_chunks, plot_index_demo=None,
                                        split_base_id_str="_lab_", new_target_prefix="lab_"):
    """
    Generate and evaluate predictions using a NeuralForecast model.
    
    Parameters:
    - model: Trained NeuralForecast model.
    - eval_manager: Evaluation manager instance for handling evaluations.
    - eval_dataset_dictionary: Dictionary containing dataset information.
    - forecast_horizon_chunks: Number of steps to forecast.
    - plot_index_demo: (Optional) Index to plot specific predictions.
    
    Returns:
    - full_df_targets: Concatenated target DataFrame.
    - full_df_prediction: Concatenated prediction DataFrame.
    """
    
    # Step 1: Initialize Evaluation Manager for Streaming
    eval_manager.evaluate_split_stream_start()
    
    # Step 2: Extract Columns Used for Inputs and Targets
    inputs_cols, future_known_inputs_cols, target_cols = eval_manager.get_column_usage()
    target_cols_original = target_cols.copy()
    target_cols = target_cols.copy()
    target_cols.extend(["ds", "unique_id"])  # 'ds' for date and 'unique_id' as per structure
    
    # Step 3: Prepare the Data for NeuralForecast
    # Concatenate all time series into a single DataFrame as expected by NeuralForecast
    # Assuming 'df' in eval_dataset_dictionary has columns: unique_id, ds, y, and any covariates
    model_input_df = eval_dataset_dictionary["df"].copy()

    #: get split_dates
    model_input_df['base_unique_id'] = model_input_df['unique_id'].str.rsplit(split_base_id_str, 0).str[0]
    split_dates = eval_dataset_dictionary["split_dates"].copy()
    split_dates = split_dates.rename(columns={"unique_id": "base_unique_id"})

    model_input_df = pd.merge(
        left=model_input_df,
        right=split_dates,
        on="base_unique_id",
        how="left",
    )

    #: get future DF - add unique_id and DS, filter for future, then drop 
    model_future_df = model_input_df[eval_dataset_dictionary["future_covariate_cols"] + ["unique_id", "ds", "split_date"]].copy()
    model_future_df = model_future_df[model_future_df["ds"] > model_future_df["split_date"]]
    model_future_df = model_future_df.drop(columns=["split_date"])
    
    #: filter so input is only history
    model_input_df = model_input_df[model_input_df["ds"] <= model_input_df["split_date"]]
    model_input_df = model_input_df.drop(columns=["split_date", "base_unique_id"])
    
    # Ensure 'unique_id', 'ds', and 'y' columns are present
    assert "unique_id" in model_input_df.columns, "unique_id column is missing in the dataset."
    assert "ds" in model_input_df.columns, "ds (date) column is missing in the dataset."
    assert "y" in model_input_df.columns, "y (target) column is missing in the dataset."

    
    # Process model specific stuff
    can_use_future = model.models[0].EXOGENOUS_FUTR
    can_use_static = model.models[0].EXOGENOUS_STAT
    can_use_past = model.models[0].EXOGENOUS_HIST
    if not can_use_future:
        model_future_df = None
        logging.info("Model does not use future covariates. Future covariates will not be used.")
        if not can_use_past:
            model_input_df = model_input_df[["ds", "unique_id", "y"]]
            logging.info("Model does not use past covariates. Past covariates will not be used.")
    
    # Step 4: Run Predictions Using the NeuralForecast Model
    logging.info("Running predictions using the NeuralForecast model...")
    predictions_df = model.predict(df=model_input_df,
                                   futr_df=model_future_df,
                                   static_df=eval_dataset_dictionary["static_df"] if can_use_static else None,)
    
    # Step 6: Plot Predictions if Requested
    if plot_index_demo is not None:
        demo_id = model_input_df['unique_id'].unique()[plot_index_demo]
        demo_pred = predictions_df[predictions_df['unique_id'] == demo_id]
        demo_pred.plot(x='ds', y='y', title=f"Prediction for {demo_id}")
        plt.show()
    
    #: split from name the target, then melt back to wide format using columns "target_variable" and "y"
    model_column_name = [col for col in predictions_df.columns if col not in ["ds", "unique_id"]][0]
    predictions_df = predictions_df.rename(columns={model_column_name: "y"})
    predictions_df['base_unique_id'] = predictions_df['unique_id'].str.rsplit(split_base_id_str, 0).str[0]
    predictions_df['target_variable'] = new_target_prefix + predictions_df['unique_id'].str.split(split_base_id_str, 1).str[1]
    predictions_df = predictions_df.drop(columns=["unique_id"])
    predictions_wide_df = predictions_df.pivot(index=['ds', 'base_unique_id'], columns='target_variable', values='y').reset_index(drop=False)
    predictions_wide_df = predictions_wide_df.rename(columns={"ds": "date"})

    #: add nans for missing targets
    for target in target_cols_original:
        if target not in predictions_wide_df.columns:
            predictions_wide_df[target] = np.nan


    #: Step 5: Inverse Transform Predictions if Scaling was Applied
    pipeline_targets = eval_dataset_dictionary.get("pipeline_targets", None)
    if pipeline_targets:
        logging.info("Applying inverse transformation to predictions...")

        predictions_ts = TimeSeries.from_group_dataframe(
            predictions_wide_df,
            group_cols=['base_unique_id'],
            time_col='date',
            value_cols=target_cols_original  # Need to make sure order is correct
        )

        # Step 3: Apply inverse_transform
        inversed_ts = pipeline_targets.inverse_transform(predictions_ts, partial=True)

        # Step 4: Convert back to dataframe
        inversed_df = [ts.pd_dataframe().reset_index(drop=False) for ts in inversed_ts]

        # Add in base_unique_id from static_covariates from the time series
        inversed_df = [df.assign(base_unique_id=ts.static_covariates["base_unique_id"].values[0]) 
                       for df, ts in zip(inversed_df, inversed_ts)]

        # Concat
        predictions_wide_df_new = pd.concat(inversed_df, ignore_index=True, axis=0)
        predictions_wide_df = predictions_wide_df_new

        

    # Step 8: Prepare Predictions for Evaluation Manager
    # Convert predictions to the required format and send them to the evaluation manager
    for unique_id, group in predictions_wide_df.groupby('base_unique_id'):
        
        #: get patientid and patient_sample_index
        patientid = str(unique_id.split("_")[0])
        patient_sample_index = str(unique_id.split("_", maxsplit=1)[1])


        # Retrieve corresponding target DataFrame
        target_df = eval_dataset_dictionary["target_original_dfs_dic"][patientid][patient_sample_index]

        #: prepare predicted dataframe
        curr_prediction = group.copy()
        curr_prediction = curr_prediction.drop(columns=["base_unique_id"])
        curr_prediction["patientid"] = patientid
        curr_prediction["patient_sample_index"] = patient_sample_index
        dates_to_use = target_df["date"]
        curr_prediction = curr_prediction[curr_prediction["date"].isin(dates_to_use)]
        
        # Send to Evaluation Manager
        eval_manager.evaluate_split_stream_prediction(curr_prediction, target_df, patientid, patient_sample_index)
    
    # Step 9: Post-process Predictions
    logging.info("Finished generating and processing predictions.")
    
    # Step 10: Perform Full Evaluation
    full_df_targets, full_df_prediction = eval_manager.concat_eval()
    
    # Step 11: Return the Evaluation Results
    return full_df_targets, full_df_prediction






def prepare_neuralforecast_data(converted_data):
    """
    Merges the NeuralForecast-formatted data and extracts column names.

    Args:
        converted_data (dict): The output of `convert_to_neuralforecast_dataset`.

    Returns:
        tuple: A tuple containing:
            - merged_df (pd.DataFrame): The merged DataFrame.
            - target_col (str): The name of the target column ('y').
            - past_covariate_cols (list): A list of past covariate column names.
            - future_covariate_cols (list): A list of future covariate column names.
            - static_covariate_cols (list): A list of static covariate column names.
    """
    target_df = converted_data["target_df"]
    past_covariates_df = converted_data["past_covariates_df"]
    future_covariates_df = converted_data["future_covariates_df"]
    static_df = converted_data["static_df"]

    # For future use, get the last date of each unique_id in  past_covariates_df
    split_dates = past_covariates_df.groupby('unique_id')['ds'].max().reset_index()
    split_dates = split_dates.rename(columns={"ds": "split_date"})

    # 1. Merge DataFrames
    merged_df = target_df.copy()  # Start with the target data

    # 3. Merge
    if past_covariates_df is not None and not past_covariates_df.empty:
        merged_df = pd.merge(merged_df, past_covariates_df, on=["unique_id", "ds"], how="left")
    if future_covariates_df is not None and not future_covariates_df.empty:
        merged_df = pd.merge(merged_df, future_covariates_df, on=["unique_id", "ds"], how="left")
    if static_df is not None and not static_df.empty:
        merged_df = pd.merge(merged_df, static_df, on=["unique_id"], how="left")

    # Identify covariate columns
    past_covariate_cols = [col for col in merged_df.columns if col not in ["unique_id", "ds", "y"] and col in past_covariates_df.columns if past_covariates_df is not None and not past_covariates_df.empty]
    future_covariate_cols = [col for col in merged_df.columns if col not in ["unique_id", "ds", "y"] and col in future_covariates_df.columns if future_covariates_df is not None and not future_covariates_df.empty]
    static_covariate_cols = [col for col in merged_df.columns if col not in ["unique_id", "ds", "y"] and col in static_df.columns if static_df is not None and not static_df.empty]

    return merged_df, past_covariate_cols, future_covariate_cols, static_covariate_cols, split_dates



def convert_to_neuralforecast_dataset(darts_dataset, split_base_id_str="_lab_", add_target_prefix=""):
    """
    Converts a Darts dataset dictionary to a NeuralForecast compatible format.

    Args:
        darts_dataset (dict): A dictionary containing Darts TimeSeries objects
                            for target, past covariates, and static covariates,
                            as output by the `convert_to_darts_dataset` function.

    Returns:
        dict: A dictionary containing the data in NeuralForecast format
            with keys: 'df', 'static_df', 'past_covariates_df', 'future_covariates_df' and the original dictionaries other parts.
    """

    target_ts = darts_dataset["target_ts"]
    past_covariate_ts = darts_dataset["past_covariate_ts"]
    future_covariates_ts = darts_dataset["future_covariates_ts"]
    patientids_and_patient_sample_index = darts_dataset["patientids_and_patient_sample_index"]

    target_column_mapping = None
    freq = None
    target_cols = None

    # 1. Prepare Target Time Series Data
    df_list = []
    for ts, (patientid, patient_sample_index) in zip(target_ts, patientids_and_patient_sample_index):
        df = ts.pd_dataframe().reset_index()
        df = df.rename(columns={"date": "ds"})
        
        # Add unique ID column
        unique_id = f"{patientid}_{patient_sample_index}"
        df["unique_id"] = unique_id
        
        # Rename target columns to y
        if target_cols is None:
            target_cols = [col for col in df.columns if col not in ["ds", "unique_id"]]

        # Reorder columns and save
        df = df[["unique_id", "ds"] + [col for col in df.columns if col not in ["unique_id", "ds"]]]
        df_list.append(df)

        # Get frequency from first TimeSeries
        if freq is None:
            freq = ts.freq_str

    target_df = pd.concat(df_list, ignore_index=True)

    # 2. Prepare Past Covariates Time Series Data
    past_covariates_df_list = []
    for ts, (patientid, patient_sample_index) in zip(past_covariate_ts, patientids_and_patient_sample_index):
        df = ts.pd_dataframe().reset_index()
        df = df.rename(columns={"date": "ds"})
        
        # Add unique ID column
        unique_id = f"{patientid}_{patient_sample_index}"
        df["unique_id"] = unique_id
        past_covariates_df_list.append(df)
    
    past_covariates_df = pd.concat(past_covariates_df_list, ignore_index=True)

    # 3. Prepare Future Covariates Time Series Data
    future_covariates_df_list = []
    for ts, (patientid, patient_sample_index) in zip(future_covariates_ts, patientids_and_patient_sample_index):
        df = ts.pd_dataframe().reset_index()
        df = df.rename(columns={"date": "ds"})
        
        # Add unique ID column
        unique_id = f"{patientid}_{patient_sample_index}"
        df["unique_id"] = unique_id
        future_covariates_df_list.append(df)
    
    future_covariates_df = pd.concat(future_covariates_df_list, ignore_index=True)
    
    # 4. Prepare Static Covariates Data
    static_df_list = []
    for ts, (patientid, patient_sample_index) in zip(target_ts, patientids_and_patient_sample_index):
        static_df = ts.static_covariates.reset_index(drop=True)
        
        # Add unique ID column
        unique_id = f"{patientid}_{patient_sample_index}"
        static_df["unique_id"] = unique_id
        static_df_list.append(static_df)

    static_df = pd.concat(static_df_list, ignore_index=True).drop_duplicates(subset=["unique_id"], keep='first')

    # 5. Return NeuralForecast compatible dictionary
    ret_dic = {
        "target_df": target_df,
        "past_covariates_df": past_covariates_df,
        "future_covariates_df": future_covariates_df,
        "static_df": static_df,
    }

    merged_full_df, past_covariate_cols, future_covariate_cols, static_covariate_cols, split_dates = prepare_neuralforecast_data(ret_dic)

    #: 6. transform into many series (due to multivariate nature) with the unique_id appended with the column name
    #: use the list of target columns from target_cols
    #: add input to static_covariate to identify which variables to predict

    # Melt the target dataframe to long format
    id_columns = ["unique_id", "ds"]

    # Identify extra columns not in id_columns or target_cols
    extra_columns = [col for col in merged_full_df.columns 
                    if col not in id_columns + target_cols]

    # Combine all id_vars
    all_id_vars = id_columns + extra_columns

    # Perform the melt
    merged_full_df_univariate = merged_full_df.melt(
        id_vars=all_id_vars,
        value_vars=target_cols,
        var_name="target",
        value_name="y",
    )

    # Append target name to unique_id
    merged_full_df_univariate["unique_id"] = merged_full_df_univariate.apply(
        lambda row: f"{row['unique_id']}_{add_target_prefix}{row['target']}", axis=1
    )

    # Reorder to be more readable
    merged_full_df_univariate = merged_full_df_univariate[["unique_id", "ds", "y"] + [col for col in merged_full_df_univariate.columns if col not in ["unique_id", "ds", "y"]]]

    # Drop the 'target' column as it's now part of unique_id
    merged_full_df_univariate = merged_full_df_univariate.drop(columns=["target"])

    # Create target_column_mapping
    target_column_mapping = {
        uid: uid.split("_")[-1] for uid in merged_full_df_univariate["unique_id"]
    }

    #: Expand static_df to match the univariate unique_id
    # Expand static_df to match the univariate unique_id
    # Extract the base unique_id by removing the target suffix
    merged_full_df_univariate['base_unique_id'] = merged_full_df_univariate['unique_id'].str.rsplit(split_base_id_str, 0).str[0]
    
    # Merge the expanded unique_id with the static covariates using the base_unique_id
    static_expanded_df = pd.merge(
        merged_full_df_univariate[['unique_id', 'base_unique_id']].drop_duplicates(),
        static_df,
        left_on='base_unique_id',
        right_on='unique_id',
        how='left'
    ).drop(columns=['base_unique_id', 'unique_id_y']).rename(columns={'unique_id_x': 'unique_id'})
    
    merged_full_df_univariate = merged_full_df_univariate.drop(columns=['base_unique_id'])

    #: add to static_df the column indicating which variable is being predicted
    static_expanded_df["target"] = "lab_" + static_expanded_df["unique_id"].apply(lambda x: x.split("lab_")[-1])

    # Convert target to one hot encoding
    static_expanded_df = pd.get_dummies(static_expanded_df, columns=["target"], drop_first=False)

    #: drop sequences that are purely 0 the entire time for y (since they're fully imputeed), since they're also not interested in test time
    # Identify unique_ids where all y values are 0
    unique_ids_to_drop = merged_full_df_univariate.groupby('unique_id')['y'].apply(lambda x: x.abs().sum())
    unique_ids_to_drop = unique_ids_to_drop[unique_ids_to_drop == 0].index.tolist()
    
    # Drop these unique_ids from the main dataframe
    merged_full_df_univariate = merged_full_df_univariate[~merged_full_df_univariate['unique_id'].isin(unique_ids_to_drop)]
    
    # Drop the corresponding entries from static_expanded_df
    static_expanded_df = static_expanded_df[~static_expanded_df['unique_id'].isin(unique_ids_to_drop)]

    # Fill in any missing values in the static covariates and merged_full_df_univariate with 0 (coming from future covariates)
    static_expanded_df = static_expanded_df.fillna(0)
    merged_full_df_univariate = merged_full_df_univariate.fillna(0)


    # For meta data, process the original target DFs to be easily accessible (for val/test sets)
    target_original_dfs = {}
    if "target_original_dfs" in darts_dataset and darts_dataset["target_original_dfs"] is not None:
        for target_df in darts_dataset["target_original_dfs"]:
            patientid = str(target_df["patientid"].iloc[0])
            patient_sample_index = str(target_df["patient_sample_index"].iloc[0])
            if patientid not in target_original_dfs:
                target_original_dfs[patientid] = {}
            target_original_dfs[patientid][patient_sample_index] = target_df

    
    final_ret_dic = {
        "df": merged_full_df_univariate,
        "static_df": static_expanded_df,
        "past_covariate_cols": past_covariate_cols,
        "future_covariate_cols": future_covariate_cols,
        "static_covariate_cols": static_covariate_cols,
        "target_column_mapping": target_column_mapping,
        "split_dates": split_dates,
        "freq": freq,
        "target_original_dfs_dic": target_original_dfs,
        **{k : v for k, v in darts_dataset.items() if k not in ["target_ts", "past_covariate_ts", "future_covariates_ts", "target_original_dfs"]} # Return the other elements
    }

    return final_ret_dic
