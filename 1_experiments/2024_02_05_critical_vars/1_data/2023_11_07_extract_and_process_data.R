library(ggplot2)

# This is a custom R library to process Flatiron data into a format suitable for ML
# Please reach out to us in case you need access to this library
library(FlatironTimeSeries)   


# 1. Connect to database
# 2. Determine hyperparameters for event extraction, e.g. using the functions `plot_top_diagnoses_excluding_cancer` and `plot_top_labs`
# 3. Extract events using the function `extract_events`
# 4. Extract constant data using the function `extract_constant`
# 5. Extract molecular events using the function `extract_molecular`
# 6. Enrich constant with further statistics using e.g. `get_further_statistics_for_constants_table`
# 7. Filter out patients
# 8. Ensure consistent filtering, e.g. using `remove_patients_not_in_constants_table`
# 9. Determine number of days for a bucket, in which all events in same bucket are merged together, e.g. using `plot_distribution_of_time_between_drug_administrations` and `plot_distribution_of_time_between_all_events`
# 10. Merge all events in one bucket using `group_all_events_into_bucket`
# 11. Explore which variables have the most change over time, using e.g. `get_all_variables_changing_on_average_from_one_visit_to_next_per_patient`, `get_r2_for_copy_forward_across_variables` and `plot_changes_in_patients_across_therapies`
# 12. Assign patients to stratified splits (e.g. for training/test sets) using `assign_patients_to_groups`
# 13. Save tables to scratch space to ensure reusability, using `save_table_to_scratch_space` and `save_local_table_to_scratch_space`
# 14. Pivot into wide table using `pivot_molecular_and_events_table_wide`, where each row is one visit, with one column per variable
# 15. Download data and split into one file per patient using `download_table` and `split_table_by_patientid`
# 16. Download description data of variables using e.g. `get_descriptive_name_mapping_events`





##################################################### 1) DB CONNECTION #####################################################



db <- RocheData::get_data(data = "flatiron_se_ptcg")
db$username <- Sys.getenv("DB_USERNAME") # <-------------------------------------- Set to your Redshift credentials here
db$password <- Sys.getenv("DB_PASSWORD")
db$schema <- "flatiron_se_ptcg_nsclc_sdm_2023_09_30"
db$connect()


con <- db$get_connection()


scratch_space_name <- "scr_scr_464_danda_exploration"   # <----------------------- Set to your scratch space here


##################################################### 2) + 3) MASTER.EVENTS #####################################################



# First find hyperparameters by plotting
diagnosis_top_k <- 50
plot_top_diagnoses_excluding_cancer(con, db, 20)


lab_top_k <- 80
plot_top_labs(con, db, 20)



# Run extractions
diagnosis_disease_name <- "non small cell lung cancer"
extracted_events <- extract_events(con, db, diagnosis_top_k, lab_top_k, diagnosis_disease_name)

master.events_long <- extracted_events[["master_events"]]
master.statistics <- extracted_events[["master_statistics"]]



##################################################### 4) MASTER.CONSTANT #####################################################


# Load constant data
extracted_constant <- extract_constant(con, db)

master.constant <-extracted_constant[["master_constant"]]
master.statistics <- master.statistics %>%
  dplyr::union_all(extracted_constant[["master_statistics"]])


##################################################### 5) MASTER.MOLECULAR #####################################################



# Load molecular data
molecular_extracted <- extract_molecular(con, db)

master.molecular_long <- molecular_extracted[["master_molecular"]]
master.statistics <- master.statistics %>%
  dplyr::union_all(molecular_extracted[["master_statistics"]])


##################################################### 6) GET FURTHER STATISTICS FOR CONSTANTS TABLE#####################################################


master.constant <- get_further_statistics_for_constants_table(con, db, master.constant,
                                                              master.events_long,
                                                              master.molecular_long)



##################################################### 7) + 8) FILTER OUT PATIENTS WITH TOO FEW DATA #####################################################


# Do plots to figure out optimal filtering values
plot_nr_visits_vs_observed(master.constant, max_val=30, step_size=1)
plot_nr_visits_vs_patients(master.constant, max_val=50, cutoff_value=NA)


explore.num_patients_no_drug_visits <- master.constant %>%
  dplyr::filter(num_drug_visits == 0) %>%
  dplyr::summarise(n = n()) %>%
  dplyr::pull(n)






# Do actual filtering
master.constant <- master.constant %>%
  dplyr::filter(num_drug_visits > 0) %>%
  dplyr::compute()




# Post filtering processing across all tables
explore.num_patients_after_filtering <- master.constant %>%
  dplyr::summarise(n = n()) %>%
  dplyr::pull(n)


filtered_results <- remove_patients_not_in_constants_table(con, db,
                                                           master.constant,
                                                           master.events_long,
                                                           master.molecular_long)


master.constant <- filtered_results[["master_constant"]]
master.events_long <- filtered_results[["master_events_long"]]
master.molecular_long <- filtered_results[["master_molecular_long"]]




# add statistics to master.statistics file
post_filter_nr_rows_events_long <- master.events_long %>%
  dplyr::summarise(n = n()) %>%
  dplyr::pull(n)

post_filter_nr_patients <- master.constant %>%
  dplyr::summarise(n = n()) %>%
  dplyr::pull(n)

post_filter_nr_rows_molecular_long <- master.molecular_long %>%
  dplyr::summarise(n = n()) %>%
  dplyr::pull(n)



master.statistics <- rbind(master.statistics, c("master_events_long", "nr_rows_post_filter", post_filter_nr_rows_events_long, "# rows in  master.events long table after filtering"))
master.statistics <- rbind(master.statistics, c("master_constants", "nr_patients_post_filter", post_filter_nr_patients, "# patients overall after filtering"))
master.statistics <- rbind(master.statistics, c("master_molecular_long", "nr_rows_post_filter", post_filter_nr_rows_molecular_long, "# rows in master.molecular after filtering"))





#################################################### 9) + 10) DO MERGING OF DATES INTO BUCKETS ##########################################


# Do it before pivot, as much more efficient


# find optimal hyperparameter for merging of dates
plot_distribution_of_time_between_drug_administrations(master.events_long)


plot_distribution_of_time_between_all_events(master.events_long)

# Result: 7 days seems like a good choice




#: do merging of dates
master.events_long <- group_all_events_into_bucket_use_bucket_date(master.events_long, window_size=7)

master.events_long <- master.events_long %>%
  dplyr::mutate(date = as.Date(date)) %>%
  dplyr::compute()


# Plot to see difference
plot_distribution_of_time_between_all_events(master.events_long)



# Save statistics
post_date_merging_nr_rows <- master.events_long %>%
  dplyr::summarise(n = n()) %>%
  dplyr::pull(n)

master.statistics <- rbind(master.statistics, c("master_events_long", "nr_rows_post_date_merging", post_date_merging_nr_rows, "# rows in master.events long table after merging dates"))





#################################################### 11) EXPLORE WHICH VARIABLES HAVE THE MOST DIFFERENCE FROM VISIT TO VISIT ##########################################



# Get general statistics about how the variables change across patients
variable_changing_statistics <- get_all_variables_changing_on_average_from_one_visit_to_next_per_patient(master.events_long)


#: check which variables have lowest R^2 for copy forward
copy_forward_r2 <- get_r2_for_copy_forward_across_variables(master.events_long)

variable_changing_statistics <- variable_changing_statistics %>%
  dplyr::left_join(copy_forward_r2, by = c("event_category", "event_name", "event_descriptive_name"))


variable_changing_statistics_labs <- variable_changing_statistics %>%
  dplyr::filter(event_category == "lab") %>%
  dplyr::arrange(r_squared)


variable_changing_statistics_labs <- variable_changing_statistics_labs %>%
  dplyr::mutate(custom_reward_function = r_squared * total_obs_of_variable) %>%
  dplyr::select(event_category, event_name, event_descriptive_name, total_obs_of_variable, r_squared, custom_reward_function, overall_mean, overall_std)






#: explore top changing variables based on highest custom_cost_function via plots
curr_event_category <- "lab"
curr_event_name <- "26499_4"
cycle_time_to_filter_for <- 1
nr_of_therapies_to_plot <- 3

plots.changes <- plot_changes_in_patients_across_therapies(con, db, master.events_long,
                                                           curr_event_category,
                                                           curr_event_name,
                                                           top_n_lots_to_plot = nr_of_therapies_to_plot,
                                                           cycle_time_to_filter_for = cycle_time_to_filter_for,
                                                           y_lim_lower = 0,
                                                           y_lim_higher = 30,
                                                           alpha_patients_trajectories = 0.09,
                                                           y_tick_distance = 1)

plots.changes[["therapy_averaged"]]

plots.changes[["patient_level"]]






##################################################### 12) ASSIGN PATIENTS TO SPLITS  ###########################################################


num_splits = 10
stratify_columns_numeric = c("nr_of_obs_per_visit", "num_drug_visits")
stratify_columns_categorical = c("groupstage", "smokingstatus")

master.constant <- assign_patients_to_groups(master.constant,
                                             num_splits = num_splits,
                                             stratify_columns_numeric = stratify_columns_numeric,
                                             stratify_columns_categorical = stratify_columns_categorical,
                                             nr_iterations = 10000)



##################################################### 13) SAVE STATISTICS FILE  ###########################################################


# save master.statistics to scratch space
master.statistics <- save_local_table_to_scratch_space(local_table = master.statistics,
                                                       con = con,
                                                       scratch_space_name = scratch_space_name,
                                                       new_table_name = "master_statistics")

# Save all other also to scratch space
master.events_long <- save_table_to_scratch_space(master.events_long, scratch_space_name, "master_events_long", replace = TRUE)
master.molecular_long <- save_table_to_scratch_space(master.molecular_long, scratch_space_name, "master_molecular_long", replace = TRUE)
master.constant <- save_table_to_scratch_space(master.constant, scratch_space_name, "master_constant", replace = TRUE)



##################################################### 14) PIVOT WIDER  ###########################################################



# NOTE: Uncomment next lines only if you really need to pivot the table, as it will take some time!


pivoted_results <- pivot_molecular_and_events_table_wide(con, db,
                                                         master.constant,
                                                         master.events_long,
                                                         master.molecular_long,
                                                         pivot_molecular=FALSE)  # Don't pivot molecular as it takes up too much disk space in wide format


master.events <- pivoted_results[["master_events"]]


# Save to scratch space
master.events <- save_table_to_scratch_space(master.events, scratch_space_name, "master_events", replace = TRUE)





##################################################### 15) SAVE LOCALLY & SPLIT  ###########################################################



# NOTE: Uncomment next lines and set to correct paths, as these are large files and the processes will take some time!


master.constant <- download_table(master.constant, "~/uc2_nsclc/2_experiments/2023_11_07_neutrophils/1_data/constant.csv")
master.events <- download_table(master.events, "~/uc2_nsclc/2_experiments/2023_11_07_neutrophils/1_data/events.csv")
master.molecular_long <- download_table(master.molecular_long, "~/uc2_nsclc/2_experiments/2023_11_07_neutrophils/1_data/molecular_long.csv")
master.events_long <- download_table(master.events_long, "~/uc2_nsclc/2_experiments/2023_11_07_neutrophils/1_data/events_long.csv")
master.statistics <- download_table(master.statistics, "~/uc2_nsclc/2_experiments/2023_11_07_neutrophils/1_data/statistics.csv")

new_col_events <- split_table_by_patientid(master.events, "~/uc2_nsclc/2_experiments/2023_11_07_neutrophils/1_data/patient_events/")

# Add to constant to see  which file is which
new_col_events <- new_col_events %>%
  dplyr::rename("path_to_events_file" = "path")

master.constant <- master.constant %>%
  dplyr::left_join(new_col_events, by = "patientid")




#new_col_molecular <- split_table_by_patientid(master.molecular_long, "~/uc2_nsclc/2_experiments/2023_11_07_neutrophils/1_data/patient_events/")


#new_col_molecular <- new_col_molecular %>%
#  dplyr::rename("path_to_molecular_file" = "path")

#%>%  dplyr::left_join(new_col_molecular, by = "patientid")




##################################################### 16) GET MAPPING FILES FOR DT-GPT  ###########################################################



#molecular_mapping <- get_descriptive_name_mapping_molecular(master.molecular_long)


events_mapping <- get_descriptive_name_mapping_events(master.events_long)


# handle ecog duplicate
events_mapping <- events_mapping %>%
  dplyr::mutate(event_descriptive_name = dplyr::case_when(
    event_descriptive_name == "ECOG Baseline" ~ "ECOG",
    TRUE ~ event_descriptive_name
  )) %>%
  dplyr::group_by(event_name, event_category, event_descriptive_name) %>%
  dplyr::filter(dplyr::row_number() == 1L) %>%
  dplyr::ungroup()
  
  


#: put it directly into correct format
events_mapping <- events_mapping %>%
  dplyr::rename("original_column_names" = "event_name",
                "group" = "event_category",
                "descriptive_column_name" = "event_descriptive_name")


write.csv(events_mapping, "~/uc2_nsclc/2_experiments/2023_11_07_neutrophils/1_data/column_descriptive_name_mapping.csv")




#################################################### Assign to train/validation/test in constant and save locally   ###########################################################



#: assign groups
master.constant <- master.constant %>%
  dplyr::mutate(dataset_split = dplyr::case_when(
    dataset_split_id == 1 ~ "TEST",
    dataset_split_id == 2 ~ "VALIDATION",
    TRUE ~ "TRAIN"
  ))



#: save
write.csv(master.constant, "~/uc2_nsclc/2_experiments/2023_11_07_neutrophils/1_data/constant.csv")




################################################ GET LOT TABLE ###################################


lot_table <- con %>%
  dplyr::tbl(dbplyr::in_schema(db$schema, "lineoftherapy")) %>%
  dplyr::collect()



#: save
write.csv(lot_table, "~/uc2_nsclc/2_experiments/2023_11_07_neutrophils/1_data/line_of_therapy.csv")









