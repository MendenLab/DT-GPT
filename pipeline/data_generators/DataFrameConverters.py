import __init__  # Do all imports
import logging
import wandb
import pandas as pd
import json
import numpy as np
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from abc import ABC, abstractmethod
import re
from datetime import timedelta, date
import random
import traceback
import os


#: abstract base class
class DataFrameConverter(ABC):

    skip_columns = ["date", "patientid", "patient_sample_index", "X", "X.1", "X.2", "Unnamed: 0", "X.5"]

    constant_columns_mapping = {
            "birthyear" : "birth year",
            "gender" : "gender",
            "sesindex2015_2019" : "ses index",
            "isadvanced" : "is cancer advanced",
            "histology" : "histology",
            "groupstage" : "cancer stage",
            "smokingstatus" : "smoking status",
            "ethnicity": "ethnicity",
        }

    @staticmethod
    @abstractmethod
    def convert_df_to_strings(column_name_mapping, constants_row, true_events_input, true_future_events_input, true_events_output):
        pass

    @staticmethod
    @abstractmethod
    def convert_from_strings_to_df(column_name_mapping, string_output, starting_date):
        pass
    

