import __init__
from abc import ABC, abstractmethod
import random
import logging



class MatchingBase(ABC):

    def __init__(self, list_of_training_events, list_of_training_meta_data):
        # list_of_training_events - list of entries of form:
        # (curr_const_row, true_events_input, true_future_events_input, target_dataframe)

        # list_of_training_meta_data - corresponding list of meta data in dic form
        pass


    @abstractmethod
    def match(self, constant_df, input_df, future_known_inputs_df, meta_data):
        # Take as input the dataframes of patient for whome we want to retrieve matching patients
        # Return as list of dataframes of matching patients from list_of_training_events, + matching meta_data
        pass







class Match_LoT_Age_Linename(MatchingBase):


    def __init__(self, list_of_training_events, list_of_training_meta_data, 
                 min_num_matching_patients, max_num_matching_patients):
        
        super().__init__(list_of_training_events, list_of_training_meta_data)

        self.list_of_training_events = list_of_training_events
        self.list_of_training_meta_data = list_of_training_meta_data

        # Min and max nr of patient matches needed
        self.min_num_matching_patients = min_num_matching_patients
        self.max_num_matching_patients = max_num_matching_patients
        
        #: do preprocessing
        self.dic_of_training_events = {}
        self._setup_training_events()
        logging.info("Finished setting up Linename, LoT & Age matching manager")



    def _setup_training_events(self):
        
        #: go through all training events and build dictionary
        #: dic of form {{line_name} : {line_number: {year_of_birth: [(patientid, <list of index in self.list_of_training_events>)] } } }

        for idx in range(len(self.list_of_training_meta_data)):

            curr_line = self.list_of_training_meta_data[idx]["line_name"]
            curr_line_number = self.list_of_training_meta_data[idx]["line_number"]
            curr_year_of_birth = self.list_of_training_events[idx][0]["birthyear"].tolist()[0]
            curr_patientid = self.list_of_training_events[idx][0]["patientid"].tolist()[0]

            if curr_line not in self.dic_of_training_events:
                self.dic_of_training_events[curr_line] = {}

            if curr_line_number not in self.dic_of_training_events[curr_line]:
                self.dic_of_training_events[curr_line][curr_line_number] = {}
            
            if curr_year_of_birth not in self.dic_of_training_events[curr_line][curr_line_number]:
                self.dic_of_training_events[curr_line][curr_line_number][curr_year_of_birth] = []
            
            # Add to dic
            self.dic_of_training_events[curr_line][curr_line_number][curr_year_of_birth].append((curr_patientid, idx))
            

    def _get_closest_year(self, ret_matches, curr_line, curr_line_number, curr_patientid, curr_year_of_birth, num_more_matches_needed):

        # Match for closest year, if possible

        available_years = list(self.dic_of_training_events[curr_line][curr_line_number].keys())
        relative_years = [abs(x - curr_year_of_birth) for x in available_years]
        relative_years_sorted = sorted(zip(relative_years, available_years))
        max_num_matches_to_take = self.max_num_matching_patients - num_more_matches_needed

        for _, year in relative_years_sorted:
            if num_more_matches_needed > 0:
                curr_potential_matches = self.dic_of_training_events[curr_line][curr_line_number][year]
                curr_potential_matches = [x for x in curr_potential_matches if x[0] != curr_patientid]

                if len(curr_potential_matches) > max_num_matches_to_take:
                    random.shuffle(curr_potential_matches)
                    curr_potential_matches = curr_potential_matches[:max_num_matches_to_take]
                
                curr_potential_matches_idx = [x[1] for x in curr_potential_matches]
                ret_matches.extend(curr_potential_matches_idx)
                num_more_matches_needed -= len(curr_potential_matches_idx)
                max_num_matches_to_take -= len(curr_potential_matches_idx)

        return ret_matches, num_more_matches_needed


    def match(self, constant_df, input_df, future_known_inputs_df, meta_data):
        
        #: get line name, line number, year of birth
        curr_line = meta_data["line_name"]
        curr_line_number = meta_data["line_number"]
        curr_year_of_birth = constant_df["birthyear"].tolist()[0]
        curr_patientid = constant_df["patientid"].tolist()[0]

        #: if exact match exists, return random subset of those
        if curr_line in self.dic_of_training_events and curr_line_number in self.dic_of_training_events[curr_line] and curr_year_of_birth in self.dic_of_training_events[curr_line][curr_line_number]:

            #: get random subset of exact matches            
            exact_matches = self.dic_of_training_events[curr_line][curr_line_number][curr_year_of_birth]
            exact_matches_no_curr_patient = [x[1] for x in exact_matches if x[0] != curr_patientid]
            random.shuffle(exact_matches_no_curr_patient)
            ret_matches = exact_matches_no_curr_patient[:self.max_num_matching_patients]

        else:

            # Else kick off with empty
            ret_matches = []

        #: if not enough matches, expand ret_matches by going up the dictionary and selecting closest match
        num_more_matches_needed = self.min_num_matching_patients - len(ret_matches)

        #: check for closest years
        if num_more_matches_needed > 0:
            if curr_line in self.dic_of_training_events and curr_line_number in self.dic_of_training_events[curr_line]:

                #: call function
                ret_matches, num_more_matches_needed = self._get_closest_year(ret_matches, curr_line, curr_line_number, curr_patientid, curr_year_of_birth, num_more_matches_needed)


        #: check for closest line number
        if num_more_matches_needed > 0:
            if curr_line in self.dic_of_training_events:

                available_line_numbers = list(self.dic_of_training_events[curr_line].keys())
                relative_line_numbers = [abs(x - curr_line_number) for x in available_line_numbers]
                relative_line_numbers_sorted = sorted(zip(relative_line_numbers, available_line_numbers))

                for _, line_number in relative_line_numbers_sorted:
                    
                    #: go iteratively through the years
                    ret_matches, num_more_matches_needed = self._get_closest_year(ret_matches, curr_line, line_number, curr_patientid, curr_year_of_birth, num_more_matches_needed)
            

        #: final backup - randomly sample from all patients
        if num_more_matches_needed > 0:

            # logging.info("Picking " + str(num_more_matches_needed) + " random matches for patient")
            
            curr_random_sample = random.sample(list(range(len(self.list_of_training_events))), num_more_matches_needed)
            sample_no_curr_patientid = [x for x in curr_random_sample if self.list_of_training_events[x][0]["patientid"].tolist()[0] != curr_patientid]
            ret_matches.extend(sample_no_curr_patientid)

            #if len(ret_matches) == 0:
            #    logging.info("Edge case: no matches for current patient: " + str(curr_patientid) + " and lot nr: " + str(curr_line_number))


        #: get data and return
        ret_matches_data = [(self.list_of_training_events[x], self.list_of_training_meta_data[x]) for x in ret_matches]
        return ret_matches_data



