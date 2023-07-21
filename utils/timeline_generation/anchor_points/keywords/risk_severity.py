import pandas as pd
import re


class KeywordsSuicideRiskSeverity:
    def __init__(self):

        r = "../" * 100  # Just to ensure you get to the root directory

        # Paths
        self.path_all_csvs = (
            r + "storage/ahills/datasets/suicide_risk_severity_lexicon/"
        )

        # List containing paths to each csv file in the lexicon.
        self.path_list_csvs = [
            self.path_all_csvs + file_name + ".csv"
            for file_name in [
                "suicidal_attempt",
                "suicidal_behavior",
                "suicidal_ideation",
                "suicidal_indicator",
            ]
        ]

        self.lexicon = self.load_lexicons(post_process=True)

    def input_text_contains_suicide_risk(self, input_text, post_process=True):
        """
        Takes an input string, and returns a Boolean which is True when
        it contains a keyword within the suicide risk severity lexicon.
        """
        if post_process:
            input_text = self.post_process_string(input_text)

        # Convert into list of substrings
        substring_list = list(self.lexicon["keywords"])

        # Check if any keyword is present in the input string
        contains_keyword = any(substring in input_text for substring in substring_list)

        return contains_keyword

    def load_lexicons(self, post_process=True):
        """
        Returns a DataFrame with columns 'keywords' and 'risk_type'. The
        'keywords' column contains
        """
        csv_paths = self.path_list_csvs

        initialized = False
        for file_path in csv_paths:
            lexicon = pd.read_csv(file_path, header=None)
            lexicon = lexicon.T
            risk_type = file_path.split("_")[-1][:-4]  # Get risk type from file name
            lexicon["risk_type"] = risk_type
            lexicon = lexicon.rename(columns={0: "keywords"})

            if not initialized:
                full_lexicon = lexicon
                initialized = True
            else:
                full_lexicon = pd.concat([full_lexicon, lexicon], axis=0)

        # Lower case
        if post_process:
            full_lexicon = self.post_process_lexicon(full_lexicon)

        return full_lexicon

    def post_process_lexicon(self, lexicon):
        # Lower case, remove punctuaton
        lexicon["keywords"] = lexicon["keywords"].apply(
            lambda x: self.post_process_string(x)
        )

        # Remove some words (in this case empty string)
        lexicon = lexicon[lexicon["keywords"] != ""]

        # Reset index
        lexicon = lexicon.reset_index().drop("index", axis=1)

        return lexicon

    def post_process_string(self, input_string):
        output_string = str(input_string).lower()  # Lower case
        # output_string = re.sub(r"[^\w\s]", "", str(output_string))  # remove punctuaton
        output_string = str.strip(output_string)  # Remove whitespace

        return output_string
