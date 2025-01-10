import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import cptac
import os
import torch

class data_collection:

    def __init__(self):

        self.lu = None         # Luad
        self.br = None         # Brca
        self.hn = None         # Hnscc
        self.gbm = None        # Gbm
        self.ov = None         # Ovarian
        self.cc = None         # Ccrcc
        self.en = None         # Endometrial
        
        self.all_data = None
        self.all_data_cleaned = None

        self.transcriptome = None
        self.proteome = None
        self.labels = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def download_data(self):
        print("Downloading CPTAC datasets. This may take a while depending on network speed...")
        cptac.download('luad')
        cptac.download('brca')
        cptac.download('hnscc')
        cptac.download('gbm')
        cptac.download('ovarian')
        cptac.download('ccrcc')
        cptac.download('endometrial')
        print("All datasets have been downloaded.")
    

    def load_data(self):
        print("Loading CPTAC datasets into memory...")
        self.lu = cptac.Luad()
        self.br = cptac.Brca()
        self.hn = cptac.Hnscc()
        self.gbm = cptac.Gbm()
        self.ov = cptac.Ovarian()
        self.cc = cptac.Ccrcc()
        self.en = cptac.Endometrial()
        print("All datasets have been loaded.")

    def combine_data(self, individual=False):
        """
        Combines transcriptomics and proteomics data for each cancer type, 
        labels them, and concatenates everything into a single DataFrame.
        """
        print("Combining transcriptomics and proteomics data...")

        # Retrieve transcriptomics and proteomics
        lu_transcriptome = self.lu.get_transcriptomics()
        lu_proteome = self.lu.get_proteomics()

        br_transcriptome = self.br.get_transcriptomics()
        br_proteome = self.br.get_proteomics()

        hn_transcriptome = self.hn.get_transcriptomics()
        hn_proteome = self.hn.get_proteomics()

        gbm_transcriptome = self.gbm.get_transcriptomics()
        gbm_proteome = self.gbm.get_proteomics()

        ov_transcriptome = self.ov.get_transcriptomics()
        ov_proteome = self.ov.get_proteomics()

        cc_transcriptome = self.cc.get_transcriptomics()
        cc_proteome = self.cc.get_proteomics()

        en_transcriptome = self.en.get_transcriptomics()
        en_proteome = self.en.get_proteomics()

        cancer_labels = ['Lung', 'Breast', 'Head and Neck', 'Brain', 'Ovarian', 'Renal Cell', 'Endometrial']

        if individual:

            label_to_id = {label: idx for idx, label in enumerate(cancer_labels)}

            lu_transcriptome['Cancer'] = cancer_labels[0]
            br_transcriptome['Cancer'] = cancer_labels[1]
            hn_transcriptome['Cancer'] = cancer_labels[2]
            gbm_transcriptome['Cancer'] = cancer_labels[3]
            ov_transcriptome['Cancer'] = cancer_labels[4]
            cc_transcriptome['Cancer'] = cancer_labels[5]
            en_transcriptome['Cancer'] = cancer_labels[6]

            lu_proteome['Cancer'] = cancer_labels[0]
            br_proteome['Cancer'] = cancer_labels[1]
            hn_proteome['Cancer'] = cancer_labels[2]
            gbm_proteome['Cancer'] = cancer_labels[3]
            ov_proteome['Cancer'] = cancer_labels[4]
            cc_proteome['Cancer'] = cancer_labels[5]
            en_proteome['Cancer'] = cancer_labels[6]

            self.transcriptome = pd.concat([lu_transcriptome, br_transcriptome, hn_transcriptome, gbm_transcriptome,
                                    ov_transcriptome, cc_transcriptome, en_transcriptome], axis = 0)
            self.proteome = pd.concat([lu_proteome, br_proteome, hn_proteome, gbm_proteome,
                                    ov_proteome, cc_proteome, en_proteome], axis = 0)
            
            self.transcriptome = self.transcriptome.dropna(subset=['Cancer'])
            self.proteome = self.proteome.dropna(subset=['Cancer'])

            all_labels = self.transcriptome['Cancer'].values

            numeric_labels = [label_to_id[label] for label in all_labels]
            self.labels = torch.tensor(numeric_labels, dtype=torch.long).to(self.device)


        # Concatenate transcriptome and proteome for each dataset
        lu_combined = pd.concat([lu_transcriptome, lu_proteome], axis=1)
        br_combined = pd.concat([br_transcriptome, br_proteome], axis=1)
        hn_combined = pd.concat([hn_transcriptome, hn_proteome], axis=1)
        gbm_combined = pd.concat([gbm_transcriptome, gbm_proteome], axis=1)
        ov_combined = pd.concat([ov_transcriptome, ov_proteome], axis=1)
        cc_combined = pd.concat([cc_transcriptome, cc_proteome], axis=1)
        en_combined = pd.concat([en_transcriptome, en_proteome], axis=1)

        # Handle duplicates where needed
        hn_combined = hn_combined.loc[:, ~hn_combined.columns.duplicated()]
        en_combined = en_combined.loc[:, ~en_combined.columns.duplicated()]

        # Optionally rename columns with a prefix
        lu_combined = lu_combined.add_prefix('Lung_')
        br_combined = br_combined.add_prefix('Breast_')
        hn_combined = hn_combined.add_prefix('HeadNeck_')
        gbm_combined = gbm_combined.add_prefix('Brain_')
        ov_combined = ov_combined.add_prefix('Ovarian_')
        cc_combined = cc_combined.add_prefix('Renal_')
        en_combined = en_combined.add_prefix('Endometrial_')

        # Add a label column for each cancer type
        
        lu_combined['Cancer'] = cancer_labels[0]
        br_combined['Cancer'] = cancer_labels[1]
        hn_combined['Cancer'] = cancer_labels[2]
        gbm_combined['Cancer'] = cancer_labels[3]
        ov_combined['Cancer'] = cancer_labels[4]
        cc_combined['Cancer'] = cancer_labels[5]
        en_combined['Cancer'] = cancer_labels[6]

        # Concatenate all
        self.all_data = pd.concat([lu_combined, br_combined, hn_combined, 
                                   gbm_combined, ov_combined, cc_combined, 
                                   en_combined], axis=0)

        print("Data has been combined into a single DataFrame.")
    
    def clean_data(self):
        """
        Cleans the combined data by filling NA/NaN values with 0.
        """
        if self.all_data is None:
            raise ValueError("Please run combine_data() before cleaning.")
        self.all_data_cleaned = self.all_data.fillna(0)
    

    def get_all_data(self):
        """
        Returns the combined data (with missing values as-is).
        """
        return self.all_data

    def get_all_data_cleaned(self):
        """
        Returns the cleaned combined data (NA/NaN filled with 0).
        """
        return self.all_data_cleaned

    def save_data_to_csv(self, filename="cancer_data.csv", individual=False):
        """
        Saves the cleaned data to a CSV file.
        """
        if individual:
            if self.transcriptome is None or self.proteome is None:
                raise ValueError("Please run combine_data() before saving.")
            else:
                self.transcriptome.to_csv("transcriptome.csv", index=True)
                self.proteome.to_csv("proteome.csv", index=True)

        elif self.all_data_cleaned is None:
            if self.all_data is None:
                raise ValueError("No data to save. Run combine_data() first.")
            else:
                self.all_data.to_csv(filename, index=True)
        else:
            self.all_data_cleaned.to_csv(filename, index=True)

    def load_data_from_csv(self, filename="cancer_data.csv", individual=False):
        """
        Loads data from a CSV file into all_data_cleaned.
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file '{filename}' does not exist.")
        
        if individual:
            self.transcriptome = pd.read_csv("transcriptome.csv", index_col=0)
            self.proteome = pd.read_csv("proteome.csv", index_col=0)
            
        else:
             self.all_data_cleaned = pd.read_csv(filename, index_col=0)

    
