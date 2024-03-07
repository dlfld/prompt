from collections import Counter
from math import log, sqrt, ceil
from tkinter import _flatten


class NER_Adaptive_Resampling():

    def __init__(self, data):
        self.data = data

    def conll_data_read(self):

        data = self.data
        x = []
        y = []
        for item in data:
            x.append(item[0].split("/"))
            y.append(item[1].split("/"))

        return x, y

    def get_stats(self):

        # Get stats of the class distribution of the dataset
        labels = list(_flatten(self.conll_data_read()[-1]))
        x = []
        for item in labels:
            x.append(item)
        num_tokens = len(x)
        ent = [label for label in labels]
        count_ent = Counter(ent)

        for key in count_ent:
            # Use frequency instead of count
            count_ent[key] = count_ent[key] / num_tokens
        print(count_ent)
        return count_ent

    def resamp(self, method):

        # Select method by setting hyperparameters listed below:
        # sc: the smoothed resampling incorporating count
        # sCR: the smoothed resampling incorporating Count & Rareness
        # sCRD: the smoothed resampling incorporating Count, Rareness, and Density
        # nsCRD: the normalized and smoothed  resampling  incorporating Count, Rareness, and Density

        if method not in ['sc', 'sCR', 'sCRD', 'nsCRD']:
            raise ValueError("Unidentified Resampling Method")

        x, y = self.conll_data_read()
        stats = self.get_stats()
        total_res = []
        for sen in range(len(x)):

            rsp_time = 0
            sen_len = len(y[sen])
            ents = Counter([label for label in y[sen]])
            # Pass if there's no entity in a sentence
            if ents:
                for ent in ents.keys():
                    # Resampling method selection and resampling time calculation,
                    # see section 'Resampling Functions' in our paper for details.
                    if method == 'sc':
                        rsp_time += ents[ent]
                    if method == 'sCR' or method == 'sCRD':
                        weight = -log(stats[ent], 2)
                        rsp_time += ents[ent] * weight
                    if method == 'nsCRD':
                        weight = -log(stats[ent], 2)
                        rsp_time += sqrt(ents[ent]) * weight
                if method == 'sCR':
                    rsp_time = sqrt(rsp_time)
                if method == 'sCRD' or method == 'nsCRD':
                    rsp_time = rsp_time / sqrt(sen_len)
                rsp_time = ceil(rsp_time)

                for t in range(rsp_time):
                    total_res.append(["/".join(x[sen]), "/".join(y[sen])])
        return total_res
