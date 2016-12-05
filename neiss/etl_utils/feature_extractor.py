import pandas as pd
from etl_constants import feConstants as fec, colNames as col

class FeatureExtractor:
    """
    # TODO: document
    """

    def __init__(self, patient_data):
        assert isinstance(patient_data, pd.DataFrame) # TODO: allow list-like/dict, convert to DF
        self.raw_df = patient_data
        self.processed_df = None

    def extract_features(self, keep_weight_vector=False):
        """
        # TODO: doc
        :param keep_weight_vector:
        :return:
        """
        # extract month information from dates
        self.raw_df[fec.MONTH] = self.raw_df[col.TRMT_DATE].apply(self._extract_month)

        # NEISS encodes age in months as 2XX where XX is the number of months
        self._standardize_age()

        # convert categorical/nominal variables into binary indicators
        self.processed_df = self._nominal_to_dummies(fec.NOMINAL_FEATS)

        # combine extracted binary features with transformed features we want to keep
        for feat in fec.TO_ADD + keep_weight_vector*[col.WEIGHT]: # only include weight vector if explictly asked
            self.processed_df[feat] = self.raw_df[feat]

        return self.processed_df



    def _extract_month(self, dt_str, fmt=fec.DT_FMT):
        """
        extract month information from dates
        :param date_str: ### TODO: doc
        :param fmt: ### TODO: doc
        :return:
        """
        from datetime import datetime as dt
        return int(dt.strptime(dt_str, fmt).strftime(fec.MONTH))

    def _standardize_age(self):
        """
        NEISS encodes age in months as 2XX where XX is the number of months.
        :return: standardized float vector representing age in years
        """
        standardize_age = lambda age: (age - 200) / 12.0 if age > 200 else float(age)
        self.raw_df[col.AGE] = self.raw_df[col.AGE].apply(standardize_age)

    def _nominal_to_dummies(self, nom_features):
        """
        convert categorical/nominal variables into binary indicators
        :param nom_features:
        :return:
        """
        safe_tostr = lambda val: str(val) if pd.notnull(val) else val
        for nom in nom_features:
            self.raw_df.loc[:, nom] = self.raw_df[nom].apply(safe_tostr)
        dummy_df = pd.get_dummies(self.raw_df.loc[:, nom_features])
        return dummy_df

