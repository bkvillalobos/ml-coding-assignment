from scorer_constants import MiscConstants as misc, ModelConstants as modc
class Scorer:
    """

    """
    def __init__(self, model_path=None, model=modc.RF):
        """
        Initializes scorere object.
        :param model_path: (optional) explicit path to pickled scikit-learn model to use
        :param model: Str key for which model to use. See scorer_constants.ModelConstants.
                        Defaults to 'rf' for Random Forest.
        """
        if not model_path:
            # use most recently developed model of specified type
            import os
            neiss_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            models_dir = os.path.join(neiss_path, misc.MODELS_DIR)
            model_path = models_dir + sorted([f for f in os.listdir(models_dir) if f.split('_')[1] == model], reverse=True)[0]
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """
        ### TODO: doc
        :return:
        """
        from sklearn.externals import joblib
        self.model = joblib.load(self.model_path)

    def score(self, input):
        """
        ### TODO: doc
        :param input:
        :return:
        """
        try:
            y_hat = self.model.predict(input)
        except ValueError:
            raise ValueError("Bad input data. Can't fit model")
        return y_hat



