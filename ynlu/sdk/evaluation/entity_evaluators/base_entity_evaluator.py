class BaseEntityEvaluator(object):

    def preprocess_predictions(self):
        """
        preprocess predictions before evaluating
        """
        pass

    def preprocess_labels(self):
        """
        preprocess labels before evaluating
        """
        pass

    def evaluate(self):
        raise NotImplementedError

    # def describe(self):
    #     """
    #     Give some description about the evaluator
    #     """
    #     raise NotImplementedError

    def save(self):
        """
        Save the output of evaluate to self.output_dir
        """
        raise NotImplementedError

    def run(
        self,
        output_dir: str,
        predictions: list,
        labels: list,
    ):
        """
        one click
        preprocess -> evaluate -> save
        """
        raise NotImplementedError
