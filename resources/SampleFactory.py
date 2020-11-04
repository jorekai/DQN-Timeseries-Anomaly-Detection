import random
from environment.Config import ConfigTimeSeries
from environment.BaseEnvironment import TimeSeriesEnvironment
from resources.Utils import pretty_print_df


class SampleFactory:
    """
    This Factory creates sample subsets of the input Timeseries. The format of the output is a standard supervised
    learning split of (training_set, testing_set, validation_set)
    """

    def __init__(self, timeseries, labelled):
        self.timeseries = timeseries
        self.labelled = labelled

    def create_sample(self, amount=1, train_percentage=0.8, split=True):
        if not split:
            return self.__get_random_sample(amount=amount, labelled=self.labelled)
        else:
            return self.__get_sample_set(pct_test=train_percentage, labelled=self.labelled)

    def __get_random_sample(self, amount, labelled):
        """
        Should not be used in production environment @see get_sample_test instead
        :param amount: size of expected sample output < complete dataframe size
        :param labelled: True if labelled dataframe
        :return: a sample pandas dataframe
        """
        random_index = random.randint(0, len(self.timeseries))
        if labelled:
            return self.timeseries.iloc[abs(len(self.timeseries)) - random_index: abs(
                len(self.timeseries) - random_index + amount)]
        return self.timeseries.iloc[
               abs(len(self.timeseries)) - random_index: abs(
                   len(self.timeseries) - random_index + amount)]

    def __get_sample_set(self, pct_test, labelled):
        """
        Provides preprocessed Tuple of TimeSeries Frames
        :param pct_test: percentage of test size from complete dataframe
        :param labelled: True if labelled dataframe
        :return: Tuple (training, testing, validation) as pandas dataframe
        """
        if labelled:
            df = self.timeseries
        else:
            df = self.timeseries
        train_size = int(len(df) * pct_test)
        test_size = len(df) - train_size
        val_size = int(test_size * 0.1)
        train, test, val = df.iloc[0:train_size], df.iloc[len(df) - test_size:len(df) - val_size], df.iloc[
                                                                                                   len(
                                                                                                       df) - val_size:len(
                                                                                                       df)]
        return train, test, val


if __name__ == '__main__':
    env = TimeSeriesEnvironment(verbose=False,
                                config=ConfigTimeSeries(normal=0, anomaly=1, reward_correct=1, reward_incorrect=-1,
                                                        action_space=[0, 1], seperator=",", boosted=False),
                                filename="Test/SmallData.csv")

    factory = SampleFactory(timeseries=env.timeseries_labeled, labelled=True)

    train, test, val = factory.create_sample(train_percentage=0.8, split=True)

    print("Training \n{} \n Testing \n{} \n Validation \n{} \n ".format(pretty_print_df(train),
                                                                        pretty_print_df(test),
                                                                        pretty_print_df(val)))
