import os

from tensorflow import keras

from agents.BinaryStateAgent import BinaryStateAgent
from environment.BaseEnvironment import TimeSeriesEnvironment
from environment.BinaryStateEnvironment import BinaryStateEnvironment
from environment.Config import ConfigTimeSeries
from environment.Simulator import Simulator


class Evaluator:

    def __init__(self, model, dataset, percentage=1):
        """
        So far my Evaluator only consists of a model directory input and a dataset folder.
        Not very modular
        """
        self.model = keras.models.load_model(model)
        self.dataset = dataset
        self.percentage = percentage

    def run(self):
        for subdir, dirs, files in os.walk(f"../ts_data/{self.dataset}"):
            for file in files:
                if file.find('.csv') != -1:
                    if files.index(file) < len(files) * self.percentage:
                        # Trained on these files
                        pass
                    else:
                        print(f"Testing on file {file}")
                        config = ConfigTimeSeries()
                        env = BinaryStateEnvironment(
                            TimeSeriesEnvironment(verbose=False, filename=f"./{self.dataset}/{file}", config=config),
                            steps=25)

                        agent = BinaryStateAgent(dqn=self.model, memory=None, alpha=0.0001, gamma=0.9, epsilon=0.2,
                                                 epsilon_end=0.01, epsilon_decay=0.5, fit_epoch=10, action_space=2,
                                                 batch_size=256)
                        simulation = Simulator(25, agent, env, 2, testing=True)
                        simulation.run()


if __name__ == '__main__':
    evaluator = Evaluator(model="../aws/lstm_binary_0.5_A2Benchmark", dataset="A2Benchmark")
    evaluator.run()
