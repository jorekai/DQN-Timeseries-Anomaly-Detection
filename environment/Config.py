class ConfigTimeSeries:
    def __init__(self, normal=0, anomaly=1, reward_correct=1, reward_incorrect=-1,
                 action_space=[0, 1], seperator=",", boosted=False, window=20):
        self.action_space = action_space
        self.reward_incorrect = reward_incorrect
        self.reward_correct = reward_correct
        self.anomaly = anomaly
        self.normal = normal
        self.boosted = boosted
        self.seperator = seperator
        self.window = window

    def __repr__(self):
        return {"normal": self.normal, "anomaly": self.anomaly, "reward_correct": self.reward_correct,
                "reward_incorrect": self.reward_incorrect, "action_space": self.action_space}
