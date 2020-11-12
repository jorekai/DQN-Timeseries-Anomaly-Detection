class ConfigTimeSeries:
    def __init__(self, normal=0, anomaly=1, action_space=[0, 1], separator=",", directory="../ts_data/", window=25):
        self.action_space = action_space
        self.directory = directory
        self.anomaly = anomaly
        self.normal = normal
        self.separator = separator
        self.window = window

    def __repr__(self):
        return {"normal": self.normal, "anomaly": self.anomaly,
                "action_space": self.action_space}
