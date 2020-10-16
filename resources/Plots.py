import matplotlib.pyplot as plt
import seaborn as sb


def plot_series(series, name="______"):
    plt.figure(figsize=(15, 7))
    chart = sb.lineplot(
        x=series.index,
        y="value",
        data=series,
        ci=None
    ).set_title('Time Series: {}'.format(name))
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.show()


def plot_actions(actions, series):
    plt.figure(figsize=(15, 7))
    plt.plot(series.index, actions, label="Actions", linestyle="solid")
    plt.plot(series.index, series["anomaly"], label="True Label", linestyle="dotted")
    plt.plot(series.index, series["value"], label="Series", linestyle="dashed")
    plt.legend()
    plt.ylabel('Reward Sum')
    plt.show()


def plot_learn(data):
    plt.figure(figsize=(15, 7))
    sb.lineplot(
        data=data,
    ).set_title("Learning")
    plt.ylabel('Reward Sum')
    plt.show()


def plot_reward(result):
    plt.figure(figsize=(15, 7))
    sb.lineplot(
        data=result,
    ).set_title("Reward Random vs Series")
    plt.ylabel('Reward Sum')
    plt.show()
