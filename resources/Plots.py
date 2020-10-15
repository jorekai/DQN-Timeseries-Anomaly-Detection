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
