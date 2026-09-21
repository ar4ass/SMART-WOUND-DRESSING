import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.visualization import plot_actual_vs_predicted, plot_channel_bar, plot_color_comparison


def test_plot_color_comparison_without_reference():
    fig = plot_color_comparison((255, 0, 0), (200, 50, 50))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_color_comparison_with_reference():
    fig = plot_color_comparison((255, 0, 0), (200, 50, 50), reference_rgb=(180, 40, 40))
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_channel_bar_rgb():
    fig = plot_channel_bar([120, 90, 60], ["R", "G", "B"], "RGB rata-rata")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_actual_vs_predicted():
    actual = np.array([4.0, 5.0, 6.0, 7.0])
    predicted = np.array([4.2, 4.8, 6.3, 6.9])
    fig = plot_actual_vs_predicted(actual, predicted)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
