import matplotlib.pyplot as plt
import numpy as np

def plot_signal_and_cam(signal, cam, title='Signal and Grad-CAM'):
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.arange(len(signal))

    # Normalize the CAM to make it easier to see on the plot
    cam_normalized = (cam - cam.min()) / (cam.max() - cam.min())

    # Plot the original signal
    ax.plot(t, signal, label='Signal')

    # Create a second y-axis to plot the heatmap
    ax2 = ax.twinx()
    ax2.plot(t, cam_normalized, color='red', linestyle='--', label='Grad-CAM Heatmap')

    # Fill under the curve
    ax2.fill_between(t, 0, cam_normalized, color='red', alpha=0.5)

    # Labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax2.set_ylabel('CAM Importance')
    ax.set_title(title)

    # Legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.show()