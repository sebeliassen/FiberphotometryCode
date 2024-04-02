import sys

from matplotlib.lines import Line2D
sys.path.append('../')

from main import load_and_prepare_sessions
from processing.session_sampling import MiceAnalysis
from analysis.timepoint_analysis import sample_signals_and_metrics
from config import all_brain_regions, all_event_types
from itertools import product
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from utils import mouse_br_events_count

from fitsne import FItSNE
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

window_size = 5
window = np.ones(window_size) / window_size

sessions = load_and_prepare_sessions("../../Baseline", load_from_pickle=True, remove_bad_signal_sessions=True)
mouse_analyser = MiceAnalysis(sessions)

# generate all aggregated signals
all_event_signals = []
labels = []

for mouse in mouse_analyser.mice_dict.values():
    mouse_sessions = mouse.sessions
    for brain_region, event in product(all_brain_regions, all_event_types):
        if mouse_br_events_count(mouse, brain_region, event) < 15:
            continue
        if brain_region != 'VS':
            continue
        if event == 'cor_reject':
            continue
        mouse_signals = [] 
        for session in mouse_sessions:
            if session.signal_info.get((brain_region, event)) is None:
                continue
            signals = sample_signals_and_metrics([session], event, brain_region)[0]
            signals = np.array([np.convolve(signal, window, mode='same') for signal in signals])
            mouse_signals.append(signals[:, 150:250])
        if len(mouse_signals) == 0:
            continue
        mouse_signals = np.vstack(mouse_signals)
        sample_idxs = np.random.choice(len(mouse_signals), 100, replace=True)
        mouse_signals = mouse_signals[sample_idxs]
        if len(mouse_signals) > 5:
            for i in range(5):
                all_event_signals.append(np.mean(mouse_signals[i::5], axis=0))
                labels.append((mouse.mouse_id, brain_region, event))
        # all_event_signals.append(mouse_signals)
        # labels.extend([(mouse.mouse_id, brain_region, event)] * len(mouse_signals))

all_event_signals = np.array(all_event_signals)
# all_event_signals = np.vstack(all_event_signals)
signals_embedded = FItSNE(all_event_signals, max_iter=750);

import matplotlib.pyplot as plt
import numpy as np

# Define mappings
colors = {'hit': 'green', 
        'reward_collect': 
        'purple', 'mistake': 
        'red', 'miss':'orange', 
        'cor_reject':'blue',
        'other': 'black'}

brain_reg_to_color = {'VS': 'purple',
                      'DMS': 'forestgreen',
                      'DLS': 'C0'}

shapes = {'DMS': 'o', 'DLS': 's', 'VS': 'X'}
linestyles = {'DMS': 'dotted', 'DLS': 'dashed', 'VS': 'solid'}

fig, ax = plt.subplots(figsize=(10, 6))

for idx, (point, label) in enumerate(zip(signals_embedded, labels)):
    color = colors[label[-1]]  # label[-1] should correspond to the event type for color selection
    # color = brain_reg_to_color[label[1]]

    # Create an inset for each signal
    ax_inset = inset_axes(ax, width=0.5, height=0.5, loc='center',
                          bbox_to_anchor=(point[0], point[1]),
                          bbox_transform=ax.transData, borderpad=0)
    ax_inset.plot(all_event_signals[idx], color=color, alpha=0.8)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_frame_on(False)

# Adjusting the axes limits based on the range of the t-SNE embeddings
x_min, x_max = signals_embedded[:, 0].min(), signals_embedded[:, 0].max()
y_min, y_max = signals_embedded[:, 1].min(), signals_embedded[:, 1].max()

# Here, you can adjust the factor to control how much to "zoom out"
padding_factor = 1.1
x_padding = (x_max - x_min) * padding_factor
y_padding = (y_max - y_min) * padding_factor

ax.set_xlim(x_min - x_padding, x_max + x_padding)
ax.set_ylim(y_min - y_padding, y_max + y_padding)

ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
plt.title('t-SNE Visualization with Signal Representations')
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[event_type], markersize=10) for event_type in colors.keys()]
plt.legend(handles, colors.keys())
plt.show()