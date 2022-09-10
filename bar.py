import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

cmap = plt.get_cmap('tab10')

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set_style('darkgrid')

SMALL_SIZE = 15

mpl.rc('font', size=SMALL_SIZE)
mpl.rc('axes', titlesize=SMALL_SIZE)


def get_color_index(label):
    if label in ['Random']:
        return 0

    if label in ['Expert']:
        return 2

labels = ['Original 100 expert',
        'Original equal LEO expert',
        'LEO with normal classifier',
        'LEO with equivariant classifier',
        'LEO with perfect classifier'
 ]

colors = ['purple','g','b','y','r']
linestyles = ['-' for _ in range(len(colors))]
alphas = [1 for _ in range(len(colors))]
# for label in labels:

#     if label not in ['Random', 'State Expert']:
#         colors.append(cmap.colors[get_color_index(label)])
#         linestyles.append('-')
#         alphas.append(1.0)
#     else:
#         if label == 'Random':
#             colors.append('b')

#         if label == 'State Expert':
#             colors.append('r')

#         linestyles.append('--')
#         alphas.append(0.2)


palette = list(zip(labels, colors, linestyles, alphas))

def export_legend(palette, dpi="figure", filename="legend.png"):
    # Create empty figure with the legend
    handles = []
    for p in palette:
        l, c, line, alpha = p
        handles.append(Line2D(xdata=[0], ydata=[1], color=c, label=l, alpha=alpha, linestyle=line))
    fig = plt.figure()
    legend = fig.gca().legend(handles=handles, framealpha=1, frameon=True, ncol=10)

    # Render the legend
    fig.canvas.draw()

    # Export the figure, limiting the bounding box to the legend area,
    # slighly extended to ensure the surrounding rounded corner box of
    # is not cropped. Transparency is enabled, so it is not an issue.
    bbox  = legend.get_window_extent().padded(2)
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=dpi, transparent=True, bbox_inches=bbox)
    # plt.show()
    # Delete the legend along with its temporary figure
    plt.close(fig)

export_legend(palette, dpi=100)