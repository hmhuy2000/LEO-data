from cProfile import label
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import os
import seaborn as sns
import matplotlib as mpl
import argparse


cmap = plt.get_cmap('tab10')

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

SMALL_SIZE = 15

mpl.rc('font', size=SMALL_SIZE)
mpl.rc('axes', titlesize=SMALL_SIZE)

def lpf(arr, weight=0.8):
  filtered_arr = arr
  for i in range(1, len(arr)):
    filtered_arr[i] = weight*arr[i - 1] + (1 - weight)*arr[i]
  return filtered_arr
#----------------------- config here ------------------------#

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--task', type=str, default='house_building_1', help="'house_building_1','house_building_2'")
parser.add_argument('--algo', type=str, default='DQN')
parser.add_argument('--show', type=int, default=0)
args = parser.parse_args()

num_experts_lookup = {
                      'house_building_1': 5,
                      'house_building_2': 5,
                      }

task = args.task
algo = args.algo
num_expert = num_experts_lookup[args.task]

if algo in ['DQN']:
  sns.set_style('darkgrid')
else:
  sns.set_style('whitegrid')

# label
# SDQfD-100
# SDQfD-5
# LEO-SDQfD-5
# LEO-SDQfD-5 (Perfect)

#------------------------------------------------------------#
parent_folder = f'new_{task}'
# color_list = [
#             'g', 
#             'purple',
#             'b',
#             'y', 
#             'r'
#             ]

color_list = [
            cmap.colors[0],
            cmap.colors[1],
            cmap.colors[2],
            cmap.colors[3],
            ]

label_list = [
            f"{algo}-{0}",
            f"LEO-{algo}-{0}",
            f"{algo}-{num_expert}",
            f"LEO-{algo}-{num_expert}",
]
# folder_list = os.listdir(parent_folder)
folder_list = [
    f"{algo}_Original_{0}",
    f"G-{algo}_Old_Classifier_{0}",
    f"{algo}_Original_{num_expert}",
    f"G-{algo}_Old_Classifier_{num_expert}",
]

file_name = 'info/eval_rewards.npy'
line_list = []

for id, folder_name in enumerate(folder_list):
  # print(folder_name)
  avg = []
  sub_folder_list = os.listdir(os.path.join(parent_folder, folder_name))

  for i, sub_folder in enumerate(sub_folder_list):
    arr = np.load(os.path.join(parent_folder, folder_name, sub_folder, file_name), allow_pickle=True)
    # print(len(arr))
    arr = arr[0:len(arr)-1]
    avg.append([])
    for j in range(len(arr)):
      avg[i].append(np.mean(arr[j]))
    avg[i] = lpf(avg[i])
  avg = np.array(avg)
  plot_avg = np.mean(avg, axis=0)
  std_arr = stats.sem(avg, axis=0)

  x = np.arange(0.5, (len(arr)+1)*0.5, 0.5)
  plt.plot(x, plot_avg, color=color_list[id], linestyle='solid', linewidth=2, label=label_list[id])
  line_list.append(plot_avg)
  plt.fill_between(x, plot_avg - std_arr, plot_avg + std_arr, color=color_list[id], alpha=0.15)
# plt.xlim([0,1])
# plt.ylim([0,25])
# plt.xticks(np.arange(0, 0.5*(len(arr)+1), 2.5)) 
# plt.yticks(np.arange(0, 1.1, 0.2)) 
# plt.legend(line_list, folder_list)plt.legend(loc='upper left')
plt.legend()
# plt.title(parent_folder)
plt.tight_layout()
# plt.grid()
if args.show == 1:
  plt.show()
else:
  plt.savefig(os.path.join('Figures', f'nonExpert_{parent_folder}_{algo}.png'),dpi=600)