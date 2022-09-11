from cProfile import label
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import os

def lpf(arr, weight=0.8):
  filtered_arr = arr
  for i in range(1, len(arr)):
    filtered_arr[i] = weight*arr[i - 1] + (1 - weight)*arr[i]
  return filtered_arr
#----------------------- config here ------------------------#

task = 'house_building_4'
# algo = 'DQN'
algo = 'SDQfD'
num_expert = 25

#------------------------------------------------------------#
parent_folder = f'new_{task}'
color_list = [
            'g', 
            'purple',
            'b',
            'r'
            ]

label_list = [
            num_expert,
            100,
            num_expert,
            num_expert
]
# folder_list = os.listdir(parent_folder)
folder_list = [
    f"{algo}_Original_{num_expert}",
    f"{algo}_Original_100",
    f"G-{algo}_Old_Classifier_{num_expert}",
    f"G-{algo}_Perfect_Classifier_{num_expert}"
]

if (task == 'house_building_4'):
  color_list.append('y')
  label_list.append(num_expert)
  folder_list.append(f"G-{algo}_equi_Classifier_{num_expert}")

file_name = 'info/eval_rewards.npy'
line_list = []

for id, folder_name in enumerate(folder_list):
  print(folder_name)
  avg = []
  sub_folder_list = os.listdir(os.path.join(parent_folder, folder_name))

  for i, sub_folder in enumerate(sub_folder_list):
    arr = np.load(os.path.join(parent_folder, folder_name, sub_folder, file_name), allow_pickle=True)
    print(len(arr))
    arr = arr[0:len(arr)-1]
    avg.append([])
    for j in range(len(arr)):
      avg[i].append(np.mean(arr[j]))
    avg[i] = lpf(avg[i])
  avg = np.array(avg)
  plot_avg = np.mean(avg, axis=0)
  std_arr = stats.sem(avg, axis=0)

  x = np.arange(0.5, (len(arr)+1)*0.5, 0.5)
  plt.plot(x, plot_avg, color=color_list[id], linewidth=3,label=label_list[id])
  line_list.append(plot_avg)
  plt.fill_between(x, plot_avg - std_arr, plot_avg + std_arr, color=color_list[id], alpha=0.3)
plt.xlim([0,1])
plt.xticks(np.arange(0, 0.5*(len(arr)+1), 2.5)) 
plt.yticks(np.arange(0, 1.1, 0.2)) 
# plt.legend(line_list, folder_list)
plt.legend(loc='lower left')
# plt.title(parent_folder)
plt.tight_layout()
plt.grid()
# plt.show()
plt.savefig(os.path.join('Figures', f'{parent_folder}_{algo}.png'),dpi=600)

