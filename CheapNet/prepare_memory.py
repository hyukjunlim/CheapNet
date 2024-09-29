import os
import numpy as np
import pandas as pd
from dataset_CheapNet import GraphDataset, PLIDataLoader
from CheapNet_nobatch import CheapNet
import torch
from utils import *

# def seed_everything(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
# seed_everything(40)

# data_root = './data'
# graph_type = 'Graph_GIGN'
# batch_size = 64

# test2016_dir = os.path.join(data_root, 'train')
# test2016_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
# test2016_set = GraphDataset(test2016_dir, test2016_df, graph_type=graph_type, create=False)
# test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=4)

# device = torch.device('cuda:0')

# model = CheapNet(35, 256).to(device)
# model = model.cuda()
# model.eval()

# total_node_num_list = []
# total = []
# for i, data in enumerate(test2016_loader):
#     data = data.to(device)
#     with torch.no_grad():
#         for j in range(data.batch.max().item() + 1):
#             mask = data.batch == j
#             node_num = mask.sum()
#             total_node_num_list.append(node_num.item())
#             total.append((node_num.item(), test2016_df.iloc[batch_size * i + j - 1]['pdbid']))

# ############################################################################################################
# # make a histogram of the number of nodes in each batch
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set the style to 'whitegrid' for a professional look
# sns.set(style="whitegrid")

# # Calculate mean and standard deviation
# mean_node_num = np.mean(total_node_num_list)
# std_node_num = np.std(total_node_num_list)

# # Create the histogram with more detailed customization
# plt.figure(figsize=(8, 6))
# plt.hist(total_node_num_list, bins=20, color='steelblue', edgecolor='black')

# # Titles and labels with larger fonts
# plt.title('Distribution of Node Counts per Graph in the Validation Set', fontsize=16)
# plt.xlabel('Number of Nodes in Each Graph', fontsize=14)
# plt.ylabel('Frequency of Graphs', fontsize=14)

# # Add gridlines for better readability
# plt.grid(True, linestyle='--', alpha=0.7)

# # Add mean and std annotations
# plt.axvline(mean_node_num, color='red', linestyle='dashed', linewidth=2)
# plt.text(mean_node_num + 20, plt.ylim()[1] * 0.9, f'Mean: {mean_node_num:.2f}', color='red', fontsize=12)
# plt.text(mean_node_num + 20, plt.ylim()[1] * 0.8, f'STD: {std_node_num:.2f}', color='red', fontsize=12)

# # Save the updated figure
# plt.savefig('histogram.png', bbox_inches='tight', dpi=300)
############################################################################################################

# selected_indices = []
# for lower_bound in range(50, 450, 50):
#     upper_bound = lower_bound + 50
#     indices_in_range = [i for i, (count, _) in enumerate(total) if lower_bound <= count < upper_bound]
#     if len(indices_in_range) >= 64:
#         selected = np.random.choice(indices_in_range, 64, replace=False)  # Select 3 random graphs
#     else:
#         selected = indices_in_range  # If less than 3, select all available graphs
    
#     selected_indices.append(list(selected))

# # Print or save the selected indices
# print("Selected Indices:", selected_indices)
# print(f"Selected count for each group: {[[total[i][0] for i in j] for j in selected_indices]}")
# # make the selected pdbids like [[a,b,c,d], [e,f,g,h], ...], not flat
# selected_pdbids = [[total[i][1] for i in j] for j in selected_indices]
# print(f'Selected pdbids: {selected_pdbids}')

############################################################################################################
selected_indices = [[4758, 5491, 5912, 994, 4647, 7869, 470, 3929, 7626, 9259, 4455, 7506, 6370, 690, 5281, 3037, 8342, 239, 6436, 5396, 5929, 1894, 6249, 2616, 11042, 8328, 6700, 9752, 6604, 11057, 8502, 6605, 5111, 9541, 4984, 3847, 1214, 5170, 5191, 4261, 8926, 9000, 9222, 4907, 11243, 2184, 2004, 10450, 6232, 663, 146, 8820, 10761, 6100, 4983, 7576, 11015, 10156, 10420, 1721, 6624, 1789, 11186, 10324, 5567, 7659, 5478, 2412, 3402, 6294, 8160, 1384, 6362, 3281, 1928, 3427, 5605, 2680, 9867, 881, 2186, 8925, 7143, 5653, 1871, 4664, 4642, 8750, 9504, 1918, 4405, 5867, 11064, 11041, 7, 10520, 9898, 8123, 1645, 9823, 553, 2494, 3936, 9995, 3325, 184, 1516, 6617, 5308, 4904, 11446, 7836, 1350, 4234, 2688, 4271, 3356, 9313, 8912, 1309, 6633, 9853, 9256, 7016, 6173, 2314, 7136, 934], [1969, 5499, 10658, 11294, 11438, 5826, 25, 6682, 6820, 11858, 3227, 8811, 864, 5476, 11058, 10784, 6965, 9007, 1157, 8413, 10387, 7005, 10782, 2569, 10123, 814, 532, 10981, 7995, 7857, 3859, 10116, 11012, 5981, 9581, 10606, 4949, 7628, 9139, 3366, 10764, 1571, 7501, 10475, 9526, 4549, 2892, 6435, 3232, 974, 7133, 3274, 4550, 1260, 11119, 1003, 7490, 3552, 3, 5356, 1713, 5467, 4395, 4224, 499, 10494, 1085, 7637, 4841, 10873, 3903, 7075, 6093, 9513, 4499, 9043, 10238, 6841, 3924, 2421, 4883, 2973, 3698, 4744, 11553, 3327, 133, 9821, 3063, 4087, 8412, 273, 1464, 4493, 6695, 5706, 9894, 7261, 4945, 8487, 4696, 11291, 3377, 6197, 7537, 9364, 524, 10392, 11193, 10753, 1802, 9472, 8738, 4709, 7236, 1389, 1224, 11881, 8044, 2558, 11788, 9366, 9569, 2443, 7192, 3155, 9984, 8204], [2859, 3404, 203, 1411, 9253, 11831, 1231, 1702, 7532, 7673, 3143, 8402, 2672, 10886, 2514, 3727, 8739, 3406, 6544, 5, 5220, 2117, 4420, 5615, 3508, 2498, 7939, 595, 1155, 10601, 7491, 10001, 11492, 9265, 1939, 494, 9442, 8339, 3550, 7087, 2479, 7767, 5619, 7145, 5171, 11185, 4281, 9134, 5852, 7268, 2350, 10303, 3500, 8226, 10393, 747, 7777, 11508, 7748, 999, 119, 6616, 6280, 9090, 1792, 3562, 6297, 7057, 5664, 8708, 7328, 892, 4118, 9655, 6764, 11048, 10332, 4411, 8109, 5473, 8392, 805, 7171, 10685, 9789, 1193, 1001, 8338, 1818, 360, 9554, 7081, 2305, 2454, 1151, 10275, 4014, 1472, 8337, 5482, 11159, 5673, 9667, 10190, 2276, 307, 7971, 6357, 2475, 1716, 10014, 8508, 6879, 4101, 6515, 1901, 5888, 7851, 10040, 11598, 11899, 10364, 5235, 5586, 5263, 1549, 7581, 10993], [6733, 1639, 5923, 7709, 3994, 5524, 4952, 11474, 909, 568, 7036, 10903, 3502, 6568, 4533, 11445, 3333, 5953, 2466, 3138, 4512, 1204, 4403, 5890, 7550, 821, 7689, 2655, 9486, 10212, 9291, 5612, 5105, 6427, 224, 1100, 7670, 9587, 10018, 427, 8152, 3195, 6573, 3394, 769, 8051, 3129, 3607, 7944, 903, 7592, 2631, 7707, 4874, 10359, 766, 5547, 8941, 8657, 6309, 9910, 10823, 1527, 4213, 9480, 4447, 10763, 7606, 1552, 5990, 512, 3918, 5671, 2143, 5321, 10735, 10059, 8111, 9066, 6728, 2201, 631, 630, 9096, 11316, 2727, 7212, 6538, 8043, 4368, 3580, 3388, 11362, 6993, 3428, 8596, 649, 4043, 11877, 3203, 1014, 4354, 67, 9793, 57, 5759, 3059, 10617, 4102, 11495, 1779, 5367, 9888, 1490, 9030, 2531, 5878, 4397, 8514, 11171, 6443, 4346, 3160, 6817, 4786, 5429, 7371, 4171], [18, 6691, 6148, 9334, 515, 2953, 10874, 7296, 7288, 1903, 3343, 121, 6227, 789, 3770, 6292, 8752, 4266, 3049, 242, 7068, 10440, 6259, 5926, 188, 3050, 4194, 6283, 7513, 9941, 1979, 3420, 7000, 5408, 244, 8782, 2986, 10800, 4092, 52, 2119, 9945, 8267, 8108, 5464, 4052, 2333, 685, 11427, 7235, 15, 7868, 9676, 11892, 11798, 4041, 6363, 5095, 2085, 636, 2211, 5905, 2055, 8908, 8083, 6877, 11753, 5685, 10573, 1601, 7623, 6489, 6618, 7195, 11659, 7819, 8292, 5289, 8628, 11797, 6857, 3915, 5811, 10530, 10131, 5782, 6415, 4178, 3241, 8265, 6788, 11382, 2819, 689, 9985, 11172, 2487, 8524, 6042, 8278, 10631, 1839, 10258, 1620, 11868, 396, 353, 10425, 9455, 10705, 2324, 1715, 1850, 10953, 10592, 1202, 10845, 4658, 1205, 4593, 4325, 2552, 1040, 832, 1563, 1583, 3560, 8179], [6584, 6731, 4393, 11580, 2326, 9857, 5719, 4310, 2835, 11528, 3297, 11045, 9805, 2141, 8707, 2543, 137, 6818, 5392, 8121, 5000, 1426, 6680, 5148, 4938, 3591, 7872, 9794, 11349, 3779, 9551, 3987, 2452, 10772, 4950, 6661, 1024, 2142, 11615, 4462, 2764, 5662, 5389, 2665, 6897, 3536, 47, 4143, 2000, 10235, 8459, 7672, 2571, 2638, 4536, 5617, 70, 10866, 6339, 7046, 5572, 1046, 9510, 10876, 1068, 2615, 2772, 4284, 1635, 5411, 5896, 8189, 9437, 9887, 1131, 644, 815, 1080, 918, 10340, 6209, 10783, 6964, 4050, 3300, 199, 6010, 3046, 9784, 616, 11867, 9023, 8081, 4832, 3017, 10139, 8377, 10951, 3978, 4460, 8933, 8880, 4785, 7803, 3489, 2594, 6822, 9964, 4872, 1840, 9570, 6144, 1727, 4013, 9942, 7522, 6754, 11354, 6126, 4170, 5955, 5378, 11285, 1953, 4089, 40, 1909, 6757], [17, 9249, 1913, 5395, 6306, 9565, 1755, 2909, 5901, 2922, 5901, 11803, 11787, 11011, 11117, 2586, 2505, 11787, 8465, 7529, 340, 8329, 9016, 7418, 1854, 6699, 5036, 8924, 10397, 6438, 8924, 4813, 2163, 1446, 4793, 8236, 11817, 9249, 10155, 6860, 4149, 7480, 4793, 8665, 10397, 2995, 8102, 9565, 6497, 912, 1446, 4793, 11817, 5256, 7529, 4793, 8102, 2586, 4149, 11803, 8329, 4965, 9892, 6726, 7183, 11787, 1194, 4020, 11799, 5901, 5862, 8665, 1521, 409, 5006, 4716, 2583, 10967, 5877, 2922, 8924, 11173, 7334, 4233, 9920, 9249, 1194, 10182, 9016, 11787, 2586, 17, 7681, 7842, 5036, 3229, 10137, 7076, 6001, 2583, 9565, 9565, 9750, 11398, 10967, 1521, 4652, 8236, 6498, 11644, 8236, 11117, 6726, 11644, 1383, 8465, 6039, 11117, 340, 5978, 11787, 11011, 11799, 2909, 1367, 11561, 11398, 5466], [11700, 3704, 1632, 7015, 5763, 4242, 5201, 9149, 4242, 1419, 6773, 1630, 8985, 3635, 1425, 7993, 8549, 2850, 10309, 6189, 6146, 5763, 9708, 9047, 8441, 9149, 3771, 10491, 3322, 9047, 8789, 9708, 3456, 2369, 9149, 6205, 6762, 9047, 9211, 6189, 8789, 9211, 1667, 3322, 9708, 7254, 6773, 11812, 10309, 7254, 7015, 1132, 8793, 10153, 8055, 8985, 4060, 9556, 8985, 11046, 3456, 9420, 8055, 1382, 9708, 8985, 10491, 8220, 4242, 8932, 935, 5069, 2031, 4910, 11641, 11641, 7015, 1382, 1073, 5763, 6205, 11046, 11641, 732, 3072, 4242, 10810, 8789, 9556, 6576, 8789, 10810, 11511, 11812, 11046, 1073, 9527, 2369, 6189, 2642, 7254, 8122, 10309, 10810, 5590, 732, 3704, 2031, 7916, 2031, 5590, 11641, 9556, 6205, 10309, 6762, 9420, 5154, 8985, 5201, 732, 9047, 10491, 10491, 4910, 8864, 3456, 8789]]

# selected_indices = [[4758, 5491, 5912, 994, 4647, 7869, 470, 3929, 7626, 9259, 4455, 7506, 6370, 690, 5281, 3037, 8342, 239, 6436, 5396, 5929, 1894, 6249, 2616, 11042, 8328, 6700, 9752, 6604, 11057, 8502, 6605, 5111, 9541, 4984, 3847, 1214, 5170, 5191, 4261, 8926, 9000, 9222, 4907, 11243, 2184, 2004, 10450, 6232, 663, 146, 8820, 10761, 6100, 4983, 7576, 11015, 10156, 10420, 1721, 6624, 1789, 11186, 10324], [1969, 5499, 10658, 11294, 11438, 5826, 25, 6682, 6820, 11858, 3227, 8811, 864, 5476, 11058, 10784, 6965, 9007, 1157, 8413, 10387, 7005, 10782, 2569, 10123, 814, 532, 10981, 7995, 7857, 3859, 10116, 11012, 5981, 9581, 10606, 4949, 7628, 9139, 3366, 10764, 1571, 7501, 10475, 9526, 4549, 2892, 6435, 3232, 974, 7133, 3274, 4550, 1260, 11119, 1003, 7490, 3552, 3, 5356, 1713, 5467, 4395, 4224], [2859, 3404, 203, 1411, 9253, 11831, 1231, 1702, 7532, 7673, 3143, 8402, 2672, 10886, 2514, 3727, 8739, 3406, 6544, 5, 5220, 2117, 4420, 5615, 3508, 2498, 7939, 595, 1155, 10601, 7491, 10001, 11492, 9265, 1939, 494, 9442, 8339, 3550, 7087, 2479, 7767, 5619, 7145, 5171, 11185, 4281, 9134, 5852, 7268, 2350, 10303, 3500, 8226, 10393, 747, 7777, 11508, 7748, 999, 119, 6616, 6280, 9090], [6733, 1639, 5923, 7709, 3994, 5524, 4952, 11474, 909, 568, 7036, 10903, 3502, 6568, 4533, 11445, 3333, 5953, 2466, 3138, 4512, 1204, 4403, 5890, 7550, 821, 7689, 2655, 9486, 10212, 9291, 5612, 5105, 6427, 224, 1100, 7670, 9587, 10018, 427, 8152, 3195, 6573, 3394, 769, 8051, 3129, 3607, 7944, 903, 7592, 2631, 7707, 4874, 10359, 766, 5547, 8941, 8657, 6309, 9910, 10823, 1527, 4213], [18, 6691, 6148, 9334, 515, 2953, 10874, 7296, 7288, 1903, 3343, 121, 6227, 789, 3770, 6292, 8752, 4266, 3049, 242, 7068, 10440, 6259, 5926, 188, 3050, 4194, 6283, 7513, 9941, 1979, 3420, 7000, 5408, 244, 8782, 2986, 10800, 4092, 52, 2119, 9945, 8267, 8108, 5464, 4052, 2333, 685, 11427, 7235, 15, 7868, 9676, 11892, 11798, 4041, 6363, 5095, 2085, 636, 2211, 5905, 2055, 8908], [6584, 6731, 4393, 11580, 2326, 9857, 5719, 4310, 2835, 11528, 3297, 11045, 9805, 2141, 8707, 2543, 137, 6818, 5392, 8121, 5000, 1426, 6680, 5148, 4938, 3591, 7872, 9794, 11349, 3779, 9551, 3987, 2452, 10772, 4950, 6661, 1024, 2142, 11615, 4462, 2764, 5662, 5389, 2665, 6897, 3536, 47, 4143, 2000, 10235, 8459, 7672, 2571, 2638, 4536, 5617, 70, 10866, 6339, 7046, 5572, 1046, 9510, 10876], [624, 2737, 367, 4139, 2766, 11803, 1666, 11644, 1707, 9470, 5978, 7396, 7279, 6582, 2914, 798, 6001, 8857, 9916, 11721, 7161, 6039, 10182, 7842, 11787, 4138, 10967, 10901, 1924, 9076, 8239, 11746, 11561, 10397, 5537, 1383, 5466, 7031, 2253, 7080, 4107, 7681, 4063, 10137, 7076, 7507, 3229, 9892, 11799, 9920, 11817, 7518, 2970, 4652, 7334, 3833, 4147, 4233, 1367, 5313, 6498, 11173, 5877, 8665], [5154, 6165, 1419, 1630, 4242, 1434, 3072, 4665, 2642, 5201, 1073, 1550, 4910, 9556, 7254, 2031, 8793, 10491, 4693, 10153, 11700, 8055, 7916, 8932, 9527, 2635, 3503, 5590, 4520, 8864, 1425, 8122, 11511, 1659, 4060, 935, 6576, 3635, 9700, 11641, 1132, 1632, 11812, 8985, 9420, 9211, 6773, 8220, 11046, 6762, 6205, 5069, 3704, 2369, 3456, 7015, 732, 1667, 8789, 10227, 3322, 1382, 10810, 3771]]
selected_pdbids = [['3tiy', '4k8a', '4gfn', '2iko', '4iq6', '3pka', '3gf2', '2qd7', '4dlj', '4kz5', '3zn0', '4tk3', '4g2w', '5ali', '3odu', '3i60', '2v2v', '2xx2', '3ekr', '3o0e', '4wy3', '4qp7', '4ba3', '2p3b', '3hcm', '2cll', '4fi9', '3ijg', '4zg6', '2avi', '3rl8', '4e5f', '1agm', '3hzv', '4nyt', '1b38', '3mg4', '1al7', '1hi3', '1gj5', '4a4h', '4ewo', '5c4o', '4whz', '4b05', '2uxz', '1bty', '3du8', '4b0g', '2h02', '2ksb', '3vfj', '3wt7', '3tkh', '4waf', '1k9r', '3rcj', '3loo', '1e4h', '2ql9', '4txc', '4l7b', '2h5e', '4qaa'], ['3at1', '4qoc', '4i9z', '5aaf', '4jof', '4e1n', '2gnl', '4obv', '1czq', '4a7i', '3cii', '4idn', '3t6r', '3sxu', '2vmd', '4i80', '3uat', '2za5', '2brm', '2vqj', '2ygu', '2chm', '3djq', '2x9e', '4j86', '1kr3', '3p4w', '1zfk', '4eqc', '4gzf', '4mcv', '4k3m', '4tzm', '4itp', '1ikv', '4mnq', '3o87', '3jy9', '3amv', '1rbo', '5bvf', '1a42', '3rz5', '3gcv', '2cfg', '4b83', '2fj0', '1ssq', '4d9p', '3so9', '4jlj', '1g52', '4mk2', '1els', '3u2q', '4z2h', '3hdn', '1i37', '1n3i', '1uwt', '1qsc', '1vik', '4n1b', '1okx'], ['3tao', '4mvy', '1h8l', '4aua', '3zyf', '4iut', '3thb', '2vip', '2h2d', '1dva', '2w8j', '1vjb', '2qhr', '4uib', '2g71', '3c56', '2yi0', '3mt8', '3ly2', '1dqx', '1jmg', '3vid', '3fn0', '1nw4', '4a4c', '2wd1', '4udb', '3e6y', '3s8o', '4mwr', '3b9s', '1w82', '3qzq', '1n94', '3cr4', '4fk6', '4ehv', '3i1y', '3fwv', '3cjf', '1yys', '1s4d', '1onp', '1vyw', '4a4e', '4fk7', '3g90', '4kn7', '1ysi', '2pow', '1g9a', '4q2k', '4ith', '4l02', '4pnl', '1x1z', '4k2g', '2hxm', '2wej', '4rwl', '3dx3', '2b53', '1fq8', '4igq'], ['2itp', '4y79', '4yth', '3grj', '4qkd', '2yfx', '1sts', '1ugx', '3rxe', '3rdo', '4r5g', '1if7', '3fxz', '1u2r', '4u69', '2ylc', '4ran', '3tl0', '4hgs', '1i7g', '4j52', '1uyk', '11gs', '4i11', '1jeu', '3uyr', '1tsi', '1uz4', '3fk1', '4d2r', '1f8b', '1gvk', '4fr3', '4n7y', '4ke0', '4zun', '1w31', '2pyn', '4hwb', '4j8t', '3src', '2xel', '1gt4', '3oui', '2vsl', '3shv', '4lkt', '4zk5', '3t1n', '2g5p', '2cmf', '1v2w', '2of4', '3f7i', '3e3b', '3cct', '3uvl', '4c52', '4f6u', '3ndm', '2ha6', '4j17', '4awf', '3ti4'], ['4gq6', '4yx9', '4ozo', '3kv2', '4qhp', '4z2b', '6gpb', '1npw', '2y67', '3qch', '1sln', '1sqt', '2zxd', '1skj', '3e3c', '3nf9', '1rhk', '3mnu', '3obx', '4uuq', '4o0v', '4bdd', '3fh7', '2p7a', '2qi5', '3o9g', '4bfz', '3ewc', '4hwt', '4asj', '4j8r', '3nu3', '1w22', '1a37', '1wcc', '4mp2', '4cft', '3o7u', '4zwx', '2gl0', '1cka', '1pmn', '1uwu', '2oz7', '3ft5', '4g3g', '4g93', '4n4v', '1o3j', '1ghw', '1q54', '3oap', '2l3r', '2f1b', '3twu', '1qr3', '4eoh', '1avp', '3qmk', '3m5e', '3plu', '2g79', '4bis', '5csz'], ['4q6e', '2on3', '4jxw', '4xkb', '3pjt', '4w53', '3d9n', '4qsh', '4hfp', '4i10', '2fmb', '4o43', '3wtm', '3vhd', '3d2t', '2ozr', '4cg8', '4msu', '1odi', '2r9m', '1a52', '1dxp', '2g19', '1l6y', '4lh7', '2v85', '2g72', '3lhj', '1yqy', '4qws', '4jjf', '4f6v', '5am0', '4z7n', '4vgc', '3kad', '4j73', '4jmh', '1g2a', '3o8p', '3nth', '2x8d', '3zmh', '1br5', '4bie', '4gj2', '2qo1', '2xv1', '1q91', '3prz', '1swg', '3suu', '5fl5', '2l6j', '4mwv', '1ghy', '3t3e', '3cdb', '1i7i', '4tln', '1i90', '4ee0', '4xub', '2d1x'], ['3q2a', '4qvl', '2d41', '3drg', '4y85', '3cph', '3q8d', '3zbf', '4aj4', '1hvj', '3d67', '4aig', '5aml', '1g53', '4qfo', '4uvu', '3kqe', '4bt9', '3oof', '1lbk', '1gfw', '2jiw', '1ppc', '2avq', '2c8y', '1cnx', '5alk', '1o79', '3hqh', '3oy8', '4dk7', '4rj6', '1df8', '3qs4', '5aol', '4pge', '3qk0', '1r5g', '2jds', '2vo7', '3hhu', '4j03', '2wos', '1b7h', '4rn0', '1bra', '3fpd', '3e0q', '2xye', '4xy8', '1ii5', '4yv0', '1sj0', '3u6i', '1np0', '1nc6', '4io3', '4d8s', '2jdm', '1g9t', '2hmh', '1iiq', '4egi', '4wnk'], ['3c7n', '3vru', '3fud', '2wtx', '4jlh', '3hbo', '3px8', '1qm4', '3ot8', '3sx9', '1b9s', '1fkg', '3q4k', '3lpj', '1yrs', '3s3n', '4rd0', '3tia', '3oku', '3t7g', '4oq5', '5ajz', '4p4d', '4ft2', '4n7m', '4g19', '3rsv', '3pwh', '3k8d', '2kdh', '3ttz', '4xqb', '2aez', '1pz5', '3ued', '1hyz', '4jsa', '2hwh', '4pm0', '4c1c', '2v83', '3t0t', '2nxm', '4a22', '4pns', '4acm', '1a28', '2nnd', '3qg6', '4ez5', '1lke', '2qf6', '4d8a', '4bfd', '4mw7', '5hvp', '3d9v', '4wy7', '1mq5', '3wnr', '4uvv', '3opm', '1mjj', '3qkk']]
# df = pd.read_csv('../DEAttentionDTA/data/seq_data_core2016.csv')
# df = df[df['PDBname'].isin(selected_pdbids)]
# print(df)
# print(df.shape)
# df = pd.read_csv('../GAABind/dataset/PDBBind/test2016.txt', sep='\t', header=None)
# df.columns = ['pdbid']
# df = df[df['pdbid'].isin(selected_pdbids)]
# print(df)
# print(df.shape)

# 1081, 1085, 1105, 1125, 1145, 1165
############################################################################################################


class GraphSubsetDataset(GraphDataset):
    def __init__(self, root, df, indices, graph_type='Graph_GIGN', create=False):
        super().__init__(root, df, graph_type=graph_type, create=create)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]  # Map the new idx to the original dataset
        return super().__getitem__(original_idx)
    


cfg = 'TrainConfig_GIGN'
graph_type = 'Graph_GIGN'
batch_size = 128
data_root = './data'
epochs = 20

import torch
import torch.nn as nn
import torch.optim as optim
from CheapNet_nobatch import CheapNet
from dataset_CheapNet import GraphDataset, PLIDataLoader
from utils import *
    
test2016_dir = os.path.join(data_root, 'train')
test2016_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
test2016_set = GraphDataset(test2016_dir, test2016_df, graph_type=graph_type, create=False)
test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CheapNet(35, 256, [28, 156])
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
criterion = nn.MSELoss()

running_loss = AverageMeter()
running_acc = AverageMeter()
running_best_mse = BestMeter("min")
best_model_list = []

# for iteration, indices in enumerate(selected_indices):
iteration = 7
indices = selected_indices[iteration]
print(f"Iteration {iteration + 1}")

# Create a subset dataset and data loader
test2016_subset = GraphSubsetDataset(test2016_dir, test2016_df, indices)
test2016_loader = PLIDataLoader(test2016_subset, batch_size=batch_size, shuffle=False, num_workers=4)

# Train the model
model.train()
l = []  # Store timing
for epoch in range(epochs):
    print(f'Epoch {epoch}')
    for batch_idx, data in enumerate(test2016_loader):
        data = data.to(device)
        pred = model(data)
        label = data.y
        MSE_loss = criterion(pred, label)
        loss = MSE_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()