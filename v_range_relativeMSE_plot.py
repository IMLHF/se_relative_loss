import matplotlib.pyplot as plt

# exp_name = "IRM_ReLU_A10"
# y1_RMAE = [0.105, 0.141, 0.088, 0.068, 0.055, 0.043]
# y2_RMAE = [0.089, 0.131, 0.084, 0.067, 0.054, 0.039]

# exp_name = "PSM_ReLU_A10"
# y1_RMAE = [0.106, 0.156, 0.09, 0.068, 0.053, 0.04]
# y2_RMAE = [0.094, 0.149, 0.092, 0.071, 0.055, 0.038]

# exp_name = "IRM_Real_A10"
# y1_RMAE = [0.112, 0.132, 0.081, 0.063, 0.051, 0.039]
# y2_RMAE = [0.091, 0.132, 0.085, 0.069, 0.056, 0.042]

exp_name = "PSM_Real_A10"
y1_RMAE = [0.112, 0.164, 0.103, 0.08, 0.064, 0.049]
y2_RMAE = [0.091, 0.161, 0.101, 0.079, 0.063, 0.047]

x=["$[0,0.1)$","$[0.1, 0.3)$","$[0.3, 0.5)$","$[0.5, 0.7)$","$[0.7, 0.9)$", r"$[0.9,+\infty)$"]
noisy_RMAE = [0.293, 0.236, 0.228, 0.226, 0.224, 0.222]
plt.figure(figsize=(5,4))
plt.plot(x, noisy_RMAE, label='Noisy', linewidth=2, color='black', marker='s',
         markerfacecolor='black', markersize=8)
plt.plot(x, y1_RMAE, label='Exp.19', linewidth=2, color='blue', marker='o',
         markerfacecolor='blue', markersize=8)
plt.plot(x, y2_RMAE, label='Exp.21', linewidth=2, color='red', marker='>',
         markerfacecolor='red', markersize=8)
plt.xlabel('Value segmentation')
plt.ylabel('RMAE')
plt.legend()
plt.subplots_adjust(top=0.98, right=0.96, left=0.13, bottom=0.12)
plt.savefig("exp/paper_sample/RMAERS_%s.jpg" % exp_name)
# plt.show()
plt.close()
