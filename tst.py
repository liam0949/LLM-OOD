import torch

ood_datasets = ['rte', 'sst2', 'mnli', '20ng', 'trec', 'imdb', 'wmt16', 'multi30k']

# taskname = "imdb"
# if taskname in ["sst2" , "imdb"]:
#     ood_datasets = list(set(ood_datasets) - set(["sst2","imdb"]))
# else:
#     ood_datasets = list(set(ood_datasets) - set([taskname]))
# print(ood_datasets)
# id=1
# id+=1
# print(id)
# import torch
# a = torch.zeros(5)
# print(a)

# import numpy as np
# import matplotlib.pyplot as plt
# MSP = [0.5,0.031595173,0.32535098,0.19921503,0.966192302,0.700228754,0.901288907,0.98246938,0.999425074,0.998102309,0.998854202,0.998568187,0.995303271,0.996445014,0.995264605,0.002271321,0.009462825,0.00660253,0.010595444,0.015408606,0.163644755,0.152112051,0.890832343,0.918915396]
# Maha = [0.507962658,1,0.594930337,0.997626437,0.763199846,0.905002178,0.986299374,0.985211793,0.999840701,0.999901814,0.999869664,0.999895442,0.999840411,0.999767712,0.999727163,0.99849824,0.996967227,0.997843374,0.997748664,0.995127173,0.996066461,0.997222396,0.999106764,0.981168192]
# Cosine = [0.489451477,1,0.999012198,0.995068522,0.99989269,0.999537452,0.999582925,0.999739907,0.999804206,0.999146734,0.998551678,0.99917787,0.999067374,0.998676945,0.997948947,0.99623416,0.996087894,0.995830118,0.997219789,0.994466515,0.99704036,0.996928706,0.991829529,0.968184125]
# Energy = [0.502585865,0.574438918,0.023634108,0.971750116,0.490353836,0.084603819,0.015104489,0.394062213,0.966283247,0.992423716,0.995780011,0.997555766,0.997970524,0.994620311,0.986805078,0.99746569,0.997997171,0.996836602,0.997691316,0.996421554,0.994370356,0.571178782,0.895107941,0.913310081]
# #x = ['REST','LAPT','AUTO']
# x = np.arange(24) #总共有几组，就设置成几，我们这里有三组，所以设置为3
# total_width, n = 0.8, 4    # 有多少个类型，只需更改n即可，比如这里我们对比了四个，那么就把n设成4
# width = total_width / n
# x = x - (total_width - width) / 2
# plt.figure(figsize=(19.2, 5.5))
# # colors = [(0.65, 0.25, 0.21), (0.94, 0.76, 0.64), (0.50, 0.76, 0.89), (0.51, 0.72, 0.61)]
# # plt.bar(x, MSP, color = "#0b967d",width=width,label='MSP')
# # plt.bar(x + width, Maha, color = "#89b4cd",width=width,label='Maha')
# # plt.bar(x + 2 * width, Cosine , color = "#f0cd84",width=width,label='Cosine')
# # plt.bar(x + 3 * width, Energy , color = "#dc8350",width=width,label='Energy')
# plt.bar(x, MSP, color = (0.65, 0.25, 0.21),width=width,label='MSP')
# plt.bar(x + width, Maha, color = (0.94, 0.76, 0.64),width=width,label='Maha')
# plt.bar(x + 2 * width, Cosine , color = (0.50, 0.76, 0.89),width=width,label='Cosine')
# plt.bar(x + 3 * width, Energy , color = (0.51, 0.72, 0.61),width=width,label='Energy')
# plt.xlabel("20NG",fontsize=20)
# plt.ylabel("AUROC", fontsize=20)
# # plt.rcParams['figure.figsize'] = (19.2,7)
# # plt.legend(loc = "best")
# plt.legend(loc="upper center", ncol = 24 , fontsize=16)
# plt.xticks(x,["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23"])
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# # my_y_ticks = np.arange(0.3, 1.2, 0.02)
# plt.ylim((0.0, 1.15))
# # plt.yticks(my_y_ticks)
# plt.show()
# plt.savefig(""20ng.png"")
import numpy as np
# a = np.logspace(0, 32, 32, endpoint=True, base=0.9)
# print(a)
# import  torch
# from torch.nn import CrossEntropyLoss, MSELoss
# rec_loss_fct = MSELoss()
#         # rec_loss = rec_loss_fct(rec_hidden, tgt).sum(dim=-1).mean()
# a = torch.randn([128,5])
# b = torch.randn([128,5])
# rec_loss = rec_loss_fct(a, b)
# print(rec_loss)

#
a= {"idx": 7, "name": "jack"}
a.pop("idx", None)
print(a)