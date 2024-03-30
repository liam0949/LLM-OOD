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
a = "re gospel dating in article mimsy umd edu mangoe cs umd edu charley wingate writes well john has a quite different not necessarily more elaborated theology there is some evidence that he must have known luke and that the content of q was known to him but not in a canonized form this is a new argument to me could you elaborate a little the argument goes as follows q oid quotes appear in john but not in the almost codified way they were in matthew or luke however they are considered to be similar enough to point to knowledge of q as such and not an entirely different source assuming that he knew luke would obviously put him after luke and would give evidence for the latter assumption i don t think this follows if you take the most traditional attributions then luke might have known john but john is an elder figure in either case we re talking spans of time here which are well within the range of lifetimes we are talking date of texts here not the age of the authors the usual explanation for the time order of mark matthew and luke does not consider their respective ages it says matthew has read the text of mark and luke that of matthew and probably that of mark as it is assumed that john knew the content of luke s text the evidence for that is not overwhelming admittedly earlier manuscripts of john have been discovered interesting where and which how are they dated how old are they unfortunately i haven t got the info at hand it was i think in the late s or early s and it was possibly as old as ce when they are from about why do they shed doubt on the order on putting john after the rest of the three i don t see your point it is exactly what james felder said they had no first hand knowledge of the events and it obvious that at least two of them used older texts as the base of their account and even the association of luke to paul or mark to peter are not generally accepted well a genuine letter of peter would be close enough wouldn t it sure an original together with id card of sender and receiver would be fine so what s that supposed to say am i missing something and i don t think a one step removed source is that bad if luke and mark and matthew learned their stories directly from diciples then i really cannot believe in the sort of big transformation from jesus to gospel that some people posit in news reports one generally gets no better information than this and if john is a diciple then there s nothing more to be said that john was a disciple is not generally accepted the style and language together with the theology are usually used as counterargument the argument that john was a disciple relies on the claim in the gospel of john itself is there any other evidence for it one step and one generation removed is bad even in our times compare that to reports of similar events in our century in almost illiterate societies not even to speak off that believers are not necessarily the best sources it is also obvious that mark has been edited how old are the oldest manuscripts to my knowledge which can be antiquated the oldest is quite after any of these estimates and it is not even complete the only clear editing is problem of the ending and it s basically a hopeless mess the oldest versions give a strong sense of incompleteness to the point where the shortest versions seem to break off in midsentence the most obvious solution is that at some point part of the text was lost the material from verse on is pretty clearly later and seems to represent a synopsys of the end of luke in other words one does not know what the original of mark did look like and arguments based on mark are pretty weak but how is that connected to a redating of john benedikt"
a = a.split(" ")
print(len(a))