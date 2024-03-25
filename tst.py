ood_datasets = ['rte', 'sst2', 'mnli', '20ng', 'trec', 'imdb', 'wmt16', 'multi30k']

taskname = "imdb"
if taskname in ["sst2" , "imdb"]:
    ood_datasets = list(set(ood_datasets) - set(["sst2","imdb"]))
else:
    ood_datasets = list(set(ood_datasets) - set([taskname]))
print(ood_datasets)