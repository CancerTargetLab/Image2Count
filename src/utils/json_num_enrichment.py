import json 
import os

# Only works for merged models performance metrics, as we do not differentiate between different models in the json file

path = 'figures/metrics/nanostring_100/6_6/metrics/mean/performance_metrics/cluster_enrichment.json'
out = 'figures/metrics/nanostring_100/6_6/metrics/mean/performance_metrics/'

dbs = ['CollecTRI', 'PROGENy', 'hallmark_msigdb', 'reactome_msigdb', 'kegg_msigdb']
niche_size = ['1', '2', '3', '5', '8', '11', 'sc']

def per_cluster_key_coverage(predicted_dict, true_dict, top_k=5):
    identified = 0
    true_total = 0
    id_set = set()
    for key in true_dict.keys():
        true_total += len(true_dict[key]['name'][:top_k])
        for val in predicted_dict[key]['name'][:top_k]:
            if val in true_dict[key]['name'][:top_k]:
                identified += 1
                id_set.add(val)
    return identified / true_total, id_set

d = 0

with open(path, 'r') as file:
    d = json.load(file)

keys = list(d.keys())

for db in dbs:
    idd_enrichmend = {}
    for niche in niche_size:
        yy = d[f'yy_{niche}'][db]
        xy = d[list(filter(lambda x: x.startswith(f'xy_{niche}'), keys))[0]][db]

        yx = d[list(filter(lambda x: x.startswith(f'yx_{niche}'), keys))[0]][db]
        xx = d[list(filter(lambda x: x.startswith(f'xx_{niche}'), keys))[0]][db]

        _, id_set1 = per_cluster_key_coverage(xy, yy)
        _, id_set2 = per_cluster_key_coverage(xx, yx)
        id_set1 = list(id_set1)
        id_set1.extend(list(id_set2))
        idd_enrichmend[niche] = id_set1
    with open(os.path.join(out, f'{db}.json'), 'w') as file:
        json.dump(idd_enrichmend, file,  indent=4)
