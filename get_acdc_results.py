# %%

import torch
import pickle
import glob
import os
import numpy as np

# %%
from acdc.ioi.utils import get_ioi_true_edges, get_all_ioi_things
from acdc.greaterthan.utils import get_all_greaterthan_things, get_greaterthan_true_edges

# %%

def get_edges(edge_list):
    edges = {
        'attn-attn': [],
        'attn-mlp': [],
        'mlp-attn': [],
        'mlp-mlp': []
    }
    circ_map = {"q": 0, "k": 1, "v": 2}
    extra_edges = 0
    print("Edge list", len(edge_list))
    for (dest, dest_idx, src, src_idx), _ in edge_list:            
        
        dest = dest.split(".")
        src = src.split(".")
        if dest[-1] == "hook_resid_post":
            dest_node = ["mlp", 13]
        elif dest[-1] == "hook_mlp_in":
            dest_node = ["mlp", int(dest[1]) + 1]
        elif dest[-1] == "hook_mlp_out":
            extra_edges += 1
            continue
        elif dest[2] == "attn":
            if src[2] == 'attn':
                extra_edges += 1
            continue
        elif dest[-1].endswith("input"):
            if isinstance(dest_idx, tuple):
                dest_idx = dest_idx[-1]
            else:
                dest_idx = dest_idx.as_index[-1]
            dest_node = ["attn", circ_map[dest[2].replace("hook_", "").replace("_input", "")], int(dest[1]), dest_idx]
        else:
            print(dest)
            raise Exception()
        
        if src[-1] == "hook_mlp_out":
            src_node = ["mlp", int(src[1]) + 1]
        elif src[-1] == "hook_result":
            if isinstance(src_idx, tuple):
                src_idx = src_idx[-1]
            else:
                src_idx = src_idx.as_index[-1]
            src_node = ["attn", int(src[1]), src_idx]
        elif src[-1] == "hook_resid_pre":
            src_node = ["mlp", 0]
        else:
            print(src)
            raise Exception()

        edges[src_node[0] + "-" + dest_node[0]].append(dest_node[1:] + src_node[1:])

    for k in edges:
        print(edges[k])
        edges[k] = torch.tensor(edges[k])
    print("Extra edges", extra_edges)
    return edges

# %%


dataset="gt"
ablation_type="mean"
acdc_ds = "greaterthan" if dataset == "gt" else dataset

results_folder = f"results/raw/{acdc_ds}/{ablation_type}"
processed_folder = f"results/processed/{dataset}/{ablation_type}"

if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)

g = glob.glob(f"{results_folder}/*.pkl")
# g = glob.glob("eap_results_ioi/*.pkl")
# g = ["another_final_edges_0.001.pkl"]

for gfile in g:
    thresh = gfile.replace(f"{results_folder}/raw_edges_", "").replace(".pkl", "")
    # thresh = gfile.replace("eap_results_ioi/", "").replace(".pkl", "")

    with open(gfile, "rb") as f:
        my_edges = pickle.load(f)
    print(thresh)
    print(len(my_edges))

    edges = get_edges(my_edges)

    # attn-attn: dest_circ dest_layer dest_head src_layer src_head
    # attn-mlp: dest_layer src_layer src_head
    # mlp-attn: dest_circ dest_layer dest_head src_layer
    # mlp-mlp: dest_layer src_layer
    # mlps are 0-13 (1-12 for the real MLPs)

    torch.save(edges, f"{processed_folder}/edges_{thresh}.pth")
    # torch.save(edges, f"acdc_gt_runs/edges_{thresh}.pth")
    # torch.save(edges, f"eap_ioi_runs/edges_{thresh}.pth")

# %%

# things = get_all_ioi_things(num_examples=100, device="cuda:0", metric_name="kl_div")
# # %%
# true_edges_raw = get_ioi_true_edges(things.tl_model)
# # %%
# true_edges = get_edges(list(true_edges_raw.items()))
# # %%
# torch.save(true_edges, f"acdc_ioi_runs/edges_manual.pth")
# # %%
# things = get_all_greaterthan_things(num_examples=100, device="cuda:0", metric_name="kl_div")
# true_edges_raw = get_greaterthan_true_edges(things.tl_model)
# true_edges = get_edges(list(true_edges_raw.items()))
# torch.save(true_edges, f"acdc_gt_runs/edges_manual.pth")

# # %%
# import torch

# gt_edges = torch.load(f"acdc_gt_runs/edges_manual.pth")
# # %%

# with open(f"../inverseprobes/results/oca/ioi/means_attention.pkl", "rb") as f:
#     #  n_layers x 10 (seq_len) x n_heads x d_head
#     init_modes_attention = pickle.load(f)
# with open(f"../inverseprobes/results/oca/ioi/means_mlp.pkl", "rb") as f:
#     # n_layers x 10 (seq_len) x d_model
#     init_modes_mlp = pickle.load(f)

# # %%
# init_modes_attention.shape
# # %%
