# %%
import torch
import pickle
# %%
with open("my_edges.pkl", "rb") as f:
    q = pickle.load(f)
# %%

for edge in q:
    dest = edge[0].split(".")
    if dest[-1] == "hook_resid_post":
        dest_code=("mlp", 13)
    elif (dest[-1] == "hook_result" and dest[-2] == "attn") or dest[-1] == "hook_v" or dest[-1] == "hook_q" or dest[-1] == "hook_k" or dest[-1] == "hook_mlp_out":
        continue
    elif (dest[-1] == "hook_mlp_in"):
        dest_code=("mlp", int(dest[1]) + 1)
    elif dest[2].endswith("input"):
        dest_code=("attn", int(dest[1]), dest[-1].replace("hook_", "").replace("_input",""), int(str(edge[1]).split(",")[-1].replace(" ", "").replace("]", "")))
        # print(dest_code)
    
    orig = edge[2].split(".")
    if orig[-1] == "hook_result" and orig[-2] == "attn":
        orig_code=("attn", orig[1], orig[-1].replace("hook_", "").replace("_input",""), str(edge[-1]).split(",")[-1].replace(" ", "").replace("]", ""))
    elif orig[-1] == "hook_mlp_out":
        dest_code=("mlp", )
        print(orig)
        break
    # print(edge[2].split("."))
    # break
# %%
from acdc.greaterthan.utils import get_greaterthan_true_edges
from acdc.ioi.utils import get_gpt2_small
model = get_gpt2_small(device="cuda:0")
all_edges = get_greaterthan_true_edges(model)

# %%
