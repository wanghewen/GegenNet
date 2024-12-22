import random
import sys
from global_parameters import find_gpus, evaluate
import torch
import numpy as np

from data_load import load_data
from new_sbg import SBG, test_and_val
import argparse
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, diags
from sklearn import preprocessing
import CommonModules as CM


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true', default=False,
                    help='debug mode')
parser.add_argument('--dataset', type=str, default='review',
                    help='choose dataset')
parser.add_argument('--seed', type=int, default=2023,
                    help='Random seed.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout parameter')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight Decay')
parser.add_argument('--predictor', type=str, default='2-linear',
                    help='decoder method')  # VERY IMPORTANT! MUCH BETTER THAN 1-linear
parser.add_argument('--lr', type=float, default=1e-2,
                    help='Initial learning rate.')
parser.add_argument('--dim_embs', type=int, default=32,
                    help='initial embedding size of node')  # Final edge embedding will be this * 2
parser.add_argument('--svd_dim', type=int, default=32,
                    help='initial embedding size of node')
parser.add_argument('--epochs', type=int, default=300,
                    help='initial embedding size of node')
parser.add_argument('--eigen_method', type=str, default="SA_LA",
                    help='eigen decomposition method')
parser.add_argument('--beta', type=float, default=0,
                    help='the proportion of SA vs LA')
parser.add_argument('--spectral_transform', type=str, default="Jacobi_paper",
                    help='spectral transform method')
parser.add_argument('--spectral_transform_layer', type=int, default=3,
                    help='spectral transform layer number')
parser.add_argument('--use_cache_eigen_results', action='store_true',
                    help='whether to use cached eigen results')
parser.add_argument('--dataset_random_seed', type=int, default=None,
                    help='whether to split the train/val/test dataset according to this seed')


args = parser.parse_args()
print(args)
print("dataset_random_seed:", getattr(args, "dataset_random_seed", None))
eigen_method_A = args.eigen_method.split('_')[0]
eigen_method_A_org = args.eigen_method.split('_')[1]
gpus, accelerator = find_gpus(1)
device = f"cuda:{gpus[0]}" if accelerator == "gpu" else "cpu"

seed_everything(args.seed)
torch.use_deterministic_algorithms(True)

dataset = args.dataset

(train_edgelist, val_edgelist, test_edgelist, set_a_num, set_b_num, edges_num,
 nodes_num) = load_data(dataset, random_seed=args.dataset_random_seed)

edges = np.concatenate((train_edgelist, val_edgelist, test_edgelist), axis=0)
num_a = edges[:, 0].max() - edges[:, 0].min() + 1
num_b = edges[:, 1].max() - edges[:, 1].min() + 1

train_edgelist[:, 1] = train_edgelist[:, 1] + num_a
val_edgelist[:, 1] = val_edgelist[:, 1] + num_a
test_edgelist[:, 1] = test_edgelist[:, 1] + num_a

val_pos_mask = val_edgelist[:, 2] > 0
val_neg_mask = val_edgelist[:, 2] < 0

test_pos_mask = test_edgelist[:, 2] > 0
test_neg_mask = test_edgelist[:, 2] < 0

pos_rows = []
pos_cols = []
pos_vals = []
neg_rows = []
neg_cols = []
neg_vals = []
for a, b, s in train_edgelist:
    if s==1:
        pos_rows.append(a)
        pos_cols.append(b)
        pos_vals.append(s)
    elif s==-1:
        neg_rows.append(a)
        neg_cols.append(b)
        neg_vals.append(s)

train_pos_edges = torch.from_numpy(np.vstack([np.array(pos_rows), np.array(pos_cols)]))
train_neg_edges = torch.from_numpy(np.vstack([np.array(neg_rows), np.array(neg_cols)]))
train_pos_edges_vals = torch.from_numpy(np.array(pos_vals))
train_neg_edges_vals = torch.from_numpy(np.array(neg_vals))

rows = np.hstack([pos_rows,neg_rows])
cols = np.hstack([pos_cols,neg_cols])
vals = np.hstack([pos_vals,neg_vals])
n = max(max(rows), max(cols))+1
A_org = csr_matrix((np.hstack([vals,vals]),(np.hstack([rows,cols]),np.hstack([cols,rows]))), dtype=float)
A_pos = csr_matrix((np.hstack([pos_vals,pos_vals]),(np.hstack([pos_rows,pos_cols]),np.hstack([pos_cols,pos_rows]))), shape=(n,n), dtype=float)
D_pos = diags(np.asarray(A_pos.sum(axis=0)).reshape(-1))
A_pos = D_pos-A_pos
A_neg = csr_matrix((np.hstack([neg_vals,neg_vals]),(np.hstack([neg_rows,neg_cols]),np.hstack([neg_cols,neg_rows]))), shape=(n,n), dtype=float)
D_neg = diags(np.asarray(A_neg.sum(axis=0)).reshape(-1))
A_neg = D_neg-A_neg

A = A_pos + A_neg

print("eigen solve for A...")  # 3.2 L
v0 = np.random.rand(min(A.shape))  # Generate reproducible results: https://stackoverflow.com/a/52403508
if args.svd_dim >= A.shape[0]:
    print("svd_dim is larger than A. Exiting...")
    sys.exit()
if eigen_method_A != "None":
    cached_eigen_file = f"eigenvec_A_svd_dim_{args.svd_dim}_dataset_{args.dataset}.pkl"
    if CM.IO.FileExist(cached_eigen_file) and args.use_cache_eigen_results:
        print("Import eigen results for A...")
        eigval, eigvec = CM.IO.ImportFromPkl(cached_eigen_file)
    else:
        eigval, eigvec = eigsh(A, k=args.svd_dim, v0=v0, which=eigen_method_A)
        if args.use_cache_eigen_results:
            print("Export eigen results for A...")
            CM.IO.ExportToPkl(cached_eigen_file, [eigval, eigvec])

A_org = preprocessing.normalize(A_org, norm='l2', axis=1)
print("eigen solve for A_org...")  # 3.2 B
if eigen_method_A_org != "None":
    cached_eigen_file = f"eigenvec_A_org_svd_dim_{args.svd_dim}_dataset_{args.dataset}.pkl"
    if CM.IO.FileExist(cached_eigen_file) and args.use_cache_eigen_results:
        print("Import eigen results for A_org...")
        eigval2, eigvec2 = CM.IO.ImportFromPkl(cached_eigen_file)
    else:
        eigval2, eigvec2 = eigsh(A_org, k=args.svd_dim, v0=v0, which=eigen_method_A_org)
        if args.use_cache_eigen_results:
            print("Export eigen results for A_org...")
            CM.IO.ExportToPkl(cached_eigen_file, [eigval2, eigvec2])

beta = args.beta

if eigen_method_A != "None" and eigen_method_A_org != "None":
    eigvec = beta*eigvec + (1-beta)*eigvec2
elif eigen_method_A != "None":
    eigvec = eigvec
else:
    eigvec = eigvec2

x = torch.from_numpy(eigvec).to(device).float()

# train and test edges
val_pos_edges = torch.from_numpy(val_edgelist[val_pos_mask, 0:2].T) # [2, edges_n]
val_neg_edges = torch.from_numpy(val_edgelist[val_neg_mask, 0:2].T)
test_pos_edges = torch.from_numpy(test_edgelist[test_pos_mask, 0:2].T)
test_neg_edges = torch.from_numpy(test_edgelist[test_neg_mask, 0:2].T)

pos_index = train_pos_edges.to(device)
neg_index = train_neg_edges.to(device)
other_index = torch.cat([val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges], dim=1).to(device)
pos_weight = train_pos_edges_vals.to(device)
neg_weight = train_neg_edges_vals.to(device)
other_weight = torch.ones(other_index.shape[1]).to(device)

model = SBG(args, num_a, num_b, x, emb_size_a=x.shape[1], emb_size_b=args.dim_embs, spectral_transform=args.spectral_transform,
            spectral_transform_layer=args.spectral_transform_layer).to(device)
x: torch.Tensor
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
cnt = 0

uids_train = torch.cat([train_pos_edges[0], train_neg_edges[0]]).long().to(device)
vids_train = torch.cat([train_pos_edges[1], train_neg_edges[1]]).long().to(device)
y_label_train = torch.cat([torch.ones(train_pos_edges.shape[1]), torch.zeros(train_neg_edges.shape[1])]).to(device)

uids_val = torch.cat([val_pos_edges[0], val_neg_edges[0]]).long().to(device)
vids_val = torch.cat([val_pos_edges[1], val_neg_edges[1]]).long().to(device)
y_label_val = torch.cat([torch.ones(val_pos_edges.shape[1]), torch.zeros(val_neg_edges.shape[1])]).to(device)

uids_test = torch.cat([test_pos_edges[0], test_neg_edges[0]]).long().to(device)
vids_test = torch.cat([test_pos_edges[1], test_neg_edges[1]]).long().to(device)
y_label_test = torch.cat([torch.ones(test_pos_edges.shape[1]), torch.zeros(test_neg_edges.shape[1])]).to(device)

res_best = {'val_auc': 0, 'val_f1':0, 'val_macro_f1':0}
print_list = ['val_epoch', 'test_auc', 'test_f1', 'test_macro_f1', 'test_micro_f1']
A_pos = csr_matrix((pos_vals, np.vstack([np.array(pos_rows), np.array(pos_cols)])), shape=(model.x.shape[0],
                                                                                          model.x.shape[0]), dtype=np.float32)
A_neg = csr_matrix((neg_vals, np.vstack([np.array(neg_rows), np.array(neg_cols)])), shape=(model.x.shape[0],
                                                                                          model.x.shape[0]), dtype=np.float32)

for epoch in range(args.epochs):

    edges = (pos_index, neg_index, other_index)
    edge_weights = (pos_weight, neg_weight, other_weight)
    model(edges, edge_weights)

    y_score = model.predict_combine(model.embs, uids_train, vids_train)

    loss_label = model.compute_label_loss(y_score, y_label_train)
    loss = loss_label
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    res_cur = dict()
    cnt += 1
    model.eval()
    y_score_train = model.predict_combine(model.embs, uids_train, vids_train)
    res = test_and_val(y_score_train, y_label_train, mode='train', epoch=epoch )
    res_cur.update(res)
    y_score_val = model.predict_combine(model.embs, uids_val, vids_val)
    res = test_and_val(y_score_val, y_label_val, mode='val', epoch=epoch)
    res_cur.update(res)
    y_score_test = model.predict_combine(model.embs, uids_test, vids_test)
    res = test_and_val(y_score_test, y_label_test, mode='test', epoch=epoch)
    res_cur.update(res)

    if res_cur['val_auc'] + res_cur['val_macro_f1'] > res_best['val_auc'] + res_best['val_macro_f1']:
        res_best = res_cur
        best_y_score_test = y_score_test
        best_y_label_test = y_label_test
        print(res_best)
        for i in print_list:
            print(i, res_best[i], end=' ')
        print("")

evaluate(best_y_score_test, best_y_label_test, mode='test', epoch=res_best["val_epoch"], print_result=True, is_best=True)




