import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_sparse import spmm

from global_parameters import evaluate


class SBG(nn.Module):
    def __init__(self, args, num_a, num_b, x, emb_size_a=32, emb_size_b=32, layer_num=2, spectral_transform="None",
                 spectral_transform_layer=3):
        super(SBG,self).__init__()
        self.emb_size_a = emb_size_a
        self.emb_size_b = emb_size_b
        self.x = nn.Parameter(x, requires_grad=False)
        self.args = args

        # initial feature transformer
        self.trans_g_org = nn.Linear(self.emb_size_a, self.emb_size_b, bias=False)
        self.trans_g_pos = nn.Linear(self.emb_size_a, self.emb_size_b, bias=False)
        self.trans_g_neg = nn.Linear(self.emb_size_a, self.emb_size_b, bias=False)
        self.x_org_batchnorm = nn.BatchNorm1d(self.emb_size_b)
        self.x_pos_batchnorm = nn.BatchNorm1d(self.emb_size_b)
        self.x_neg_batchnorm = nn.BatchNorm1d(self.emb_size_b)

        # emb trans
        self.mlp_emb = nn.Linear(self.emb_size_b*3, self.emb_size_b, bias=False)
        self.mlp_emb_batchnorm = nn.BatchNorm1d(self.emb_size_b)

        self.update_func = nn.Sequential(
            nn.Dropout(args.dropout),
        )
        self.activation = nn.PReLU()
        self.link_predictor = ScorePredictor(args, dim_embs=self.emb_size_b)
        self.alpha1 = 1
        self.alpha2 = 1
        self.alpha3 = 1

        self.pos_p1 = 1
        self.neg_p1 = -1

        self.spectral_transform = spectral_transform
        self.spectral_transform_layer = spectral_transform_layer
        assert spectral_transform_layer in list(range(1, 10))

    # def JacobiConv(self, L, xs, adj, a=1.0, b=1.0):
    def JacobiConv(self, L, xs, adj_index, adj_value, a=1.0, b=1.0):
        '''
        Jacobi Bases. Please refer to our paper for the form of the bases.
        '''
        if L == 0: return xs[0]
        if L == 1:
            coef1 = (a - b) / 2
            coef2 = (a + b + 2) / 2
            return coef1 * xs[-1] + coef2 * spmm(index=adj_index, value=adj_value, n=xs[-1].shape[0], m=xs[-1].shape[0],
                                                 matrix=xs[-1])
        coef_1 = (2 * L + a + b - 1) * (2 * L + a + b) / (2 * L) / (L + a + b)
        coef_2 = (2 * L + a + b - 1) * (a**2 - b**2) / (2 * L) / (L + a + b) / (2 * L + a + b - 2)
        coef_3 = (L + a) * (L + b) * (2 * L + a + b) / L / (L + a + b) / (2 * L + a + b - 2)
        X = coef_1 * spmm(index=adj_index, value=adj_value, n=xs[-1].shape[0], m=xs[-1].shape[0],
                          matrix=xs[-1]) + coef_2 * xs[-1]
        X -= coef_3 * xs[-2]
        return X

    def forward(self, edges, edge_weights):
        pos_index, neg_index, other_index = edges
        pos_weight, neg_weight, other_weight = edge_weights

        x_org = self.trans_g_org(self.x)
        x_pos = self.trans_g_pos(self.x)
        x_neg = self.trans_g_neg(self.x)

        x_org = self.update_func(x_org)
        x_pos = self.update_func(x_pos)
        x_neg = self.update_func(x_neg)

        alpha1 = self.alpha1
        alpha2 = self.alpha2
        alpha3 = self.alpha3

        a_b_dict = {
            "Chebyshev": (-1/2, -1/2),
            "Legendre": (0, 0),
            "Jacobi_paper": (1, 1),
            "Jacobi_gossip": (1/2, -1/2)
        }
        if self.spectral_transform != "None":
            a, b = a_b_dict[self.spectral_transform]
        if self.spectral_transform == "None":
            x_pos_1 = alpha1*spmm(index=pos_index, value=pos_weight, n=x_pos.shape[0], m=x_pos.shape[0],
                                  matrix=x_pos).float()
        else:
            xs = [self.JacobiConv(0, [x_pos], None, None, a=a, b=b)]
            for L in range(1, self.spectral_transform_layer):
                x = self.JacobiConv(L, xs, pos_index, pos_weight, a=a, b=b)
                xs.append(x)
            x_pos_1 = xs[-1]

        x_pos = x_pos_1
        if self.spectral_transform == "None":
            x_neg_1 = alpha2*spmm(index=neg_index, value=neg_weight, n=x_neg.shape[0], m=x_neg.shape[0],
                                  matrix=x_neg).float()
        else:
            xs = [self.JacobiConv(0, [x_neg], None, None, a=a, b=b)]
            for L in range(1, self.spectral_transform_layer):
                x = self.JacobiConv(L, xs, neg_index, neg_weight, a=a, b=b)
                xs.append(x)
            x_neg_1 = xs[-1]

        x_neg = x_neg_1
        x_org = self.x_org_batchnorm(x_org)
        x_pos = self.x_pos_batchnorm(x_pos)
        x_neg = self.x_neg_batchnorm(x_neg)
        x_org = alpha3 * x_org
        x_org = self.activation(x_org)
        x_pos = self.activation(x_pos)
        x_neg = self.activation(x_neg)


        self.embs = torch.cat([x_org, x_pos, x_neg], dim=1)
        self.embs = self.mlp_emb(self.embs)
        self.embs = F.normalize(self.embs, p=2, dim=1)
        self.embs = self.update_func(self.embs)


    def predict_combine(self, embs, uids, vids):
        # u_embs = self.combine(embs, uids)
        # v_embs = self.combine(embs, vids)
        u_embs = embs[uids]
        v_embs = embs[vids]
        score = self.link_predictor(u_embs, v_embs)
        return score

    def compute_label_loss(self, score, y_label):
        return F.binary_cross_entropy_with_logits(score, y_label)

    def compute_attention(self, embs):
        attn = self.attention(embs).softmax(dim=0)
        return attn

    def combine(self, embs, nids, device):
        if self.args.sign_conv == 'sign':
            if self.args.sign_aggre == 'pos':
                embs = (embs[0], embs[1])
            elif self.args.sign_aggre == 'neg':
                embs = (embs[2], embs[3])

        if self.combine_type == 'concat':
            embs = torch.cat(embs, dim=-1)
            sub_embs = embs[nids].to(device)
            out_embs = self.transform(sub_embs)
            return out_embs
        elif self.combine_type == 'attn':
            embs = torch.stack(embs, dim=0)
            sub_embs = embs[:, nids].to(device)
            attn = self.compute_attention(sub_embs)
            # attn: (2,n,1)   sub_embs: (2,n,feature)
            out_embs = (attn * sub_embs).sum(dim=0)
            return out_embs
        elif self.combine_type == 'mean':
            embs = torch.stack(embs, dim=0).mean(dim=0)
            sub_embs = embs[nids].to(device)
            return sub_embs
        elif self.combine_type == 'pos':
            sub_embs = embs[0][nids].to(device)
            return sub_embs


class ScorePredictor(nn.Module):
    def __init__(self, args, dim_embs, **params):
        super().__init__()
        self.args = args
        self.dim_embs = dim_embs

        if self.args.predictor == 'dot':
            pass
        elif self.args.predictor == '1-linear':
            self.predictor = nn.Linear(self.dim_embs * 2, 1)
        elif self.args.predictor == '2-linear':
            self.predictor = nn.Sequential(nn.Linear(self.dim_embs * 2, self.dim_embs),
                                           nn.LeakyReLU(),
                                           # nn.BatchNorm1d(self.dim_embs),
                                           nn.Linear(self.dim_embs, 1))
        elif self.args.predictor == '3-linear':
            self.predictor = nn.Sequential(nn.Linear(self.dim_embs * 2, self.dim_embs),
                                           nn.LeakyReLU(),
                                           nn.Linear(self.dim_embs, self.dim_embs),
                                           nn.LeakyReLU(),
                                           nn.Linear(self.dim_embs, 1)
                                           )
        elif self.args.predictor == '4-linear':
            self.predictor = nn.Sequential(nn.Linear(self.dim_embs * 2, self.dim_embs),
                                           nn.LeakyReLU(),
                                           nn.Linear(self.dim_embs, self.dim_embs),
                                           nn.LeakyReLU(),
                                           nn.Linear(self.dim_embs, self.dim_embs),
                                           nn.LeakyReLU(),
                                           nn.Linear(self.dim_embs, 1)
                                           )
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, u_e, u_v):
        if self.args.predictor == 'dot':
            score = u_e.mul(u_v).sum(dim=-1)
        else:
            x = torch.cat([u_e, u_v], dim=-1)
            score = self.predictor(x).flatten()
        return score


@torch.no_grad()
def test_and_val(pred_y, y, mode='val', epoch=0):
    res = evaluate(pred_y, y, mode, epoch)
    return res





