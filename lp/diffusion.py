import mkl
import numpy as np
import scipy
from scipy.sparse import csr_matrix, eye, diags
from scipy.sparse import linalg as s_linalg
import faiss
from faiss import normalize_L2
import time
import scipy.stats
import torch.nn.functional as F
import torch
from collections import Counter

mkl.get_max_threads()


def normalize_connection_graph(W):
    W = W - diags(W.diagonal())
    D = np.array(1. / np.sqrt(W.sum(axis=1)))
    D[np.isnan(D)] = 0
    D[np.isinf(D)] = 0
    D_mh = diags(D.reshape(-1))
    Wn = D_mh * W * D_mh
    return Wn


def cg_diffusion(qsims, Wn, alpha=0.99, maxiter=20, tol=1e-6):
    Wnn = eye(Wn.shape[0]) - alpha * Wn
    out_sims = []
    for i in range(qsims.shape[0]):
        f, inf = s_linalg.cg(Wnn, qsims[i, :], tol=tol, maxiter=maxiter)
        out_sims.append(f.reshape(-1, 1))
    out_sims = np.concatenate(out_sims, axis=1)
    ranks = np.argsort(-out_sims, axis=0)
    return ranks, out_sims


def cal_confidence(prob, k, threshold):
    flag = 1
    porder = np.sort(prob)
    sum = 0
    for i in range(k):
        sum += flag * porder[i]
        flag *= -1
    return sum > threshold



def part_diffusion_1(X, labels, symbol, labeled_idx, alpha, k, classes, max_iter, p_labels, weights, th):
    N = X.shape[0]
    d = X.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index
    # index = faiss.IndexFlatIP(d)
    normalize_L2(X)
    index.add(X)

    c = time.time()
    D, I = index.search(X, k + 1)
    elapsed = time.time() - c
    print('kNN Search done in %d seconds' % elapsed)
    # Create the graph
    D = D[:, 1:] ** 3
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T

    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))

    W = W + W.T

    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    Z = np.zeros((N, d))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(len(classes)):
        cur_idx = np.where(p_labels == i)
        y = np.zeros((N,))
        y[cur_idx] = 1.0 / len(cur_idx)
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
        Z[:, i] = f
    # Handle numberical errorindex
    Z[Z < 0] = 0
    probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
    probs_l1[probs_l1 < 0] = 0
    entropy = scipy.stats.entropy(probs_l1.T)
    print("entropy", entropy)

    temp = 0
    for i in range(N):
        weights[i] = 1 - entropy[i] / np.log(len(classes))
        plabel = np.argmax(probs_l1[i])
        if entropy[i] < th:
            temp += 1
            flag = True
            if symbol[i] == 1:
                p_labels[i] = np.argmax(probs_l1[i])
                symbol[i] = -1
            else:
                p_labels[i] = np.argmax(probs_l1[i])
                symbol[i] = 1
    if temp == 0:
        flag = False
    p_labels[labeled_idx] = labels[labeled_idx]
    weights = weights / np.max(weights)
    correct_idx = (p_labels == labels)
    correct_idx = correct_idx + 0
    account = Counter(correct_idx)[1]
    p_f = Counter(p_labels)[-1]
    acc = account / (len(labels) - p_f)
    print("acc", acc)
    return acc, weights, symbol, p_labels
