"""
蒸馏选号器 —— distill_select.py（3D 组选号 + 快乐8 选号）

「蒸馏」在这里的意思
  老师：每晚（甚至一天训练很多版本）的 RL、以及共识投票(已含 ML/DL)、采样 给出的推荐。
        同一期有多个 RL 版本时，全部记录下来，求"多版本一致度"——多个版本都看好的号码得分高。
  学生：一个很轻的选号器 = 「开奖号码信号」模型（用几千期开奖号码训练，冷热/遗漏/重号/邻号等）
        + 「多版本一致度」两者按权重融合，直接给出选号。
  检验：滚动前推——每一期只用它之前的数据训练、预测它，对比随机期望和朴素频率基线。

标准（已调低、调灵敏）
  选号器【始终输出蒸馏结果】，不再因为没通过严格检验就退回共识。
  检验结果用「证据强度」标出来，不隐藏：
    strong 显著(z≥2) / weak 有迹象(1≤z<2) / none 没有证据(z<1，仍输出，仅供参考)
  训练用了验证集自动选正则强度，开奖信号也加了更多维度，提高灵敏度。

数据来源
  history.json                      开奖号码（几千期）
  dl_rl.json                        当前 RL 结果
  rl_archive/*/*.json               当天各个 RL 版本
  git 历史里的 dl_rl.json 与 rl_archive 的各个旧版本（--backfill-git）
  prediction.json、dl_lstm_tfm.json（含 git 历史）  ML、DL 各目标概率，按 RL 共识同一套换算成候选号码的对数提升，作为独立两路老师
  同一期：RL 多版本合并成「一致度」；RL、ML、DL 三路各自标准化后等权相加

用法
  python distill_select.py                 # 正常运行
  python distill_select.py --backfill-git  # 从 git 历史补录所有 RL 版本
  python distill_select.py --selftest      # 合成数据自检

输出
  distill_log.jsonl      每行=某游戏某一期，所有 RL 版本的信号累计（同版本不重复计）
  distill_result.json    两个游戏的选号、证据强度、检验结果（网页读这个）
"""

import json, os, math, subprocess, argparse, itertools, random, glob
from datetime import datetime, timezone, timedelta
import numpy as np

LOG, OUT = 'distill_log.jsonl', 'distill_result.json'
BJ = timezone(timedelta(hours=8))
Z_STRONG, Z_WEAK = 2.0, 1.0           # 证据强度档位（只用来标注，不再决定"用不用蒸馏结果"）
L2_GRID = (0.3, 1.0, 3.0, 10.0)       # 正则强度候选，用验证集自动选
D3_TOPK = 12
D3_EVAL, D3_TRAIN_WIN, D3_REFIT, D3_VALID = 1000, 3000, 100, 300
K8_TOPN = 10
K8_EVAL, K8_TRAIN_WIN, K8_REFIT, K8_VALID = 400, 1500, 50, 150
K8_PLAYS = [4, 5, 6, 9, 10]
LAMBDA_DEFAULT = 0.5                  # 老师(多版本一致度)在融合里的默认权重
LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
LOGGED_FIT_MIN = 10                   # 有标签的"有推荐记录的期"≥它，才用这些期去挑融合权重


# ═════════════════════════ 通用 ═════════════════════════
def _z(hits, exp, var):
    return round(float((hits - exp) / math.sqrt(var)), 2) if var > 0 else 0.0


def _read_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def zs(v):
    v = np.asarray(v, dtype=np.float64); sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-12 else np.zeros_like(v)


def zs2(v):
    """30 维（3D 的 位×数字）按位各自标准化；其它长度（快乐8 的 80）整体标准化"""
    v = np.asarray(v, dtype=np.float64)
    return np.concatenate([zs(r) for r in v.reshape(3, 10)]) if v.size == 30 else zs(v)


def newton_clogit(X, t, l2, iters=15):
    n, C, F = X.shape; w = np.zeros(F)
    for _ in range(iters):
        s = X @ w; s -= s.max(1, keepdims=True)
        pi = np.exp(s); pi /= pi.sum(1, keepdims=True)
        m = np.einsum('nc,ncf->nf', pi, X)
        g = (X[np.arange(n), t] - m).sum(0) - l2 * w
        H = -(np.einsum('nc,ncf,ncg->fg', pi, X, X) - np.einsum('nf,ng->fg', m, m)) - l2 * np.eye(F)
        try: step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError: break
        w = w - step
        if np.abs(step).max() < 1e-6: break
    return w


def clogit_loglik(X, t, w):
    s = X @ w; s -= s.max(1, keepdims=True)
    return float((s[np.arange(len(X)), t] - np.log(np.exp(s).sum(1))).mean())


def newton_logit(X, y, l2, iters=15):
    N, F = X.shape; w = np.zeros(F)
    for _ in range(iters):
        z = X @ w; p = 1 / (1 + np.exp(-z))
        g = X.T @ (y - p) - l2 * w
        H = -(X.T * (p * (1 - p))) @ X - l2 * np.eye(F)
        try: step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError: break
        w = w - step
        if np.abs(step).max() < 1e-6: break
    return w


def logit_loglik(X, y, w):
    z = X @ w
    return float((y * z - np.log1p(np.exp(z))).mean())


# ═════════════════════════ 3D：组选号 ═════════════════════════
GROUPS = sorted({''.join(map(str, sorted((a, b, c)))) for a in range(10) for b in range(10) for c in range(10)})
G_IDX = {g: i for i, g in enumerate(GROUPS)}
G_Q = np.array([(6 if len(set(g)) == 3 else 3 if len(set(g)) == 2 else 1) / 1000.0 for g in GROUPS])
G_KIND3 = np.array([1.0 if len(set(g)) == 2 else 0.0 for g in GROUPS])
G_KIND1 = np.array([1.0 if len(set(g)) == 1 else 0.0 for g in GROUPS])
G_SUM = np.array([sum(map(int, g)) for g in GROUPS])
G_SPAN = np.array([int(max(g)) - int(min(g)) for g in GROUPS])
_allsum = np.array([a + b + c for a in range(10) for b in range(10) for c in range(10)])
_allspan = np.array([max(a, b, c) - min(a, b, c) for a in range(10) for b in range(10) for c in range(10)])
Q_SUM = np.bincount(_allsum, minlength=28) / 1000.0
Q_SPAN = np.bincount(_allspan, minlength=10) / 1000.0
_PERM = np.zeros((220, 6, 3), dtype=np.int64); _PM = np.zeros((220, 6))
for i, g in enumerate(GROUPS):
    ps = sorted(set(itertools.permutations(map(int, g))))
    for j, p in enumerate(ps): _PERM[i, j] = p; _PM[i, j] = 1.0
_PN = _PM.sum(1)
D3_HF = ['ln_f30', 'ln_f100', 'ln_f300', 'omit_ratio', 'pos_hot', 'share_last', 'ln_sum100', 'ln_span100',
         'ln_f10', 'omit_sq', 'is_zu3', 'is_baozi', 'ln_q']


def d3_group_seq(hist):
    return np.array([G_IDX[''.join(map(str, sorted(int(x) for x in r['digits'])))] for r in hist])


def d3_hist_features(hist, t0):
    """t∈[t0,T]：预测第 t 期，只用 <t 的开奖。返回 (特征 (T+1-t0,220,F), 组选序号序列)"""
    T = len(hist)
    gseq = d3_group_seq(hist)
    digs = np.array([[int(x) for x in r['digits']] for r in hist])
    oh = np.zeros((T + 1, 220), dtype=np.int32)
    ohp = np.zeros((T + 1, 3, 10), dtype=np.int32)
    ohs = np.zeros((T + 1, 28), dtype=np.int32); ohk = np.zeros((T + 1, 10), dtype=np.int32)
    for i in range(T):
        oh[i + 1] = oh[i]; oh[i + 1, gseq[i]] += 1
        ohp[i + 1] = ohp[i]
        for p in range(3): ohp[i + 1, p, digs[i, p]] += 1
        ohs[i + 1] = ohs[i]; ohs[i + 1, int(digs[i].sum())] += 1
        ohk[i + 1] = ohk[i]; ohk[i + 1, int(digs[i].max() - digs[i].min())] += 1
    last = np.full(220, -1)
    for i in range(t0): last[gseq[i]] = i
    out = np.zeros((T + 1 - t0, 220, len(D3_HF)), dtype=np.float32)
    lnq = np.log(G_Q)
    for t in range(t0, T + 1):
        r = out[t - t0]
        for k, w in enumerate((30, 100, 300)):
            c = oh[t] - oh[max(0, t - w)]
            r[:, k] = np.log((c + 0.5) / ((min(t, w)) * G_Q + 0.5))
        om = np.minimum((t - 1 - last) * G_Q, 6.0)
        r[:, 3] = om
        pc = ohp[t] - ohp[max(0, t - 50)]
        lnf = np.log((pc + 0.5) / (min(t, 50) / 10.0 + 0.5))
        val = lnf[0][_PERM[:, :, 0]] + lnf[1][_PERM[:, :, 1]] + lnf[2][_PERM[:, :, 2]]
        r[:, 4] = (val * _PM).sum(1) / _PN
        if t >= 1:
            ld = list(digs[t - 1]); sh = np.zeros(220)
            for i, g in enumerate(GROUPS):
                a = [int(c) for c in g]; n = 0
                for d in ld:
                    if d in a: a.remove(d); n += 1
                sh[i] = n
            r[:, 5] = sh
        w100 = min(t, 100)
        cs_ = ohs[t] - ohs[max(0, t - 100)]; ck_ = ohk[t] - ohk[max(0, t - 100)]
        r[:, 6] = np.log((cs_[G_SUM] + 0.5) / (w100 * Q_SUM[G_SUM] + 0.5))
        r[:, 7] = np.log((ck_[G_SPAN] + 0.5) / (w100 * Q_SPAN[G_SPAN] + 0.5))
        c10 = oh[t] - oh[max(0, t - 10)]
        r[:, 8] = np.log((c10 + 0.5) / (min(t, 10) * G_Q + 0.5))
        r[:, 9] = om ** 2 / 6.0
        r[:, 10] = G_KIND3; r[:, 11] = G_KIND1; r[:, 12] = lnq
        if t < T: last[gseq[t]] = t
    return out, gseq


def choose_l2_clogit(X, y, start, valid):
    """用 [start-valid, start) 做验证，挑对数似然最高的正则强度（只用评估窗口之前的数据）"""
    lo = max(0, start - valid)
    best, bl = L2_GRID[1], -1e18
    tr_lo = max(0, lo - 3000)
    for l2 in L2_GRID:
        w = newton_clogit(X[tr_lo:lo].astype(np.float64), y[tr_lo:lo], l2)
        ll = clogit_loglik(X[lo:start].astype(np.float64), y[lo:start], w)
        if ll > bl: bl, best = ll, l2
    return best


def d3_eval(X, y, eval_n, train_win, refit, valid=D3_VALID, topk=D3_TOPK):
    N = len(X); start = N - eval_n
    l2 = choose_l2_clogit(X, y, start, valid)
    acc = {'model': [0, 0.0, 0.0], 'freq': [0, 0.0, 0.0]}
    w = None; diff = []
    for t in range(start, N):
        if w is None or (t - start) % refit == 0:
            lo = max(0, t - train_win); w = newton_clogit(X[lo:t].astype(np.float64), y[lo:t], l2)
        hm = hf = 0
        for name, sc in (('model', X[t] @ w), ('freq', X[t][:, 0])):
            pk = np.argsort(-sc, kind='stable')[:topk]; e = float(G_Q[pk].sum())
            h = int(y[t] in pk); acc[name][0] += h; acc[name][1] += e; acc[name][2] += e * (1 - e)
            if name == 'model': hm = h
            else: hf = h
        diff.append(hm - hf)
    out = {k: {'hits': int(v[0]), 'expected': round(v[1], 2), 'z': _z(*v)} for k, v in acc.items()}
    out['n_eval'] = eval_n; out['l2'] = l2
    d = np.array(diff, dtype=float)
    out['model_minus_freq'] = {'mean_hits': round(float(d.mean()), 4),
                               'z': round(float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))), 2) if d.std(ddof=1) > 0 else 0.0}
    lo = max(0, N - train_win)
    return out, newton_clogit(X[lo:].astype(np.float64), y[lo:], l2)


# ═════════════════════════ 3D：直选（百位 / 十位 / 个位 各自 10 选 1）═════════════════════════
D3P_HF = ['ln_f10', 'ln_f30', 'ln_f100', 'ln_f300', 'omit', 'omit_sq', 'same_prev', 'in_prev_any', 'ln_fall100', 'is_odd', 'level']
D3P_TOPK = 3                                   # 每位取 3 个数字算命中（随机期望 30%）
POS_NAMES = ['百位', '十位', '个位']


def d3p_hist_features(hist, t0):
    """预测第 t 期（t∈[t0,T]），只用 <t 的开奖。返回 (特征 (T+1-t0,3,10,F), 开奖数字 (T,3))"""
    T = len(hist)
    digs = np.array([[int(x) for x in r['digits']] for r in hist])
    cs = np.zeros((T + 1, 3, 10), dtype=np.int32)
    for i in range(T):
        cs[i + 1] = cs[i]
        for p in range(3): cs[i + 1, p, digs[i, p]] += 1
    last = np.full((3, 10), -1)
    for i in range(t0):
        for p in range(3): last[p, digs[i, p]] = i
    out = np.zeros((T + 1 - t0, 3, 10, len(D3P_HF)), dtype=np.float32)
    odd = np.tile((np.arange(10) % 2).astype(np.float32), (3, 1)); lvl = np.tile(np.arange(10) / 9.0, (3, 1))
    for t in range(t0, T + 1):
        r = out[t - t0]
        for k, w in enumerate((10, 30, 100, 300)):
            c = cs[t] - cs[max(0, t - w)]
            r[:, :, k] = np.log((c + 0.5) / (min(t, w) * 0.1 + 0.5))
        om = np.minimum((t - 1 - last) * 0.1, 6.0)
        r[:, :, 4] = om; r[:, :, 5] = om ** 2 / 6.0
        if t >= 1:
            for p in range(3): r[p, digs[t - 1, p], 6] = 1.0
            for d in set(digs[t - 1].tolist()): r[:, d, 7] = 1.0
        c100 = (cs[t] - cs[max(0, t - 100)]).sum(0)
        r[:, :, 8] = np.log((c100 + 0.5) / (3 * min(t, 100) * 0.1 + 0.5))[None, :]
        r[:, :, 9] = odd; r[:, :, 10] = lvl
        if t < T:
            for p in range(3): last[p, digs[t, p]] = t
    return out, digs


def d3_hits(sc, y):
    """sc: 30 维（3位×10数字）得分；y: 开奖 3 个数字。返回 3 位里"开奖数字落在该位得分前3"的位数"""
    sc = np.asarray(sc).reshape(3, 10)
    return int(sum(y[p] in np.argsort(-sc[p], kind='stable')[:D3P_TOPK] for p in range(3)))


D3_EXP, D3_VAR = 0.9, 0.63                     # 每期期望命中位数 / 方差（3 位各自 30% 命中）


def _flat(X, y, lo, hi):
    return X[lo:hi].reshape(-1, 10, X.shape[-1]).astype(np.float64), y[lo:hi].reshape(-1)


def choose_l2_pos(X, y, start, valid):
    lo = max(0, start - valid); tr_lo = max(0, lo - 1500)
    best, bl = L2_GRID[1], -1e18
    Xv, yv = _flat(X, y, lo, start)
    for l2 in L2_GRID:
        w = newton_clogit(*_flat(X, y, tr_lo, lo), l2)
        ll = clogit_loglik(Xv, yv, w)
        if ll > bl: bl, best = ll, l2
    return best


def d3p_eval(X, y, eval_n, train_win, refit, valid=D3_VALID):
    N = len(X); start = N - eval_n
    l2 = choose_l2_pos(X, y, start, valid)
    acc = {'model': [0, 0.0, 0.0], 'freq': [0, 0.0, 0.0]}; w = None; diff = []
    for t in range(start, N):
        if w is None or (t - start) % refit == 0:
            w = newton_clogit(*_flat(X, y, max(0, t - train_win), t), l2)
        hh = {}
        for name, sc in (('model', X[t] @ w), ('freq', X[t][:, :, 1])):
            h = d3_hits(sc, y[t]); hh[name] = h
            acc[name][0] += h; acc[name][1] += D3_EXP; acc[name][2] += D3_VAR
        diff.append(hh['model'] - hh['freq'])
    out = {k: {'hits': int(v[0]), 'expected': round(v[1], 2), 'z': _z(*v)} for k, v in acc.items()}
    out['n_eval'] = eval_n; out['l2'] = l2
    d = np.array(diff, dtype=float)
    out['model_minus_freq'] = {'mean_hits': round(float(d.mean()), 4),
                               'z': round(float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))), 2) if d.std(ddof=1) > 0 else 0.0}
    return out, newton_clogit(*_flat(X, y, max(0, N - train_win), N), l2)


# ═════════════════════════ 快乐8：逐球 ═════════════════════════
K8_HF = ['ln_f3', 'ln_f5', 'ln_f10', 'ln_f20', 'ln_f30', 'ln_f100', 'omit_ratio', 'omit_sq',
         'in_prev', 'nb_prev', 'zone_hot30', 'is_odd', 'ball_n']


def k8_matrix(hist):
    M = np.zeros((len(hist), 80), dtype=np.int8)
    for i, r in enumerate(hist):
        for b in r['numbers']: M[i, b - 1] = 1
    return M


def k8_hist_features(M, t0):
    T = len(M)
    cs = np.zeros((T + 1, 80), dtype=np.int32); cs[1:] = np.cumsum(M, axis=0)
    last = np.full(80, -1)
    for i in range(t0): last[M[i] == 1] = i
    out = np.zeros((T + 1 - t0, 80, len(K8_HF)), dtype=np.float32)
    ball_n = np.arange(1, 81) / 80.0; odd = (np.arange(1, 81) % 2).astype(np.float32); zone = (np.arange(80) // 20)
    for t in range(t0, T + 1):
        r = out[t - t0]
        for k, w in enumerate((3, 5, 10, 20, 30, 100)):
            c = cs[t] - cs[max(0, t - w)]
            r[:, k] = np.log((c + 0.5) / (min(t, w) * 0.25 + 0.5))
        om = np.minimum((t - 1 - last) / 4.0, 6.0)
        r[:, 6] = om; r[:, 7] = om ** 2 / 6.0
        if t >= 1:
            prev = M[t - 1].astype(np.float32); r[:, 8] = prev
            nb = np.zeros(80, dtype=np.float32); nb[1:] += prev[:-1]; nb[:-1] += prev[1:]
            r[:, 9] = ((nb > 0) & (prev == 0)).astype(np.float32)
        c30 = cs[t] - cs[max(0, t - 30)]
        zc = np.array([c30[zone == z].sum() for z in range(4)])
        r[:, 10] = np.log((zc[zone] + 0.5) / (min(t, 30) * 5.0 + 0.5))
        r[:, 11] = odd; r[:, 12] = ball_n
        if t < T: last[M[t] == 1] = t
    return out


def choose_l2_logit(X, Y, start, valid):
    lo = max(0, start - valid); tr_lo = max(0, lo - 1500)
    best, bl = L2_GRID[1], -1e18
    Xv = X[lo:start].reshape(-1, X.shape[2]).astype(np.float64); Yv = Y[lo:start].reshape(-1).astype(np.float64)
    for l2 in L2_GRID:
        w = newton_logit(X[tr_lo:lo].reshape(-1, X.shape[2]).astype(np.float64), Y[tr_lo:lo].reshape(-1).astype(np.float64), l2)
        ll = logit_loglik(Xv, Yv, w)
        if ll > bl: bl, best = ll, l2
    return best


def k8_eval(X, Y, eval_n, train_win, refit, valid=K8_VALID, topn=K8_TOPN):
    N = len(X); start = N - eval_n
    l2 = choose_l2_logit(X, Y, start, valid)
    exp = topn * 20 / 80.0
    var = topn * 0.25 * 0.75 * (80 - topn) / 79.0
    acc = {'model': 0.0, 'freq': 0.0}; w = None; diff = []
    for t in range(start, N):
        if w is None or (t - start) % refit == 0:
            lo = max(0, t - train_win)
            w = newton_logit(X[lo:t].reshape(-1, X.shape[2]).astype(np.float64), Y[lo:t].reshape(-1).astype(np.float64), l2)
        hm = Y[t][np.argsort(-(X[t] @ w), kind='stable')[:topn]].sum()
        hf = Y[t][np.argsort(-X[t][:, 4], kind='stable')[:topn]].sum()
        acc['model'] += hm; acc['freq'] += hf; diff.append(hm - hf)
    n = eval_n
    out = {k: {'avg_hits': round(float(v) / n, 3), 'expected': round(exp, 3), 'z': _z(float(v), exp * n, var * n)} for k, v in acc.items()}
    out['n_eval'] = n; out['l2'] = l2
    d = np.array(diff, dtype=float)
    out['model_minus_freq'] = {'mean_hits': round(float(d.mean()), 4),
                               'z': round(float(d.mean() / (d.std(ddof=1) / math.sqrt(n))), 2) if d.std(ddof=1) > 0 else 0.0}
    lo = max(0, N - train_win)
    return out, newton_logit(X[lo:].reshape(-1, X.shape[2]).astype(np.float64), Y[lo:].reshape(-1).astype(np.float64), l2)


# ═════════════════════════ 老师信号（每个 RL 版本一份，同一期累加）═════════════════════════
def d3_version(rl):
    """从一个 RL 版本取 3D 的"位置×数字"信号。返回 (版本号, 期号, 日期, 4×30)：[ln(RL概率/随机), 推荐12注计数, 共识直选计数, 采样计数]"""
    r = (rl.get('results') or {}).get('3d')
    if not r or not r.get('base_qihao'): return None
    pp = r.get('rl_pos_probs')
    if not pp or len(pp) != 3 or any(len(p) != 10 for p in pp): return None
    ln = [math.log(max(float(x), 1e-6) / 0.1) for p in pp for x in p]
    def cnt(lst):
        c = [0.0] * 30
        for g in lst or []:
            g = [int(x) for x in g]
            if len(g) == 3:
                for p in range(3): c[p * 10 + g[p]] += 1
        return c
    rec = cnt(r.get('ppo_groups'))
    con = cnt([b['digits'] for b in ((r.get('consensus') or {}).get('bets') or [])])
    samp = cnt(list(r.get('sampled_groups') or []) + list(r.get('sampled_extra') or []))
    vid = str(rl.get('version_time_bj') or rl.get('updated_at') or '')
    return vid, str(r['base_qihao']), r.get('base_date'), [ln, rec, con, samp]


def k8_version(rl):
    r = (rl.get('results') or {}).get('kl8')
    if not r or not r.get('base_qihao'): return None
    scores = r.get('rl_scores') or []
    if len(scores) != 80: return None
    final = ((r.get('structure') or {}).get('ball_final')) or []
    votes, lift = {}, {}
    for e in ((r.get('consensus') or {}).get('table')) or []:
        votes[e['ball']] = e.get('votes', 0); lift[e['ball']] = e.get('lift', 1.0)
    pick = {}
    for lst in (r.get('picks_by_n') or {}).values():
        for b in lst: pick[b] = pick.get(b, 0) + 1
    samp = {b: c for b, c in ((r.get('sampled_stats') or {}).get('ball_top') or [])}
    sz = zs(scores)
    balls = [[float(sz[i]), math.log(max(float(final[i]), 1e-6)) if i < len(final) else 0.0, float(votes.get(i + 1, 0)), float(lift.get(i + 1, 1.0)),
              float(pick.get(i + 1, 0)), float(samp.get(i + 1, 0))] for i in range(80)]
    vid = str(rl.get('version_time_bj') or rl.get('updated_at') or '')
    return vid, str(r['base_qihao']), r.get('base_date'), balls


# ═════════════════════════ ML / DL 信号（与 RL 共识同一套换算：目标概率 → 候选号码的对数权重）═════════════════════════
VOTE_LAYOUT = {
    '3d':  [('sum_grp', 3), ('odd', 4), ('group_type', 3), ('big', 4), ('span_grp', 3), ('road_dom', 3), ('arith', 2)],
    'kl8': [('odd_grp', 3), ('zone_dom', 4), ('tot_grp', 3), ('big_grp', 3), ('five_dom', 5), ('consec_grp', 3), ('range_grp', 3),
            ('repeat_grp', 3), ('diag_grp', 3), ('hot_grp', 3)],
}
VOTE_MIN_TARGETS = 4
K8_MC = 40000                                  # 快乐8 蒙特卡洛样本数（固定种子，所有快照用同一批样本）
_D3_ALL = np.array([[b, s, g] for b in range(10) for s in range(10) for g in range(10)], dtype=np.int64)


def _grp3(x, lo, hi): return np.where(x <= lo, 0, np.where(x <= hi, 1, 2))


def vote_feats_3d(C):
    sm = C.sum(axis=1)
    tri = (C[:, 0] == C[:, 1]) & (C[:, 1] == C[:, 2])
    eq = (C[:, 0] == C[:, 1]) | (C[:, 1] == C[:, 2]) | (C[:, 0] == C[:, 2])
    s3 = np.sort(C, axis=1); span = C.max(axis=1) - C.min(axis=1)
    rc = np.stack([((C % 3) == r).sum(axis=1) for r in range(3)], axis=1)
    return {'sum_grp': _grp3(sm, 9, 17), 'odd': (C % 2 != 0).sum(axis=1), 'group_type': np.where(tri, 0, np.where(eq, 1, 2)),
            'big': (C >= 5).sum(axis=1), 'span_grp': _grp3(span, 3, 6), 'road_dom': rc.argmax(axis=1),
            'arith': (((s3[:, 1] - s3[:, 0]) == (s3[:, 2] - s3[:, 1])) & ((s3[:, 2] - s3[:, 0]) > 0)).astype(np.int64)}


def _k8_nb(prev):
    nb = set()
    for x in prev:
        if x > 1: nb.add(x - 1)
        if x < 80: nb.add(x + 1)
    return nb - set(prev)


def _k8_hot(draws, k=20):
    cnt = {}
    for r in draws:
        for n in r['numbers']: cnt[n] = cnt.get(n, 0) + 1
    return set(n for n, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:k])


def _cnt_grp(S, ball_set, lo, hi):
    m = np.zeros(81, dtype=bool)
    if ball_set: m[list(ball_set)] = True
    x = m[S].sum(axis=1)
    return np.where(x <= lo, 0, np.where(x <= hi, 1, 2))


def vote_feats_kl8(S, prev, hot):
    sm = S.sum(axis=1); odd = (S % 2 != 0).sum(axis=1); big = (S > 40).sum(axis=1)
    zone = np.stack([((S >= lo) & (S <= hi)).sum(axis=1) for lo, hi in [(1, 20), (21, 40), (41, 60), (61, 80)]], axis=1)
    five = np.stack([((S >= lo) & (S <= hi)).sum(axis=1) for lo, hi in [(1, 16), (17, 32), (33, 48), (49, 64), (65, 80)]], axis=1)
    adj = (np.diff(S, axis=1) == 1); runs = adj[:, 0].astype(np.int64) + (adj[:, 1:] & ~adj[:, :-1]).sum(axis=1)
    rng = S[:, -1] - S[:, 0]
    return {'odd_grp': np.where(odd < 9, 0, np.where(odd <= 11, 1, 2)), 'zone_dom': zone.argmax(axis=1),
            'tot_grp': np.where(sm < 640, 0, np.where(sm < 820, 1, 2)), 'big_grp': np.where(big < 9, 0, np.where(big <= 11, 1, 2)),
            'five_dom': five.argmax(axis=1), 'consec_grp': np.where(runs == 0, 0, np.where(runs <= 2, 1, 2)),
            'range_grp': np.where(rng < 60, 0, np.where(rng < 70, 1, 2)),
            'repeat_grp': _cnt_grp(S, prev, 4, 5), 'diag_grp': _cnt_grp(S, _k8_nb(prev) if prev else set(), 5, 7), 'hot_grp': _cnt_grp(S, hot, 4, 5)}


def _vprobs(dct, n):
    if not isinstance(dct, dict) or not dct: return None
    v = np.array([float(dct.get(str(i), 0.0)) for i in range(n)], dtype=np.float64) / 100.0
    return v / v.sum() if v.sum() > 0 else None


def _logw(feats, tgt, layout):
    n = len(next(iter(feats.values()))); logw = np.zeros(n); ncs = dict(layout)
    for name, p in tgt.items():
        nc = ncs[name]; f = feats[name]
        q = np.bincount(f, minlength=nc)[:nc] / n
        p = np.clip(np.asarray(p, dtype=np.float64)[:nc], 1e-3, 1.0); p = p / p.sum()
        logw += np.log(np.where(q > 0, p / np.maximum(q, 1e-12), 1.0)[np.minimum(f, nc - 1)])
    return logw


_D3F = None; _K8S = None; _D3GI = None


def ml_dl_vector(game, probs_by_target, hist, idx):
    """一个来源(ML 或 DL)的目标概率 → 候选号码的对数提升：3D 返回(220,)；快乐8 返回(80,)。目标不足返回 None。
    hist[:idx+1] 是该快照所基于的开奖史（快乐8 的重号/斜连/热点目标需要上期与近30期）"""
    global _K8S
    layout = VOTE_LAYOUT[game]; tgt = {}
    for name, n in layout:
        pv = _vprobs((probs_by_target or {}).get(name), n)
        if pv is not None: tgt[name] = pv
    if len(tgt) < VOTE_MIN_TARGETS: return None
    if game == '3d':
        global _D3F
        if _D3F is None: _D3F = vote_feats_3d(_D3_ALL)
        w = np.exp(_logw(_D3F, tgt, layout)); w /= w.sum()
        return np.concatenate([np.log(np.maximum(np.bincount(_D3_ALL[:, c], weights=w, minlength=10) / 0.1, 1e-6)) for c in range(3)])
    if _K8S is None:
        rng = np.random.default_rng(20260101)
        _K8S = np.argpartition(rng.random((K8_MC, 80)), 20, axis=1)[:, :20] + 1; _K8S.sort(axis=1)
    prev = set(hist[idx]['numbers']); hot = _k8_hot(hist[max(0, idx - 29):idx + 1])
    w = np.exp(_logw(vote_feats_kl8(_K8S, prev, hot), tgt, layout)); w /= w.sum()
    inc = np.zeros(81)
    for j in range(20): np.add.at(inc, _K8S[:, j], w)
    return np.log(np.maximum(inc[1:] / 0.25, 1e-6))


def md_snapshots(pred, dl, hists):
    """从一份 prediction.json / dl_lstm_tfm.json 取出 [(game, 来源, 快照id, 基准期号, 向量)]"""
    out = []
    for game in ('3d', 'kl8'):
        hist = hists[game]
        pg = ((pred or {}).get('predictions') or {}).get(game)
        if pg:
            dc = pg.get('data_count')
            if dc and 0 < dc <= len(hist):
                idx = dc - 1
                tg = {k: ((m.get('prediction') or {}).get('probs')) for k, m in (pg.get('models') or {}).items()}
                v = ml_dl_vector(game, tg, hist, idx)
                if v is not None: out.append((game, 'ml', 'ml' + str(pg.get('updated_at') or pred.get('updated_at')), str(hist[idx]['qihao']), v))
        dg = ((dl or {}).get('results') or {}).get(game)
        if dg:
            ua = str(dl.get('updated_at') or '')
            try:
                bj = datetime.strptime(ua[:19], '%Y-%m-%d %H:%M:%S') + timedelta(hours=8)      # dl 文件里是 UTC 时间
                dt = bj.strftime('%Y-%m-%d'); late = bj.hour >= 22
                idx = max(i for i, r in enumerate(hist) if (str(r['date']) <= dt if late else str(r['date']) < dt))
            except Exception: continue
            v = ml_dl_vector(game, {k: (m or {}).get('probs') for k, m in dg.items()}, hist, idx)
            if v is not None: out.append((game, 'dl', 'dl' + ua, str(hist[idx]['qihao']), v))
    return out


def new_rec(log, game, bq, bd=None):
    key = (game, bq); rec = log.get(key)
    if rec is None:
        rec = {'game': game, 'base_qihao': bq, 'base_date': bd, 'versions': [], 'n': 0}
        if game == '3d': rec['fmt'] = 2; rec['pos'] = [[0.0] * 30 for _ in range(4)]
        else: rec['balls'] = [[0.0] * 6 for _ in range(80)]
        log[key] = rec
    return rec


def add_md(log, game, src, sid, bq, vec):
    rec = new_rec(log, game, bq)
    s = rec.setdefault(src, {'ids': [], 'v': [0.0] * len(vec)})
    if sid in s['ids']: return False
    s['ids'].append(sid); s['v'] = [round(a + float(b), 4) for a, b in zip(s['v'], vec)]
    return True


def backfill_md(log, hists, use_git):
    n = 0
    def feed(pred, dl):
        nonlocal n
        for game, src, sid, bq, vec in md_snapshots(pred, dl, hists): n += add_md(log, game, src, sid, bq, vec)
    feed(_read_json('prediction.json') if os.path.exists('prediction.json') else None, _read_json('dl_lstm_tfm.json') if os.path.exists('dl_lstm_tfm.json') else None)
    if use_git:
        for fn, kind in (('prediction.json', 'ml'), ('dl_lstm_tfm.json', 'dl')):
            try:
                for sha in subprocess.run(['git', 'log', '--format=%H', '--', fn], capture_output=True, text=True, check=True).stdout.split():
                    try:
                        d = json.loads(subprocess.run(['git', 'show', f'{sha}:{fn}'], capture_output=True, text=True, check=True).stdout)
                        feed(d, None) if kind == 'ml' else feed(None, d)
                    except Exception: continue
            except Exception as e:
                print('  git 历史不可用：', e)
    return n


def load_log():
    d = {}
    if os.path.exists(LOG):
        with open(LOG, encoding='utf-8') as f:
            for line in f:
                try: r = json.loads(line)
                except Exception: continue
                if 'versions' not in r: continue          # 旧格式（只保留最新一版的记录）丢弃，由补录重新生成
                if r.get('game') == '3d' and r.get('fmt') != 2: continue     # 3D 改成直选(位置)后，旧的组选格式丢弃重新补录
                d[(r['game'], r['base_qihao'])] = r
    return d


def save_log(d):
    with open(LOG, 'w', encoding='utf-8') as f:
        for k in sorted(d, key=lambda x: (x[0], int(x[1]) if x[1].isdigit() else 0)):
            f.write(json.dumps(d[k], ensure_ascii=False, separators=(',', ':')) + '\n')


def add_version(log, game, ver):
    """把一个版本累加进对应期的记录；同一个版本号只算一次。返回是否新增"""
    if ver is None: return False
    vid, bq, bd, payload = ver
    rec = new_rec(log, game, bq, bd)
    if vid and vid in rec['versions']: return False
    rec['versions'].append(vid); rec['n'] += 1
    if game == '3d':
        for j in range(4):
            a = rec['pos'][j]
            for i in range(30): a[i] += payload[j][i]
    else:
        for i, v in enumerate(payload):
            a = rec['balls'][i]
            for j in range(6): a[j] += v[j]
    return True


def versions_from_rl(rl):
    return [('3d', d3_version(rl)), ('kl8', k8_version(rl))]


def backfill(log, use_git):
    n = 0
    files = sorted(glob.glob('rl_archive/*/*.json'))
    for p in files:
        try:
            for g, v in versions_from_rl(_read_json(p)): n += add_version(log, g, v)
        except Exception: continue
    if use_git:
        try:
            out = subprocess.run(['git', 'log', '--name-status', '--format=C %H', '--', 'dl_rl.json', 'rl_archive/'], capture_output=True, text=True, check=True).stdout.splitlines()
            sha, seen = None, set()
            for ln in out:
                if ln.startswith('C '): sha = ln[2:].strip(); continue
                parts = ln.split('\t')
                if len(parts) == 2 and parts[0] in ('A', 'M') and (parts[1] == 'dl_rl.json' or (parts[1].startswith('rl_archive/') and parts[1].endswith('.json') and 'index' not in parts[1])):
                    if (sha, parts[1]) in seen: continue
                    seen.add((sha, parts[1]))
                    try:
                        blob = subprocess.run(['git', 'show', f'{sha}:{parts[1]}'], capture_output=True, text=True, check=True).stdout
                        for g, v in versions_from_rl(json.loads(blob)): n += add_version(log, g, v)
                    except Exception: continue
        except Exception as e:
            print('  git 历史不可用：', e)
    return n


def _md_parts(rec, out):
    for src in ('ml', 'dl'):
        s = rec.get(src)
        if s and s['ids']:
            v = np.array(s['v'], dtype=np.float64) / len(s['ids'])
            if v.std() > 1e-12: out[src] = zs2(v)
    return out


def teacher_parts_3d(rec):
    """老师的三路：RL(多版本一致度) / ML / DL，各自按位标准化成 30 维；没有的路为 None"""
    out = {'rl': None, 'ml': None, 'dl': None}
    if not rec: return out
    if rec['n'] > 0:
        M = np.array(rec['pos'], dtype=np.float64) / rec['n']
        cols = [zs2(M[j]) for j in range(4) if M[j].std() > 1e-12]
        if cols: out['rl'] = zs2(np.sum(cols, axis=0))
    return _md_parts(rec, out)


def teacher_parts_k8(rec):
    out = {'rl': None, 'ml': None, 'dl': None}
    if not rec: return out
    if rec['n'] > 0:
        B = np.array(rec['balls'], dtype=np.float64) / rec['n']
        cols = [zs(B[:, j]) for j in range(6) if B[:, j].std() > 1e-12]
        if cols: out['rl'] = zs(np.sum(cols, axis=0))
    return _md_parts(rec, out)


def combine_teacher(parts):
    """RL、ML、DL 三路各自标准化后等权相加（缺哪路就少哪路）"""
    cols = [v for v in parts.values() if v is not None]
    return zs2(np.sum(cols, axis=0)) if cols else None


def teacher_3d(rec): return combine_teacher(teacher_parts_3d(rec))
def teacher_k8(rec): return combine_teacher(teacher_parts_k8(rec))


# ═════════════════════════ 融合与流程 ═════════════════════════
def tier(z):
    return 'strong' if z >= Z_STRONG else ('weak' if z >= Z_WEAK else 'none')


def logged_stats(items, lam, game):
    """items: [(sH, T, y)]；在有推荐记录的已开奖期上，比较 融合(λ) 与 仅开奖信号 的命中"""
    res = {'blend': [0.0, 0.0, 0.0], 'only_H': [0.0, 0.0, 0.0]}
    for sH, T, y, q in items:
        for name, sc in (('blend', zs2(sH) + lam * zs2(T)), ('only_H', zs2(sH))):
            if game == '3d':
                res[name][0] += d3_hits(sc, y); res[name][1] += D3_EXP; res[name][2] += D3_VAR
            else:
                pk = np.argsort(-sc, kind='stable')[:K8_TOPN]
                res[name][0] += float(y[pk].sum()); res[name][1] += K8_TOPN * 0.25
                res[name][2] += K8_TOPN * 0.25 * 0.75 * (80 - K8_TOPN) / 79.0
    return {k: {'hits': round(v[0], 2), 'expected': round(v[1], 2), 'z': _z(*v)} for k, v in res.items()}


def source_stats(items, game):
    """各路老师单独的表现（只在该路有记录的已开奖期上统计）——看 ML / DL / RL 哪路有用"""
    names = {'rl': 'RL', 'ml': 'ML', 'dl': 'DL'}; out = {}
    for src in ('rl', 'ml', 'dl'):
        h = e = v = 0.0; n = 0
        for sH, T, y, pp in items:
            sc = pp.get(src)
            if sc is None: continue
            n += 1
            if game == '3d':
                h += d3_hits(sc, y); e += D3_EXP; v += D3_VAR
            else:
                pk = np.argsort(-sc, kind='stable')[:K8_TOPN]
                h += float(y[pk].sum()); e += K8_TOPN * 0.25; v += K8_TOPN * 0.25 * 0.75 * (80 - K8_TOPN) / 79.0
        out[src] = {'name': names[src], 'n_periods': n, 'hits': round(h, 2), 'expected': round(e, 2), 'z': _z(h, e, v) if n else 0.0}
    return out


def run_game(game, hist, log, rl_latest_key):
    T = len(hist)
    if game == '3d':
        t0 = max(310, T - (D3_EVAL + D3_TRAIN_WIN + D3_VALID + 300))
        Xall, digs = d3p_hist_features(hist, t0)
        X, y = Xall[:-1], digs[t0:]
        he, w_h = d3p_eval(X, y, min(D3_EVAL, len(X) - 600), D3_TRAIN_WIN, D3_REFIT)
        hw = {f: round(float(v), 4) for f, v in zip(D3P_HF, w_h)}
        res = {'game': '3d', 'unit': '直选（百位/十位/个位）'}
        label = lambda idx: digs[idx]
        sH_of = lambda row: (Xall[row].astype(np.float64) @ w_h).ravel()
        teach = teacher_3d; parts_of = teacher_parts_3d
    else:
        M = k8_matrix(hist); t0 = max(110, T - (K8_EVAL + K8_TRAIN_WIN + K8_VALID + 150))
        Xall = k8_hist_features(M, t0)
        X, Y = Xall[:-1], M[t0:].astype(np.float32)
        he, w_h = k8_eval(X, Y, min(K8_EVAL, len(X) - 400), K8_TRAIN_WIN, K8_REFIT)
        hw = {f: round(float(v), 4) for f, v in zip(K8_HF, w_h)}
        res = {'game': 'kl8', 'unit': '球号'}
        label = lambda idx: M[idx].astype(np.float32)
        sH_of = lambda row: Xall[row] @ w_h
        teach = teacher_k8; parts_of = teacher_parts_k8
    res.update({'n_history': T, 'history_eval': he, 'history_weights': hw})
    # 有推荐记录、且下一期已开奖的期
    items = []; n_rec = 0; n_ver_total = 0
    for (g, bq), rec in sorted(log.items(), key=lambda kv: int(kv[0][1]) if kv[0][1].isdigit() else 0):
        if g != game: continue
        n_rec += 1; n_ver_total += rec['n']
        idx = next((i for i, r in enumerate(hist) if str(r['qihao']) == bq), None)
        if idx is None or idx + 1 >= len(hist) or idx + 1 < t0: continue
        pp = parts_of(rec); Tt = combine_teacher(pp)
        if Tt is None: continue
        items.append((sH_of(idx + 1 - t0), Tt, label(idx + 1), pp))
    res['n_logged_periods'] = n_rec; res['n_versions'] = n_ver_total; res['n_labeled'] = len(items)
    res['n_ml_snapshots'] = sum(len((r.get('ml') or {}).get('ids', [])) for (g, _), r in log.items() if g == game)
    res['n_dl_snapshots'] = sum(len((r.get('dl') or {}).get('ids', [])) for (g, _), r in log.items() if g == game)
    res['source_eval'] = source_stats(items, game)
    # 融合权重
    lam, lam_src = LAMBDA_DEFAULT, f'默认（已标注的推荐期不足 {LOGGED_FIT_MIN} 期，暂不据此调权重）'
    if len(items) >= LOGGED_FIT_MIN:
        best, bz = LAMBDA_DEFAULT, -1e9
        for L in LAMBDA_GRID:
            z = logged_stats(items, L, game)['blend']['z']
            if z > bz: bz, best = z, L
        lam, lam_src = best, f'在 {len(items)} 个已开奖的推荐期上挑选'
    if items:
        res['logged_eval'] = logged_stats(items, lam, game)
    res['lambda'] = lam; res['lambda_note'] = lam_src
    # 证据强度：以开奖号码信号在上千期滚动检验里的 z 为准；若推荐期够多(≥LOGGED_FIT_MIN)且融合更强，取两者较大
    z_h = he['model']['z']; z_use = z_h; basis = f'开奖号码信号 {he["n_eval"]} 期滚动检验 z={z_h}'
    if 'logged_eval' in res and len(items) >= LOGGED_FIT_MIN and res['logged_eval']['blend']['z'] > z_h:
        z_use = res['logged_eval']['blend']['z']; basis = f'融合选号在 {len(items)} 个推荐期的检验 z={z_use}'
    res['evidence'] = tier(z_use); res['evidence_z'] = z_use; res['evidence_basis'] = basis
    # 今日选号
    rec = log.get((game, rl_latest_key)) if rl_latest_key else None
    sH = sH_of(T - t0); Tt = teach(rec) if rec else None
    lam_eff = lam if Tt is not None else 0.0
    score = zs2(sH) + lam_eff * (zs2(Tt) if Tt is not None else 0.0)
    res['source'] = 'blend' if Tt is not None else 'history'
    pp_today = parts_of(rec) if rec else {}
    res['sources_today'] = [k for k, v in pp_today.items() if v is not None]
    res['n_ml_today'] = len((rec.get('ml') or {}).get('ids', [])) if rec else 0
    res['n_dl_today'] = len((rec.get('dl') or {}).get('ids', [])) if rec else 0
    res['base_qihao'] = rl_latest_key; res['n_versions_today'] = rec['n'] if rec else 0
    order = np.argsort(-score, kind='stable')
    if game == '3d':
        sc = score.reshape(3, 10)
        res['positions'] = [{'name': POS_NAMES[p], 'digits': [{'d': int(d), 'score': round(float(sc[p, d]), 3)} for d in np.argsort(-sc[p], kind='stable')[:5]]} for p in range(3)]
        J = sc[0][:, None, None] + sc[1][None, :, None] + sc[2][None, None, :]
        top = np.argsort(-J.ravel(), kind='stable')[:D3_TOPK]
        res['bets'] = [{'digits': [int(top_i // 100), int(top_i // 10 % 10), int(top_i % 10)], 'score': round(float(J.ravel()[top_i]), 3)} for top_i in top]
    else:
        res['ranking'] = [{'ball': int(i + 1), 'score': round(float(score[i]), 3)} for i in order[:20]]
        res['picks_by_n'] = {str(n): sorted(int(i + 1) for i in order[:n]) for n in K8_PLAYS}
    return res


def run(args):
    log = load_log()
    n_new = backfill(log, args.backfill_git)
    hist = _read_json('history.json')
    # 当前 RL 结果对应的期号（尚未开奖的那期）= 今日选号用的老师信号
    if os.path.exists('dl_rl.json'):
        rl = _read_json('dl_rl.json')
        for g, v in versions_from_rl(rl):
            if v: n_new += add_version(log, g, v)
    n_md = backfill_md(log, hist, args.backfill_git)       # ML / DL 的各目标概率 → 候选对数提升，按快照累计
    save_log(log)
    latest = {'3d': str(hist['3d'][-1]['qihao']), 'kl8': str(hist['kl8'][-1]['qihao'])}   # 今日信号 = 基于最新已开奖期的各路推荐
    out = {'updated_at': datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S'), 'games': {},
           'disclaimer': '彩票开奖具有随机性，选号器只是对开奖号码与各路推荐信号的统计整合，不代表更容易中奖，仅供娱乐参考。'}
    out['games']['3d'] = run_game('3d', hist['3d'], log, latest['3d'])
    out['games']['kl8'] = run_game('kl8', hist['kl8'], log, latest['kl8'])
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f'[蒸馏] 日志 {len(log)} 期，本次新增 RL 版本 {n_new} 个、ML/DL 快照 {n_md} 个')
    for g, r in out['games'].items():
        he = r['history_eval']; m = he['model']
        nm = '3D' if g == '3d' else '快乐8'
        hit = f"命中{m['hits']}/期望{m['expected']}" if g == '3d' else f"前10球均中{m['avg_hits']}/期望{m['expected']}"
        print(f"  {nm}: 证据={r['evidence']}（{r['evidence_basis']}）  来源={r['source']}  λ={r['lambda']}  今日用到 RL{r['n_versions_today']}/ML{r['n_ml_today']}/DL{r['n_dl_today']}  开奖信号检验 {hit} z={m['z']}（频率基线 z={he['freq']['z']}，正则={he['l2']}）")
        print('     今日选号：', ' | '.join(p['name'] + ':' + ''.join(str(x['d']) for x in p['digits']) for p in r['positions']) + '  直选:' + ' '.join(''.join(map(str, b['digits'])) for b in r['bets']) if g == '3d' else r['picks_by_n']['10'])
        if 'logged_eval' in r: print('     推荐期检验：', r['logged_eval'])
        print('     各路老师单独表现：', {k: (v['n_periods'], v['z']) for k, v in r['source_eval'].items()})


# ═════════════════════════ 自检 ═════════════════════════
def selftest():
    random.seed(1)
    for name, bias in (('3D 纯随机', 0.0), ('3D 含结构(25%概率重复上期号码)', 0.25)):
        hist = []
        for i in range(2600):
            if i and random.random() < bias: d = list(hist[-1]['digits']); random.shuffle(d)
            else: d = [random.randrange(10) for _ in range(3)]
            hist.append({'qihao': str(i), 'digits': d})
        Xall, digs = d3p_hist_features(hist, 310)
        ev, _ = d3p_eval(Xall[:-1], digs[310:], 600, 1200, 100, valid=200)
        print(f"[{name}] model z={ev['model']['z']} freq z={ev['freq']['z']} 差z={ev['model_minus_freq']['z']} 正则={ev['l2']}  → 证据={tier(ev['model']['z'])}")
    for name, bias in (('快乐8 纯随机', 0.0), ('快乐8 含结构(重号偏多)', 0.25)):
        hist = []
        for i in range(1800):
            if i and bias:
                prev = hist[-1]['numbers']; keep = [b for b in prev if random.random() < 0.25 + bias]
                rest = [b for b in range(1, 81) if b not in keep]; nums = sorted(keep + random.sample(rest, 20 - len(keep)))
            else: nums = sorted(random.sample(range(1, 81), 20))
            hist.append({'qihao': str(i), 'numbers': nums})
        M = k8_matrix(hist); Xall = k8_hist_features(M, 110)
        ev, _ = k8_eval(Xall[:-1], M[110:].astype(np.float32), 400, 1000, 50, valid=100)
        print(f"[{name}] model avg={ev['model']['avg_hits']} (期望{ev['model']['expected']}) z={ev['model']['z']} 差z={ev['model_minus_freq']['z']} 正则={ev['l2']}  → 证据={tier(ev['model']['z'])}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--backfill-git', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest() if a.selftest else run(a)
