"""
蒸馏选号器 —— distill_select.py（3D 组选号 + 快乐8 选号）

「蒸馏」在这里的意思
  把两类信息压缩成一个"选号器"：
  ① 开奖号码本身（history.json，几千期）——每一期都能回头算出 冷热/遗漏/重号/邻号 等信号，
     所以这一层有几千期样本，可以马上训练、马上做诚实的滚动检验；
  ② 每晚各路推荐（ML/DL/RL、共识投票、采样）——这些推荐以前没有留存，
     从今天起每晚记录进 distill_log.jsonl，样本一期一期攒，攒够后再把它们并进来。

三层选号（谁通过检验用谁）
  full     = 开奖信号 + 各路推荐信号 一起学（需要累计足够多期的推荐记录）
  history  = 只用开奖号码信号（现在就能检验）
  fallback = 以上都没通过检验 → 沿用共识/RL 的排序（页面上会标明）
  检验方式：滚动前推——每一期只用它之前的数据训练、预测它，
  对比"随机期望"和"最朴素的频率基线"；显著好于随机(z≥2)且好于频率基线才启用。
  彩票开奖是随机的，大概率一直是 fallback；这个结论本身就是脚本的重要输出。

用法
  python distill_select.py                 # 正常运行（每晚流水线之后）
  python distill_select.py --backfill-git  # 另外从 dl_rl.json 的 git 历史补录推荐记录
  python distill_select.py --selftest      # 合成数据自检

输出
  distill_log.jsonl      每行=某游戏某一期的各路推荐信号（同一期只留最新一版）
  distill_result.json    两个游戏各自的状态、检验结果、今日选号（网页读这个）
"""
import json, os, math, subprocess, argparse, itertools, random
from datetime import datetime, timezone, timedelta
import numpy as np

LOG, OUT = 'distill_log.jsonl', 'distill_result.json'
BJ = timezone(timedelta(hours=8))
Z_NEED = 2.0
L2 = 1.0

# 3D
D3_TOPK = 12
D3_EVAL, D3_TRAIN_WIN, D3_REFIT = 1000, 3000, 100
# 快乐8
K8_TOPN = 10                   # 排名取前几个球（选四..选十都从这个排名里取前N个）
K8_EVAL, K8_TRAIN_WIN, K8_REFIT = 400, 1500, 50
K8_PLAYS = [4, 5, 6, 9, 10]
# 推荐记录并入（full 层）
L_MIN_TRAIN, L_MIN_EVAL = 40, 60


# ═════════════════════════ 通用 ═════════════════════════
def _z(hits, exp, var):
    return round(float((hits - exp) / math.sqrt(var)), 2) if var > 0 else 0.0


def _read_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def newton_clogit(X, t, iters=15):
    """条件 logit：X (n,C,F)，t (n,) 真实候选下标。最大化对数似然 − L2"""
    n, C, F = X.shape
    w = np.zeros(F)
    for _ in range(iters):
        s = X @ w; s -= s.max(1, keepdims=True)
        pi = np.exp(s); pi /= pi.sum(1, keepdims=True)
        m = np.einsum('nc,ncf->nf', pi, X)
        g = (X[np.arange(n), t] - m).sum(0) - L2 * w
        H = -(np.einsum('nc,ncf,ncg->fg', pi, X, X) - np.einsum('nf,ng->fg', m, m)) - L2 * np.eye(F)
        try: step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError: break
        w = w - step
        if np.abs(step).max() < 1e-6: break
    return w


def newton_logit(X, y, iters=15):
    """逻辑回归：X (N,F)，y (N,) 0/1"""
    N, F = X.shape
    w = np.zeros(F)
    for _ in range(iters):
        z = X @ w; p = 1 / (1 + np.exp(-z))
        g = X.T @ (y - p) - L2 * w
        H = -(X.T * (p * (1 - p))) @ X - L2 * np.eye(F)
        try: step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError: break
        w = w - step
        if np.abs(step).max() < 1e-6: break
    return w


# ═════════════════════════ 3D：组选号 ═════════════════════════
GROUPS = sorted({''.join(map(str, sorted((a, b, c)))) for a in range(10) for b in range(10) for c in range(10)})
G_IDX = {g: i for i, g in enumerate(GROUPS)}
G_Q = np.array([(6 if len(set(g)) == 3 else 3 if len(set(g)) == 2 else 1) / 1000.0 for g in GROUPS])
G_KIND3 = np.array([1.0 if len(set(g)) == 2 else 0.0 for g in GROUPS])
G_KIND1 = np.array([1.0 if len(set(g)) == 1 else 0.0 for g in GROUPS])
# 每个组选号的全部排列(补齐到6个)
_PERM = np.zeros((220, 6, 3), dtype=np.int64); _PM = np.zeros((220, 6))
for i, g in enumerate(GROUPS):
    ps = sorted(set(itertools.permutations(map(int, g))))
    for j, p in enumerate(ps): _PERM[i, j] = p; _PM[i, j] = 1.0
_PN = _PM.sum(1)
D3_HF = ['ln_f30', 'ln_f100', 'ln_f300', 'omit_ratio', 'pos_hot', 'share_last', 'is_zu3', 'is_baozi', 'ln_q']


def d3_group_seq(hist):
    return np.array([G_IDX[''.join(map(str, sorted(int(x) for x in r['digits'])))] for r in hist])


def d3_hist_features(hist, t0):
    """为 t∈[t0, T] 计算每个组选号的开奖信号（预测第 t 期，只用 <t 的开奖）。返回 (T+1-t0, 220, F) float32"""
    T = len(hist)
    gseq = d3_group_seq(hist)
    digs = np.array([[int(x) for x in r['digits']] for r in hist])
    oh = np.zeros((T + 1, 220), dtype=np.int32)
    for i, g in enumerate(gseq): oh[i + 1] = oh[i]; oh[i + 1, g] += 1            # 前缀和：oh[t]=前t期各组出现次数
    ohp = np.zeros((T + 1, 3, 10), dtype=np.int32)
    for i in range(T):
        ohp[i + 1] = ohp[i]
        for p in range(3): ohp[i + 1, p, digs[i, p]] += 1
    last = np.full(220, -1)
    # 先把 t0 之前的 last_seen 推进
    for i in range(t0): last[gseq[i]] = i
    out = np.zeros((T + 1 - t0, 220, len(D3_HF)), dtype=np.float32)
    lnq = np.log(G_Q)
    for t in range(t0, T + 1):
        r = out[t - t0]
        for k, w in enumerate((30, 100, 300)):
            c = oh[t] - oh[max(0, t - w)]
            r[:, k] = np.log((c + 0.5) / ((min(t, w)) * G_Q + 0.5))
        r[:, 3] = np.minimum((t - 1 - last) * G_Q, 6.0)          # 遗漏/期望间隔
        w50 = min(t, 50)
        pc = ohp[t] - ohp[max(0, t - 50)]
        lnf = np.log((pc + 0.5) / (w50 / 10.0 + 0.5))             # (3,10)
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
        r[:, 6] = G_KIND3; r[:, 7] = G_KIND1; r[:, 8] = lnq
        if t < T: last[gseq[t]] = t
    return out, gseq


def d3_eval(X, y, eval_n, train_win, refit, topk=D3_TOPK, base_col=0):
    """X (N,220,F) 对应连续的期；y (N,) 真实下标。最后 eval_n 期滚动评估。
    返回 dict：model / freq(最朴素：近30期频率)排序 的 命中、期望、z"""
    N = len(X); start = N - eval_n
    acc = {'model': [0, 0.0, 0.0], 'freq': [0, 0.0, 0.0]}
    w = None; diff = []
    for t in range(start, N):
        if w is None or (t - start) % refit == 0:
            lo = max(0, t - train_win)
            w = newton_clogit(X[lo:t].astype(np.float64), y[lo:t])
        s_m = X[t] @ w; s_f = X[t][:, base_col]
        hm = hf = 0
        for name, s in (('model', s_m), ('freq', s_f)):
            pk = np.argsort(-s, kind='stable')[:topk]; e = float(G_Q[pk].sum())
            h = int(y[t] in pk); acc[name][0] += h; acc[name][1] += e; acc[name][2] += e * (1 - e)
            if name == 'model': hm = h
            else: hf = h
        diff.append(hm - hf)
    out = {k: {'hits': v[0], 'expected': round(v[1], 2), 'z': _z(*v)} for k, v in acc.items()}
    out['n_eval'] = eval_n
    d = np.array(diff, dtype=float)
    out['model_minus_freq'] = {'mean_hits': round(float(d.mean()), 4),
                               'z': round(float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))), 2) if d.std(ddof=1) > 0 else 0.0}
    return out, newton_clogit(X[max(0, N - train_win):].astype(np.float64), y[max(0, N - train_win):])


# ── 3D 推荐信号（日志）──
D3_LF = ['ln_cons', 'samp_k', 'ln_rl', 'in_rec', 'in_cons']


def d3_record(rl):
    r = (rl.get('results') or {}).get('3d')
    if not r or not r.get('base_qihao'): return None
    cons = r.get('consensus') or {}
    gt = cons.get('group_table')
    if not gt: return None
    rec, con = {}, {}
    for g in r.get('ppo_groups') or []:
        k = ''.join(map(str, sorted(int(x) for x in g))); rec[k] = rec.get(k, 0) + 1
    for b in cons.get('bets') or []:
        k = ''.join(map(str, sorted(int(x) for x in b['digits']))); con[k] = con.get(k, 0) + 1
    cands = {}
    for sec in gt['sections']:
        for row in sec['rows']:
            for e in row:
                if e: cands[e['d']] = [e['p'], e['q'], e.get('k', 0), e.get('rp'), rec.get(e['d'], 0), con.get(e['d'], 0)]
    if len(cands) < 100: return None
    return {'game': '3d', 'base_qihao': str(r['base_qihao']), 'base_date': r.get('base_date'),
            'logged_at': rl.get('updated_at'), 'cands': cands}


def d3_lfeat(cands):
    X = np.zeros((220, len(D3_LF)))
    for g, (p, q, k, rp, rec, con) in cands.items():
        i = G_IDX.get(g)
        if i is None: continue
        q = max(float(q), 1e-6)
        X[i] = [math.log(max(float(p), 1e-6) / q), float(k or 0), math.log(max(float(rp), 1e-6) / q) if rp is not None else 0.0, rec, con]
    return X


# ═════════════════════════ 快乐8：逐球 ═════════════════════════
K8_HF = ['ln_f5', 'ln_f10', 'ln_f30', 'ln_f100', 'omit_ratio', 'in_prev', 'nb_prev', 'ball_n']
K8_LF = ['rl_score', 'ball_final', 'votes', 'lift', 'pick_cnt', 'samp_cnt']


def k8_matrix(hist):
    M = np.zeros((len(hist), 80), dtype=np.int8)
    for i, r in enumerate(hist):
        for b in r['numbers']: M[i, b - 1] = 1
    return M


def k8_hist_features(M, t0):
    """t∈[t0,T]：预测第 t 期，只用 <t 的开奖。返回 (T+1-t0, 80, F) float32"""
    T = len(M)
    cs = np.zeros((T + 1, 80), dtype=np.int32); cs[1:] = np.cumsum(M, axis=0)
    last = np.full(80, -1)
    for i in range(t0):
        last[M[i] == 1] = i
    out = np.zeros((T + 1 - t0, 80, len(K8_HF)), dtype=np.float32)
    ball_n = np.arange(1, 81) / 80.0
    for t in range(t0, T + 1):
        r = out[t - t0]
        for k, w in enumerate((5, 10, 30, 100)):
            c = cs[t] - cs[max(0, t - w)]
            r[:, k] = np.log((c + 0.5) / (min(t, w) * 0.25 + 0.5))
        r[:, 4] = np.minimum((t - 1 - last) / 4.0, 6.0)
        if t >= 1:
            prev = M[t - 1].astype(np.float32); r[:, 5] = prev
            nb = np.zeros(80, dtype=np.float32); nb[1:] += prev[:-1]; nb[:-1] += prev[1:]
            r[:, 6] = ((nb > 0) & (prev == 0)).astype(np.float32)
        r[:, 7] = ball_n
        if t < T: last[M[t] == 1] = t
    return out


def k8_eval(X, Y, eval_n, train_win, refit, topn=K8_TOPN):
    """X (N,80,F)，Y (N,80)。返回评估与最终权重。命中=前topn个球里中了几个，期望 topn*20/80"""
    N = len(X); start = N - eval_n
    exp = topn * 20 / 80.0
    var = topn * 0.25 * 0.75 * (80 - topn) / 79.0   # 超几何方差（每期 20 个球）
    acc = {'model': 0.0, 'freq': 0.0}; w = None; diff = []
    for t in range(start, N):
        if w is None or (t - start) % refit == 0:
            lo = max(0, t - train_win)
            w = newton_logit(X[lo:t].reshape(-1, X.shape[2]).astype(np.float64), Y[lo:t].reshape(-1).astype(np.float64))
        s_m = X[t] @ w; s_f = X[t][:, 2]                  # freq 基线：近30期频率
        hm = Y[t][np.argsort(-s_m, kind='stable')[:topn]].sum(); hf = Y[t][np.argsort(-s_f, kind='stable')[:topn]].sum()
        acc['model'] += hm; acc['freq'] += hf; diff.append(hm - hf)
    n = eval_n
    out = {k: {'avg_hits': round(float(v) / n, 3), 'expected': round(exp, 3), 'z': _z(float(v), exp * n, var * n)} for k, v in acc.items()}
    out['n_eval'] = n
    d = np.array(diff, dtype=float)
    out['model_minus_freq'] = {'mean_hits': round(float(d.mean()), 4),
                               'z': round(float(d.mean() / (d.std(ddof=1) / math.sqrt(n))), 2) if d.std(ddof=1) > 0 else 0.0}
    lo = max(0, N - train_win)
    return out, newton_logit(X[lo:].reshape(-1, X.shape[2]).astype(np.float64), Y[lo:].reshape(-1).astype(np.float64))


def k8_record(rl):
    r = (rl.get('results') or {}).get('kl8')
    if not r or not r.get('base_qihao'): return None
    scores = r.get('rl_scores') or []
    final = ((r.get('structure') or {}).get('ball_final')) or []
    if len(scores) != 80: return None
    votes, lift = {}, {}
    for e in ((r.get('consensus') or {}).get('table')) or []:
        votes[e['ball']] = e.get('votes', 0); lift[e['ball']] = e.get('lift', 1.0)
    pick_cnt = {}
    for lst in (r.get('picks_by_n') or {}).values():
        for b in lst: pick_cnt[b] = pick_cnt.get(b, 0) + 1
    samp = {b: c for b, c in ((r.get('sampled_stats') or {}).get('ball_top') or [])}
    balls = [[scores[i], final[i] if i < len(final) else 1.0, votes.get(i + 1, 0), lift.get(i + 1, 1.0), pick_cnt.get(i + 1, 0), samp.get(i + 1, 0)] for i in range(80)]
    return {'game': 'kl8', 'base_qihao': str(r['base_qihao']), 'base_date': r.get('base_date'),
            'logged_at': rl.get('updated_at'), 'balls': balls}


def k8_lfeat(balls):
    X = np.array(balls, dtype=np.float64)
    X[:, 0] = (X[:, 0] - X[:, 0].mean()) / (X[:, 0].std() + 1e-9)        # rl_score 按期标准化
    X[:, 1] = np.log(np.maximum(X[:, 1], 1e-6))
    return X


# ═════════════════════════ 日志 ═════════════════════════
def load_log():
    d = {}
    if os.path.exists(LOG):
        with open(LOG, encoding='utf-8') as f:
            for line in f:
                try: r = json.loads(line)
                except Exception: continue
                d[(r['game'], r['base_qihao'])] = r
    return d


def save_log(d):
    with open(LOG, 'w', encoding='utf-8') as f:
        for k in sorted(d, key=lambda x: (x[0], int(x[1]) if x[1].isdigit() else 0)):
            f.write(json.dumps(d[k], ensure_ascii=False, separators=(',', ':')) + '\n')


def merge(d, rec):
    if rec is None: return False
    k = (rec['game'], rec['base_qihao']); old = d.get(k)
    if old is None or str(rec.get('logged_at') or '') >= str(old.get('logged_at') or ''):
        d[k] = rec; return True
    return False


def records_from_rl(rl):
    return [r for r in (d3_record(rl), k8_record(rl)) if r]


def backfill_git(d):
    try: shas = subprocess.run(['git', 'log', '--format=%H', '--', 'dl_rl.json'], capture_output=True, text=True, check=True).stdout.split()
    except Exception as e:
        print('  git 历史不可用：', e); return 0
    n = 0
    for sha in shas:
        try:
            blob = subprocess.run(['git', 'show', f'{sha}:dl_rl.json'], capture_output=True, text=True, check=True).stdout
            for r in records_from_rl(json.loads(blob)): n += merge(d, r)
        except Exception: continue
    return n


# ═════════════════════════ 每个游戏的流程 ═════════════════════════
def decide(he, fe, kind):
    """he/fe: history/full 的检验结果。返回 (status, reason)"""
    def ok(ev):
        if not ev: return False
        key = 'model'
        return ev[key]['z'] >= Z_NEED and ev['model_minus_freq']['mean_hits'] > 0 and ev['model_minus_freq']['z'] >= 1.0
    if ok(fe): return 'full', '开奖信号+各路推荐信号 的滚动检验显著好于随机且好于频率基线，启用。'
    if ok(he): return 'history', '仅用开奖号码信号的滚动检验显著好于随机且好于频率基线，启用。'
    return 'fallback', '滚动检验未达到「显著好于随机(z≥2)且好于频率基线」，沿用共识/RL 排序。'


def run_3d(hist, log, rl_latest):
    T = len(hist)
    n_need = D3_EVAL + D3_TRAIN_WIN + 300
    t0 = max(310, T - n_need)
    Xall, gseq = d3_hist_features(hist, t0)         # t0..T
    X, y = Xall[:-1], gseq[t0:]                      # 有标签的期
    he, w_h = d3_eval(X, y, min(D3_EVAL, len(X) - 200), D3_TRAIN_WIN, D3_REFIT)
    res = {'game': '3d', 'unit': '组选号码', 'n_history': T, 'history_eval': he,
           'history_weights': {f: round(float(v), 4) for f, v in zip(D3_HF, w_h)}}
    # full：把推荐记录并进来
    items = []
    for (g, bq), rec in sorted(log.items(), key=lambda kv: int(kv[0][1]) if kv[0][1].isdigit() else 0):
        if g != '3d': continue
        idx = next((i for i, r in enumerate(hist) if str(r['qihao']) == bq), None)
        if idx is None or idx + 1 >= len(hist) or idx + 1 < t0: continue
        Xh = Xall[idx + 1 - t0]; Xl = d3_lfeat(rec['cands'])
        items.append((np.hstack([Xh, Xl]), int(gseq[idx + 1])))
    res['n_logged'] = sum(1 for k in log if k[0] == '3d'); res['n_labeled'] = len(items)
    fe, w_f = None, None
    if len(items) >= L_MIN_TRAIN + L_MIN_EVAL:
        Xf = np.stack([a for a, _ in items]); yf = np.array([b for _, b in items])
        fe, w_f = d3_eval(Xf, yf, len(items) - L_MIN_TRAIN, 10 ** 9, 5, base_col=0)
        res['full_eval'] = fe; res['full_weights'] = {f: round(float(v), 4) for f, v in zip(D3_HF + D3_LF, w_f)}
    else:
        res['full_note'] = f'推荐信号记录 {len(items)} 期，需要 {L_MIN_TRAIN + L_MIN_EVAL} 期才能把各路推荐并进来检验。'
    status, reason = decide(he, fe, '3d'); res['status'] = status; res['reason'] = reason
    # 今日选号：下一期 = 第 T 期
    xh = Xall[T - t0]
    rec = rl_latest.get('3d') if rl_latest else None
    if status == 'full' and rec is not None:
        score = np.hstack([xh, d3_lfeat(rec['cands'])]) @ w_f; res['source'] = 'full'
    elif status == 'history':
        score = xh @ w_h; res['source'] = 'history'
    elif rec is not None:
        score = d3_lfeat(rec['cands'])[:, 0]; res['source'] = 'consensus'
    else:
        score = xh[:, 0]; res['source'] = 'freq30'
    order = np.argsort(-score, kind='stable')[:D3_TOPK]
    res['picks'] = [{'d': GROUPS[i], 'score': round(float(score[i]), 3), 'kind': '组三' if len(set(GROUPS[i])) == 2 else ('豹子' if len(set(GROUPS[i])) == 1 else '组六')} for i in order]
    if rec is not None: res['base_qihao'] = rec['base_qihao']
    return res


def run_kl8(hist, log, rl_latest):
    T = len(hist)
    M = k8_matrix(hist)
    t0 = max(110, T - (K8_EVAL + K8_TRAIN_WIN + 150))
    Xall = k8_hist_features(M, t0)
    X, Y = Xall[:-1], M[t0:].astype(np.float32)
    he, w_h = k8_eval(X, Y, min(K8_EVAL, len(X) - 200), K8_TRAIN_WIN, K8_REFIT)
    res = {'game': 'kl8', 'unit': '球号', 'n_history': T, 'history_eval': he,
           'history_weights': {f: round(float(v), 4) for f, v in zip(K8_HF, w_h)}}
    items = []
    for (g, bq), rec in sorted(log.items(), key=lambda kv: int(kv[0][1]) if kv[0][1].isdigit() else 0):
        if g != 'kl8': continue
        idx = next((i for i, r in enumerate(hist) if str(r['qihao']) == bq), None)
        if idx is None or idx + 1 >= len(hist) or idx + 1 < t0: continue
        items.append((np.hstack([Xall[idx + 1 - t0], k8_lfeat(rec['balls'])]), M[idx + 1].astype(np.float32)))
    res['n_logged'] = sum(1 for k in log if k[0] == 'kl8'); res['n_labeled'] = len(items)
    fe, w_f = None, None
    if len(items) >= L_MIN_TRAIN + L_MIN_EVAL:
        Xf = np.stack([a for a, _ in items]); Yf = np.stack([b for _, b in items])
        fe, w_f = k8_eval(Xf, Yf, len(items) - L_MIN_TRAIN, 10 ** 9, 5)
        res['full_eval'] = fe; res['full_weights'] = {f: round(float(v), 4) for f, v in zip(K8_HF + K8_LF, w_f)}
    else:
        res['full_note'] = f'推荐信号记录 {len(items)} 期，需要 {L_MIN_TRAIN + L_MIN_EVAL} 期才能把各路推荐并进来检验。'
    status, reason = decide(he, fe, 'kl8'); res['status'] = status; res['reason'] = reason
    xh = Xall[T - t0]
    rec = rl_latest.get('kl8') if rl_latest else None
    if status == 'full' and rec is not None:
        score = np.hstack([xh, k8_lfeat(rec['balls'])]) @ w_f; res['source'] = 'full'
    elif status == 'history':
        score = xh @ w_h; res['source'] = 'history'
    elif rec is not None:
        b = np.array(rec['balls'], dtype=float); score = b[:, 1] + 1e-3 * b[:, 0]; res['source'] = 'consensus'   # 共识/RL 的最终逐球打分
    else:
        score = xh[:, 2]; res['source'] = 'freq30'
    order = np.argsort(-score, kind='stable')
    res['ranking'] = [{'ball': int(i + 1), 'score': round(float(score[i]), 3)} for i in order[:20]]
    res['picks_by_n'] = {str(n): sorted(int(i + 1) for i in order[:n]) for n in K8_PLAYS}
    if rec is not None: res['base_qihao'] = rec['base_qihao']
    return res


def run(args):
    log = load_log(); n_new = 0
    if args.backfill_git: n_new += backfill_git(log)
    rl_latest = {}
    if os.path.exists('dl_rl.json'):
        for r in records_from_rl(_read_json('dl_rl.json')):
            n_new += merge(log, r)
        rl_latest = {r['game']: r for r in records_from_rl(_read_json('dl_rl.json'))}
    save_log(log)
    hist = _read_json('history.json')
    # 只用"该游戏最新一期开奖之后"的推荐记录作为今日选号输入；已经被新开奖覆盖的旧记录不再当作今日信号
    for g, key in (('3d', '3d'), ('kl8', 'kl8')):
        rec = rl_latest.get(g)
        if rec and str(hist[key][-1]['qihao']) != rec['base_qihao']:
            rl_latest.pop(g, None)
    out = {'updated_at': datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S'), 'games': {},
           'disclaimer': '彩票开奖具有随机性，选号器只是对开奖号码与各路推荐信号的统计整合，不代表更容易中奖，仅供娱乐参考。'}
    out['games']['3d'] = run_3d(hist['3d'], log, rl_latest)
    out['games']['kl8'] = run_kl8(hist['kl8'], log, rl_latest)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f'[蒸馏] 日志 {len(log)} 条（本次新增/更新 {n_new}）')
    for g, r in out['games'].items():
        he = r['history_eval']
        if g == '3d':
            print(f"  3D  状态={r['status']}  来源={r['source']}  开奖信号检验: 命中{he['model']['hits']}/期望{he['model']['expected']} z={he['model']['z']}（频率基线 z={he['freq']['z']}）")
            print('      今日12个组选号：', ' '.join(p['d'] for p in r['picks']))
        else:
            print(f"  快乐8 状态={r['status']}  来源={r['source']}  开奖信号检验: 前10球平均命中{he['model']['avg_hits']}/期望{he['model']['expected']} z={he['model']['z']}（频率基线 z={he['freq']['z']}）")
            print('      选十：', r['picks_by_n']['10'])
        print('     ', r['reason'])


# ═════════════════════════ 自检 ═════════════════════════
def selftest():
    rng = np.random.default_rng(1); random.seed(1)
    # 3D：纯随机历史 → 不应启用；加入"上期号码会重复"的弱结构 → 应能检出
    for name, bias in (('3D 纯随机', 0.0), ('3D 含结构(25%概率重复上期组选号)', 0.25)):
        hist = []
        for i in range(2600):
            if i and random.random() < bias: d = list(hist[-1]['digits']); random.shuffle(d)
            else: d = [random.randrange(10) for _ in range(3)]
            hist.append({'qihao': str(i), 'digits': d})
        Xall, gseq = d3_hist_features(hist, 310)
        ev, _ = d3_eval(Xall[:-1], gseq[310:], 600, 1200, 100)
        print(f"[{name}] model z={ev['model']['z']} freq z={ev['freq']['z']} 差z={ev['model_minus_freq']['z']}")
    for name, bias in (('快乐8 纯随机', 0.0), ('快乐8 含结构(重号偏多)', 0.25)):
        hist = []
        for i in range(1800):
            if i and bias:
                prev = hist[-1]['numbers']
                keep = [b for b in prev if random.random() < 0.25 + bias]
                rest = [b for b in range(1, 81) if b not in keep]
                nums = sorted(keep + random.sample(rest, 20 - len(keep)))
            else:
                nums = sorted(random.sample(range(1, 81), 20))
            hist.append({'qihao': str(i), 'numbers': nums})
        M = k8_matrix(hist); Xall = k8_hist_features(M, 110)
        ev, _ = k8_eval(Xall[:-1], M[110:].astype(np.float32), 400, 1000, 50)
        print(f"[{name}] model avg={ev['model']['avg_hits']} (期望{ev['model']['expected']}) z={ev['model']['z']} 差z={ev['model_minus_freq']['z']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--backfill-git', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest() if a.selftest else run(a)
