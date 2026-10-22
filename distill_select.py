"""
蒸馏选号器（3D，组选号码级别）—— distill_select.py

做什么
  把每晚 ML/DL/RL/共识/采样 给出的"各路推荐信号"，连同下一期的真实开奖，
  一期一期记录下来(distill_log.jsonl)，再学一个"该信哪一路、各信多少"的小模型，
  用它从 220 个组选号里挑出 12 个。这就是"蒸馏"：把多路推荐压缩成一个选号器。

为什么是 3D 组选号
  dl_rl.json 里只有 3D 带"逐个组选号"的完整信号表(group_table：共识概率、RL联合概率、
  采样次数)；双色球/快乐8没有逐号信号表，暂不支持。

诚实的前提（一定要看）
  1) 历史推荐没有被留存过(rl_archive 只留当天)，所以日志从今天才开始积累，
     每个交易日 1 期。样本攒够之前，选号器【不会启用】，输出的是共识排序（fallback）。
  2) 彩票开奖是随机的。ML 的走势回测(*_ml_walkforward.json)里几乎所有目标的收益都是负的。
     本脚本做的是"诚实检验"：滚动前推(只用过去训练、预测下一期)，
     对比 随机期望 和 共识排序。只有 样本够多 且 显著好于随机 且 好于共识 才启用蒸馏结果，
     否则状态写 fallback，并说明原因。大概率会一直是 fallback——这本身就是有价值的结论。

用法
  python distill_select.py                 # 记录今天的信号 + 打标签 + 评估 + 输出 distill_result.json
  python distill_select.py --backfill-git  # 另外从 dl_rl.json 的 git 历史版本补录(需 fetch-depth: 0)
  python distill_select.py --selftest      # 合成数据自检(无信号→不启用；有信号→启用)

输出
  distill_log.jsonl      每行 = 一期的候选信号（同一期只保留最新一版）
  distill_result.json    状态、评估结果、今日 12 个组选号
"""
import json, os, sys, math, subprocess, argparse, random
from datetime import datetime, timezone, timedelta
import numpy as np

LOG = 'distill_log.jsonl'
OUT = 'distill_result.json'
TOP_K = 12            # 选几个组选号
MIN_TRAIN = 40        # 至少积累多少期已标注样本才开始滚动评估
MIN_EVAL = 100        # 至少评估多少期才允许启用蒸馏结果
Z_NEED = 2.0          # 相对随机的显著性门槛
L2 = 1.0              # 正则
REFIT_EVERY = 5       # 滚动评估时每隔几期重训一次
FEATS = ['ln_cons', 'samp_k', 'ln_rl', 'in_rec', 'in_cons', 'is_zu3', 'is_baozi']

BJ = timezone(timedelta(hours=8))


# ───────────────────────── 记录 ─────────────────────────
def _read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def record_from_rl(rl):
    """从 dl_rl.json 取 3D 的候选信号。返回 dict 或 None"""
    r = (rl.get('results') or {}).get('3d')
    if not r: return None
    cons = r.get('consensus') or {}
    gt = cons.get('group_table')
    if not gt or not r.get('base_qihao'): return None
    rec, con = {}, {}
    for g in r.get('ppo_groups') or []:
        k = ''.join(map(str, sorted(int(x) for x in g))); rec[k] = rec.get(k, 0) + 1
    for b in cons.get('bets') or []:
        k = ''.join(map(str, sorted(int(x) for x in b['digits']))); con[k] = con.get(k, 0) + 1
    cands = {}
    for sec in gt['sections']:
        for row in sec['rows']:
            for e in row:
                if not e: continue
                cands[e['d']] = [e['p'], e['q'], e.get('k', 0), e.get('rp'), rec.get(e['d'], 0), con.get(e['d'], 0)]
    if len(cands) < 100: return None
    return {'game': '3d', 'base_qihao': str(r['base_qihao']), 'base_date': r.get('base_date'),
            'logged_at': rl.get('updated_at'), 'n_sampled': gt.get('n_sampled'), 'cands': cands}


def load_log():
    d = {}
    if os.path.exists(LOG):
        with open(LOG, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: r = json.loads(line)
                except Exception: continue
                d[r['base_qihao']] = r
    return d


def save_log(d):
    with open(LOG, 'w', encoding='utf-8') as f:
        for k in sorted(d, key=lambda x: int(x) if x.isdigit() else 0):
            f.write(json.dumps(d[k], ensure_ascii=False, separators=(',', ':')) + '\n')


def merge(d, rec):
    """同一期多版本：保留 logged_at 最新的"""
    if rec is None: return False
    old = d.get(rec['base_qihao'])
    if old is None or str(rec.get('logged_at') or '') >= str(old.get('logged_at') or ''):
        d[rec['base_qihao']] = rec; return True
    return False


def backfill_git(d):
    try:
        shas = subprocess.run(['git', 'log', '--format=%H', '--', 'dl_rl.json'], capture_output=True, text=True, check=True).stdout.split()
    except Exception as e:
        print('  git 历史不可用：', e); return 0
    n = 0
    for sha in shas:
        try:
            blob = subprocess.run(['git', 'show', f'{sha}:dl_rl.json'], capture_output=True, text=True, check=True).stdout
            if merge(d, record_from_rl(json.loads(blob))): n += 1
        except Exception:
            continue
    return n


# ───────────────────────── 标注 ─────────────────────────
def label_of(history3d, base_qihao):
    """下一期(base 之后那一期)开出的组选号，如 '368'；还没开则 None"""
    idx = {str(r['qihao']): i for i, r in enumerate(history3d)}
    i = idx.get(str(base_qihao))
    if i is None or i + 1 >= len(history3d): return None
    return ''.join(map(str, sorted(int(x) for x in history3d[i + 1]['digits'])))


# ───────────────────────── 模型：条件 logit ─────────────────────────
def featurize(cands):
    keys = sorted(cands)
    X = np.zeros((len(keys), len(FEATS)))
    q = np.zeros(len(keys))
    for i, k in enumerate(keys):
        p, qq, ks, rp, rec, con = cands[k]
        qq = max(float(qq), 1e-6)
        X[i, 0] = math.log(max(float(p), 1e-6) / qq)
        X[i, 1] = float(ks or 0)
        X[i, 2] = math.log(max(float(rp), 1e-6) / qq) if rp is not None else 0.0
        X[i, 3] = float(rec); X[i, 4] = float(con)
        nu = len(set(k))
        X[i, 5] = 1.0 if nu == 2 else 0.0
        X[i, 6] = 1.0 if nu == 1 else 0.0
        q[i] = qq / 100.0
    return keys, X, q


def fit_clogit(periods, iters=15):
    """periods: [(X, true_index)]。牛顿法最大化条件对数似然 − L2"""
    d = len(FEATS); w = np.zeros(d)
    for _ in range(iters):
        g = -L2 * w; H = -L2 * np.eye(d)
        for X, t in periods:
            s = X @ w; s -= s.max(); pi = np.exp(s); pi /= pi.sum()
            m = pi @ X
            g += X[t] - m
            H -= (X * pi[:, None]).T @ X - np.outer(m, m)
        try: step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError: break
        w = w - step
        if np.abs(step).max() < 1e-6: break
    return w


def _score(stats):
    hits, exp, var = stats
    z = (hits - exp) / math.sqrt(var) if var > 0 else 0.0
    return {'hits': int(hits), 'expected': round(exp, 2), 'z': round(z, 2)}


def walk_forward(items):
    """items: [(keys, X, q, true_index)]，按时间顺序。返回各方法的 命中/期望/方差"""
    n = len(items)
    acc = {'distilled': [0, 0.0, 0.0], 'consensus': [0, 0.0, 0.0]}
    diff = []   # 蒸馏 − 共识 的逐期命中差
    w = None; ne = 0
    for t in range(MIN_TRAIN, n):
        if w is None or (t - MIN_TRAIN) % REFIT_EVERY == 0:
            w = fit_clogit([(X, ti) for _, X, _, ti in items[:t]])
        keys, X, q, ti = items[t]
        picks = {'distilled': np.argsort(-(X @ w), kind='stable')[:TOP_K],
                 'consensus': np.argsort(-X[:, 0], kind='stable')[:TOP_K]}
        h = {}
        for name, pk in picks.items():
            e = float(q[pk].sum()); acc[name][0] += int(ti in pk); acc[name][1] += e; acc[name][2] += e * (1 - e)
            h[name] = int(ti in pk)
        diff.append(h['distilled'] - h['consensus']); ne += 1
    out = {k: _score(v) for k, v in acc.items()}
    out['n_eval'] = ne
    if diff:
        md = float(np.mean(diff)); sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        out['distilled_minus_consensus'] = {'mean_hits': round(md, 4), 'z': round(md / (sd / math.sqrt(len(diff))), 2) if sd > 0 else 0.0}
    return out


# ───────────────────────── 主流程 ─────────────────────────
def run(args):
    log = load_log()
    n_new = 0
    if args.backfill_git: n_new += backfill_git(log)
    if os.path.exists('dl_rl.json'):
        if merge(log, record_from_rl(_read_json('dl_rl.json'))): n_new += 1
    save_log(log)

    hist = _read_json('history.json').get('3d', [])
    items, latest = [], None
    for bq in sorted(log, key=lambda x: int(x) if x.isdigit() else 0):
        keys, X, q = featurize(log[bq]['cands'])
        lab = label_of(hist, bq)
        if lab is None:
            latest = (bq, keys, X, q, log[bq]); continue
        if lab not in keys: continue
        items.append((keys, X, q, keys.index(lab)))
    if latest is None and log:   # 最新一条已被标注过(新一期已开)，说明今天还没有新信号
        pass

    res = {'updated_at': datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S'), 'game': '3d', 'unit': '组选号码',
           'n_logged': len(log), 'n_labeled': len(items), 'top_k': TOP_K,
           'need': {'min_train': MIN_TRAIN, 'min_eval': MIN_EVAL, 'z_vs_random': Z_NEED}, 'features': FEATS}
    status, reason, ev, w = 'fallback', '', None, None
    if len(items) < MIN_TRAIN + MIN_EVAL:
        reason = f'已标注样本 {len(items)} 期，至少需要 {MIN_TRAIN + MIN_EVAL} 期（{MIN_TRAIN}期训练起点 + {MIN_EVAL}期滚动评估）才能判断蒸馏是否有效；目前每个开奖日积累 1 期。'
    else:
        ev = walk_forward(items)
        res['walk_forward'] = ev
        d, c = ev['distilled'], ev['consensus']
        dm = ev.get('distilled_minus_consensus', {})
        if d['z'] >= Z_NEED and dm.get('mean_hits', 0) > 0 and dm.get('z', 0) >= 1.0:
            status = 'distilled'; reason = f"滚动评估 {ev['n_eval']} 期：蒸馏命中 {d['hits']}（随机期望 {d['expected']}，z={d['z']}），且优于共识排序，启用蒸馏选号。"
        else:
            reason = (f"滚动评估 {ev['n_eval']} 期：蒸馏命中 {d['hits']}（随机期望 {d['expected']}，z={d['z']}）；"
                      f"共识排序命中 {c['hits']}（z={c['z']}）。未达到「显著好于随机且好于共识」，沿用共识排序。")
        w = fit_clogit([(X, t) for _, X, _, t in items])
        res['weights'] = {f: round(float(v), 4) for f, v in zip(FEATS, w)}
    res['status'] = status; res['reason'] = reason

    # 今日选号
    if latest is not None:
        bq, keys, X, q, rec = latest
        use_w = w if (status == 'distilled' and w is not None) else None
        score = (X @ use_w) if use_w is not None else X[:, 0]
        order = np.argsort(-score, kind='stable')[:TOP_K]
        res['source'] = 'distilled' if use_w is not None else 'consensus'
        res['base_qihao'] = bq; res['base_date'] = rec.get('base_date')
        res['picks'] = [{'d': keys[i], 'score': round(float(score[i]), 3), 'cons_p': rec['cands'][keys[i]][0],
                         'base_p': rec['cands'][keys[i]][1], 'rl_p': rec['cands'][keys[i]][3],
                         'samp': rec['cands'][keys[i]][2], 'rec': rec['cands'][keys[i]][4], 'cons': rec['cands'][keys[i]][5]} for i in order]
    else:
        res['picks'] = []; res['note'] = 'dl_rl.json 里没有尚未开奖的 3D 信号记录（可能今晚还没跑 RL）。'
    res['disclaimer'] = '彩票开奖具有随机性，选号器只是对各路信号的统计整合，不代表更容易中奖，仅供娱乐参考。'
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[蒸馏] 记录 {len(log)} 期，已标注 {len(items)} 期，本次新增/更新 {n_new} 条；状态={status}")
    print('      ', reason)
    if res.get('picks'): print('       今日12个组选号：', ' '.join(p['d'] for p in res['picks']), f"（来源：{res['source']}）")


# ───────────────────────── 自检 ─────────────────────────
def _synth(n, signal, seed=0):
    """合成 n 期：每期 220 个候选。signal>0 时，开奖号的 RL联合概率会被额外抬高(模拟 RL 真有信号)"""
    rng = random.Random(seed); np_rng = np.random.default_rng(seed)
    groups = sorted({''.join(map(str, sorted((a, b, c)))) for a in range(10) for b in range(10) for c in range(10)})
    qs = {g: (6 if len(set(g)) == 3 else 3 if len(set(g)) == 2 else 1) / 10.0 for g in groups}   # 单位：%
    hist, log = [], {}
    for t in range(n + 1):
        draw = [rng.randrange(10) for _ in range(3)]
        hist.append({'qihao': str(1000 + t), 'digits': draw})
    for t in range(n):
        true = ''.join(map(str, sorted(hist[t + 1]['digits'])))
        cands = {}
        for g in groups:
            q = qs[g]
            noise = float(np_rng.normal(0, 0.25)); boost = signal if g == true else 0.0
            rp = q * math.exp(noise + boost); p = q * math.exp(float(np_rng.normal(0, 0.25)))
            cands[g] = [round(p, 4), q, int(np_rng.poisson(q * 12 / 10)), round(rp, 4), 0, 0]
        top = sorted(cands, key=lambda g: -cands[g][3])[:12]
        for g in top: cands[g][4] = 1
        log[str(1000 + t)] = {'game': '3d', 'base_qihao': str(1000 + t), 'cands': cands, 'logged_at': f'{t:06d}'}
    return hist, log


def selftest():
    for signal, name in [(0.0, '无信号'), (0.6, '有信号(RL联合概率里含真号信息)')]:
        hist, log = _synth(400, signal)
        items = []
        for bq in sorted(log, key=int):
            keys, X, q = featurize(log[bq]['cands']); lab = label_of(hist, bq)
            items.append((keys, X, q, keys.index(lab)))
        ev = walk_forward(items)
        d, c = ev['distilled'], ev['consensus']
        print(f"[{name}] 评估{ev['n_eval']}期  蒸馏 hits={d['hits']} 期望={d['expected']} z={d['z']} | 共识 z={c['z']} | 差 {ev.get('distilled_minus_consensus')}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--backfill-git', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest() if a.selftest else run(a)
