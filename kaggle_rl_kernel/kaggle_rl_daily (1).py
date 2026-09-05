"""
福彩 PPO 强化学习 — 每天增量微调
kaggle_rl_daily.py

功能：
  1. 加载上次训练好的 PPO 模型（从 Kaggle Dataset "fucai-rl-cache"）
     若没有则首次全量训练
  2. 加载本周训练好的 LSTM/Transformer 权重（从 "fucai-dl-cache"）
     计算最新一期的隐层状态，构建 RL 状态
  3. 用最近新增的数据做增量微调（几千步，而非从头训练）
  4. 保存更新后的 PPO 模型回 Kaggle Dataset
  5. 更新 prediction.json 的 dl_result.rl 字段（每天更新）

Kaggle Secrets: GH_TOKEN, GH_REPO, KAGGLE_TOKEN
Kaggle 设置: 不需要GPU（PPO在CPU训练也很快），Internet开启
"""
import os, json, sys, time, warnings, base64, urllib.request, random, shutil, subprocess, copy, math
from datetime import datetime, date
from collections import Counter, defaultdict
from itertools import combinations
warnings.filterwarnings('ignore')

_secrets_client = None
_secrets_client_ready = False

def _get_secrets_client(retries=5, delay=4):
    global _secrets_client, _secrets_client_ready
    if _secrets_client_ready:
        return _secrets_client
    for attempt in range(retries):
        try:
            from kaggle_secrets import UserSecretsClient
            _secrets_client = UserSecretsClient()
            _secrets_client_ready = True
            print(f"  [Secrets] Client 连接成功（第{attempt+1}次尝试）")
            return _secrets_client
        except Exception as e:
            if attempt < retries - 1:
                print(f"  [Secrets] Client 连接失败（第{attempt+1}次）: {e}，{delay}秒后重试…")
                time.sleep(delay)
            else:
                print(f"  [Secrets] Client 连接彻底失败（{retries}次均失败）: {e}")
    return None

SECRETS_DATASET_MOUNT = '/kaggle/input/fucai-secrets/secrets.json'
_dataset_secrets = None

def _load_secrets_from_dataset():
    try:
        with open(SECRETS_DATASET_MOUNT) as f:
            return json.load(f)
    except Exception:
        return {}

def get_secret(name, retries=3, delay=3):
    global _dataset_secrets
    client = _get_secrets_client()
    if client is not None:
        for attempt in range(retries):
            try:
                v = client.get_secret(name)
                if v:
                    return v
                else:
                    break
            except Exception as e:
                if attempt < retries - 1:
                    print(f"  [Secret] {name} 第{attempt+1}次读取失败: {e}，{delay}秒后重试…")
                    time.sleep(delay)
                else:
                    print(f"  [Secret] {name} kaggle_secrets重试{retries}次仍失败: {e}")

    if _dataset_secrets is None:
        _dataset_secrets = _load_secrets_from_dataset()
    if name in _dataset_secrets and _dataset_secrets[name]:
        print(f"  [Secret] {name} 从 fucai-secrets Dataset 读取成功")
        return _dataset_secrets[name]

    return os.environ.get(name, '')

_HARDCODED_GH_TOKEN = ''  # 不要在这里写Token！写了会被GitHub自动吊销，必须用Kaggle Secrets      # ← 新的 GitHub Token
_HARDCODED_GH_REPO  = 'wa121325/fucai-data'
_HARDCODED_KAGGLE_TOKEN = 'KGAT_0847d8a3c8619a4db2ff2c7c3e9e824f'

GH_TOKEN = get_secret('GH_TOKEN') or get_secret('gh_token') or _HARDCODED_GH_TOKEN
GH_REPO  = get_secret('GH_REPO')  or get_secret('gh_repo')  or _HARDCODED_GH_REPO
KAGGLE_TOKEN = get_secret('KAGGLE_TOKEN') or get_secret('kaggle_token') or _HARDCODED_KAGGLE_TOKEN
print(f"GitHub: {GH_REPO}  GH_TOKEN: {'✓('+str(len(GH_TOKEN))+')' if GH_TOKEN else '✗'}")

try:
    import torch, torch.nn as nn
    DEVICE = torch.device('cpu')   # PPO在CPU更稳定
    print("PyTorch ✓")
except ImportError:
    print("PyTorch ✗"); sys.exit(1)

try:
    subprocess.run(['pip','install','stable-baselines3','gymnasium','-q'], capture_output=True, timeout=180)
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    print("SB3 ✓")
except Exception as e:
    print(f"SB3 ✗: {e}"); sys.exit(1)

import numpy as np

DL_DATASET_SLUG = 'fucai-dl-cache'
DL_MOUNTED      = f'/kaggle/input/{DL_DATASET_SLUG}'
RL_DATASET_SLUG = 'fucai-rl-cache'
RL_DATASET_ID   = f'megskfdbbskeb/{RL_DATASET_SLUG}'
RL_LOCAL_DIR    = '/kaggle/working/rl_cache'
RL_MOUNTED      = f'/kaggle/input/{RL_DATASET_SLUG}'
# 特征窗口。原来是50期——8742期历史里99.4%的数据从没进过特征，
# 每条统计量都只看最近50期，模型的"视野"极短。
# 加到300期后，长周期统计量（road/prime/repeat等）才有足够样本，
# 短窗口(3/5/10/20)照常保留对最新开奖的敏感度，两头都要。
# 注意：窗口只影响统计量的计算范围，不增减特征个数，维度仍是82/133/131。
WINDOW = 300; SEQ_LEN = 20

# ══════════════════════════════════════════════════════
#  新增特征辅助函数（三个脚本共用，务必保持完全一致）
#  补齐之前的空缺：遗漏统计、质合比、012路、和值尾数、重号/邻号、上期号码编码
# ══════════════════════════════════════════════════════
_PRIMES = set([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79])

def _omission_stats(w, pool_max, get_nums, prefix):
    """
    遗漏统计：之前遗漏信息只有强化学习在用，ML/DL的特征函数里一个都没有，
    这是最明显的空缺。这里把它提炼成聚合统计量补进来。
    - max/mean/std：整体遗漏分布的形态
    - overdue_cnt：遗漏值超过"该号码理论平均间隔"的号码个数（即所谓"超期"号码有多少）
    - last_draw_omit_mean：上期开出的那批号码，在开出之前平均冷了多久
      （衡量"这期开的是热号还是冷号"，这个序列本身可能比单纯遗漏值更有结构）
    """
    f = {}
    if not w:
        for k in ['omit_max','omit_mean','omit_std','overdue_cnt','last_draw_omit_mean']:
            f[f'{prefix}{k}'] = 0.0
        return f
    last_seen = {}
    for i, rec in enumerate(w):
        for n in get_nums(rec):
            last_seen[n] = i
    total = len(w)
    omits = []
    for n in range(1, pool_max+1):
        omits.append(total - 1 - last_seen[n] if n in last_seen else total)
    omits = np.array(omits, dtype=np.float32)
    # 理论平均间隔 = 号池大小 / 每期开出个数
    per_draw = len(get_nums(w[-1])) if w else 1
    theoretical_gap = pool_max / max(per_draw, 1)
    f[f'{prefix}omit_max']  = float(omits.max())
    f[f'{prefix}omit_mean'] = float(omits.mean())
    f[f'{prefix}omit_std']  = float(omits.std())
    f[f'{prefix}overdue_cnt'] = float((omits > theoretical_gap).sum())
    # 上期号码在开出前的遗漏（需要看倒数第二期为止的状态）
    if len(w) >= 2:
        prev_seen = {}
        for i, rec in enumerate(w[:-1]):
            for n in get_nums(rec):
                prev_seen[n] = i
        base = len(w) - 1
        vals = [base - 1 - prev_seen[n] if n in prev_seen else base for n in get_nums(w[-1])]
        f[f'{prefix}last_draw_omit_mean'] = float(np.mean(vals)) if vals else 0.0
    else:
        f[f'{prefix}last_draw_omit_mean'] = 0.0
    return f

def _prime_ratio(nums):
    """质合比：彩票分析里的经典维度，号池内质数分布不均匀，这个比例的波动是真实统计量"""
    return sum(1 for n in nums if n in _PRIMES) / max(len(nums), 1)

def _road012_counts(nums):
    """012路：按除3余数分三组。之前只有3D做了，双色球/快乐8同样适用"""
    c = [0,0,0]
    for n in nums: c[n % 3] += 1
    return c

def _repeat_neighbor(cur, prev):
    """
    重号：本期与上期重复的号码个数
    邻号：本期号码中，是上期某号码±1的个数
    这是走势图里很常见的跨期观察角度，之前只有3D零星涉及
    """
    if not prev: return 0, 0
    ps = set(prev)
    rep = len(set(cur) & ps)
    nb = sum(1 for n in cur if (n-1 in ps or n+1 in ps) and n not in ps)
    return rep, nb


# ══════════════════════════════════════════════════════
#  GitHub 工具
# ══════════════════════════════════════════════════════
def gh_raw(path):
    url = f'https://raw.githubusercontent.com/{GH_REPO}/main/{path}?t={int(time.time())}'
    req = urllib.request.Request(url, headers={'Cache-Control':'no-cache','User-Agent':'rl-bot'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r: return r.read().decode('utf-8')
    except Exception as e: print(f"  gh_raw({path}) 失败: {e}"); return None

def gh_put(path, content_str, message):
    url = f'https://api.github.com/repos/{GH_REPO}/contents/{path}'
    sha = None
    try:
        req = urllib.request.Request(url, headers={'Authorization':f'token {GH_TOKEN}',
            'Accept':'application/vnd.github.v3+json','User-Agent':'rl-bot'})
        with urllib.request.urlopen(req, timeout=15) as r: sha = json.loads(r.read()).get('sha')
    except Exception: pass
    data = {'message':message,'branch':'main','content':base64.b64encode(content_str.encode()).decode()}
    if sha: data['sha'] = sha
    req = urllib.request.Request(url, data=json.dumps(data).encode(), method='PUT', headers={
        'Authorization':f'token {GH_TOKEN}','Content-Type':'application/json',
        'Accept':'application/vnd.github.v3+json','User-Agent':'rl-bot'})
    with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())

# ══════════════════════════════════════════════════════
#  特征工程（与LSTM训练脚本一致）
# ══════════════════════════════════════════════════════
def f3d(records, idx):
    w=records[max(0,idx-WINDOW):idx]
    if len(w)<5: return None
    f={}
    for ws,sfx in [(3,'3'),(5,'5'),(10,'10'),(20,'20'),(WINDOW,'W')]:
        chunk=w[-ws:]
        sms=[sum(x['digits']) for x in chunk]
        sps=[max(x['digits'])-min(x['digits']) for x in chunk]
        odds=[sum(1 for d in x['digits'] if d%2!=0) for x in chunk]
        bigs=[sum(1 for d in x['digits'] if d>=5) for x in chunk]
        tails=[sum(x['digits'])%10 for x in chunk]
        gbs=[abs(x['digits'][0]-x['digits'][1]) for x in chunk]
        gsg=[abs(x['digits'][1]-x['digits'][2]) for x in chunk]
        f[f'sm{sfx}']=float(np.mean(sms)); f[f'ss{sfx}']=float(np.std(sms)) if len(sms)>1 else 0.0
        f[f'sp{sfx}']=float(np.mean(sps))
        f[f'odd{sfx}']=float(np.mean(odds)); f[f'big{sfx}']=float(np.mean(bigs))
        f[f'tail{sfx}']=float(np.mean(tails))
        f[f'gbs{sfx}']=float(np.mean(gbs)); f[f'gsg{sfx}']=float(np.mean(gsg))
        for ci,cn in enumerate(['b','s','g']):
            vals=[x['digits'][ci] for x in chunk]
            f[f'{cn}m{sfx}']=float(np.mean(vals))
            f[f'{cn}s{sfx}']=float(np.std(vals)) if len(vals)>1 else 0.0
    if len(w)>=3:
        s3=[sum(x['digits']) for x in w[-3:]]
        f['sm_trend']=1 if s3[-1]>s3[-2] else(-1 if s3[-1]<s3[-2] else 0)
    else: f['sm_trend']=0
    w20 = w[-20:]; n20 = len(w20) or 1
    r0=r1=r2=0
    for x in w20:
        for d in x['digits']:
            if d%3==0: r0+=1
            elif d%3==1: r1+=1
            else: r2+=1
    total=r0+r1+r2 or 1
    f['road0']=r0/total; f['road1']=r1/total; f['road2']=r2/total
    g3=g6=gt=0
    for x in w20:
        b2,s2,g2=x['digits']
        if b2==s2==g2: gt+=1
        elif b2==s2 or s2==g2 or b2==g2: g3+=1
        else: g6+=1
    f['grp3']=g3/n20; f['grp6']=g6/n20; f['grpt']=gt/n20
    # 重号比例（与各自前一期比较，近20期）
    rep_cnt=0; rep_n=0
    for i in range(max(1,len(w)-20), len(w)):
        prev=w[i-1]['digits']; cur=w[i]['digits']
        rep_cnt += sum(1 for k in range(3) if prev[k]==cur[k])
        rep_n += 1
    f['repeat_ratio'] = rep_cnt/rep_n if rep_n else 0.0
    # 斜连（三位等差数列）比例，近20期
    arith_cnt=0
    for x in w20:
        s3d=sorted(x['digits'])
        if (s3d[1]-s3d[0])==(s3d[2]-s3d[1]) and s3d[2]-s3d[0]>0: arith_cnt+=1
    f['arith_ratio'] = arith_cnt/n20
    # ── 新增特征：遗漏统计 / 质合比 / 和值尾数 / 重号邻号 ──
    # 3D按位处理：把每期三位数字当作号码集合（0-9映射到1-10避免0号问题）
    f.update(_omission_stats(w, 10, lambda r: [d+1 for d in r['digits']], 'g_'))
    primes = [_prime_ratio([d for d in x['digits'] if d>1]) for x in w[-20:]]
    f['prime_ratio20'] = float(np.mean(primes)) if primes else 0.0
    tails = [sum(x['digits']) % 10 for x in w[-20:]]
    f['sumtail_mean20'] = float(np.mean(tails)) if tails else 0.0
    f['sumtail_std20']  = float(np.std(tails)) if len(tails)>1 else 0.0
    reps, nbs = [], []
    for i in range(1, len(w[-20:])):
        chunk = w[-20:]
        r, n = _repeat_neighbor(chunk[i]['digits'], chunk[i-1]['digits'])
        reps.append(r); nbs.append(n)
    f['repeat_mean20']   = float(np.mean(reps)) if reps else 0.0
    f['neighbor_mean20'] = float(np.mean(nbs)) if nbs else 0.0
    # 上期号码原始编码：让模型能自己学出跨期关系，而不必全靠手工设计的聚合量
    last = w[-1]['digits'] if w else [0,0,0]
    for pi in range(3):
        f[f'prev_pos{pi}'] = float(last[pi]) if pi < len(last) else 0.0

    return f


def fssq(records, idx):
    w=records[max(0,idx-WINDOW):idx]
    if len(w)<5: return None
    f={}
    for ws,sfx in [(3,'3'),(5,'5'),(10,'10'),(20,'20'),(WINDOW,'W')]:
        chunk=w[-ws:]
        sms=[sum(x['red']) for x in chunk]
        bls=[x['blue'] for x in chunk]
        odds=[sum(1 for n in x['red'] if n%2!=0) for x in chunk]
        bigs=[sum(1 for n in x['red'] if n>16) for x in chunk]
        z1s=[sum(1 for n in x['red'] if n<=11) for x in chunk]
        z2s=[sum(1 for n in x['red'] if 12<=n<=22) for x in chunk]
        z3s=[sum(1 for n in x['red'] if n>=23) for x in chunk]
        consecs=[sum(1 for i in range(len(sorted(x['red']))-1) if sorted(x['red'])[i+1]-sorted(x['red'])[i]==1) for x in chunk]
        ac_vals=[]; max_gaps=[]
        for x in chunk:
            sred=sorted(x['red'])
            diffs=set()
            for i in range(len(sred)):
                for j in range(i+1,len(sred)):
                    diffs.add(sred[j]-sred[i])
            ac_vals.append(len(diffs)-(len(sred)-1))
            max_gaps.append(max(sred[k+1]-sred[k] for k in range(len(sred)-1)) if len(sred)>1 else 0)
        blue_odds=[x['blue']%2 for x in chunk]
        blue_bigs=[1 if x['blue']>=9 else 0 for x in chunk]
        f[f'sm_mean{sfx}']=float(np.mean(sms)); f[f'sm_std{sfx}']=float(np.std(sms)) if len(sms)>1 else 0.0
        f[f'bl_mean{sfx}']=float(np.mean(bls)); f[f'bl_std{sfx}']=float(np.std(bls)) if len(bls)>1 else 0.0
        f[f'odd_mean{sfx}']=float(np.mean(odds))
        f[f'big_mean{sfx}']=float(np.mean(bigs))
        f[f'z1_mean{sfx}']=float(np.mean(z1s))
        f[f'z2_mean{sfx}']=float(np.mean(z2s))
        f[f'z3_mean{sfx}']=float(np.mean(z3s))
        f[f'consec_mean{sfx}']=float(np.mean(consecs))
        f[f'ac_mean{sfx}']=float(np.mean(ac_vals)); f[f'ac_std{sfx}']=float(np.std(ac_vals)) if len(ac_vals)>1 else 0.0
        f[f'gap_mean{sfx}']=float(np.mean(max_gaps))
        f[f'blodd_mean{sfx}']=float(np.mean(blue_odds))
        f[f'blbig_mean{sfx}']=float(np.mean(blue_bigs))
    if len(w)>=3:
        s3=[sum(x['red']) for x in w[-3:]]
        f['sm_trend']=1 if s3[-1]>s3[-2] else(-1 if s3[-1]<s3[-2] else 0)
        b3=[x['blue'] for x in w[-3:]]
        f['bl_trend']=1 if b3[-1]>b3[-2] else(-1 if b3[-1]<b3[-2] else 0)
    else:
        f['sm_trend']=0; f['bl_trend']=0
    cnt=Counter(n for x in w[-20:] for n in x['red'])
    f['hot_z1']=sum(cnt.get(n,0) for n in range(1,12))
    f['hot_z2']=sum(cnt.get(n,0) for n in range(12,23))
    f['hot_z3']=sum(cnt.get(n,0) for n in range(23,34))
    bcnt=Counter(x['blue'] for x in w[-20:])
    f['hot_bl_lo']=sum(bcnt.get(n,0) for n in range(1,9))
    f['hot_bl_hi']=sum(bcnt.get(n,0) for n in range(9,17))
    # ── 新增特征：遗漏统计 / 质合比 / 012路 / 和值尾数 / 重号邻号 / 上期编码 ──
    f.update(_omission_stats(w, 33, lambda r: r['red'], 'r_'))
    f.update(_omission_stats(w, 16, lambda r: [r['blue']], 'b_'))
    primes = [_prime_ratio(x['red']) for x in w[-20:]]
    f['prime_ratio20'] = float(np.mean(primes)) if primes else 0.0
    r0s, r1s, r2s = [], [], []
    for x in w[-20:]:
        c = _road012_counts(x['red']); r0s.append(c[0]); r1s.append(c[1]); r2s.append(c[2])
    f['road0_mean20'] = float(np.mean(r0s)) if r0s else 0.0
    f['road1_mean20'] = float(np.mean(r1s)) if r1s else 0.0
    f['road2_mean20'] = float(np.mean(r2s)) if r2s else 0.0
    tails = [sum(x['red']) % 10 for x in w[-20:]]
    f['sumtail_mean20'] = float(np.mean(tails)) if tails else 0.0
    reps, nbs = [], []
    chunk = w[-20:]
    for i in range(1, len(chunk)):
        r, n = _repeat_neighbor(chunk[i]['red'], chunk[i-1]['red'])
        reps.append(r); nbs.append(n)
    f['repeat_mean20']   = float(np.mean(reps)) if reps else 0.0
    f['neighbor_mean20'] = float(np.mean(nbs)) if nbs else 0.0
    # 上期红球二值编码(33维)+上期蓝球，让模型自行学习跨期规律
    prev_red = set(w[-1]['red']) if w else set()
    for n in range(1, 34):
        f[f'prev_r{n}'] = 1.0 if n in prev_red else 0.0
    f['prev_blue'] = float(w[-1]['blue']) if w else 0.0

    return f


def fkl8(records, idx):
    w=records[max(0,idx-WINDOW):idx]
    if len(w)<5: return None
    f={}
    for ws,sfx in [(3,'3'),(5,'5'),(10,'10'),(20,'20'),(WINDOW,'W')]:
        chunk=w[-ws:]
        tots=[sum(x['numbers']) for x in chunk]
        odds=[sum(1 for n in x['numbers'] if n%2!=0) for x in chunk]
        bigs=[sum(1 for n in x['numbers'] if n>40) for x in chunk]
        mins=[min(x['numbers']) for x in chunk]
        maxs=[max(x['numbers']) for x in chunk]
        cgs=[]
        for x in chunk:
            sn=sorted(x['numbers']); cg=0; inc=False
            for i in range(len(sn)-1):
                if sn[i+1]-sn[i]==1:
                    if not inc: cg+=1; inc=True
                else: inc=False
            cgs.append(cg)
        f[f'tm{sfx}']=float(np.mean(tots)); f[f'ts{sfx}']=float(np.std(tots)) if len(tots)>1 else 0.0
        f[f'odd{sfx}']=float(np.mean(odds)); f[f'big{sfx}']=float(np.mean(bigs))
        f[f'mn{sfx}']=float(np.mean(mins)); f[f'mx{sfx}']=float(np.mean(maxs))
        f[f'cg{sfx}']=float(np.mean(cgs))
        for zi,(lo,hi) in enumerate([(1,20),(21,40),(41,60),(61,80)]):
            zv=[sum(1 for n in x['numbers'] if lo<=n<=hi) for x in chunk]
            f[f'z{zi+1}m{sfx}']=float(np.mean(zv))
        for fi2,(lo,hi) in enumerate([(1,16),(17,32),(33,48),(49,64),(65,80)]):
            fv=[sum(1 for n in x['numbers'] if lo<=n<=hi) for x in chunk]
            f[f'f{fi2+1}m{sfx}']=float(np.mean(fv))
    if len(w)>=3:
        t3=[sum(x['numbers']) for x in w[-3:]]
        f['tot_trend']=1 if t3[-1]>t3[-2] else(-1 if t3[-1]<t3[-2] else 0)
    else: f['tot_trend']=0
    cnt=Counter(n for x in w[-20:] for n in x['numbers'])
    for zi,(lo,hi) in enumerate([(1,20),(21,40),(41,60),(61,80)]):
        f[f'hz{zi+1}']=sum(cnt.get(n,0) for n in range(lo,hi+1))
    # ── 新增特征：遗漏统计 / 质合比 / 012路 / 和值尾数 / 重号邻号 / 上期编码 ──
    f.update(_omission_stats(w, 80, lambda r: r['numbers'], 'n_'))
    primes = [_prime_ratio(x['numbers']) for x in w[-20:]]
    f['prime_ratio20'] = float(np.mean(primes)) if primes else 0.0
    r0s, r1s, r2s = [], [], []
    for x in w[-20:]:
        c = _road012_counts(x['numbers']); r0s.append(c[0]); r1s.append(c[1]); r2s.append(c[2])
    f['road0_mean20'] = float(np.mean(r0s)) if r0s else 0.0
    f['road1_mean20'] = float(np.mean(r1s)) if r1s else 0.0
    f['road2_mean20'] = float(np.mean(r2s)) if r2s else 0.0
    tails = [sum(x['numbers']) % 10 for x in w[-20:]]
    f['sumtail_mean20'] = float(np.mean(tails)) if tails else 0.0
    reps, nbs = [], []
    chunk = w[-20:]
    for i in range(1, len(chunk)):
        r, n = _repeat_neighbor(chunk[i]['numbers'], chunk[i-1]['numbers'])
        reps.append(r); nbs.append(n)
    f['repeat_mean20']   = float(np.mean(reps)) if reps else 0.0
    f['neighbor_mean20'] = float(np.mean(nbs)) if nbs else 0.0
    # 快乐8每期开20个球，上期二值编码就是80维，维度偏大且信息稀疏，
    # 改用"上期号码按四区分布"这种压缩表示，兼顾跨期信息与维度控制
    prev = w[-1]['numbers'] if w else []
    for zi,(lo,hi) in enumerate([(1,20),(21,40),(41,60),(61,80)]):
        f[f'prev_z{zi}'] = float(sum(1 for n in prev if lo<=n<=hi))

    # ══════════════════════════════════════════════════════
    #  快乐8专属新增特征（仅本游戏，3D/双色球不加）
    # ══════════════════════════════════════════════════════

    # ── ① 冷热变化率：这是维度上的真正空白 ──
    # 现有信号（频率、遗漏）回答的都是"现在怎样"，没有一个回答"正在往哪变"。
    # 一个球从冷转热、和一直是热的，含义完全不同，但之前的特征区分不出来。
    _f10 = Counter(n for r in w[-10:] for n in r['numbers'])
    _f50 = Counter(n for r in w[-50:] for n in r['numbers'])
    _n10, _n50 = max(len(w[-10:]),1), max(len(w[-50:]),1)
    _rates = [ _f10.get(n,0)/_n10 - _f50.get(n,0)/_n50 for n in range(1,81) ]
    _rates = np.array(_rates, dtype=np.float32)
    f['heat_rate_mean'] = float(_rates.mean())          # 整体升温还是降温
    f['heat_rate_std']  = float(_rates.std())           # 冷热分化程度
    f['heat_rising_cnt']  = float((_rates > 0.05).sum())  # 明显转热的球数
    f['heat_falling_cnt'] = float((_rates < -0.05).sum()) # 明显转冷的球数
    # 上期开出的号码，此前是在升温还是降温（判断"追热"还是"追冷"更奏效）
    _prev_nums = w[-1]['numbers'] if w else []
    f['prev_heat_rate'] = float(np.mean([_rates[n-1] for n in _prev_nums])) if _prev_nums else 0.0

    # ── ② 尾数分布：80球按尾数正好分10组各8个，是干净的统计维度 ──
    _tail_cnt = [0]*10
    for n in _prev_nums: _tail_cnt[n % 10] += 1
    for t in range(10): f[f'tail{t}'] = float(_tail_cnt[t])
    f['tail_std']  = float(np.std(_tail_cnt))            # 尾数分布均匀还是集中
    f['tail_zero'] = float(sum(1 for c in _tail_cnt if c == 0))  # 有几个尾数完全没出

    # ── ③ 间隔分布：之前只有"连续号组数"，丢了间隔的整体形态 ──
    _gaps_all = []
    for x in w[-20:]:
        s = sorted(x['numbers'])
        _gaps_all.append([s[i+1]-s[i] for i in range(len(s)-1)])
    if _gaps_all:
        _flat = [g for gs in _gaps_all for g in gs]
        f['gap_mean20'] = float(np.mean(_flat))
        f['gap_std20']  = float(np.std(_flat))
        f['gap_max20']  = float(np.mean([max(gs) for gs in _gaps_all if gs]))
    else:
        f['gap_mean20'] = f['gap_std20'] = f['gap_max20'] = 0.0

    # ── ④ AC值：两两差值的离散度，双色球有、快乐8之前没有 ──
    if _prev_nums and len(_prev_nums) > 1:
        _d = set()
        _ps = sorted(_prev_nums)
        for i in range(len(_ps)):
            for j in range(i+1, len(_ps)): _d.add(_ps[j]-_ps[i])
        f['ac_value'] = float(len(_d) - (len(_ps)-1))
    else:
        f['ac_value'] = 0.0

    # ── ⑤ 同尾号对数：彩民常看的形态维度 ──
    f['same_tail_pairs'] = float(sum(c*(c-1)//2 for c in _tail_cnt))

    # ── ⑥ 区间转移：号码在四个区之间的流动方向 ──
    # 比"各区多少个"多一层信息：是从哪个区流向哪个区
    if len(w) >= 2:
        _pz = [sum(1 for n in w[-2]['numbers'] if lo<=n<=hi) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]]
        _cz = [sum(1 for n in _prev_nums          if lo<=n<=hi) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]]
        for zi in range(4): f[f'zone_delta{zi}'] = float(_cz[zi] - _pz[zi])
    else:
        for zi in range(4): f[f'zone_delta{zi}'] = 0.0

    # ── ⑦ 重号的区间分布：比"重了几个"更细 ──
    if len(w) >= 2:
        _rep = set(w[-1]['numbers']) & set(w[-2]['numbers'])
        for zi,(lo,hi) in enumerate([(1,20),(21,40),(41,60),(61,80)]):
            f[f'repeat_z{zi}'] = float(sum(1 for n in _rep if lo<=n<=hi))
    else:
        for zi in range(4): f[f'repeat_z{zi}'] = 0.0


    return f


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=10, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim,64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64,output_dim))
    def forward(self, x, return_hidden=False):
        out,_ = self.lstm(x); last = self.norm(out[:,-1,:]); logits = self.head(last)
        return (logits,last) if return_hidden else logits

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, output_dim=10, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(d_model,32), nn.GELU(), nn.Linear(32,output_dim))
    def forward(self, x, return_hidden=False):
        x = self.proj(x); x = self.transformer(x)
        pooled = self.pool(x.transpose(1,2)).squeeze(-1); logits = self.head(pooled)
        return (logits,pooled) if return_hidden else logits


def load_lstm_tfm(game, current_feat_dim=None):
    """
    从挂载的 Dataset 加载本周训练好的LSTM/TFM权重。

    返回 (lstm_list, tfm_list, meta)：
    ── 之前只加载 {game}_lstm.pt（每个游戏第一个训练目标的模型），
       另外6个目标训练出来的模型RL完全接触不到，等于DL 6/7的成果被浪费。
       现在把7个目标的模型全部加载，各自的隐层表征都会进入RL状态。
    """
    meta_path = f'{DL_MOUNTED}/{game}_meta.json'
    if not os.path.exists(meta_path):
        print(f"  ! 找不到 {game} 的LSTM/TFM权重（先运行 kaggle_lstm_tfm.py 并挂载 {DL_DATASET_SLUG}）")
        return [], [], None
    with open(meta_path) as f: meta = json.load(f)
    if current_feat_dim is not None and meta.get('feat_dim') != current_feat_dim:
        print(f"  ! {game} 的LSTM/TFM权重特征维度({meta.get('feat_dim')})与当前特征工程({current_feat_dim})不一致")
        print(f"    请先重新运行 kaggle_lstm_tfm.py 生成新权重，本次跳过LSTM/TFM隐层")
        return [], [], None

    # 该游戏的全部训练目标（顺序需与 kaggle_lstm_tfm.py 的 configs 一致）
    TARGETS = {
        '3d':  ['sum_grp','group_type','odd','big','span_grp','road_dom','arith'],
        'ssq': ['odd','sum_grp','ac_grp','red_zone_dom','gap_grp','big','consec'],
        'kl8': ['odd_grp','zone_dom','tot_grp','big_grp','five_dom','consec_grp','range_grp'],
    }.get(game, [])

    lstm_list, tfm_list, loaded, skipped = [], [], [], []
    for tname in TARGETS:
        tmeta_path = f'{DL_MOUNTED}/{game}_{tname}_meta.json'
        lpath = f'{DL_MOUNTED}/{game}_{tname}_lstm.pt'
        tpath = f'{DL_MOUNTED}/{game}_{tname}_tfm.pt'
        if not (os.path.exists(tmeta_path) and os.path.exists(lpath) and os.path.exists(tpath)):
            skipped.append(tname); continue
        try:
            with open(tmeta_path) as f: tm = json.load(f)
            if current_feat_dim is not None and tm.get('feat_dim') != current_feat_dim:
                skipped.append(tname); continue
            l = LSTMEncoder(tm['feat_dim'], hidden_dim=tm['hidden_dim'], output_dim=tm['n_classes'])
            l.load_state_dict(torch.load(lpath, map_location='cpu')); l.eval()
            t = TransformerEncoder(tm['feat_dim'], d_model=tm['d_model'], output_dim=tm['n_classes'])
            t.load_state_dict(torch.load(tpath, map_location='cpu')); t.eval()
            lstm_list.append(l); tfm_list.append(t); loaded.append(tname)
        except Exception as e:
            print(f"    ! 加载 {game}_{tname} 失败: {e}")
            skipped.append(tname)

    # 兼容旧版Dataset：若按目标命名的文件一个都没有，回退到只加载第一个目标的固定名文件
    if not lstm_list:
        try:
            l = LSTMEncoder(meta['feat_dim'], hidden_dim=meta['hidden_dim'], output_dim=meta['n_classes'])
            l.load_state_dict(torch.load(f'{DL_MOUNTED}/{game}_lstm.pt', map_location='cpu')); l.eval()
            t = TransformerEncoder(meta['feat_dim'], d_model=meta['d_model'], output_dim=meta['n_classes'])
            t.load_state_dict(torch.load(f'{DL_MOUNTED}/{game}_tfm.pt', map_location='cpu')); t.eval()
            lstm_list, tfm_list = [l], [t]
            print(f"    (旧版Dataset：仅加载到第一个目标的模型，重新运行DL脚本后可加载全部7个)")
        except Exception as e:
            print(f"    ! 回退加载也失败: {e}")
            return [], [], None
    else:
        print(f"    ✓ 已加载 {len(loaded)}/{len(TARGETS)} 个目标的LSTM+TFM模型: {loaded}"
              + (f"（跳过: {skipped}）" if skipped else ""))
    return lstm_list, tfm_list, meta


def precompute_hidden_multi(records, feat_fn, models, seq_len=SEQ_LEN, batch_size=256):
    """
    批量计算多个模型的隐层状态，序列只构造一次供所有模型共用。

    ── 为什么必须共用序列 ──
    构造序列要对每一期调用一次特征函数，是整个流程最慢的一步（实测占500秒以上）。
    现在要加载7个目标的模型，若每个模型各构造一遍序列，耗时会翻7倍。
    这里改成：序列构造一次，7个模型依次前向推理（推理很快），
    最后把各模型的隐层横向拼接成一个大向量。

    返回 (hidden_array, idx_to_row)：hidden_array 每行是所有模型隐层的拼接
    """
    models = [m for m in (models or []) if m is not None]
    if not models:
        return None, {}
    X, idxs = [], []
    # 上界用 len(records)+1：idx=len(records) 对应"用全部已知数据预测下一期"，
    # 这是生成推荐时要用的那一步。若只算到 len(records)-1，推荐时会查不到隐层、
    # 被迫回退成零向量，白白丢掉LSTM/TFM的信息。
    for idx in range(seq_len, len(records)+1):
        seq = []; valid = True
        for j in range(idx-seq_len, idx):
            feat = feat_fn(records, j)
            if feat is None: valid=False; break
            seq.append(list(feat.values()))
        if not valid: continue
        X.append(seq); idxs.append(idx)
    if not X:
        return None, {}
    X = np.array(X, dtype=np.float32)

    per_model = []
    for model in models:
        model.eval()
        hs = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.FloatTensor(X[i:i+batch_size])
                _, h = model(xb, return_hidden=True)
                hs.append(h.numpy())
        per_model.append(np.vstack(hs))
    hidden = np.concatenate(per_model, axis=1)   # 横向拼接所有模型的隐层
    idx_to_row = {idx:i for i,idx in enumerate(idxs)}
    return hidden, idx_to_row


def precompute_omission_kl8(records):
    """
    批量一次性计算快乐8所有期数的遗漏向量（O(N*80)，而非原来每次调用O(N*80)倒序扫描导致的O(N²*80)）
    返回数组 omit_arr[idx] = 80维遗漏归一化向量，与 omission_vec_kl8(records, idx) 完全等价
    """
    N = len(records)
    last_seen = {}   # 号码 -> 最后出现的下标
    omit_arr = np.zeros((N+1, 80), dtype=np.float32)
    for idx in range(N+1):
        avg = max(idx*20/80, 1)
        for n in range(1, 81):
            if n in last_seen:
                omit_arr[idx, n-1] = min((idx-1-last_seen[n]) / avg, 3)
            else:
                omit_arr[idx, n-1] = 2.0
        if idx < N:
            for n in records[idx]['numbers']:
                last_seen[n] = idx
    return omit_arr


def precompute_freq_kl8(records, window=30):
    """
    批量一次性计算快乐8所有期数的"近N期逐球出现频率"（80维，每个球一个独立数值）
    这是遗漏向量之外，第二个真正能逐球区分号码的信号。
    ── 为什么需要这个 ──
    快乐8 PPO 需要给80个球各自打分，但状态向量里绝大部分内容
    （走势聚合特征、ML分组概率、LSTM/TFM隐层）对80个球来说都是完全相同的共享信息，
    根本不携带"这是哪个球"的区分度。之前只有遗漏值这一个80维信号能区分个体球，
    信号太单薄，导致网络最后一层的输出权重很容易在训练中自己走出一个
    跟真实状态无关、只是训练过程偶然形成的固定偏好（表现为持续偏向某个号码区间）。
    加入"近期出现频率"作为第二个逐球信号，让网络有更充分的依据去真正学习"选哪个球"，
    而不是在信息不足的情况下被迫依赖训练过程中的随机偏置。
    """
    N = len(records)
    freq_arr = np.zeros((N+1, 80), dtype=np.float32)
    from collections import deque
    recent = deque(maxlen=window)
    for idx in range(N+1):
        cnt = Counter(n for r in recent for n in r['numbers'])
        for n in range(1, 81):
            freq_arr[idx, n-1] = cnt.get(n, 0)
        if idx < N:
            recent.append(records[idx])
    return freq_arr


def precompute_omission_ssq(records):
    """双色球：33红球+16蓝球=49维遗漏向量，批量预计算"""
    N = len(records)
    last_seen_r = {}; last_seen_b = {}
    omit_arr = np.zeros((N+1, 49), dtype=np.float32)
    for idx in range(N+1):
        avg_r = max(idx*6/33, 1); avg_b = max(idx/16, 1)
        for n in range(1,34):
            if n in last_seen_r:
                omit_arr[idx, n-1] = min((idx-1-last_seen_r[n])/avg_r, 3)
            else:
                omit_arr[idx, n-1] = 2.0
        for n in range(1,17):
            if n in last_seen_b:
                omit_arr[idx, 33+n-1] = min((idx-1-last_seen_b[n])/avg_b, 3)
            else:
                omit_arr[idx, 33+n-1] = 2.0
        if idx < N:
            for n in records[idx]['red']: last_seen_r[n]=idx
            last_seen_b[records[idx]['blue']] = idx
    return omit_arr


# ══════════════════════════════════════════════════════
#  马尔可夫转移 + 贝叶斯后验不确定性
#  这两类信号在RL现有状态里都没有对应物：
#  · 马尔可夫建模的是 P(下期=Y | 上期=X) 这种【跨期条件转移】，
#    而走势特征是窗口聚合量、遗漏是"多久没出"，都不是转移关系。
#  · 贝叶斯给的是【不确定性】——现有所有信号都只说"是什么"，
#    没有一个告诉模型"这个判断有多可信"。样本少时后验方差大，
#    模型可以据此少依赖该信号，这是点估计给不了的信息。
# ══════════════════════════════════════════════════════
MARKOV_WINDOW_RL = 200          # 转移统计只看最近N期，太长会让概率长期不变
BAYES_PRIOR = 1.0               # Beta先验的伪计数（拉普拉斯平滑）


def precompute_markov_3d(records, window=MARKOV_WINDOW_RL):
    """
    3D马尔可夫：对每一位，给出"上期该位是X时，下期各数字的转移概率"。
    返回 arr[idx] = 30维（百/十/个位各10个数字的转移概率）
    """
    N = len(records)
    arr = np.zeros((N+1, 30), dtype=np.float32)
    for idx in range(1, N+1):
        w = records[max(0, idx-window):idx]
        if len(w) < 2:
            arr[idx] = 0.1; continue
        last = records[idx-1]['digits']
        for p in range(3):
            cnt = np.full(10, BAYES_PRIOR, dtype=np.float32)   # 拉普拉斯平滑，避免零概率
            for i in range(1, len(w)):
                if w[i-1]['digits'][p] == last[p]:
                    cnt[w[i]['digits'][p]] += 1.0
            arr[idx, p*10:(p+1)*10] = cnt / cnt.sum()
    return arr


def precompute_markov_balls(records, pool_max, get_nums, window=MARKOV_WINDOW_RL):
    """
    号码型玩法的马尔可夫：给出"上期开出的那批号码之后，各号码出现的条件概率"。
    对每个号码 n，统计"上期出现过X且本期出现n"的比例（X遍历上期号码）。
    返回 arr[idx] = pool_max 维
    """
    N = len(records)
    arr = np.zeros((N+1, pool_max), dtype=np.float32)
    for idx in range(1, N+1):
        w = records[max(0, idx-window):idx]
        if len(w) < 2:
            arr[idx] = 1.0/pool_max; continue
        prev_set = set(get_nums(records[idx-1]))
        cnt = np.full(pool_max, BAYES_PRIOR, dtype=np.float32)
        base = BAYES_PRIOR * pool_max
        for i in range(1, len(w)):
            # 上一期与"当前上期"有重叠时，本期的号码作为转移证据（重叠越多权重越高）
            overlap = len(set(get_nums(w[i-1])) & prev_set)
            if overlap == 0: continue
            wt = overlap / max(len(prev_set), 1)
            for n in get_nums(w[i]):
                cnt[n-1] += wt
            base += wt * len(get_nums(w[i]))
        arr[idx] = cnt / max(base, 1e-6)
    return arr


def precompute_bayes(records, pool_max, get_nums, window=MARKOV_WINDOW_RL):
    """
    贝叶斯后验：对每个号码用 Beta-Binomial 估计出现率，同时给出【不确定性】。
    返回 arr[idx] = pool_max*2 维（前半是后验均值，后半是后验标准差）

    后验标准差是关键：观测样本少时它大，模型可以学会"这个估计不可信，别太依赖"。
    现有的频率/遗漏都是点估计，给不了这个信息。
    """
    N = len(records)
    arr = np.zeros((N+1, pool_max*2), dtype=np.float32)
    for idx in range(1, N+1):
        w = records[max(0, idx-window):idx]
        n_draw = len(w)
        if n_draw < 2:
            arr[idx, :pool_max] = 0.5; arr[idx, pool_max:] = 0.5; continue
        hits = np.full(pool_max, 0.0, dtype=np.float32)
        for r in w:
            for n in get_nums(r): hits[n-1] += 1.0
        a = hits + BAYES_PRIOR                       # Beta后验的alpha
        b = (n_draw - hits) + BAYES_PRIOR            # Beta后验的beta
        mean = a / (a + b)
        var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
        arr[idx, :pool_max] = mean
        arr[idx, pool_max:] = np.sqrt(var)
    return arr


def precompute_omission_3d(records):
    """福彩3D：百十个位各10个数字，共30维遗漏向量"""
    N = len(records)
    last_seen = [{}, {}, {}]  # 每位一个字典：数字 -> 最后出现下标
    omit_arr = np.zeros((N+1, 30), dtype=np.float32)
    for idx in range(N+1):
        avg = max(idx/10, 1)
        for pos in range(3):
            for d in range(10):
                if d in last_seen[pos]:
                    omit_arr[idx, pos*10+d] = min((idx-1-last_seen[pos][d])/avg, 3)
                else:
                    omit_arr[idx, pos*10+d] = 2.0
        if idx < N:
            for pos in range(3):
                last_seen[pos][records[idx]['digits'][pos]] = idx
    return omit_arr

# ══════════════════════════════════════════════════════
#  ML概率向量 + 遗漏向量
# ══════════════════════════════════════════════════════
def extract_ml_prob_vec(ml_pred, game, verbose=True):
    vec = []
    models_data = ml_pred.get('models', {})
    # blue 目标在传统ML里标签范围是1-16（未做偏移），其余目标都是0起始的分组标签
    if game=='3d': tk=['sum_grp','odd','group_type','big','span_grp','road_dom','arith']; nc=[3,4,3,4,3,3,2]; offsets=[0,0,0,0,0,0,0]
    elif game=='ssq': tk=['odd','sum_grp','ac_grp','red_zone_dom','gap_grp','big','consec']; nc=[7,3,3,3,3,7,6]; offsets=[0,0,0,0,0,0,0]
    else: tk=['odd_grp','zone_dom','tot_grp','big_grp','five_dom','consec_grp','range_grp']; nc=[3,4,3,3,5,3,3]; offsets=[0,0,0,0,0,0,0]
    found, missing = [], []
    for tkey,n,off in zip(tk,nc,offsets):
        m = models_data.get(tkey,{}); probs = m.get('prediction',{}).get('probs',{})
        seg = [float(probs.get(str(i+off),0.0))/100.0 for i in range(n)]
        vec.extend(seg)
        (found if any(v>0 for v in seg) else missing).append(tkey)
    if verbose:
        print(f"    [传统ML状态注入验证] {game}: 成功读到概率的目标={found}")
        if missing:
            print(f"    ⚠️ [传统ML状态注入验证] 以下目标概率全为0，未生效={missing}（请确认 kaggle_fucai.py 已用最新目标重新跑过）")
    return np.array(vec, dtype=np.float32)

def extract_dl_prob_vec(dl_pred, game, verbose=True):
    """
    提取深度学习(LSTM+Transformer集成)对各目标的预测概率，注入RL状态。

    ── 为什么要补这个 ──
    之前 RL 读了传统ML全部7个目标的概率，却一条深度学习的预测都没读——
    DL 训练出的7组预测写在 dl_lstm_tfm.json 里，RL 脚本对它零引用。
    RL 能接触到 DL 的唯一渠道是"第一个训练目标的隐层状态"，
    另外6个目标学到的东西完全浪费。这里把 DL 的预测概率也接进来。

    结构与 extract_ml_prob_vec 对称：同样7个目标，同样按概率展开成向量。
    """
    vec = []
    game_data = (dl_pred or {}).get(game, {})
    if game=='3d': tk=['sum_grp','odd','group_type','big','span_grp','road_dom','arith']; nc=[3,4,3,4,3,3,2]
    elif game=='ssq': tk=['odd','sum_grp','ac_grp','red_zone_dom','gap_grp','big','consec']; nc=[7,3,3,3,3,7,6]
    else: tk=['odd_grp','zone_dom','tot_grp','big_grp','five_dom','consec_grp','range_grp']; nc=[3,4,3,3,5,3,3]
    found, missing = [], []
    for tkey, n in zip(tk, nc):
        probs = (game_data.get(tkey, {}) or {}).get('probs', {})
        seg = [float(probs.get(str(i), 0.0))/100.0 for i in range(n)]
        vec.extend(seg)
        (found if any(v > 0 for v in seg) else missing).append(tkey)
    if verbose:
        if found:
            print(f"    [深度学习状态注入验证] {game}: 成功读到概率的目标={found}")
        if missing:
            print(f"    ⚠️ [深度学习状态注入验证] 以下目标概率全为0，未生效={missing}"
                  f"（若全部未生效，通常是 dl_lstm_tfm.json 尚未生成或该游戏未训练）")
    return np.array(vec, dtype=np.float32)


# （omission_vec_kl8/omission_vec_ssq 已被上方 precompute_omission_* 批量预计算版本取代）

# ══════════════════════════════════════════════════════
#  快乐8真实赔率
# ══════════════════════════════════════════════════════
KL8_PAYOUT = {(1,1):2,(2,2):10,(3,3):30,(4,4):100,(4,3):3,(4,2):1,(5,5):200,(5,4):8,(5,3):1,
    (6,6):1000,(6,5):20,(6,4):2,(6,3):1,(7,7):2000,(7,6):50,(7,5):4,(7,4):1,
    (8,8):5000,(8,7):100,(8,6):8,(8,5):1,(9,9):10000,(9,8):300,(9,7):20,(9,6):2,
    (10,10):18000,(10,9):600,(10,8):30,(10,7):3,(10,6):1}
TICKET_PRICE = 2.0
def calc_payout(n,h): return KL8_PAYOUT.get((n,h),0)*TICKET_PRICE - TICKET_PRICE

# ══════════════════════════════════════════════════════
#  回测隔离：训练时必须留出最后 HOLDOUT_N 期不参与训练，专门用于回测。
#  之前训练覆盖全部历史（第25期~最后一期），而回测取的是最近30期——
#  也就是说回测数据本身就在训练集里，考的是"背没背下来"而不是"会不会做新题"。
#  修好模型持久化后，训练量逐日累积，模型开始记住这30期，
#  回测平均命中从0.28位(=随机)涨到0.57位，这不是学会了，是背下来了。
#  留出holdout后，回测才是真正的样本外评估。
# ══════════════════════════════════════════════════════
# holdout期数。30期时标准误约0.095，能分辨的最小差异要0.19——
# 比3D随机基准0.30的一半还大，早停在这个精度下等于抛硬币，
# 它每次保留"训练前的随机权重"只是因为那次运气好，不代表模型更优。
# 提到150期后标准误降到约0.042，早停才开始有判断力。
HOLDOUT_N = 150

# 3D推荐注数。候选池是每位Top3，最多能组合出 3×3×3=27 种，
# 前3注用于保证每位的3个候选都出场，其余按联合概率从高到低补足。
D3_N_BETS = 12

def holdout_size(n_records):
    """按数据量自适应：至少80期保证测量精度，最多不超过总数的8%避免过度损失训练数据"""
    return max(80, min(HOLDOUT_N, int(n_records * 0.08)))

# ══════════════════════════════════════════════════════
#  熵系数（entropy coefficient）
#  PPO的损失里有一项 -ent_coef × entropy，也就是"分布越均匀奖励越高"。
#  这个值当初调高是为了防止策略过早收敛到单一偏好（比如快乐8"全是大号"那次），
#  但副作用是：3D在 ent_coef=0.03 下，保持均匀分布能白拿 0.03×ln(10)=0.069 的奖励，
#  而做出判断的期望收益只有 0.1 分且方差极大 —— 模型算下来发现"不判断"更划算，
#  于是熵稳定在 2.303（=ln10，完全均匀），输出各候选都是 10.0~10.1%，毫无区分度。
#
#  现在大幅调低，让模型有动力去做判断。注意这是把双刃剑：
#  如果数据里真没规律，降低熵只会让模型"自信地瞎猜"——
#  所以必须靠 holdout 评分来验证，而不是看分布拉开了就以为变好了。
# ══════════════════════════════════════════════════════
ENT_COEF = {
    '3d':  0.005,   # 原 0.03
    'ssq': 0.005,   # 原 0.02
    'kl8': 0.010,   # 原 0.05（快乐8要从80球选6个，保留稍多探索）
}


def report_state_sensitivity(model, build_state_fn, idx_list, game, top_k=3):
    """
    状态敏感度诊断：喂入多个【时间跨度很大、内容差异明显】的历史状态，
    看模型输出的Top候选变不变。

    ── 为什么必须查这个 ──
    如果模型输出的Top候选在完全不同的历史时点都一样，
    说明它压根没在用状态——学到的是一个固定偏好，跟输入无关。
    这种情况下"每天喂新数据"毫无意义，推荐永远不变，
    今天开什么奖对预测没有任何影响（实测出现过连续两天12注完全相同）。
    """
    tops, probs_all = [], []
    for i in idx_list:
        st = build_state_fn(i)
        if st is None: continue
        try:
            obs_t, _ = model.policy.obs_to_tensor(np.array(st).reshape(1, -1))
            with torch.no_grad():
                dist = model.policy.get_distribution(obs_t)
            dl = getattr(dist, 'distribution', None)
            if isinstance(dl, (list, tuple)):     # MultiDiscrete
                ps = [d.probs.detach().cpu().numpy()[0] for d in dl]
                tops.append(tuple(int(np.argmax(p)) for p in ps))
                probs_all.append(np.concatenate(ps))
            else:                                  # Box
                a, _ = model.predict(st, deterministic=True)
                a = np.asarray(a).ravel()
                tops.append(tuple(int(x) for x in np.argsort(a)[::-1][:top_k]))
                probs_all.append(a)
        except Exception:
            continue
    if len(tops) < 2:
        return
    uniq = len(set(tops))
    arr = np.array(probs_all)
    var = float(np.mean(np.std(arr, axis=0)))     # 各维度在不同状态间的平均波动
    print(f"    [状态敏感度] 在{len(tops)}个差异很大的历史时点上测试：")
    print(f"      输出Top组合共{uniq}种（{len(tops)}次测试）  各维度平均波动 {var:.4f}")
    if uniq == 1:
        print(f"      ⚠️ 所有时点输出完全相同 —— 模型忽略了状态，学到的是固定偏好，")
        print(f"         这意味着每天的新开奖对预测没有任何影响，推荐永远不会变")
    elif uniq <= max(2, len(tops)//4):
        print(f"      ⚠️ 输出种类很少，模型对状态的响应很弱")
    else:
        print(f"      ✓ 模型输出随状态变化，确实在用输入信息")


def eval_logprob_3d(model, build_state_fn, records, rng_idx, use_model=None):
    """
    用【平均对数概率】评分，比"平均命中位数"灵敏得多。

    命中位数每期只有0/1/2/3四个取值：模型把正确数字的概率从10%提到15%，
    只要还不是最大值，这个指标完全看不见。
    对数概率用的是全部10个概率值，这种改善能被捕捉到。
    （机器学习里评估分类模型普遍用对数损失而非准确率，正是这个原因。）

    返回值越大越好；均匀分布的基准是 ln(0.1) = -2.303。
    """
    m = use_model if use_model is not None else model
    tot, n = 0.0, 0
    for i in rng_idx:
        st = build_state_fn(i)
        if st is None: continue
        try:
            obs_t, _ = m.policy.obs_to_tensor(np.array(st).reshape(1, -1))
            with torch.no_grad():
                dist = m.policy.get_distribution(obs_t)
            dl = getattr(dist, 'distribution', None)
            if not isinstance(dl, (list, tuple)): continue
            act = records[i]['digits']
            for p_i, d in enumerate(dl):
                p = d.probs.detach().cpu().numpy()[0]
                tot += math.log(float(p[act[p_i]]) + 1e-12)
                n += 1
        except Exception:
            continue
    return tot / n if n else -99.0


def policy_dependence_test(model, build_state_fn, idx_list, game, n_pos=3, n_val=10):
    """
    策略依赖性检测：模型到底有没有在用状态？

    ── 为什么这是最关键的诊断 ──
    如果模型对差异很大的历史时点都输出几乎相同的分布，
    说明它收敛成了"常数策略"——不管输入什么都推荐同一组号码，
    那么权重学到的东西就没有意义，新开奖也永远影响不了推荐。
    这跟"输入变化太小"是完全不同的两个问题，处理方式也完全不同。

    做法：取若干相隔很远的历史时点，各算一次输出分布，
    统计"最高概率数字"是否都一样、分布之间差异有多大。
    """
    dists, tops = [], []
    for i in idx_list:
        st = build_state_fn(i)
        if st is None: continue
        try:
            obs_t, _ = model.policy.obs_to_tensor(np.array(st).reshape(1, -1))
            with torch.no_grad():
                dist = model.policy.get_distribution(obs_t)
            dl = getattr(dist, 'distribution', None)
            if isinstance(dl, (list, tuple)):
                p = np.concatenate([d.probs.detach().cpu().numpy()[0] for d in dl])
                tops.append(tuple(int(np.argmax(d.probs.detach().cpu().numpy()[0])) for d in dl))
            else:
                p = model.predict(st, deterministic=True)[0].astype(np.float32)
                tops.append(tuple(np.argsort(p)[-6:].tolist()))
            dists.append(p)
        except Exception:
            continue
    if len(dists) < 3:
        print(f"    [策略依赖性] 样本不足，跳过"); return None
    D = np.array(dists)
    # 各维度在不同时点之间的标准差：越大说明输出越随状态变化
    spread = float(np.mean(np.std(D, axis=0)))
    mean_lvl = float(np.mean(np.abs(D)))
    ratio = spread / (mean_lvl + 1e-9)
    uniq = len(set(tops))
    print(f"    [策略依赖性] 取{len(dists)}个相隔较远的历史时点分别预测：")
    print(f"      最优组合出现 {uniq} 种不同结果（{len(dists)}个时点）")
    print(f"      输出分布跨时点波动/均值 = {ratio*100:.1f}%")
    if uniq <= 1 or ratio < 0.05:
        print(f"      ⚠️ 模型基本是常数策略——不管输入什么都给同样的答案，")
        print(f"         权重没有学到状态相关的规律，新开奖自然影响不了推荐")
    elif ratio < 0.20:
        print(f"      · 模型对状态有响应但较弱，相邻两天差异小属正常")
    else:
        print(f"      ✓ 模型输出明显随状态变化，权重是有内容的")
    return {'n_points': len(dists), 'unique_best': uniq, 'spread_ratio': round(ratio, 4)}


def report_entropy(model, state, game, n_out=None):
    """
    打印策略输出的熵，用来判断模型到底有没有在做判断。
    熵接近理论最大值 = 完全均匀 = 什么都没学到。
    """
    try:
        obs_t, _ = model.policy.obs_to_tensor(np.array(state).reshape(1, -1))
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_t)
        dl = getattr(dist, 'distribution', None)
        if isinstance(dl, (list, tuple)):     # MultiDiscrete：每位一个分布
            ents, maxent = [], math.log(10)
            for d in dl:
                p = d.probs.detach().cpu().numpy()[0]
                ents.append(float(-(p*np.log(p+1e-12)).sum()))
            print(f"    [熵监控] {game} 各位熵: {[round(e,3) for e in ents]}  "
                  f"（均匀={maxent:.3f}，越低说明模型越有主见）")
            return ents
        else:                                  # Box：高斯策略，看动作标准差
            std = float(np.mean(np.exp(model.policy.log_std.detach().cpu().numpy())))
            print(f"    [熵监控] {game} 动作分布平均标准差: {std:.4f}（越小说明模型越确定）")
            return [std]
    except Exception as e:
        print(f"    [熵监控] 读取失败: {e}")
        return []

# ══════════════════════════════════════════════════════
#  状态向量分段开关：改这里就能开关任意一类信号，不用动其它代码。
#  False = 该段清零（维度保留，不触发模型重训，随时可切回来）
#
#  ⚠️ 判断依据要看历史积累，别只看单日消融数字：
#     holdout只有30期×3位=90次预测，随机波动的标准差就有3次命中，
#     单日消融里 ±0.1 的"贡献"只有约1个标准差，属于纯噪声，
#     明天再跑很可能正负号就翻转了。
#     应该看 {game}_history.json 里连续多天的 ablation 记录，
#     某段连续一两周都是负贡献，关掉才有依据。
# ══════════════════════════════════════════════════════
# 分段开关：把某段清零来测试它的贡献。清零不改变维度，切换不触发重训。
SEGMENT_ENABLE = {
    '3d':  {'走势特征':True, 'ML+DL概率':True, 'LSTM隐层':True, 'TFM隐层':True,
            '遗漏':True, '马尔可夫':True, '贝叶斯':True},
    'ssq': {'走势特征':True, 'ML+DL概率':True, 'LSTM隐层':True, 'TFM隐层':True,
            '遗漏':True, '马尔可夫':True, '贝叶斯':True},
    'kl8': {'走势特征':True, 'ML+DL概率':True, 'LSTM隐层':True, 'TFM隐层':True,
            '遗漏':True, '频率':True, '马尔可夫':True, '贝叶斯':True},
}


def apply_segment_switches(state, segs, game):
    """按 SEGMENT_ENABLE 把关闭的段清零。维度不变，所以切换开关不会触发模型重训。"""
    sw = SEGMENT_ENABLE.get(game, {})
    if all(sw.get(n, True) for n, _, _ in segs):
        return state
    state = state.copy()
    for name, s, e in segs:
        if not sw.get(name, True):
            state[s:e] = 0.0
    return state


def segment_ablation(eval_with_mask, segments, base_score, label, game=None, se=0.05):
    """
    消融诊断：逐段把输入清零，看holdout评分掉多少。掉得越多说明这段越重要。

    ── 为什么需要 ──
    现在状态向量有800~980维，其中隐层占70~80%，但这个配比是人为定的、没有验证依据。
    参数量已是训练样本的6倍，很可能大部分维度只是在制造过拟合。
    与其靠猜，不如直接测：把某段清零，模型表现掉多少，就是这段的真实贡献。

    eval_with_mask: 接受 (start, end) 元组，把该区间清零后评估，返回分数
    segments: [(名称, 起点, 终点), ...]
    """
    _se = se
    print(f"    [{label} 消融诊断] 基准分 {base_score:.4f}，逐段清零后的变化"
          f"（噪声水平±{2*_se:.3f}，差异小于此值不可信）：")
    results = []
    _sw = SEGMENT_ENABLE.get(game, {}) if game else {}
    for name, s, e in segments:
        if e <= s: continue
        if not _sw.get(name, True):
            # 该段已被开关关闭（值已全为0），再清零一次当然毫无变化，
            # 测出来的"贡献0.0000/几乎无影响"是假象，会误导判断，直接跳过
            print(f"      {name:12}({e-s:4}维)  已通过开关关闭，本次不参与消融测量")
            results.append({'segment': name, 'dims': e-s, 'disabled': True})
            continue
        try:
            score = eval_with_mask((s, e))
        except Exception as ex:
            print(f"      {name}: 评估失败({ex})"); continue
        drop = base_score - score
        results.append({'segment': name, 'dims': e-s, 'score_without': round(score,4),
                        'contribution': round(drop,4)})
        # 用测量噪声水平作为判断门槛，而不是拍脑袋定的固定值。
        # 差异小于2倍标准误时无法与随机波动区分，标成"不可分辨"而非"无影响"，
        # 避免把噪声当成结论去删特征。
        if abs(drop) < 2 * _se:
            mark = f"— 不可分辨（噪声±{2*_se:.3f}）"
        elif drop > 0:
            mark = "★ 重要"
        else:
            mark = "⚠ 去掉反而更好"
        print(f"      {name:12}({e-s:4}维)  清零后{score:.4f}  贡献{drop:+.4f}  {mark}")
    return results


def append_history(game, record):
    """
    把每日成绩追加到历史文件，随RL_LOCAL_DIR一起持久化到Dataset。
    单日数字波动大说明不了问题，长期曲线才能看出模型是在进步、退步还是原地打转。
    """
    path = f'{RL_LOCAL_DIR}/{game}_history.json'
    hist = []
    for p in (f'{RL_MOUNTED}/{game}_history.json', path):
        if os.path.exists(p):
            try:
                with open(p) as f: hist = json.load(f)
                break
            except Exception: pass
    hist.append(record)
    hist = hist[-400:]   # 只保留最近400条，避免文件无限膨胀
    os.makedirs(RL_LOCAL_DIR, exist_ok=True)
    try:
        with open(path, 'w') as f: json.dump(hist, f, ensure_ascii=False, indent=1)
        if len(hist) >= 3:
            sel = [h.get('holdout_score') for h in hist[-10:] if h.get('holdout_score') is not None]
            cln = [h.get('clean_score')   for h in hist[-10:] if h.get('clean_score')   is not None]
            if len(sel) >= 3:
                print(f"    [历史] 已积累{len(hist)}天记录")
                print(f"      选择分(会虚高): {[round(r,3) for r in sel]}")
                if len(cln) >= 3:
                    print(f"      干净分(可信)  : {[round(r,3) for r in cln]}")
                    # 两条曲线背离 = 涨的是筛选假象，不是真本事
                    d_sel = sel[-1] - sum(sel[:-1])/len(sel[:-1])
                    d_cln = cln[-1] - sum(cln[:-1])/len(cln[:-1])
                    if d_sel > 0.02 and d_cln <= 0:
                        print(f"      ⚠ 选择分在涨({d_sel:+.3f})但干净分没涨({d_cln:+.3f})"
                              f" —— 涨的是对holdout的过拟合，不是真实能力")
                    elif d_cln > 0.02:
                        print(f"      ✓ 干净分也在涨({d_cln:+.3f})，是真实提升")
    except Exception as e:
        print(f"    ! 写入历史失败: {e}")


# ══════════════════════════════════════════════════════
#  训练模式
#  'frozen'      首训一次得到权重后冻结，之后每天只加载权重 + 喂新数据 + 出预测，
#                不再训练。这样既不会过拟合（根本没训练），输出也稳定
#                （权重固定，结果只随新数据变，不会今天有偏好明天变均匀）。
#  'incremental' 每天在旧权重上微调（实测第2天六段全部低于基准、零提升）
#  'fresh'       每天从零全量训练
# ══════════════════════════════════════════════════════
TRAIN_MODE = 'frozen'

# ══════════════════════════════════════════════════════
#  挑战者机制：定期训练一个新模型去挑战现任，赢了才换
#
#  ── 为什么不是"冻结半年" ──
#  之前设成180期才重训一次，隐含假设是"首训就是最好的"，但这没有依据：
#  RL训练带随机性，一次首训完全可能落在很差的局部最优，
#  冻结半年等于把一个可能很烂的模型锁死半年。
#
#  ── 现在的做法 ──
#  每 CHALLENGE_EVERY 期就重训一个【全新随机初始化】的挑战者，
#  在【从未参与训练、也从未参与早停选择】的干净holdout上跟现任比。
#  只有赢过现任且幅度超过噪声水平(2倍标准误)，挑战者才上位。
#
#  三个效果同时成立：
#    进化   —— 每次挑战都是一次找到更好模型的机会，不用等半年
#    不过拟合 —— 上位门槛是干净数据上的真实提升，靠运气赢不了
#    稳定   —— 挑战失败就完全不动，权重和输出保持不变
# ══════════════════════════════════════════════════════
# 每期都挑战（新数据一到就发起），让模型尽快稳定到较好状态。
# 注意：挑战次数变多会带来"多重比较"问题——
# 实测所有模型真实水平完全相同时，每天挑战365次，
# 有82%的概率纯靠运气发生一次"假上位"。挑得越勤，蒙对的机会越多。
# 解决办法是把门槛从2倍标准误提到3.5倍，并加一道复检（见 run_challenge）：
#   门槛2.0SE：假上位66.5%，真提升采纳99.7%
#   门槛3.5SE：假上位17.7%，真提升采纳91.1%
#   门槛3.5SE+复检：假上位 4.8%，真提升采纳89.5%  ← 采用这个
CHALLENGE_EVERY = {'3d': 1, 'ssq': 1, 'kl8': 1}

# 上位门槛（标准误的倍数）。每期挑战下必须调高，否则运气会不断"换人"
# 上位门槛（标准误的倍数）。
# 原本设3.5是为了防每日挑战的多重比较问题，但实测 SE≈0.06 时
# 3.5SE=0.21，等于要求挑战者比随机基准0.30高出70%——这几乎不可能发生，
# 挑战机制形同虚设、权重永远不换。
# 改为2.0，多重比较的防护交给"复检"那一关（两轮独立数据都要赢）。
# 上位门槛（标准误的倍数）。这个值必须和 holdout 大小配套看：
#   干净区 75期时，2.0倍SE = 0.120，要求挑战者达到0.42（随机基准的1.4倍）—— 太苛刻
#   干净区500期时，2.0倍SE = 0.046，要求达到0.346 —— 一个真有提升的模型过得去
# 所以真正的解法是把 holdout 扩大（见 holdout_size），而不是调低这个数。
# 配合每期挑战+复检，纯运气很难连续两轮都赢这么多。
CHALLENGE_MARGIN_K = 2.0


def run_challenge(cur_model, make_fresh_model, train_fn, eval_clean, se, label, recheck_fn=None):
    """
    挑战流程：训练一个全新模型，在干净holdout上与现任比较，赢了才替换。

    cur_model:        现任模型（可能为None，即首次运行）
    make_fresh_model: 无参函数，返回一个全新随机初始化的模型
    train_fn:         接受模型，训练后返回 (model, best_score, history)
    eval_clean:       接受模型，在【干净holdout】上评分（该数据从未参与训练和早停选择）
    se:               该评分的标准误，用来定"赢多少才算真赢"

    返回 (最终采用的模型, 是否换人, 说明文字)
    """
    if cur_model is None:
        m, _b, _h = train_fn(make_fresh_model())
        return m, True, "首次训练，直接采用"

    cur_score = eval_clean(cur_model)
    print(f"    [挑战赛] 现任模型干净评分: {cur_score:.4f}")
    print(f"    [挑战赛] 训练挑战者（全新随机初始化）…")
    challenger, _b, _h = train_fn(make_fresh_model())
    new_score = eval_clean(challenger)

    margin = new_score - cur_score
    need = CHALLENGE_MARGIN_K * se
    print(f"    [挑战赛] 挑战者干净评分: {new_score:.4f}  差距 {margin:+.4f}"
          f"（需超过 {need:.4f} = {CHALLENGE_MARGIN_K}倍标准误）")
    if margin <= need:
        return cur_model, False, f"✗ 挑战失败，保留现任（差距 {margin:+.4f} 未达门槛 {need:.4f}）"

    # 复检：赢了先别急着换。用另一半holdout（早停选权重用的那半，
    # 对挑战者和现任同样都是"没参与过自己训练"的数据）独立再比一次。
    # 单次赢可能是运气，两次都赢才可信——实测把假上位率从17.7%压到4.8%。
    if recheck_fn is not None:
        c2, n2 = recheck_fn(cur_model), recheck_fn(challenger)
        m2 = n2 - c2
        print(f"    [复检] 另一半数据上 现任{c2:.4f} vs 挑战者{n2:.4f}  差距 {m2:+.4f}"
              f"（需超过 {need*0.5:.4f}）")
        if m2 <= need * 0.5:
            return cur_model, False, f"✗ 复检未通过，保留现任（首轮{margin:+.4f} 但复检仅{m2:+.4f}）"
        return challenger, True, f"✓ 两轮均胜，新模型上位（首轮{margin:+.4f} 复检{m2:+.4f}）"
    return challenger, True, f"✓ 挑战成功，新模型上位（提升 {margin:+.4f}）"


def should_challenge(game, cur_n):
    """是否到了发起挑战的时候。返回 (是否挑战, 说明)"""
    last = get_last_trained_n(game)
    if last <= 0:
        return True, "首次运行，需要完整训练"
    grew = cur_n - last
    need = CHALLENGE_EVERY.get(game, 20)
    if grew >= need:
        return True, f"距上次挑战已新增{grew}期（周期{need}期），发起新挑战"
    return False, f"距下次挑战还差{need-grew}期（已积累{grew}/{need}），保持现任模型"



def train_with_early_stop(model, total_steps, eval_fn, label,
                          n_chunks=8, patience=3, reset_timesteps=True, warmup_chunks=0):
    """
    分段训练 + 早停 + 保留最佳模型。

    ── 解决的问题 ──
    原来是"练满N步 → 无条件保存 → 覆盖昨天的模型"，
    哪怕今天练完变差了也照样覆盖，而且旧版本被覆盖后不可恢复。
    强化学习训练不稳定是常态，很容易越练越差。

    ── 现在的做法 ──
    把训练预算切成 n_chunks 段，每段结束后在 holdout（模型未训练过的最近30期）
    上评估一次，记录最佳分数和当时的权重快照。
    连续 patience 段没有提升就提前停止，最后把权重恢复到最佳状态再返回。

    warmup_chunks: 前N段不触发早停。RL训练前期策略震荡是正常现象，
    刚开始几段分数掉下去很常见，这时候判死刑会让模型永远停在随机初始状态
    （实测3D连续三次运行都因为前3段下滑而回滚到未训练权重）。
    热身期内照常记录最佳分，但不累计"无提升"计数。

    eval_fn: 无参函数，返回一个分数（越大越好）
    返回 (model, best_score, history)
    """
    chunk = max(1, total_steps // n_chunks)
    best_score = eval_fn()
    best_params = copy.deepcopy(model.get_parameters())
    history = [round(best_score, 4)]
    no_improve = 0
    print(f"    [{label}] 训练前基准分: {best_score:.4f}")

    for i in range(n_chunks):
        model.learn(total_timesteps=chunk,
                    reset_num_timesteps=(reset_timesteps and i == 0),
                    progress_bar=False)
        score = eval_fn()
        history.append(round(score, 4))
        if score > best_score:
            best_score = score
            best_params = copy.deepcopy(model.get_parameters())
            no_improve = 0
            flag = "✓ 新最佳"
        elif i < warmup_chunks:
            flag = f"（热身期第{i+1}/{warmup_chunks}段，不计入早停）"
        else:
            no_improve += 1
            flag = f"（无提升 {no_improve}/{patience}）"
        print(f"    [{label}] 第{i+1}/{n_chunks}段({chunk}步) 评分 {score:.4f}  {flag}")
        if i >= warmup_chunks and no_improve >= patience:
            print(f"    [{label}] 连续{patience}段无提升，提前停止（省下{(n_chunks-i-1)*chunk}步）")
            break

    # 恢复到最佳状态，而不是用最后一段训练完的（可能更差的）权重
    model.set_parameters(best_params)
    print(f"    [{label}] 已恢复到最佳权重，最终评分 {best_score:.4f}  评分轨迹: {history}")
    return model, best_score, history



def diverse_picks(scores, n_pick, n_bets, pool_mult=2.2, core_ratio=0.34):
    """
    多样化选号：胆码 + 拖码轮转。

    ── 为什么不用"联合得分Top-N" ──
    按组合总分排序取前N注，结果必然是N注共享同样的头几个高分球、只有末尾一两个位置在微调
    （实测双色球6注平均重合4.8/6个球，只用到8个不同号码），
    这跟只推1注没有本质区别，完全体现不出多元化，实际参考意义很低。

    ── 现在的做法 ──
    1) 候选池：按模型打分取前 pool_mult 倍于所需球数的号码，超出的说明模型不看好，不予考虑
    2) 胆码：池中打分最高的那少数几个球，模型最确信，进入每一注
    3) 拖码：池中其余球轮转填充各注剩余位置，让模型看好的号码都有机会出场

    这样既尊重模型的置信度层次（越看好的球出现越频繁），又保证各注之间有实质差异。
    返回 (各注号码列表, 胆码, 候选池)
    """
    order = np.argsort(scores)[::-1]
    pool_size = min(len(scores), max(n_pick + 2, int(round(n_pick * pool_mult))))
    pool = [int(order[i]) + 1 for i in range(pool_size)]
    core_n = max(1, min(n_pick - 1, int(round(n_pick * core_ratio))))
    core, rest = pool[:core_n], pool[core_n:]
    bets, ri = [], 0
    for _ in range(n_bets):
        sel = list(core)
        guard = 0
        while len(sel) < n_pick and rest and guard < len(rest) * 3:
            cand = rest[ri % len(rest)]; ri += 1; guard += 1
            if cand not in sel: sel.append(cand)
        # 池子不够时（理论上不会），用全局排序补齐
        oi = 0
        while len(sel) < n_pick and oi < len(order):
            c = int(order[oi]) + 1; oi += 1
            if c not in sel: sel.append(c)
        bets.append(sorted(sel))
    return bets, sorted(core), sorted(pool)


# ══════════════════════════════════════════════════════
#  新数据检测：避免拿完全相同的数据反复训练（过拟合防护）
#  三个游戏共用。记录"上次训练时用了多少期数据"，
#  下次运行时若期数没增加，说明没有新开奖，跳过训练直接沿用上次结果。
#  这对手动重复触发尤其重要——否则同一批数据被训练N次，
#  模型会对这批数据过度拟合，反而降低泛化能力。
# ══════════════════════════════════════════════════════
def get_last_trained_n(game):
    """读取上次训练时的数据期数（优先读挂载的Dataset，那是上次运行持久化的结果）"""
    for path in (f'{RL_MOUNTED}/{game}_last_trained_n.json',
                 f'{RL_LOCAL_DIR}/{game}_last_trained_n.json'):
        if os.path.exists(path):
            try:
                with open(path) as f: return json.load(f).get('n', 0)
            except Exception: pass
    return 0

def get_fixed_eval_range(game, records, size=50):
    """
    固定评测集：锁定一批期数作为长期不变的"考题"。

    ── 为什么需要 ──
    原来的回测窗口是 len(records)-30，每天新增1期就整体后移1期，
    于是"模型权重"和"考题"两个变量同时在变——
    今天净收益-1.72、明天-1.38，你根本分不清是模型进步了，
    还是今天这30期恰好比昨天好猜。这样就永远无法验证微调有没有用。

    固定评测集把考题锁死：首次运行时记录一个区间，之后一直用它。
    这样分数变化只可能来自模型本身，跨天可比。
    （注意：随着训练数据增长，这批期数会逐渐变成"训练集内的数据"，
     所以它衡量的是"拟合能力是否提升"，不能当作纯样本外泛化指标——
     真正的样本外表现仍看每日滚动回测。两者结合看才完整。）
    """
    path_mount = f'{RL_MOUNTED}/{game}_eval_range.json'
    path_local = f'{RL_LOCAL_DIR}/{game}_eval_range.json'
    rng = None
    for p in (path_mount, path_local):
        if os.path.exists(p):
            try:
                with open(p) as f: rng = json.load(f)
                break
            except Exception: pass
    if not rng or 'start' not in rng or 'end' not in rng:
        end = len(records) - 1
        start = max(SEQ_LEN + 30, end - size)
        rng = {'start': start, 'end': end, 'locked_at': str(date.today()), 'locked_n': len(records)}
        print(f"  [固定评测集] 首次锁定: 第{start}~{end}期（共{end-start}期），"
              f"之后每天用同一批考题评测，分数变化才可比")
    if rng['end'] > len(records) - 1:   # 数据被回滚等异常情况的保护
        return None
    os.makedirs(RL_LOCAL_DIR, exist_ok=True)
    try:
        with open(path_local, 'w') as f: json.dump(rng, f)
    except Exception: pass
    return rng


def append_eval_history(game, score_dict):
    """把每天在固定评测集上的成绩追加到历史曲线，便于观察长期是否进步"""
    path_mount = f'{RL_MOUNTED}/{game}_eval_history.json'
    path_local = f'{RL_LOCAL_DIR}/{game}_eval_history.json'
    hist = []
    for p in (path_mount, path_local):
        if os.path.exists(p):
            try:
                with open(p) as f: hist = json.load(f)
                break
            except Exception: pass
    if not isinstance(hist, list): hist = []
    hist.append({'date': str(date.today()), **score_dict})
    hist = hist[-200:]   # 只留最近200条，避免文件无限膨胀
    os.makedirs(RL_LOCAL_DIR, exist_ok=True)
    try:
        with open(path_local, 'w') as f: json.dump(hist, f, ensure_ascii=False)
    except Exception: pass
    # 打印近期趋势，让"有没有进步"一眼可见
    if len(hist) >= 2:
        key = [k for k in score_dict if k != 'date']
        if key:
            k = key[0]
            vals = [h.get(k) for h in hist[-10:] if h.get(k) is not None]
            if len(vals) >= 2:
                trend = "↑ 上升" if vals[-1] > vals[0] else ("↓ 下降" if vals[-1] < vals[0] else "→ 持平")
                print(f"  [固定评测集·历史] 近{len(vals)}次 {k}: {vals[0]} → {vals[-1]}  {trend}")
    return hist


def save_last_trained_n(game, n):
    """记录本次训练用到的数据期数，随RL_LOCAL_DIR一起推送到Dataset持久化"""
    os.makedirs(RL_LOCAL_DIR, exist_ok=True)
    try:
        with open(f'{RL_LOCAL_DIR}/{game}_last_trained_n.json', 'w') as f:
            json.dump({'n': n}, f)
    except Exception as e:
        print(f"  ! 记录{game}训练期数失败: {e}")

def carry_over_result(game_key, display_name, prev_result, cur_n, last_n, reason):
    """
    无新数据时，沿用上次的完整结果（字段结构保持一致，前端渲染无需感知变化）。

    注意 game_key 与 display_name 必须分开：
    game_key 用于拼文件名（'3d'/'ssq'/'kl8'），display_name 只用于日志展示（中文）。
    之前两者混用，把中文名传进 save_last_trained_n，生成了带中文的文件名，
    挂载后变成乱码（日志里看到的 '_last_trained_n.json'），
    导致双色球的"已训练期数"永远读不回来，开奖日检测形同虚设。
    """
    print(f"  {display_name} 无新开奖数据（当前{cur_n}期，上次训练时已是{last_n}期），"
          f"跳过训练避免重复数据过拟合")
    save_last_trained_n(game_key, cur_n)
    if prev_result:
        carried = dict(prev_result)
        carried['skipped'] = True
        carried['carried_over'] = True
        carried['note'] = (carried.get('note','') or '') + f'（{reason}，以上为上次训练结果，本次未重新训练）'
        return carried
    return {'skipped': True, 'games_tested': 0, 'reason': reason,
            'note': f'{reason}，且未找到上次训练结果可沿用（可能是首次运行）'}

# 各段的归一化基准（首次计算后缓存，之后固定不变）
_SEG_SCALE_CACHE = {}
_SCALE_FILE = 'seg_scales.json'


def load_seg_scales():
    """
    从Dataset读取归一化基准。基准必须跨天保持一致——
    如果每天重新计算，就退化回原来那个"输入天天漂移"的问题，
    首训学到的权重第二天照样对不上。
    """
    global _SEG_SCALE_CACHE
    for p in (f'{RL_MOUNTED}/{_SCALE_FILE}', f'{RL_LOCAL_DIR}/{_SCALE_FILE}'):
        if os.path.exists(p):
            try:
                with open(p) as f:
                    raw = json.load(f)
                _SEG_SCALE_CACHE = {k: {int(i): float(v) for i, v in d.items()} for k, d in raw.items()}
                print(f"  ✓ 已加载归一化基准（{len(_SEG_SCALE_CACHE)}个游戏），保证与首训时的输入口径一致")
                return
            except Exception as e:
                print(f"  ! 读取归一化基准失败: {e}")
    print("  ! 未找到归一化基准，本次将新建（首次运行属正常）")


def save_seg_scales():
    os.makedirs(RL_LOCAL_DIR, exist_ok=True)
    try:
        with open(f'{RL_LOCAL_DIR}/{_SCALE_FILE}', 'w') as f:
            json.dump({k: {str(i): v for i, v in d.items()} for k, d in _SEG_SCALE_CACHE.items()}, f)
    except Exception as e:
        print(f"  ! 保存归一化基准失败: {e}")


def normalize_state_segments(*segments, scale_key=None):
    """
    分段归一化，每段除以一个【固定基准】而不是"当前这一段自己的最大值"。

    ── 为什么必须改 ──
    原来每次调用都拿 seg.max() 当分母。新开一期后，某个特征成了新的最大值，
    整段的缩放基准就跟着变——结果是【本来没变的特征，归一化后也变了】。
    实测：一段5个值只有最后一个从812变成850，前4个值归一化后偏移了0.044，
    模型每天看到的输入一直在漂移，首训学到的东西第二天就对不上，
    表现为"同一份权重，昨天有偏好、今天变均匀"。

    现在改成：首次调用时按该段的实际量级确定一个基准并缓存，之后固定使用。
    同样的底层数据永远映射到同样的向量，模型学到的规律才能延用。
    """
    key = scale_key or 'default'
    cache = _SEG_SCALE_CACHE.setdefault(key, {})
    normed = []
    for i, seg in enumerate(segments):
        seg = np.asarray(seg, dtype=np.float32)
        if seg.size == 0:
            normed.append(seg); continue
        if i not in cache:
            m = float(np.abs(seg).max())
            # 用当次最大值的1.5倍作为固定基准，留出后续波动空间，避免频繁越界被截断
            cache[i] = max(m * 1.5, 1e-6)
        normed.append(seg / cache[i])
    state = np.concatenate(normed).astype(np.float32)
    return np.clip(state, -5, 5)


# ══════════════════════════════════════════════════════
#  集成环境
# ══════════════════════════════════════════════════════
KL8_TRAIN_N = 6    # 训练时用"选六"作为奖励标准，推荐时对同一个排序取不同TopN即可覆盖所有玩法

class IntegratedKL8Env(gym.Env):
    """
    快乐8环境 v4：全号码打分排序 + 双重逐球差异化信号（遗漏+近期频率）
    动作空间设计对比：
    - MultiBinary(80)：2^80种组合，训练几十万步也探索不到万分之一，学不出东西
    - 候选池Box(30)：只对预筛的30个候选打分，覆盖面受限，真正该选的号码若不在候选池里则永远选不到
    - 本版 Box(80)：对全部80个球直接打连续分数，取Top6，
      既保留全覆盖（跟MultiBinary一样能选中任意号码），
      又是标准的排序学习问题（跟候选池一样好训练，PPO能有效利用梯度）

    ⚠️ 关键修复：网络要输出80个球各自的分数，但状态向量里绝大部分内容
    （走势聚合特征/ML分组概率/LSTM/TFM隐层）对80个球来说完全相同，不携带"选哪个球"的区分度，
    之前只有遗漏值这一个80维信号能区分个体球，信号太单薄，网络最后一层容易在训练中
    自己走出一个跟真实状态无关的固定偏好（表现为持续偏向某个号码区间）。
    这版加入"近期出现频率"作为第二个逐球信号，并对这两个逐球信号整体加权(×2)，
    确保网络有足够强的信号去真正学习"选哪个球"，而不是被迫依赖训练偶然性。
    """
    metadata={'render_modes':[]}
    PERBALL_WEIGHT = 2.0   # 逐球信号（遗漏+频率）额外加权，突出其重要性

    def __init__(self, records, feat_fn, ml_vec, lstm_hidden, lstm_idx2row,
                 tfm_hidden, tfm_idx2row, omit_arr, freq_arr, train_n=KL8_TRAIN_N,
                 mk_arr=None, by_arr=None):
        super().__init__()
        self.records=records; self.feat_fn=feat_fn; self.ml_vec=ml_vec
        # 训练上界：留出最后 HOLDOUT_N 期给回测，训练时绝不触碰
        self.train_end = max(SEQ_LEN + 40, len(records) - holdout_size(len(records)))
        self.lstm_hidden=lstm_hidden; self.lstm_idx2row=lstm_idx2row
        self.tfm_hidden=tfm_hidden;   self.tfm_idx2row=tfm_idx2row
        self.omit_arr=omit_arr; self.freq_arr=freq_arr
        self.mk_arr=mk_arr; self.by_arr=by_arr
        self.train_n=train_n
        self.start=SEQ_LEN+30; self.idx=self.start
        sample=feat_fn(records,self.start); feat_dim=len(sample)
        lstm_dim = lstm_hidden.shape[1] if lstm_hidden is not None else 0
        tfm_dim  = tfm_hidden.shape[1]  if tfm_hidden  is not None else 0
        omit_dim = 80; freq_dim = 80
        mk_dim = mk_arr.shape[1] if mk_arr is not None else 0
        by_dim = by_arr.shape[1] if by_arr is not None else 0
        self.state_dim = feat_dim+len(ml_vec)+lstm_dim+tfm_dim+omit_dim+freq_dim+mk_dim+by_dim
        self.observation_space = spaces.Box(low=-5.,high=5.,shape=(self.state_dim,),dtype=np.float32)
        self.action_space = spaces.Box(low=-1.,high=1.,shape=(80,),dtype=np.float32)

    def _state(self):
        feat=self.feat_fn(self.records,self.idx)
        if feat is None: return np.zeros(self.state_dim,dtype=np.float32)
        raw=np.array(list(feat.values()),dtype=np.float32)
        if self.lstm_hidden is not None and self.idx in self.lstm_idx2row:
            lh = self.lstm_hidden[self.lstm_idx2row[self.idx]]
        else:
            lh = np.zeros(self.lstm_hidden.shape[1] if self.lstm_hidden is not None else 0, dtype=np.float32)
        if self.tfm_hidden is not None and self.idx in self.tfm_idx2row:
            th = self.tfm_hidden[self.tfm_idx2row[self.idx]]
        else:
            th = np.zeros(self.tfm_hidden.shape[1] if self.tfm_hidden is not None else 0, dtype=np.float32)
        om = self.omit_arr[self.idx] if self.omit_arr is not None else np.zeros(80,dtype=np.float32)
        fr = self.freq_arr[self.idx] if self.freq_arr is not None else np.zeros(80,dtype=np.float32)

        mk = self.mk_arr[self.idx] if self.mk_arr is not None else np.zeros(0,dtype=np.float32)
        by = self.by_arr[self.idx] if self.by_arr is not None else np.zeros(0,dtype=np.float32)
        state = normalize_state_segments(raw,self.ml_vec,lh,th,om,fr,mk,by, scale_key='kl8')
        _sg = self._segs()
        state = apply_segment_switches(state, _sg, 'kl8')
        # 逐球信号（遗漏+频率）加权。用精确段边界定位，
        # 不能再写 state[-160:] —— 马尔可夫/贝叶斯加进来后末尾160维已不是遗漏+频率了。
        state = state.copy()
        state[_sg[4][1]:_sg[5][2]] *= self.PERBALL_WEIGHT
        return np.clip(state, -5, 5)

    def _segs(self):
        lh = self.lstm_hidden.shape[1] if self.lstm_hidden is not None else 0
        th = self.tfm_hidden.shape[1] if self.tfm_hidden is not None else 0
        mk = self.mk_arr.shape[1] if self.mk_arr is not None else 0
        by = self.by_arr.shape[1] if self.by_arr is not None else 0
        base = self.state_dim-len(self.ml_vec)-lh-th-160-mk-by
        dims = [('走势特征', base), ('ML+DL概率', len(self.ml_vec)),
                ('LSTM隐层', lh), ('TFM隐层', th), ('遗漏', 80), ('频率', 80),
                ('马尔可夫', mk), ('贝叶斯', by)]
        out, p = [], 0
        for n, d in dims: out.append((n, p, p+d)); p += d
        return out

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.idx=self.start
        return self._state(), {}

    def step(self, action):
        # 对全部80个球的打分，取分数最高的 train_n 个作为本期选号
        top_idx = np.argsort(action)[-self.train_n:]
        selected = sorted([int(i)+1 for i in top_idx])

        actual=set(self.records[self.idx]['numbers'])
        hit=len(actual&set(selected)); n_sel=len(selected)
        net=calc_payout(n_sel,hit)
        # 快乐8真实赔率表在命中0-2个球时统统赔0（选6时命中期望仅1.5个，绝大多数样本落在这个区间），
        # 导致奖励大范围完全一样、没有梯度信号，PPO学不到跟状态相关的规律，只会随机收敛到某个固定偏向。
        # 加一个连续塑形项：命中率越高奖励越好，覆盖赔率表的"平坦区"，让policy始终有梯度可学。
        shaping = (hit / n_sel) * 0.3
        reward = net/(TICKET_PRICE*100) + shaping
        self.idx+=1
        terminated=(self.idx >= self.train_end)
        obs=self._state() if not terminated else np.zeros(self.state_dim,dtype=np.float32)
        return obs, reward, terminated, False, {'hit':hit,'n_sel':n_sel,'net':net}


SSQ_RED_PICK_N = 6    # 双色球固定选6个红球

class IntegratedSSQEnv(gym.Env):
    """
    双色球环境 v3：红球全号码打分排序（33个全打分，不再预筛候选池）+ 蓝球打分
    动作向量 = [33个红球分数, 16个蓝球分数]，共49维
    - 红球：33个球全部打分，取Top6
    - 蓝球：16个分数argmax
    理由同快乐8：候选池预筛会限制覆盖面，全量打分既保留完整覆盖又保持可学习的连续参数化。
    奖励按双色球真实奖级结构分级，同时激励红球和蓝球命中。
    """
    metadata={'render_modes':[]}
    def __init__(self, records, feat_fn, ml_vec, lstm_hidden, lstm_idx2row,
                 tfm_hidden, tfm_idx2row, omit_arr, red_pick_n=SSQ_RED_PICK_N,
                 mk_arr=None, by_arr=None):
        super().__init__()
        self.records=records; self.feat_fn=feat_fn; self.ml_vec=ml_vec
        # 训练上界：留出最后 HOLDOUT_N 期给回测，训练时绝不触碰
        self.train_end = max(SEQ_LEN + 40, len(records) - holdout_size(len(records)))
        self.lstm_hidden=lstm_hidden; self.lstm_idx2row=lstm_idx2row
        self.tfm_hidden=tfm_hidden;   self.tfm_idx2row=tfm_idx2row
        self.omit_arr=omit_arr; self.mk_arr=mk_arr; self.by_arr=by_arr
        self.red_pick_n=red_pick_n
        self.start=SEQ_LEN+30; self.idx=self.start
        sample=feat_fn(records,self.start); feat_dim=len(sample)
        lstm_dim = lstm_hidden.shape[1] if lstm_hidden is not None else 0
        tfm_dim  = tfm_hidden.shape[1]  if tfm_hidden  is not None else 0
        omit_dim = 49   # 33红球+16蓝球遗漏值，已含每个号码的差异化信息
        mk_dim = mk_arr.shape[1] if mk_arr is not None else 0
        by_dim = by_arr.shape[1] if by_arr is not None else 0
        self.state_dim = feat_dim+len(ml_vec)+lstm_dim+tfm_dim+omit_dim+mk_dim+by_dim
        self.observation_space = spaces.Box(low=-5.,high=5.,shape=(self.state_dim,),dtype=np.float32)
        self.action_space = spaces.Box(low=-1.,high=1.,shape=(33+16,),dtype=np.float32)

    def _state(self):
        feat=self.feat_fn(self.records,self.idx)
        if feat is None: return np.zeros(self.state_dim,dtype=np.float32)
        raw=np.array(list(feat.values()),dtype=np.float32)
        if self.lstm_hidden is not None and self.idx in self.lstm_idx2row:
            lh = self.lstm_hidden[self.lstm_idx2row[self.idx]]
        else:
            lh = np.zeros(self.lstm_hidden.shape[1] if self.lstm_hidden is not None else 0, dtype=np.float32)
        if self.tfm_hidden is not None and self.idx in self.tfm_idx2row:
            th = self.tfm_hidden[self.tfm_idx2row[self.idx]]
        else:
            th = np.zeros(self.tfm_hidden.shape[1] if self.tfm_hidden is not None else 0, dtype=np.float32)
        om = self.omit_arr[self.idx] if self.omit_arr is not None else np.zeros(49,dtype=np.float32)

        mk = self.mk_arr[self.idx] if self.mk_arr is not None else np.zeros(0,dtype=np.float32)
        by = self.by_arr[self.idx] if self.by_arr is not None else np.zeros(0,dtype=np.float32)
        st = normalize_state_segments(raw,self.ml_vec,lh,th,om,mk,by, scale_key='ssq')
        return apply_segment_switches(st, self._segs(), 'ssq')

    def _segs(self):
        lh = self.lstm_hidden.shape[1] if self.lstm_hidden is not None else 0
        th = self.tfm_hidden.shape[1] if self.tfm_hidden is not None else 0
        mk = self.mk_arr.shape[1] if self.mk_arr is not None else 0
        by = self.by_arr.shape[1] if self.by_arr is not None else 0
        base = self.state_dim-len(self.ml_vec)-lh-th-49-mk-by
        dims = [('走势特征', base), ('ML+DL概率', len(self.ml_vec)),
                ('LSTM隐层', lh), ('TFM隐层', th), ('遗漏', 49),
                ('马尔可夫', mk), ('贝叶斯', by)]
        out, p = [], 0
        for n, d in dims: out.append((n, p, p+d)); p += d
        return out

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.idx=self.start
        return self._state(), {}

    @staticmethod
    def _tier_reward(red_hit, blue_hit):
        """按双色球真实奖级结构给分级奖励，激励红球和蓝球同时命中"""
        if red_hit==6 and blue_hit: return 50.0   # 一等奖
        if red_hit==6:               return 20.0   # 二等奖
        if red_hit==5 and blue_hit:  return 10.0   # 三等奖
        if red_hit==5 or (red_hit==4 and blue_hit): return 5.0   # 四等奖
        if red_hit==4 or (red_hit==3 and blue_hit): return 2.0   # 五等奖
        if blue_hit:                 return 1.0    # 六等奖（仅蓝球）
        return -1.0   # 未中奖（成本）

    def step(self, action):
        red_scores  = action[:33]
        blue_scores = action[33:]

        top_idx = np.argsort(red_scores)[-self.red_pick_n:]
        red_selected = sorted([int(i)+1 for i in top_idx])
        blue_pred = int(np.argmax(blue_scores)) + 1

        actual_red = set(self.records[self.idx]['red'])
        actual_blue = self.records[self.idx]['blue']
        red_hit = len(actual_red & set(red_selected))
        blue_hit = int(blue_pred == actual_blue)
        reward = self._tier_reward(red_hit, blue_hit)
        # 红球选6个从33个里选，命中期望约1.09个，0-2命中区间同样存在奖励梯度不足问题，加小塑形项
        reward += (red_hit / 6.0) * 0.5

        self.idx+=1
        terminated=(self.idx >= self.train_end)
        obs=self._state() if not terminated else np.zeros(self.state_dim,dtype=np.float32)
        return obs, reward, terminated, False, {'red_hit':red_hit,'blue_hit':blue_hit,
                                                  'red_selected':red_selected,'blue_pred':blue_pred}


class Integrated3DEnv(gym.Env):
    """
    福彩3D环境：动作空间 MultiDiscrete([10,10,10])（百十个位各选一个数字，共1000种组合）
    比快乐8的2^80小得多，PPO能够正常学习。
    奖励：按位命中数给分，三位全中给大奖励（对应"直选"），
    位置命中但顺序不对不加分（3D不看"组选"，只关心百十个精确对应）。
    """
    metadata={'render_modes':[]}
    def __init__(self, records, feat_fn, ml_vec, lstm_hidden, lstm_idx2row,
                 tfm_hidden, tfm_idx2row, omit_arr, mk_arr=None, by_arr=None):
        super().__init__()
        self.records=records; self.feat_fn=feat_fn; self.ml_vec=ml_vec
        # 训练上界：留出最后 HOLDOUT_N 期给回测，训练时绝不触碰
        self.train_end = max(SEQ_LEN + 40, len(records) - holdout_size(len(records)))
        self.lstm_hidden=lstm_hidden; self.lstm_idx2row=lstm_idx2row
        self.tfm_hidden=tfm_hidden;   self.tfm_idx2row=tfm_idx2row
        self.omit_arr=omit_arr; self.mk_arr=mk_arr; self.by_arr=by_arr
        self.start=SEQ_LEN+5; self.idx=self.start
        sample=feat_fn(records,self.start); feat_dim=len(sample)
        lstm_dim = lstm_hidden.shape[1] if lstm_hidden is not None else 0
        tfm_dim  = tfm_hidden.shape[1]  if tfm_hidden  is not None else 0
        omit_dim = 30
        mk_dim = mk_arr.shape[1] if mk_arr is not None else 0
        by_dim = by_arr.shape[1] if by_arr is not None else 0
        self.state_dim = feat_dim+len(ml_vec)+lstm_dim+tfm_dim+omit_dim+mk_dim+by_dim
        self.observation_space = spaces.Box(low=-5.,high=5.,shape=(self.state_dim,),dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([10,10,10])

    def _state(self):
        feat=self.feat_fn(self.records,self.idx)
        if feat is None: return np.zeros(self.state_dim,dtype=np.float32)
        raw=np.array(list(feat.values()),dtype=np.float32)
        if self.lstm_hidden is not None and self.idx in self.lstm_idx2row:
            lh = self.lstm_hidden[self.lstm_idx2row[self.idx]]
        else:
            lh = np.zeros(self.lstm_hidden.shape[1] if self.lstm_hidden is not None else 0, dtype=np.float32)
        if self.tfm_hidden is not None and self.idx in self.tfm_idx2row:
            th = self.tfm_hidden[self.tfm_idx2row[self.idx]]
        else:
            th = np.zeros(self.tfm_hidden.shape[1] if self.tfm_hidden is not None else 0, dtype=np.float32)
        om = self.omit_arr[self.idx] if self.omit_arr is not None else np.zeros(30,dtype=np.float32)
        mk = self.mk_arr[self.idx] if self.mk_arr is not None else np.zeros(0,dtype=np.float32)
        by = self.by_arr[self.idx] if self.by_arr is not None else np.zeros(0,dtype=np.float32)
        st = normalize_state_segments(raw,self.ml_vec,lh,th,om,mk,by, scale_key='3d')
        return apply_segment_switches(st, self._segs(), '3d')

    def _segs(self):
        """状态向量各段的起止下标，供开关和消融诊断共用"""
        lh = self.lstm_hidden.shape[1] if self.lstm_hidden is not None else 0
        th = self.tfm_hidden.shape[1] if self.tfm_hidden is not None else 0
        mk = self.mk_arr.shape[1] if self.mk_arr is not None else 0
        by = self.by_arr.shape[1] if self.by_arr is not None else 0
        dims = [('走势特征', self.state_dim-len(self.ml_vec)-lh-th-30-mk-by),
                ('ML+DL概率', len(self.ml_vec)), ('LSTM隐层', lh), ('TFM隐层', th),
                ('遗漏', 30), ('马尔可夫', mk), ('贝叶斯', by)]
        out, p = [], 0
        for n, d in dims: out.append((n, p, p+d)); p += d
        return out

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.idx=self.start
        return self._state(), {}

    def step(self, action):
        pred = [int(action[0]), int(action[1]), int(action[2])]
        actual = self.records[self.idx]['digits']
        matches = sum(1 for i in range(3) if pred[i]==actual[i])
        reward = matches * 1.0
        if matches == 3:
            reward += 20.0   # 三位全中（直选）额外大奖励
        self.idx+=1
        terminated=(self.idx >= self.train_end)
        obs=self._state() if not terminated else np.zeros(self.state_dim,dtype=np.float32)
        return obs, reward, terminated, False, {'pred':pred,'actual':actual,'matches':matches}

# ══════════════════════════════════════════════════════
#  加载/保存 PPO 模型
# ══════════════════════════════════════════════════════
def load_ppo(game):
    path = f'{RL_MOUNTED}/{game}_ppo.zip'
    if os.path.exists(path):
        try:
            model = PPO.load(path, device='cpu')
            print(f"  ✓ 加载已有PPO模型: {path}")
            return model
        except Exception as e:
            print(f"  ! 加载PPO失败: {e}，将重新训练")
        return None

    # ── 关键兼容：Kaggle 上传 Dataset 时会自动把 .zip 解压成同名目录 ──
    # SB3 存的是 3d_ppo.zip，挂载后变成 3d_ppo/ 目录（里面是 policy.pth 等文件），
    # 于是 os.path.exists('3d_ppo.zip') 永远为 False，模型明明在却读不到，
    # 每天都静默退回"首次训练"，微调机制形同虚设。
    # 这里把解压出来的目录重新打包成 zip 再交给 SB3 加载。
    dir_path = f'{RL_MOUNTED}/{game}_ppo'
    if os.path.isdir(dir_path):
        try:
            tmp_base = f'/kaggle/working/_restore_{game}_ppo'
            zip_path = shutil.make_archive(tmp_base, 'zip', dir_path)
            model = PPO.load(zip_path, device='cpu')
            print(f"  ✓ 加载已有PPO模型（从被Kaggle解压的目录 {dir_path} 重新打包恢复）")
            return model
        except Exception as e:
            print(f"  ! 从解压目录恢复PPO失败: {e}，将重新训练")
            return None

    print(f"  ! 未找到PPO模型文件: {path}")
    if os.path.isdir(RL_MOUNTED):
        try:
            entries = sorted(os.listdir(RL_MOUNTED))
            print(f"    挂载目录 {RL_MOUNTED} 实际内容({len(entries)}项): {entries[:20]}")
            for e in entries:
                sub = os.path.join(RL_MOUNTED, e)
                if os.path.isdir(sub):
                    print(f"    子目录 {e}/ 内容: {sorted(os.listdir(sub))[:20]}")
        except Exception as e:
            print(f"    读取挂载目录失败: {e}")
    else:
        print(f"    ⚠️ 挂载目录 {RL_MOUNTED} 不存在——"
              f"说明 kernel-metadata.json 的 dataset_sources 里没挂载 {RL_DATASET_SLUG}，"
              f"或该Dataset尚未创建")
    return None

def save_ppo(model, game):
    os.makedirs(RL_LOCAL_DIR, exist_ok=True)
    model.save(f'{RL_LOCAL_DIR}/{game}_ppo')
    print(f"  ✓ PPO模型已保存到本地: {RL_LOCAL_DIR}/{game}_ppo.zip")

def push_rl_dataset():
    """把RL_LOCAL_DIR整体推送到Kaggle Dataset"""
    try:
        meta = {"title":"Fucai RL Cache","id":RL_DATASET_ID,"licenses":[{"name":"CC0-1.0"}]}
        with open(f'{RL_LOCAL_DIR}/dataset-metadata.json','w') as f: json.dump(meta,f)
        # 打印本次要上传的文件清单，便于跟下次运行时挂载目录的内容对照排查
        try:
            files = sorted(os.listdir(RL_LOCAL_DIR))
            print(f"  本次上传文件({len(files)}项): {files[:20]}")
        except Exception: pass
        env = os.environ.copy(); env['KAGGLE_API_TOKEN']=KAGGLE_TOKEN
        # 注意：不要加 --dir-mode tar/zip。模型文件本来就直接放在 RL_LOCAL_DIR 下，
        # 加了归档参数会把内容打包，挂载后看到的是压缩包而非 xxx_ppo.zip 独立文件，
        # 导致 load_ppo 每次都找不到文件、静默退回"首次训练"，模型永远无法累积。
        for cmd in [
            ['kaggle','datasets','version','-p',RL_LOCAL_DIR,'-m',f'daily-{date.today()}'],
            ['kaggle','datasets','create','-p',RL_LOCAL_DIR],
        ]:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
            if r.returncode==0:
                print(f"  ✓ RL模型已推送到 {RL_DATASET_ID}"); return True
            print(f"  [{cmd[1]} {cmd[2]}] rc={r.returncode}  {r.stderr[:150]}")
        return False
    except Exception as e:
        print(f"  ! 推送异常: {e}"); return False

# ══════════════════════════════════════════════════════
#  主流程：kl8 增量微调
# ══════════════════════════════════════════════════════
def run_kl8_daily(records, ml_pred, prev_result=None, dl_pred=None):
    print(f"\n{'='*50}\n快乐8 PPO 每日增量微调（全号码打分排序，{len(records)}期）\n{'='*50}")

    # 新数据检测：快乐8虽然每天开奖，但手动重复触发时数据是完全相同的，
    # 反复训练会让模型对同一批数据过拟合，这里直接跳过
    last_trained_n = get_last_trained_n('kl8')
    if len(records) <= last_trained_n:
        return carry_over_result('kl8', '快乐8', prev_result, len(records), last_trained_n,
                                 '本次运行无新开奖数据（可能是当日已训练过或重复手动触发）')

    ml_vec = extract_ml_prob_vec(ml_pred, 'kl8')
    dl_vec = extract_dl_prob_vec(dl_pred, 'kl8')
    # 传统ML概率 + 深度学习概率 拼成统一的外部模型信号向量
    ml_vec = np.concatenate([ml_vec, dl_vec]).astype(np.float32)
    _cur_feat_dim = len(fkl8(records, len(records)-1) or {})
    lstm, tfm, meta = load_lstm_tfm('kl8', current_feat_dim=_cur_feat_dim)

    print("  批量预计算 LSTM/TFM 隐层状态…")
    t0 = time.time()
    lstm_hidden, lstm_idx2row = precompute_hidden_multi(records, fkl8, lstm)
    tfm_hidden,  tfm_idx2row  = precompute_hidden_multi(records, fkl8, tfm)
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [LSTM隐层] {'✓已加载' if lstm_hidden is not None else '✗未加载（回退为0向量）'}  [TFM隐层] {'✓已加载' if tfm_hidden is not None else '✗未加载（回退为0向量）'}")

    print("  批量预计算遗漏向量…")
    t0 = time.time()
    omit_arr = precompute_omission_kl8(records)
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [遗漏向量] ✓已加载（80维，覆盖全部号码）")

    print("  批量预计算近30期频率向量…")
    t0 = time.time()
    freq_arr = precompute_freq_kl8(records, window=30)
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [频率向量] ✓已加载（80维，第二个逐球差异化信号）")

    print("  批量预计算马尔可夫转移 + 贝叶斯后验…")
    t0 = time.time()
    mk_arr = precompute_markov_balls(records, 80, lambda r: r['numbers'])   # 80维
    by_arr = precompute_bayes(records, 80, lambda r: r['numbers'])          # 160维
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [马尔可夫] 80维  [贝叶斯均值+不确定性] 160维")

    print(f"  [状态向量组成] 原始统计特征{_cur_feat_dim}维 + 外部模型概率{len(ml_vec)}维(传统ML+深度学习) + LSTM隐层 + TFM隐层 + 遗漏80维 + 频率80维（逐球信号×2加权）")

    def make_env():
        return IntegratedKL8Env(records, fkl8, ml_vec,
                                lstm_hidden, lstm_idx2row, tfm_hidden, tfm_idx2row, omit_arr, freq_arr,
                                mk_arr=mk_arr, by_arr=by_arr)
    vec_env = make_vec_env(make_env, n_envs=4)

    model = load_ppo('kl8')
    _do = False   # 是否触发了定期重训（冻结模式下由 should_retrain 决定）
    is_new = model is None
    t0 = time.time()
    if not is_new:
        try:
            model.set_env(vec_env)
            # PPO保存时会把超参一起存进去，加载后必须显式覆盖，
            # 否则改了 ENT_COEF 对已有模型完全不生效，还以为调了参
            model.ent_coef = ENT_COEF['kl8']
            print(f"    熵系数已设为 {model.ent_coef}")
        except Exception as e:
            print(f"  ! 旧PPO模型与当前环境结构不兼容（{e}），改为全新训练")
            model = None; is_new = True
    if is_new:
        print("  首次训练（20万步，全80球连续打分排序，兼顾全覆盖与可学习性）…")
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=512, batch_size=128,
                    n_epochs=10, gamma=0.95, gae_lambda=0.95, clip_range=0.2, ent_coef=ENT_COEF['kl8'],
                    verbose=0, device='cpu')

    def build_state(idx):
        feat = fkl8(records, idx)
        if feat is None: return None
        raw = np.array(list(feat.values()),dtype=np.float32)
        lh = lstm_hidden[lstm_idx2row[idx]] if (lstm_hidden is not None and idx in lstm_idx2row) else np.zeros(lstm_hidden.shape[1] if lstm_hidden is not None else 0,dtype=np.float32)
        th = tfm_hidden[tfm_idx2row[idx]]   if (tfm_hidden  is not None and idx in tfm_idx2row)  else np.zeros(tfm_hidden.shape[1] if tfm_hidden is not None else 0,dtype=np.float32)
        om = omit_arr[idx]
        fr = freq_arr[idx]
        mk = mk_arr[idx]; by = by_arr[idx]
        state = normalize_state_segments(raw,ml_vec,lh,th,om,fr,mk,by, scale_key='kl8')
        state = apply_segment_switches(state, _segs_kl8, 'kl8')
        # 逐球信号（遗漏+频率）加权。必须用精确段边界定位：
        # 之前写 state[-160:]，在马尔可夫/贝叶斯加进来之后，末尾160维已经变成它们了，
        # 加权加错了对象（训练和推理都错、方向一致所以没报错，但含义已偏）。
        state = state.copy()
        _s0 = _segs_kl8[4][1]; _s1 = _segs_kl8[5][2]   # 遗漏起点 ~ 频率终点
        state[_s0:_s1] *= IntegratedKL8Env.PERBALL_WEIGHT
        return np.clip(state, -5, 5)

    # 分段边界（开关与消融诊断共用，必须与环境类 _segs() 一致）
    _lh_d = lstm_hidden.shape[1] if lstm_hidden is not None else 0
    _th_d = tfm_hidden.shape[1] if tfm_hidden is not None else 0
    _segs_kl8, _pp = [], 0
    for _n, _d in [('走势特征', _cur_feat_dim), ('ML+DL概率', len(ml_vec)),
                   ('LSTM隐层', _lh_d), ('TFM隐层', _th_d), ('遗漏', 80), ('频率', 80),
                   ('马尔可夫', mk_arr.shape[1]), ('贝叶斯', by_arr.shape[1])]:
        _segs_kl8.append((_n, _pp, _pp+_d)); _pp += _d
    _off = [n for n,_,_ in _segs_kl8 if not SEGMENT_ENABLE.get('kl8',{}).get(n, True)]
    if _off: print(f"  [分段开关] 快乐8 已关闭: {_off}（维度保留并清零，可随时切回，不触发重训）")

    _hold_start = max(SEQ_LEN+40, len(records)-holdout_size(len(records)))
    _hold_mid = (_hold_start + len(records)) // 2   # 前半选权重，后半只报分
    def _eval_holdout(mask=None, half='select', use_model=None):
        """在holdout上评分：选六标准的平均命中球数。mask=(s,e)时清零该区间用于消融诊断。
           half='select'用前一半(早停选权重)，half='report'用后一半(从不参与选择，评分干净)"""
        tot, n = 0, 0
        _rng = range(_hold_start, _hold_mid) if half=='select' else range(_hold_mid, len(records))
        for i in _rng:
            st = build_state(i)
            if st is None: continue
            if mask is not None:
                st = st.copy(); st[mask[0]:mask[1]] = 0.0
            _m = use_model if use_model is not None else model
            a,_ = _m.predict(st, deterministic=True)
            sel = set(int(x)+1 for x in np.argsort(a)[-6:])
            tot += len(set(records[i]['numbers']) & sel); n += 1
        return tot/n if n else 0.0

    if is_new:
        model, _best, _hist = train_with_early_stop(
            model, 200000, _eval_holdout, '快乐8首训', n_chunks=16, patience=6,
            reset_timesteps=True, warmup_chunks=5)
    else:
        if TRAIN_MODE == 'frozen':
            # 冻结模式：平时不训练（输出稳定、无过拟合），
            # 但攒够足够新数据后触发一次全量重训——这才是真正的"进化"，
            # 而不是每天拿万分之一的新数据空转。
            _do, _why = should_challenge('kl8', len(records))
            print(f"  [挑战周期] {_why}")
            if _do:
                _n_clean = max(len(records) - _hold_mid, 1)
                _se_clean = math.sqrt(6*0.25*0.75) / math.sqrt(_n_clean)
                def _mk():
                    m = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=256, batch_size=64,
                            n_epochs=8, gamma=0.9, gae_lambda=0.9, clip_range=0.2,
                            ent_coef=ENT_COEF['kl8'], verbose=0, device='cpu')
                    return m
                def _tr(m):
                    return train_with_early_stop(m, 200000, _eval_holdout, '快乐8挑战者',
                                                 n_chunks=16, patience=6,
                                                 reset_timesteps=True, warmup_chunks=5)
                def _ev(m):
                    return _eval_holdout(half='report', use_model=m)
                def _rc(m):
                    # 复检用另一半数据：对现任和挑战者是同一批题，公平比较
                    return _eval_holdout(half='select', use_model=m)
                model, _swapped, _msg = run_challenge(
                    model, _mk, _tr, _ev, _se_clean, recheck_fn=_rc, label='快乐8')
                print(f"  [挑战结果] {_msg}")
                _do = _swapped   # 只有换人了才需要保存
                _best = _eval_holdout(); _hist = [round(_best,4)]
            else:
                _best = _eval_holdout(); _hist = [round(_best,4)]
                print(f"  [现任模型] 沿用已有权重出预测，holdout评分 {_best:.4f}")
        else:
            print("  增量微调（2万步，带早停）…")
            model, _best, _hist = train_with_early_stop(
                model, 20000, _eval_holdout, '快乐8微调', n_chunks=8, patience=4,
                reset_timesteps=False, warmup_chunks=2)
    print(f"    PPO训练完成，耗时 {time.time()-t0:.1f}s")
    print(f"    [训练/回测隔离] 训练止于第{_hold_start}期，回测第{_hold_start}~{len(records)-1}期（样本外）")
    _st_probe = build_state(len(records))
    if _st_probe is not None: report_entropy(model, _st_probe, '快乐8')

    # 策略依赖性：模型到底有没有在用状态（比任何评分都更根本）
    try:
        _pts = [int(x) for x in np.linspace(SEQ_LEN+60, len(records)-1, 24).astype(int)]
        _dep = policy_dependence_test(model, build_state, _pts, 'kl8')
    except Exception as _e:
        _dep = None; print(f"    [策略依赖性] 检测异常: {_e}")

    # 干净评分：用从未参与早停选择的那半holdout评分，不会被筛选污染
    try:
        _clean = _eval_holdout(half='report')
        print(f"    [干净评分] 选权重用第{_hold_start}~{_hold_mid-1}期，"
              f"未参与选择的第{_hold_mid}~{len(records)-1}期得分 {_clean:.4f}")
    except Exception as _e:
        _clean = float('nan'); print(f"    [干净评分] 计算失败: {_e}")

    _lh_dim = lstm_hidden.shape[1] if lstm_hidden is not None else 0
    _th_dim = tfm_hidden.shape[1] if tfm_hidden is not None else 0
    _p = 0; _segs = []
    for _nm, _d in [('走势特征', _cur_feat_dim), ('ML+DL概率', len(ml_vec)),
                    ('LSTM隐层', _lh_dim), ('TFM隐层', _th_dim), ('遗漏', 80), ('频率', 80),
                    ('马尔可夫', 80), ('贝叶斯', 160)]:
        _segs.append((_nm, _p, _p+_d)); _p += _d
    # 冻结模式下权重不变，消融结论不会变，跳过以节省十几次评估的时间
    _ablation = []
    if TRAIN_MODE != 'frozen':
        _ablation = segment_ablation(_eval_holdout, _segs, _best, '快乐8', 'kl8',
                                     se=math.sqrt(6*0.25*0.75)/math.sqrt(max(len(records)-_hold_start,1)))
    append_history('kl8', {'date': str(date.today()), 'holdout_score': round(_best,4),
                          'clean_score': round(_clean,4),
                          'policy_dependence': _dep,
                          'ent_coef': ENT_COEF['kl8'],
                           'state_dim': _p, 'score_history': _hist, 'ablation': _ablation})

    # 只有真正训练过才保存：冻结且未触发重训时权重没变，
    # 重复推送没意义，还可能意外覆坏已有权重
    _trained = (TRAIN_MODE != 'frozen') or is_new or _do
    if _trained:
        save_ppo(model, 'kl8')
        save_last_trained_n('kl8', len(records))   # 记录本次训练时的期数，供下次判断
    else:
        print("  [冻结] 权重未改动，跳过保存")

    # 回测：同一次预测，同时评估选四/五/六/九/十全部玩法（几乎零额外开销，只是截取不同长度TopN）
    start = max(SEQ_LEN+30, len(records)-30)
    play_sizes = [4,5,6,9,10]
    net_by_size = {n: 0.0 for n in play_sizes}
    hit_by_size = {n: 0.0 for n in play_sizes}
    games=0
    # 上界用 len(records)：idx 最大取到 len(records)-1，即拿"倒数第二期及之前"的特征
    # 去预测最后一期。原来写 len(records)-1 会让最后一期永远不参与回测，白白少一个样本。
    for idx in range(start, len(records)):
        state = build_state(idx)
        if state is None: continue
        action,_ = model.predict(state, deterministic=True)
        order = np.argsort(action)[::-1]
        ranked_all = [int(i)+1 for i in order]   # 一次打分，全部玩法复用同一个排序结果
        actual=set(records[idx]['numbers'])
        for n in play_sizes:
            sel = set(ranked_all[:n])
            hit = len(actual & sel)
            net_by_size[n] += calc_payout(n, hit)
            hit_by_size[n] += hit
        games+=1

    backtest_by_play = {}
    for n in play_sizes:
        avg_net = round(net_by_size[n]/games,2) if games else 0
        avg_hit = round(hit_by_size[n]/games,2) if games else 0
        backtest_by_play[n] = {'avg_net_per_game':avg_net,'avg_hit':avg_hit}

    # ── 固定评测集：考题锁死，只有模型在变，跨天成绩才可比 ──
    fixed_eval = None
    _rng = get_fixed_eval_range('kl8', records)
    if _rng:
        _net = 0.0; _hit = 0; _ft = 0
        for idx in range(_rng['start'], _rng['end']+1):
            st = build_state(idx)
            if st is None: continue
            act,_ = model.predict(st, deterministic=True)
            order = np.argsort(act)[::-1]
            sel = set(int(i)+1 for i in order[:KL8_TRAIN_N])
            h = len(set(records[idx]['numbers']) & sel)
            _net += calc_payout(KL8_TRAIN_N, h); _hit += h; _ft += 1
        if _ft:
            f_net = round(_net/_ft, 2); f_hit = round(_hit/_ft, 3)
            print(f"  [固定评测集] 第{_rng['start']}~{_rng['end']}期({_ft}期，选六标准): "
                  f"平均净收益{f_net}元/期  平均命中{f_hit}个")
            append_eval_history('kl8', {'avg_net': f_net, 'avg_hit': f_hit, 'n': _ft})
            fixed_eval = {'range': [_rng['start'], _rng['end']], 'n': _ft,
                          'avg_net': f_net, 'avg_hit': f_hit}

    # 找出净收益回测表现最好的玩法（仅供参考，彩票本质随机，历史回测不代表未来）
    best_play_n = max(play_sizes, key=lambda n: backtest_by_play[n]['avg_net_per_game'])
    avg_net = backtest_by_play[6]['avg_net_per_game']   # 兼容旧字段：保留选六作为默认展示值
    print(f"  回测（近{games}期，全玩法对比）：" + "  ".join(
        f"选{['','','','','四','五','六','','','九','十'][n]}净收益{backtest_by_play[n]['avg_net_per_game']}元/期" for n in play_sizes))
    print(f"  回测表现最好的玩法：选{['','','','','四','五','六','','','九','十'][best_play_n]}")

    # 今日推荐：以RL自己的判断为主——它的状态输入已经融合了ML概率/LSTM/TFM隐层/遗漏/频率/走势特征，
    # 训练过程中神经网络自己学会了怎么综合这些信息，不再用人工权重公式二次加工跟它的判断"打架"。
    # 多组推荐用同一份RL排序做滑动窗口切分（保持100%由RL主导，不引入外部信号重新排序）；
    # 遗漏/频率/ML预测只作为"参考信息"附加展示，帮助理解RL为什么这么选，不参与决策计算。
    # ⚠️ 这里必须用 len(records) 而不是 len(records)-1。
    # 训练时的约定是"特征取 records[:idx]、答案取 records[idx]"，
    # 所以 idx=len(records)-1 输出的是对【最后一期】的预测——而最后一期早就开出来了，
    # 等于让模型复述已知答案（实测表现为推荐号码与最新开奖高度重合）。
    # idx=len(records) 才是"用全部已知数据预测下一期（尚未开奖）"。
    idx = len(records)
    state = build_state(idx)
    rl_order = []
    ref_info = {}   # 参考信息：遗漏/频率/ML预测，仅用于展示说明，不影响排序
    if state is not None:
        base_action,_ = model.predict(state, deterministic=True)
        rl_order = [int(i)+1 for i in np.argsort(base_action)[::-1]]

        # 区分度诊断：模型对80个球的打分，Top6跟中位区拉不拉得开？
        # 如果差距接近0，说明模型其实没在区分号码好坏，选Top6跟随便选6个没实质区别，
        # 这比"回测净收益"更能直接反映模型到底学到没有。
        _srt = np.sort(base_action)[::-1]
        _top6, _mid = float(_srt[:6].mean()), float(_srt[34:40].mean())
        _spread = float(_srt.max() - _srt.min())
        _gap_ratio = (_top6 - _mid) / (_spread + 1e-9)
        print(f"  [区分度] Top6均分{_top6:.4f}  中位区均分{_mid:.4f}  "
              f"差距占全域{_gap_ratio*100:.1f}%")
        if _gap_ratio < 0.15:
            print(f"    ⚠️ 差距很小，说明模型对各号码的偏好不明显，本次推荐参考价值有限")
        else:
            print(f"    ✓ 模型对号码有明显区分")

        om_now = omit_arr[idx] if omit_arr is not None else np.zeros(80)
        fr_now = freq_arr[idx] if freq_arr is not None else np.zeros(80)
        models_data = ml_pred.get('models', {})
        zone_probs_raw = models_data.get('zone_dom', {}).get('prediction', {}).get('probs', {})
        five_probs_raw = models_data.get('five_dom', {}).get('prediction', {}).get('probs', {})
        zone_pred = models_data.get('zone_dom', {}).get('prediction', {}).get('value')
        five_pred = models_data.get('five_dom', {}).get('prediction', {}).get('value')
        zone_names = ['1-20区','21-40区','41-60区','61-80区']
        five_names = ['1-16','17-32','33-48','49-64','65-80']

        top6 = rl_order[:6]
        avg_omission = round(float(np.mean([om_now[b-1] for b in top6])), 2)
        avg_freq = round(float(np.mean([fr_now[b-1] for b in top6])), 2)
        ref_info = {
            'avg_omission_top6': avg_omission,
            'avg_freq_top6': avg_freq,
            'ml_zone_pred': zone_names[zone_pred] if zone_pred is not None and 0<=zone_pred<4 else None,
            'ml_five_pred': five_names[five_pred] if five_pred is not None and 0<=five_pred<5 else None,
        }
        print(f"  [主推荐] RL确定性排序Top6: {sorted(top6)}")
        print(f"  [参考信息] 该注平均遗漏{avg_omission}期，平均近期频率{avg_freq}次；"
              f"ML预测主力区间={ref_info['ml_zone_pred']}，主力五行段={ref_info['ml_five_pred']}")

        # ── 诊断：检验打分是否跟球号系统性绑定（即"是否还存在偏向大号/小号"的机制性bug）──
        ball_idx = np.arange(1, 81)
        corr = float(np.corrcoef(ball_idx, base_action)[0, 1])
        print(f"  [诊断1] RL打分与球号(1-80)的相关系数: {corr:.3f}")
        if abs(corr) > 0.3:
            direction = '偏向大号' if corr > 0 else '偏向小号'
            print(f"  ⚠️ [诊断1警告] 相关系数绝对值>0.3，RL打分可能仍跟球号系统性绑定（{direction}），建议人工复查")
        else:
            print(f"  ✓ [诊断1通过] RL打分与球号无明显系统性相关")

        # ── 诊断2（关键）：对比多个不同历史时间点的推荐结果 ──
        if games >= 4:
            test_points = sorted(set([
                max(SEQ_LEN+30, len(records)-200),
                max(SEQ_LEN+30, len(records)-100),
                max(SEQ_LEN+30, len(records)-50),
                len(records)-1,
            ]))
            snapshot_top6 = {}
            for tp in test_points:
                st = build_state(tp)
                if st is None: continue
                act,_ = model.predict(st, deterministic=True)
                t6 = set(int(i)+1 for i in np.argsort(act)[-6:])
                snapshot_top6[tp] = t6
            print(f"  [诊断2] 不同历史时期(共{len(snapshot_top6)}个采样点)的Top6对比：")
            for tp, t6 in snapshot_top6.items():
                print(f"    第{tp}期状态 → {sorted(t6)}")
            if len(snapshot_top6) >= 2:
                all_sets = list(snapshot_top6.values())
                pairwise_overlaps = []
                for i in range(len(all_sets)):
                    for j in range(i+1, len(all_sets)):
                        pairwise_overlaps.append(len(all_sets[i] & all_sets[j]))
                avg_overlap = sum(pairwise_overlaps)/len(pairwise_overlaps)
                print(f"  [诊断2] 不同时期推荐重合数: {avg_overlap:.1f}/6")
                if avg_overlap >= 4:
                    print(f"  ⚠️⚠️ [诊断2警告] 不同状态推荐重合度高(≥4/6)，模型区分度不足，建议人工复查训练情况")
                elif avg_overlap <= 1:
                    print(f"  ✓ [诊断2通过] 不同状态推荐差异明显，模型确实在响应状态变化")
                else:
                    print(f"  ⚠️ [诊断2中性] 重合度中等")

    # 各玩法预先用"胆码+拖码轮转"算好各注（原因同双色球：
    # 按联合得分取Top-N会让各注共享同样的高分球、只在末位微调，体现不出多元化）
    # ── 提取ML的7条预测作为选号条件（这些本来就已注入RL状态，
    #    但之前只在训练时被"看到"，选号环节完全没检验，这里补上）──
    def _kl8_cond_feats(c):
        cs = sorted(c)
        odd = sum(1 for x in c if x % 2 != 0)
        big = sum(1 for x in c if x > 40)
        zn = [sum(1 for x in c if lo <= x <= hi) for lo, hi in [(1,20),(21,40),(41,60),(61,80)]]
        fv = [sum(1 for x in c if lo <= x <= hi) for lo, hi in [(1,16),(17,32),(33,48),(49,64),(65,80)]]
        tt = sum(c)
        cg, inc = 0, False
        for i in range(len(cs)-1):
            if cs[i+1]-cs[i] == 1:
                if not inc: cg += 1; inc = True
            else: inc = False
        rng = cs[-1] - cs[0]
        # 分档阈值按每期20球定义，选4~10球时需按占比折算回20球口径，否则永远匹配不上
        k = 20.0 / max(len(c), 1)
        o20, b20, t20 = odd*k, big*k, tt*k
        return {'odd_grp': 0 if o20 < 9 else (1 if o20 <= 11 else 2),
                'zone_dom': int(max(range(4), key=lambda i: zn[i])),
                'tot_grp': 0 if t20 < 640 else (1 if t20 < 820 else 2),
                'big_grp': 0 if b20 < 9 else (1 if b20 <= 11 else 2),
                'five_dom': int(max(range(5), key=lambda i: fv[i])),
                'consec_grp': 0 if cg == 0 else (1 if cg <= 2 else 2),
                'range_grp': 0 if rng < 60 else (1 if rng < 70 else 2)}

    _kl8_conds = {}
    _md = ml_pred.get('models', {})
    for _k in ['odd_grp','zone_dom','tot_grp','big_grp','five_dom','consec_grp','range_grp']:
        _p = (_md.get(_k, {}) or {}).get('prediction', {})
        if _p.get('value') is not None:
            _kl8_conds[_k] = (int(_p['value']), float(_p.get('confidence', 50))/100.0)

    _play_bets = {}
    _play_core = {}
    for _n, _cnt in [(4,3), (5,3), (6,3), (8,1), (9,2), (10,1)]:
        if rl_order:
            # 完全按RL自己的球分选号。ML的那些预测已经在状态向量里，
            # 训练时模型看得见、奖励信号会告诉它有没有用；
            # 如果在外面再套一层手写规则去筛，等于用"我认为对的"覆盖"模型学到的"，
            # 而且"符合条件更容易中"这个假设本身从未被验证过。
            _b, _c, _p = diverse_picks(base_action, _n, _cnt)
            _play_bets[_n], _play_core[_n] = _b, _c
        else:
            _play_bets[_n], _play_core[_n] = [[] for _ in range(_cnt)], []
    if rl_order:
        _o6 = [len(set(_play_bets[6][i]) & set(_play_bets[6][j]))
               for i in range(len(_play_bets[6])) for j in range(i+1, len(_play_bets[6]))]
        _a6 = set()
        for c in _play_bets[6]: _a6 |= set(c)
        print(f"  [选六] 胆码{_play_core[6]}  3注共用到{len(_a6)}个号码"
              + (f"，两两平均重合{np.mean(_o6):.1f}/6" if _o6 else ""))
        if _kl8_conds:
            _cavg = sum(len([1 for k,(v,w) in _kl8_conds.items()
                             if _kl8_cond_feats(b).get(k)==v]) for b in _play_bets[6]) / max(len(_play_bets[6]),1)
            print(f"  [诊断·仅参考] RL自选的选六3注，平均符合{_cavg:.1f}/{len(_kl8_conds)}条ML预测条件"
                  f"（不参与筛选，仅用于观察RL判断与ML预测的一致程度）")

    def group(n, rank=0):
        """取该玩法第 rank+1 注（已由 diverse_picks 保证各注之间有实质差异）"""
        bets = _play_bets.get(n, [])
        if rank < len(bets): return bets[rank]
        return sorted(rl_order[:n]) if rl_order else []

    # 与传统ML的分组结构完全一致：选四3组/选五3组/复式1组8球/选六3组/选九2组/选十1组
    # 每注由"胆码+拖码轮转"生成：模型最确信的球进每注，其余候选轮转分配，兼顾置信度与多样性
    plays = {
        'xuan4':    {'name':'选四','balls':4, 'tip':'胆码+拖码轮转3注',
                     'groups':[group(4,0), group(4,1), group(4,2)]},
        'xuan5':    {'name':'选五','balls':5, 'tip':'胆码+拖码轮转3注',
                     'groups':[group(5,0), group(5,1), group(5,2)]},
        'xuan5_fu': {'name':'选五复式','balls':5, 'tip':'8球覆盖C(8,5)=56注',
                     'groups':[group(8,0)]},
        'xuan6':    {'name':'选六','balls':6, 'tip':'胆码+拖码轮转3注（回测标准）',
                     'groups':[group(6,0), group(6,1), group(6,2)]},
        'xuan9':    {'name':'选九','balls':9, 'tip':'胆码+拖码轮转2注',
                     'groups':[group(9,0), group(9,1)]},
        'xuan10':   {'name':'选十','balls':10,'tip':'RL打分最高的10球',
                     'groups':[group(10,0)]},
    }
    picks_by_n = {4:group(4,0), 5:group(5,0), 6:group(6,0), 9:group(9,0), 10:group(10,0)}

    # 记录本次训练时的期数，供下次运行判断是否有新数据
    save_last_trained_n('kl8', len(records))

    return {'avg_net_per_game':avg_net,'games_tested':games,
            'ppo_selected':picks_by_n[6],   # 兼容旧字段
            'picks_by_n':picks_by_n,        # 兼容旧字段
            'plays':plays,                  # 新结构：与传统ML的plays字段完全一致的分组格式
            'fixed_eval':fixed_eval,           # 固定评测集成绩（考题不变，跨天可比）
            'backtest_by_play':backtest_by_play,   # 选四/五/六/九/十 各玩法回测对比
            'best_play_n':best_play_n,             # 回测表现最好的玩法（仅供参考，不代表未来）
            'ref_info':ref_info,            # 参考信息：遗漏/频率/ML预测，仅供理解RL判断依据，不影响排序
            'is_first_train':is_new,
            'note':f'以RL自身综合判断为主排序（已融合ML/DL/遗漏/频率/走势特征），选六净收益{avg_net}元/期，遗漏/频率/ML预测仅作参考展示'}


def run_ssq_daily(records, ml_pred, prev_result=None, dl_pred=None):
    print(f"\n{'='*50}\n双色球 PPO 每日增量微调（红球33全量打分+蓝球，{len(records)}期）\n{'='*50}")

    # 开奖日感知：双色球只在周二/四/日开奖，其余4天没有新数据；
    # 手动重复触发时也会命中这个检查，避免同一批数据被反复训练导致过拟合
    last_trained_n = get_last_trained_n('ssq')
    if len(records) <= last_trained_n:
        return carry_over_result('ssq', '双色球', prev_result, len(records), last_trained_n,
                                 '双色球周二/四/日开奖，本次运行无新开奖数据')

    ml_vec = extract_ml_prob_vec(ml_pred, 'ssq')
    dl_vec = extract_dl_prob_vec(dl_pred, 'ssq')
    # 传统ML概率 + 深度学习概率 拼成统一的外部模型信号向量
    ml_vec = np.concatenate([ml_vec, dl_vec]).astype(np.float32)
    _cur_feat_dim = len(fssq(records, len(records)-1) or {})
    lstm, tfm, meta = load_lstm_tfm('ssq', current_feat_dim=_cur_feat_dim)

    print("  批量预计算 LSTM/TFM 隐层状态…")
    t0 = time.time()
    lstm_hidden, lstm_idx2row = precompute_hidden_multi(records, fssq, lstm)
    tfm_hidden,  tfm_idx2row  = precompute_hidden_multi(records, fssq, tfm)
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [LSTM隐层] {'✓已加载' if lstm_hidden is not None else '✗未加载（回退为0向量）'}  [TFM隐层] {'✓已加载' if tfm_hidden is not None else '✗未加载（回退为0向量）'}")

    print("  批量预计算遗漏向量…")
    t0 = time.time()
    omit_arr = precompute_omission_ssq(records)
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [遗漏向量] ✓已加载（49维：33红球+16蓝球）")

    print("  批量预计算马尔可夫转移 + 贝叶斯后验…")
    t0 = time.time()
    mk_arr = precompute_markov_balls(records, 33, lambda r: r['red'])   # 33维
    by_arr = precompute_bayes(records, 33, lambda r: r['red'])          # 66维
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [马尔可夫] 33维  [贝叶斯均值+不确定性] 66维")

    print(f"  [状态向量组成] 原始统计特征{_cur_feat_dim}维 + 外部模型概率{len(ml_vec)}维(传统ML+深度学习) + LSTM隐层 + TFM隐层 + 遗漏49维")

    def make_env():
        return IntegratedSSQEnv(records, fssq, ml_vec,
                                lstm_hidden, lstm_idx2row, tfm_hidden, tfm_idx2row, omit_arr,
                                mk_arr=mk_arr, by_arr=by_arr)
    vec_env = make_vec_env(make_env, n_envs=4)

    model = load_ppo('ssq')
    _do = False   # 是否触发了定期重训（冻结模式下由 should_retrain 决定）
    is_new = model is None
    t0 = time.time()
    if not is_new:
        try:
            model.set_env(vec_env)
            # PPO保存时会把超参一起存进去，加载后必须显式覆盖，
            # 否则改了 ENT_COEF 对已有模型完全不生效，还以为调了参
            model.ent_coef = ENT_COEF['ssq']
            print(f"    熵系数已设为 {model.ent_coef}")
        except Exception as e:
            print(f"  ! 旧PPO模型与当前环境结构不兼容（{e}），改为全新训练")
            model = None; is_new = True
    if is_new:
        print("  首次训练（15万步，红球33全量打分+蓝球联合优化）…")
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=512, batch_size=128,
                    n_epochs=10, gamma=0.95, gae_lambda=0.95, clip_range=0.2, ent_coef=ENT_COEF['ssq'],
                    verbose=0, device='cpu')

    def build_state(idx):
        feat = fssq(records, idx)
        if feat is None: return None
        raw = np.array(list(feat.values()),dtype=np.float32)
        lh = lstm_hidden[lstm_idx2row[idx]] if (lstm_hidden is not None and idx in lstm_idx2row) else np.zeros(lstm_hidden.shape[1] if lstm_hidden is not None else 0,dtype=np.float32)
        th = tfm_hidden[tfm_idx2row[idx]]   if (tfm_hidden  is not None and idx in tfm_idx2row)  else np.zeros(tfm_hidden.shape[1] if tfm_hidden is not None else 0,dtype=np.float32)
        om = omit_arr[idx]
        mk = mk_arr[idx]; by = by_arr[idx]
        st = normalize_state_segments(raw,ml_vec,lh,th,om,mk,by, scale_key='ssq')
        return apply_segment_switches(st, _segs_ssq, 'ssq')

    # 分段边界（开关与消融诊断共用，必须与环境类 _segs() 一致）
    _lh_d = lstm_hidden.shape[1] if lstm_hidden is not None else 0
    _th_d = tfm_hidden.shape[1] if tfm_hidden is not None else 0
    _segs_ssq, _pp = [], 0
    for _n, _d in [('走势特征', _cur_feat_dim), ('ML+DL概率', len(ml_vec)),
                   ('LSTM隐层', _lh_d), ('TFM隐层', _th_d), ('遗漏', 49),
                   ('马尔可夫', mk_arr.shape[1]), ('贝叶斯', by_arr.shape[1])]:
        _segs_ssq.append((_n, _pp, _pp+_d)); _pp += _d
    _off = [n for n,_,_ in _segs_ssq if not SEGMENT_ENABLE.get('ssq',{}).get(n, True)]
    if _off: print(f"  [分段开关] 双色球 已关闭: {_off}（维度保留并清零，可随时切回，不触发重训）")

    _hold_start = max(SEQ_LEN+40, len(records)-holdout_size(len(records)))
    _hold_mid = (_hold_start + len(records)) // 2   # 前半选权重，后半只报分
    def _eval_holdout(mask=None, half='select', use_model=None):
        """在holdout上评分：红球平均命中数 + 蓝球命中率加权。mask=(s,e)时清零该区间。
           half='select'用前一半(早停选权重)，half='report'用后一半(从不参与选择，评分干净)"""
        tot, n = 0.0, 0
        _rng = range(_hold_start, _hold_mid) if half=='select' else range(_hold_mid, len(records))
        for i in _rng:
            st = build_state(i)
            if st is None: continue
            if mask is not None:
                st = st.copy(); st[mask[0]:mask[1]] = 0.0
            _m = use_model if use_model is not None else model
            a,_ = _m.predict(st, deterministic=True)
            rsel = set(int(x)+1 for x in np.argsort(a[:33])[-SSQ_RED_PICK_N:])
            bpred = int(np.argmax(a[33:]))+1
            tot += len(set(records[i]['red']) & rsel) + 0.5*int(bpred==records[i]['blue'])
            n += 1
        return tot/n if n else 0.0

    if is_new:
        model, _best, _hist = train_with_early_stop(
            model, 150000, _eval_holdout, '双色球首训', n_chunks=16, patience=6,
            reset_timesteps=True, warmup_chunks=5)
    else:
        if TRAIN_MODE == 'frozen':
            # 冻结模式：平时不训练（输出稳定、无过拟合），
            # 但攒够足够新数据后触发一次全量重训——这才是真正的"进化"，
            # 而不是每天拿万分之一的新数据空转。
            _do, _why = should_challenge('ssq', len(records))
            print(f"  [挑战周期] {_why}")
            if _do:
                _n_clean = max(len(records) - _hold_mid, 1)
                _se_clean = math.sqrt(6*(6/33)*(27/33)) / math.sqrt(_n_clean)
                def _mk():
                    m = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=256, batch_size=64,
                            n_epochs=8, gamma=0.9, gae_lambda=0.9, clip_range=0.2,
                            ent_coef=ENT_COEF['ssq'], verbose=0, device='cpu')
                    return m
                def _tr(m):
                    return train_with_early_stop(m, 150000, _eval_holdout, '双色球挑战者',
                                                 n_chunks=16, patience=6,
                                                 reset_timesteps=True, warmup_chunks=5)
                def _ev(m):
                    return _eval_holdout(half='report', use_model=m)
                def _rc(m):
                    # 复检用另一半数据：对现任和挑战者是同一批题，公平比较
                    return _eval_holdout(half='select', use_model=m)
                model, _swapped, _msg = run_challenge(
                    model, _mk, _tr, _ev, _se_clean, recheck_fn=_rc, label='双色球')
                print(f"  [挑战结果] {_msg}")
                _do = _swapped   # 只有换人了才需要保存
                _best = _eval_holdout(); _hist = [round(_best,4)]
            else:
                _best = _eval_holdout(); _hist = [round(_best,4)]
                print(f"  [现任模型] 沿用已有权重出预测，holdout评分 {_best:.4f}")
        else:
            print("  增量微调（1.5万步，带早停）…")
            model, _best, _hist = train_with_early_stop(
                model, 15000, _eval_holdout, '双色球微调', n_chunks=8, patience=4,
                reset_timesteps=False, warmup_chunks=2)
    print(f"    PPO训练完成，耗时 {time.time()-t0:.1f}s")
    print(f"    [训练/回测隔离] 训练止于第{_hold_start}期，回测第{_hold_start}~{len(records)-1}期（样本外）")
    _st_probe = build_state(len(records))
    if _st_probe is not None: report_entropy(model, _st_probe, '双色球')

    # 策略依赖性：模型到底有没有在用状态（比任何评分都更根本）
    try:
        _pts = [int(x) for x in np.linspace(SEQ_LEN+60, len(records)-1, 24).astype(int)]
        _dep = policy_dependence_test(model, build_state, _pts, 'ssq')
    except Exception as _e:
        _dep = None; print(f"    [策略依赖性] 检测异常: {_e}")

    # 干净评分：用从未参与早停选择的那半holdout评分，不会被筛选污染
    try:
        _clean = _eval_holdout(half='report')
        print(f"    [干净评分] 选权重用第{_hold_start}~{_hold_mid-1}期，"
              f"未参与选择的第{_hold_mid}~{len(records)-1}期得分 {_clean:.4f}")
    except Exception as _e:
        _clean = float('nan'); print(f"    [干净评分] 计算失败: {_e}")

    _lh_dim = lstm_hidden.shape[1] if lstm_hidden is not None else 0
    _th_dim = tfm_hidden.shape[1] if tfm_hidden is not None else 0
    _p = 0; _segs = []
    for _nm, _d in [('走势特征', _cur_feat_dim), ('ML+DL概率', len(ml_vec)),
                    ('LSTM隐层', _lh_dim), ('TFM隐层', _th_dim), ('遗漏', 49),
                    ('马尔可夫', 33), ('贝叶斯', 66)]:
        _segs.append((_nm, _p, _p+_d)); _p += _d
    # 冻结模式下权重不变，消融结论不会变，跳过以节省十几次评估的时间
    _ablation = []
    if TRAIN_MODE != 'frozen':
        _ablation = segment_ablation(_eval_holdout, _segs, _best, '双色球', 'ssq',
                                     se=math.sqrt(6*(6/33)*(27/33))/math.sqrt(max(len(records)-_hold_start,1)))
    append_history('ssq', {'date': str(date.today()), 'holdout_score': round(_best,4),
                          'clean_score': round(_clean,4),
                          'policy_dependence': _dep,
                          'ent_coef': ENT_COEF['ssq'],
                           'state_dim': _p, 'score_history': _hist, 'ablation': _ablation})

    # 只有真正训练过才保存：冻结且未触发重训时权重没变，
    # 重复推送没意义，还可能意外覆坏已有权重
    _trained = (TRAIN_MODE != 'frozen') or is_new or _do
    if _trained:
        save_ppo(model, 'ssq')
        save_last_trained_n('ssq', len(records))   # 记录本次训练时的期数，供下次判断
    else:
        print("  [冻结] 权重未改动，跳过保存")

    # 回测最近30期：红球命中数分布 + 蓝球命中率
    start=max(SEQ_LEN+30, len(records)-30)
    total=0; blue_correct=0; red_hit_dist={0:0,1:0,2:0,3:0,4:0,5:0,6:0}
    # 上界用 len(records)：idx 最大取到 len(records)-1，即拿"倒数第二期及之前"的特征
    # 去预测最后一期。原来写 len(records)-1 会让最后一期永远不参与回测，白白少一个样本。
    for idx in range(start, len(records)):
        state = build_state(idx)
        if state is None: continue
        action,_ = model.predict(state, deterministic=True)
        red_scores = action[:33]; blue_scores = action[33:]
        top_idx = np.argsort(red_scores)[-SSQ_RED_PICK_N:]
        red_selected = set(int(i)+1 for i in top_idx)
        blue_pred = int(np.argmax(blue_scores))+1

        actual_red = set(records[idx]['red']); actual_blue = records[idx]['blue']
        rh = len(actual_red & red_selected); bh = int(blue_pred==actual_blue)
        red_hit_dist[rh]+=1
        if bh: blue_correct+=1
        total+=1

    blue_acc = round(blue_correct/total*100,1) if total else 0
    avg_red_hit = round(sum(k*v for k,v in red_hit_dist.items())/total,2) if total else 0

    # ── 固定评测集：考题锁死，只有模型在变，跨天成绩才可比 ──
    fixed_eval = None
    _rng = get_fixed_eval_range('ssq', records)
    if _rng:
        _rh = 0; _bh = 0; _ft = 0
        for idx in range(_rng['start'], _rng['end']+1):
            st = build_state(idx)
            if st is None: continue
            act,_ = model.predict(st, deterministic=True)
            rs = act[:33]; bs = act[33:]
            sel = set(int(i)+1 for i in np.argsort(rs)[-SSQ_RED_PICK_N:])
            _rh += len(set(records[idx]['red']) & sel)
            _bh += int(int(np.argmax(bs))+1 == records[idx]['blue']); _ft += 1
        if _ft:
            f_rh = round(_rh/_ft, 3); f_ba = round(_bh/_ft*100, 1)
            print(f"  [固定评测集] 第{_rng['start']}~{_rng['end']}期({_ft}期): "
                  f"红球平均命中{f_rh}个  蓝球准确率{f_ba}%")
            append_eval_history('ssq', {'avg_red_hit': f_rh, 'blue_acc': f_ba, 'n': _ft})
            fixed_eval = {'range': [_rng['start'], _rng['end']], 'n': _ft,
                          'avg_red_hit': f_rh, 'blue_acc': f_ba}
    print(f"  回测（近{total}期）：红球平均命中{avg_red_hit}个  蓝球准确率{blue_acc}%（随机基准6.25%）")

    # 今日推荐：以RL自己的判断为主，红球排序滑动窗口切分成6注，遗漏/ML预测仅作参考展示
    # ⚠️ 这里必须用 len(records) 而不是 len(records)-1。
    # 训练时的约定是"特征取 records[:idx]、答案取 records[idx]"，
    # 所以 idx=len(records)-1 输出的是对【最后一期】的预测——而最后一期早就开出来了，
    # 等于让模型复述已知答案（实测表现为推荐号码与最新开奖高度重合）。
    # idx=len(records) 才是"用全部已知数据预测下一期（尚未开奖）"。
    idx = len(records)
    state = build_state(idx)
    groups=[]
    ref_info = {}
    red_core_info, red_pool_info = [], []
    if state is not None:
        base_action,_ = model.predict(state, deterministic=True)
        red_scores = base_action[:33]; blue_scores = base_action[33:]

        # 蓝球排序：不再只取argmax(唯一最优解)，而是拿到RL对全部16个蓝球的完整打分排序，
        # 让不同注轮流用排名靠前的几个候选蓝球，把模型对次优选项的判断也利用起来，
        # 而不是把"分数第二、第三高"的蓝球完全浪费掉、6注全部锁死在同一个号码上。
        blue_order = [int(i)+1 for i in np.argsort(blue_scores)[::-1]]

        rl_red_order = [int(i)+1 for i in np.argsort(red_scores)[::-1]]

        # 区分度诊断（红球/蓝球分开看）：模型的打分能不能把好坏号码拉开差距？
        # 差距接近0说明模型没在真正区分，推荐等同于随机选，比看回测数字更直接。
        _rs = np.sort(red_scores)[::-1]
        _r_gap = (float(_rs[:6].mean()) - float(_rs[13:19].mean())) / (float(_rs.max()-_rs.min()) + 1e-9)
        _bs = np.sort(blue_scores)[::-1]
        _b_gap = (float(_bs[:3].mean()) - float(_bs[6:9].mean())) / (float(_bs.max()-_bs.min()) + 1e-9)
        print(f"  [区分度] 红球Top6与中位区差距占全域{_r_gap*100:.1f}%  "
              f"蓝球Top3与中位区差距占全域{_b_gap*100:.1f}%")
        if _r_gap < 0.15:
            print(f"    ⚠️ 红球区分度偏低，模型对各红球偏好不明显，推荐参考价值有限")
        if _b_gap < 0.15:
            print(f"    ⚠️ 蓝球区分度偏低，模型对16个蓝球基本无偏好")

        # ── 红球：枚举组合，取模型联合得分最高的6注 ──
        # 之前是把排序切成梯队(1-6名/7-12名/…)，但第2注开始就是模型认为"第7到12好"的球，
        # 等于故意给出越来越差的推荐，这不是"最可能出现的6注"。
        # ── 红球：胆码+拖码轮转 + ML条件校验 ──
        # ML的7个目标(奇数/和值/AC值/主力区/间距/大数/连号)本就已注入RL状态，
        # 但之前选号只看每个球的分数，"这一注整体符不符合那些预测"没人检验，这里补上。
        def _ssq_cond_feats(red):
            r = sorted(red); sm = sum(r)
            d = set()
            for i in range(len(r)):
                for j in range(i+1, len(r)): d.add(r[j]-r[i])
            ac = len(d) - (len(r)-1)
            z = [sum(1 for x in r if x <= 11), sum(1 for x in r if 12 <= x <= 22),
                 sum(1 for x in r if x >= 23)]
            mg = max(r[i+1]-r[i] for i in range(len(r)-1)) if len(r) > 1 else 0
            return {'odd': sum(1 for x in r if x % 2 != 0),
                    'sum_grp': 0 if sm < 70 else (1 if sm < 100 else 2),
                    'ac_grp': 0 if ac <= 2 else (1 if ac <= 5 else 2),
                    'red_zone_dom': int(max(range(3), key=lambda i: z[i])),
                    'gap_grp': 0 if mg <= 5 else (1 if mg <= 10 else 2),
                    'big': sum(1 for x in r if x > 16),
                    'consec': sum(1 for i in range(len(r)-1) if r[i+1]-r[i] == 1)}

        _ssq_conds = {}
        _md = ml_pred.get('models', {})
        for _k in ['odd','sum_grp','ac_grp','red_zone_dom','gap_grp','big','consec']:
            _p = (_md.get(_k, {}) or {}).get('prediction', {})
            if _p.get('value') is not None:
                _ssq_conds[_k] = (int(_p['value']), float(_p.get('confidence', 50))/100.0)

        # 完全按RL自己的球分选号（理由同快乐8：ML预测已在状态里，不在外面二次干预）
        red_top6, red_core, red_pool = diverse_picks(red_scores, 6, 6)
        if _ssq_conds:
            _cavg = sum(len([1 for k,(v,w) in _ssq_conds.items()
                             if _ssq_cond_feats(b).get(k)==v]) for b in red_top6) / max(len(red_top6),1)
            print(f"  [诊断·仅参考] RL自选的6注，平均符合{_cavg:.1f}/{len(_ssq_conds)}条ML预测条件"
                  f"（不参与筛选，仅用于观察RL判断与ML预测的一致程度）")
        _ov = [len(set(red_top6[i]) & set(red_top6[j])) for i in range(6) for j in range(i+1,6)]
        _allb = set()
        for c in red_top6: _allb |= set(c)
        print(f"  [红球] 候选池{len(red_pool)}球 胆码{red_core}  "
              f"6注共用到{len(_allb)}个号码，两两平均重合{np.mean(_ov):.1f}/6")

        # ── 蓝球：模型预测几个算几个，全部展示 ──
        # 对16个蓝球分数做softmax，把高于均匀分布(1/16=6.25%)的候选都算作模型的预测，最多3个
        _bexp = np.exp(blue_scores - np.max(blue_scores))
        _bprob = _bexp / (_bexp.sum() + 1e-12)
        _border = np.argsort(_bprob)[::-1]
        blue_cands, blue_probs = [], []
        for bi in _border[:3]:
            p = float(_bprob[bi])
            if p >= (1.0/16) or not blue_cands:   # 至少给1个，其余只需高于均匀分布即可
                blue_cands.append(int(bi) + 1); blue_probs.append(round(p*100, 1))
        print(f"  [蓝球预测] 共{len(blue_cands)}个候选: "
              + "  ".join(f"{b:02d}({p}%)" for b, p in zip(blue_cands, blue_probs)))

        for red_sel in red_top6:
            groups.append({
                'red': red_sel,
                'blue': blue_cands[0] if blue_cands else None,   # 兼容旧字段
                'blues': blue_cands,          # 模型预测的全部蓝球候选，前端有几个显示几个
                'blue_probs': blue_probs,
            })
        blue_sel = blue_cands[0] if blue_cands else blue_order[0]
        red_core_info, red_pool_info = red_core, red_pool

        # 参考信息：遗漏值+ML主力区预测，仅用于展示说明，不参与排序计算
        om_now = omit_arr[idx] if omit_arr is not None else np.zeros(49)
        top6_red = rl_red_order[:6]
        avg_omission = round(float(np.mean([om_now[b-1] for b in top6_red])), 2)
        models_data = ml_pred.get('models', {})
        zone_pred = models_data.get('red_zone_dom', {}).get('prediction', {}).get('value')
        zone_names = ['一区(1-11)','二区(12-22)','三区(23-33)']
        ref_info = {
            'avg_omission_top6': avg_omission,
            'ml_zone_pred': zone_names[zone_pred] if zone_pred is not None and 0<=zone_pred<3 else None,
        }
        print(f"  [主推荐] RL红球排序Top6: {sorted(top6_red)}  蓝球Top3候选: {blue_order[:3]}（6注轮流分配）")
        print(f"  [参考信息] 该注平均遗漏{avg_omission}期；ML预测红球主力区={ref_info['ml_zone_pred']}")

    # 兼容旧字段：主推荐仍取第一注
    red_selected = groups[0]['red'] if groups else []
    blue_pred = groups[0]['blue'] if groups else None

    # 记录本次训练时的期数，供下次运行判断是否有新开奖
    save_last_trained_n('ssq', len(records))

    return {'blue_acc_pct':blue_acc,'games_tested':total,
            'avg_red_hit':avg_red_hit,'red_hit_distribution':red_hit_dist,
            'ppo_red_selected':red_selected,'ppo_blue_pred':blue_pred,
            'ppo_groups':groups,'ref_info':ref_info,'fixed_eval':fixed_eval,
            'red_core':red_core_info,'red_pool':red_pool_info,
            'is_first_train':is_new,
            'note':f'以RL自身综合判断为主排序（已融合ML/DL/遗漏/走势特征），红球平均命中{avg_red_hit}个，蓝球准确率{blue_acc}%，遗漏/ML预测仅作参考展示'}


def run_3d_daily(records, ml_pred, prev_result=None, dl_pred=None):
    print(f"\n{'='*50}\n福彩3D PPO 每日增量微调（{len(records)}期）\n{'='*50}")

    # 新数据检测：避免重复手动触发时拿完全相同的数据反复训练导致过拟合
    last_trained_n = get_last_trained_n('3d')
    if len(records) <= last_trained_n:
        return carry_over_result('3d', '福彩3D', prev_result, len(records), last_trained_n,
                                 '本次运行无新开奖数据（可能是当日已训练过或重复手动触发）')

    ml_vec = extract_ml_prob_vec(ml_pred, '3d')
    dl_vec = extract_dl_prob_vec(dl_pred, '3d')
    # 传统ML概率 + 深度学习概率 拼成统一的外部模型信号向量
    ml_vec = np.concatenate([ml_vec, dl_vec]).astype(np.float32)
    _cur_feat_dim = len(f3d(records, len(records)-1) or {})
    lstm, tfm, meta = load_lstm_tfm('3d', current_feat_dim=_cur_feat_dim)

    print("  批量预计算 LSTM/TFM 隐层状态…")
    t0 = time.time()
    lstm_hidden, lstm_idx2row = precompute_hidden_multi(records, f3d, lstm)
    tfm_hidden,  tfm_idx2row  = precompute_hidden_multi(records, f3d, tfm)
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [LSTM隐层] {'✓已加载' if lstm_hidden is not None else '✗未加载（回退为0向量）'}  [TFM隐层] {'✓已加载' if tfm_hidden is not None else '✗未加载（回退为0向量）'}")

    print("  批量预计算遗漏向量…")
    t0 = time.time()
    omit_arr = precompute_omission_3d(records)
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [遗漏向量] ✓已加载（30维：百十个位各10个数字）")

    print("  批量预计算马尔可夫转移 + 贝叶斯后验…")
    t0 = time.time()
    mk_arr = precompute_markov_3d(records)                                    # 30维
    by_arr = precompute_bayes(records, 10, lambda r: [d+1 for d in r['digits']])  # 20维
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [马尔可夫] 30维  [贝叶斯均值+不确定性] 20维")

    print(f"  [状态向量组成] 原始统计特征{_cur_feat_dim}维 + 外部模型概率{len(ml_vec)}维(传统ML+深度学习)"
          f" + LSTM隐层 + TFM隐层 + 遗漏30维 + 马尔可夫30维 + 贝叶斯20维")

    def make_env():
        return Integrated3DEnv(records, f3d, ml_vec,
                               lstm_hidden, lstm_idx2row, tfm_hidden, tfm_idx2row, omit_arr,
                               mk_arr, by_arr)
    vec_env = make_vec_env(make_env, n_envs=4)

    # fresh 模式才不加载；frozen 和 incremental 都需要旧权重
    model = None if TRAIN_MODE == 'fresh' else load_ppo('3d')
    _do = False   # 是否触发了定期重训（冻结模式下由 should_retrain 决定）
    if TRAIN_MODE == 'fresh':
        print("  [训练模式] fresh：不加载旧权重，本次从零全量训练")
    is_new = model is None
    t0 = time.time()
    if not is_new:
        try:
            model.set_env(vec_env)
            # PPO保存时会把超参一起存进去，加载后必须显式覆盖，
            # 否则改了 ENT_COEF 对已有模型完全不生效，还以为调了参
            model.ent_coef = ENT_COEF['3d']
            print(f"    熵系数已设为 {model.ent_coef}")
        except Exception as e:
            print(f"  ! 旧PPO模型与当前环境结构不兼容（{e}），改为全新训练")
            model = None; is_new = True
    if is_new:
        print("  首次训练（10万步，MultiDiscrete([10,10,10])共1000种组合）…")
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=256, batch_size=64,
                    n_epochs=8, gamma=0.9, gae_lambda=0.9, clip_range=0.2, ent_coef=ENT_COEF['3d'],
                    verbose=0, device='cpu')

    # 分段边界（开关与消融诊断共用，必须与环境类 _segs() 一致）
    _lh_d = lstm_hidden.shape[1] if lstm_hidden is not None else 0
    _th_d = tfm_hidden.shape[1] if tfm_hidden is not None else 0
    _segs_3d, _pp = [], 0
    for _n, _d in [('走势特征', _cur_feat_dim), ('ML+DL概率', len(ml_vec)),
                   ('LSTM隐层', _lh_d), ('TFM隐层', _th_d), ('遗漏', 30),
                   ('马尔可夫', mk_arr.shape[1]), ('贝叶斯', by_arr.shape[1])]:
        _segs_3d.append((_n, _pp, _pp+_d)); _pp += _d
    _off = [n for n,_,_ in _segs_3d if not SEGMENT_ENABLE.get('3d',{}).get(n, True)]
    if _off: print(f"  [分段开关] 3D 已关闭: {_off}（维度保留并清零，可随时切回，不触发重训）")

    # build_state 提前定义：早停要在每段训练后用它在holdout上评分
    def build_state(idx):
        feat = f3d(records, idx)
        if feat is None: return None
        raw = np.array(list(feat.values()),dtype=np.float32)
        lh = lstm_hidden[lstm_idx2row[idx]] if (lstm_hidden is not None and idx in lstm_idx2row) else np.zeros(lstm_hidden.shape[1] if lstm_hidden is not None else 0,dtype=np.float32)
        th = tfm_hidden[tfm_idx2row[idx]]   if (tfm_hidden  is not None and idx in tfm_idx2row)  else np.zeros(tfm_hidden.shape[1] if tfm_hidden is not None else 0,dtype=np.float32)
        om = omit_arr[idx]
        mk = mk_arr[idx]; by = by_arr[idx]
        st = normalize_state_segments(raw,ml_vec,lh,th,om,mk,by, scale_key='3d')
        return apply_segment_switches(st, _segs_3d, '3d')

    _hold_start = max(SEQ_LEN+40, len(records)-holdout_size(len(records)))
    _hold_mid = (_hold_start + len(records)) // 2
    def _eval_holdout(mask=None, half='select', use_model=None):
        """
        holdout评分。关键：分成两半用途不同——
        · half='select'：前一半，用于早停挑权重
        · half='report'：后一半，只报分、绝不参与选择

        为什么要分：早停每天在同一批期数上挑分数最高的权重，
        连续几十天下来，选出的是"最会做这批题"的模型，
        它在这批题上的分数会稳步虚高，不代表真实泛化能力。
        留一半从不参与选择的题，报出来的分才干净。
        """
        rng = range(_hold_start, _hold_mid) if half=='select' else range(_hold_mid, len(records))
        tot, n = 0, 0
        for i in rng:
            st = build_state(i)
            if st is None: continue
            if mask is not None:
                st = st.copy(); st[mask[0]:mask[1]] = 0.0
            _m = use_model if use_model is not None else model
            a,_ = _m.predict(st, deterministic=True)
            act = records[i]['digits']
            tot += sum(1 for k in range(3) if int(a[k])==act[k]); n += 1
        return tot/n if n else 0.0

    if is_new:
        model, _best, _hist = train_with_early_stop(
            model, 100000, _eval_holdout, '3D首训', n_chunks=16, patience=6,
            reset_timesteps=True, warmup_chunks=5)
    else:
        if TRAIN_MODE == 'frozen':
            # 冻结模式：平时不训练（输出稳定、无过拟合），
            # 但攒够足够新数据后触发一次全量重训——这才是真正的"进化"，
            # 而不是每天拿万分之一的新数据空转。
            _do, _why = should_challenge('3d', len(records))
            print(f"  [挑战周期] {_why}")
            if _do:
                _n_clean = max(len(records) - _hold_mid, 1)
                _se_clean = math.sqrt(3*0.1*0.9) / math.sqrt(_n_clean)
                def _mk():
                    m = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=256, batch_size=64,
                            n_epochs=8, gamma=0.9, gae_lambda=0.9, clip_range=0.2,
                            ent_coef=ENT_COEF['3d'], verbose=0, device='cpu')
                    return m
                def _tr(m):
                    return train_with_early_stop(m, 100000, _eval_holdout, '3D挑战者',
                                                 n_chunks=16, patience=6,
                                                 reset_timesteps=True, warmup_chunks=5)
                def _ev(m):
                    return _eval_holdout(half='report', use_model=m)
                def _rc(m):
                    # 复检用另一半数据：对现任和挑战者是同一批题，公平比较
                    return _eval_holdout(half='select', use_model=m)
                model, _swapped, _msg = run_challenge(
                    model, _mk, _tr, _ev, _se_clean, recheck_fn=_rc, label='3D')
                print(f"  [挑战结果] {_msg}")
                _do = _swapped   # 只有换人了才需要保存
                _best = _eval_holdout(); _hist = [round(_best,4)]
            else:
                _best = _eval_holdout(); _hist = [round(_best,4)]
                print(f"  [现任模型] 沿用已有权重出预测，holdout评分 {_best:.4f}")
        else:
            print("  增量微调（1万步，带早停）…")
            model, _best, _hist = train_with_early_stop(
                model, 10000, _eval_holdout, '3D微调', n_chunks=8, patience=4,
                reset_timesteps=False, warmup_chunks=2)
    print(f"    PPO训练完成，耗时 {time.time()-t0:.1f}s")
    # 策略依赖性：模型到底有没有在用状态（比任何评分都更根本）
    try:
        _pts = [int(x) for x in np.linspace(SEQ_LEN+60, len(records)-1, 24).astype(int)]
        _dep = policy_dependence_test(model, build_state, _pts, '3d')
    except Exception as _e:
        _dep = None; print(f"    [策略依赖性] 检测异常: {_e}")

    _clean = _eval_holdout(half='report')
    # 对数概率评分：比命中位数灵敏，能看到"概率提升但还不是最大值"的改善
    try:
        _lp = eval_logprob_3d(model, build_state, records, range(_hold_mid, len(records)))
        print(f"    [对数概率·干净区] {_lp:.4f}（均匀基准 {math.log(0.1):.4f}，越大越好，"
              f"高出基准 {_lp-math.log(0.1):+.4f}）")
    except Exception as _e:
        _lp = None; print(f"    [对数概率] 计算失败: {_e}")
    print(f"    [训练/回测隔离] 训练止于第{_hold_start}期（样本外holdout {len(records)-_hold_start}期）")
    print(f"    [干净评分] 选权重用第{_hold_start}~{_hold_mid-1}期，"
          f"未参与选择的第{_hold_mid}~{len(records)-1}期得分 {_clean:.4f}（随机基准0.30）")
    print(f"    → 这个数才是没被筛选污染的。若它长期不涨而选择分在涨，说明涨的是假象")

    # ── 消融诊断：测量状态向量各段的真实贡献 ──
    _segs = _segs_3d; _p = _segs[-1][2]
    # 冻结模式下权重不变，消融结论不会变，跳过以节省十几次评估的时间
    _ablation = []
    if TRAIN_MODE != 'frozen':
        _ablation = segment_ablation(_eval_holdout, _segs, _best, '3D', '3d',
                                     se=math.sqrt(3*0.1*0.9)/math.sqrt(max(len(records)-_hold_start,1)))
    append_history('3d', {'date': str(date.today()),
                          'holdout_score': round(_best,4),        # 早停选出来的，会虚高
                          'clean_score': round(_clean,4),         # 未参与选择，这个才可信
                          'policy_dependence': _dep,              # 模型是否真的在用状态
                          'logprob_clean': (round(_lp,4) if _lp is not None else None),
                          'train_mode': TRAIN_MODE,
                          'ent_coef': ENT_COEF['3d'],
                          'state_dim': _p, 'score_history': _hist, 'ablation': _ablation})

    # 只有真正训练过才保存：冻结且未触发重训时权重没变，
    # 重复推送没意义，还可能意外覆坏已有权重
    _trained = (TRAIN_MODE != 'frozen') or is_new or _do
    if _trained:
        save_ppo(model, '3d')
        save_last_trained_n('3d', len(records))   # 记录本次训练时的期数，供下次判断
    else:
        print("  [冻结] 权重未改动，跳过保存")

    # 回测最近30期：统计位命中数分布 + 全中次数
    start=max(SEQ_LEN+5, len(records)-30); total=0
    match_dist={0:0,1:0,2:0,3:0}
    # 上界用 len(records)：idx 最大取到 len(records)-1，即拿"倒数第二期及之前"的特征
    # 去预测最后一期。原来写 len(records)-1 会让最后一期永远不参与回测，白白少一个样本。
    for idx in range(start, len(records)):
        state = build_state(idx)
        if state is None: continue
        action,_ = model.predict(state, deterministic=True)
        pred=[int(action[0]),int(action[1]),int(action[2])]
        actual=records[idx]['digits']
        m = sum(1 for i in range(3) if pred[i]==actual[i])
        match_dist[m]+=1; total+=1
    exact_hit_rate = round(match_dist[3]/total*100,2) if total else 0
    avg_match = round(sum(k*v for k,v in match_dist.items())/total,2) if total else 0

    # ── 固定评测集：考题锁死，只有模型在变，跨天成绩才可比 ──
    fixed_eval = None
    _rng = get_fixed_eval_range('3d', records)
    if _rng:
        _fm = {0:0,1:0,2:0,3:0}; _ft = 0
        for idx in range(_rng['start'], _rng['end']+1):
            st = build_state(idx)
            if st is None: continue
            act,_ = model.predict(st, deterministic=True)
            p=[int(act[0]),int(act[1]),int(act[2])]; a=records[idx]['digits']
            _fm[sum(1 for i in range(3) if p[i]==a[i])] += 1; _ft += 1
        if _ft:
            f_avg = round(sum(k*v for k,v in _fm.items())/_ft, 3)
            f_exact = round(_fm[3]/_ft*100, 2)
            print(f"  [固定评测集] 第{_rng['start']}~{_rng['end']}期({_ft}期): "
                  f"平均命中{f_avg}位  全中率{f_exact}%")
            append_eval_history('3d', {'avg_match': f_avg, 'exact_rate': f_exact, 'n': _ft})
            fixed_eval = {'range': [_rng['start'], _rng['end']], 'n': _ft,
                          'avg_match': f_avg, 'exact_rate': f_exact}

    # ⚠️ 这里必须用 len(records) 而不是 len(records)-1。
    # 训练时的约定是"特征取 records[:idx]、答案取 records[idx]"，
    # 所以 idx=len(records)-1 输出的是对【最后一期】的预测——而最后一期早就开出来了，
    # 等于让模型复述已知答案（实测表现为推荐号码与最新开奖高度重合）。
    # idx=len(records) 才是"用全部已知数据预测下一期（尚未开奖）"。
    idx=len(records); state=build_state(idx)
    groups=[]; pos_candidates=[]
    if state is not None:
        # 明确提取百/十/个位各自的完整概率分布（而非随机采样撞运气），
        # 用联合概率排序生成多注真正的次优组合，能说清楚"这是第几优的组合"
        try:
            obs_tensor, _ = model.policy.obs_to_tensor(np.array(state).reshape(1, -1))
            with torch.no_grad():
                dist = model.policy.get_distribution(obs_tensor)
            # MultiDiscrete动作空间下，dist.distribution是[百位分布,十位分布,个位分布]三个独立分类分布
            pos_probs = [d.probs.detach().cpu().numpy()[0] for d in dist.distribution]  # 每个是长度10的概率数组

            # 每位取Top3候选。之前用纯联合概率取Top6有个问题：
            # 联合概率是相乘的，某一位第1名只要比第2名高出一截，乘法会把优势放大，
            # 导致6注里那一位全被同一个数字垄断（比如十位0.200 vs 0.129，6注十位全是同一个），
            # 模型对第2、3候选的判断就被白白浪费了。
            # 改成"轮转+择优"：前3注让每位的3个候选各当一次主角（保证全部候选都露面），
            # 后3注再从27种组合里按联合概率择优补足。
            p_bai, p_shi, p_ge = pos_probs[0], pos_probs[1], pos_probs[2]
            top3 = []
            for p in (p_bai, p_shi, p_ge):
                idx3 = np.argsort(p)[::-1][:3]
                top3.append([(int(d), float(p[d])) for d in idx3])

            # ML的7条预测(和值/奇数/组型/大数/跨度/012路/斜连)已注入RL状态，
            # 这里在选号环节也做校验，让"这一注整体像不像模型预测的样子"参与排序
            def _d3_feats(c):
                b, s, g = c; sm = b+s+g
                tri = (b == s == g); g3 = (b == s or s == g or b == g) and not tri
                s3 = sorted(c); rd = [x % 3 for x in c]
                return {'sum_grp': 0 if sm <= 9 else (1 if sm <= 17 else 2),
                        'odd': sum(1 for x in c if x % 2 != 0),
                        'group_type': 0 if tri else (1 if g3 else 2),
                        'big': sum(1 for x in c if x >= 5),
                        'span_grp': (lambda sp: 0 if sp <= 3 else (1 if sp <= 6 else 2))(max(c)-min(c)),
                        'road_dom': max(set(rd), key=rd.count),
                        'arith': int((s3[1]-s3[0]) == (s3[2]-s3[1]) and s3[2]-s3[0] > 0)}
            _d3_conds = {}
            _md3 = ml_pred.get('models', {})
            for _k in ['sum_grp','odd','group_type','big','span_grp','road_dom','arith']:
                _p = (_md3.get(_k, {}) or {}).get('prediction', {})
                if _p.get('value') is not None:
                    _d3_conds[_k] = (int(_p['value']), float(_p.get('confidence', 50))/100.0)

            picked, seen = [], set()
            for i in range(3):   # 前3注：各位第i候选组合，确保候选全覆盖
                c = [top3[0][i][0], top3[1][i][0], top3[2][i][0]]
                pr = top3[0][i][1] * top3[1][i][1] * top3[2][i][1]
                picked.append((c, pr)); seen.add(tuple(c))
            # 其余注数：从27种组合里按联合概率补足
            # 按RL自己的联合概率排序（不用ML条件干预，理由同上）
            all27 = sorted(
                (([b, s, g], pb*ps*pg)
                 for b, pb in top3[0] for s, ps in top3[1] for g, pg in top3[2]),
                key=lambda x: -x[1])
            for c, pr in all27:
                if len(picked) >= D3_N_BETS: break
                if tuple(c) not in seen:
                    picked.append((c, pr)); seen.add(tuple(c))
            picked.sort(key=lambda x: -x[1])
            groups = [c for c, _ in picked]
            if _d3_conds:
                _cavg = sum(len([1 for k,(v,w) in _d3_conds.items()
                                 if _d3_feats(g).get(k)==v]) for g in groups) / max(len(groups),1)
                print(f"  [诊断·仅参考] RL自选的{len(groups)}注，平均符合{_cavg:.1f}/{len(_d3_conds)}条ML预测条件"
                      f"（不参与筛选，仅用于观察RL判断与ML预测的一致程度）")
            top_probs = [pr for _, pr in picked]

            # 每位候选明细，供前端展示"模型认为这位可能是哪几个数字"
            pos_candidates = [
                [{'digit': d, 'prob': round(pv*100, 1)} for d, pv in t] for t in top3
            ]

            names = ['百位','十位','个位']
            for ni, t in enumerate(top3):
                print(f"  [{names[ni]}候选] " + "  ".join(f"{d}({pv*100:.1f}%)" for d, pv in t))
            print(f"  [推荐6注] {groups}")
            print(f"    对应联合概率: {[round(x,5) for x in top_probs]}")
            _cov = [len(set(c[i] for c in groups)) for i in range(3)]
            print(f"    候选覆盖: 百位{_cov[0]}/3  十位{_cov[1]}/3  个位{_cov[2]}/3")
            # 熵越接近均匀分布(约2.303)，说明模型对该位越没有明确偏好，推荐参考价值越低
            ent = [float(-(p*np.log(p+1e-12)).sum()) for p in pos_probs]
            print(f"    各位分布熵: 百{ent[0]:.3f} 十{ent[1]:.3f} 个{ent[2]:.3f}（均匀分布=2.303，越接近说明该位越没学到偏好）")
            _emax = math.log(10)
            _gap = _emax - float(np.mean(ent))
            # 把熵差换算成"最高候选概率"，比抽象的熵值直观；
            # 注意：熵只反映模型敢不敢下判断，不代表判断正确——
            # 之前那个 0.05 的门槛是拍脑袋定的，会把17%这种明显有倾向的情况误判成"没偏好"。
            _top_p = float(np.max([np.max(p) for p in pos_probs]))
            print(f"    → 平均熵比均匀低 {_gap:.4f}，最高候选 {_top_p*100:.1f}%（均匀基准10.0%，"
                  f"ent_coef={ENT_COEF['3d']}）")
            print(f"    → 注意：熵低只说明模型敢下判断，不代表判断对。"
                  f"是否真有价值看 holdout 评分是否稳定高于随机基准 0.30")
            # 输入敏感度检测：直接量化"今天的新开奖"对状态向量的影响。
            # 如果推荐没变，这个数能立刻区分是【输入没变】还是【输入变了但模型不敏感】。
            try:
                _st_now = build_state(len(records))
                _st_prev = build_state(len(records)-1)   # 少用最新一期算的状态
                if _st_now is not None and _st_prev is not None:
                    _d = np.abs(_st_now - _st_prev)
                    _chg = float((_d > 1e-6).sum())
                    _rel = float(_d.sum() / (np.abs(_st_prev).sum() + 1e-9))
                    print(f"    [输入敏感度] 最新一期使 {int(_chg)}/{len(_st_now)} 维发生变化，"
                          f"整体变化幅度 {_rel*100:.2f}%")
                    if _rel < 0.005:
                        print(f"      ⚠️ 变化过小，模型很难对新开奖产生反应")
            except Exception as e:
                print(f"    [输入敏感度] 检测失败: {e}")

            # 与上次推荐对比：微调有没有产生实际变化，一眼可见
            try:
                _prev = (prev_result or {}).get('ppo_groups') or []
                if _prev:
                    _now_set = {tuple(g) for g in groups}
                    _pre_set = {tuple(g) for g in _prev}
                    _same = len(_now_set & _pre_set)
                    print(f"    [与上次对比] 本次{len(groups)}注中有 {_same} 注与上次相同，"
                          f"{len(groups)-_same} 注是新的")
                    if _same >= len(groups):
                        # 概率其实是变了的（新数据进来了），但各位Top3的数字排序没变，
                        # 组合出来自然还是同样12注。说清楚原因，不要只给个警告符号。
                        _tp = [f"{int(np.argmax(p))}({np.max(p)*100:.1f}%)" for p in pos_probs]
                        print(f"      ⚠️ 推荐完全没变。原因：各位概率随新数据有微小变化"
                              f"（当前各位首选 {' / '.join(_tp)}），"
                              f"但Top3的数字排序未变，组合结果自然相同。")
                        print(f"      → 若连续多天如此，说明模型对状态的响应太弱，"
                              f"新开奖信息实际上没有影响预测（见上方[状态敏感度]诊断）")
            except Exception: pass
        except Exception as e:
            print(f"  ! 提取概率分布失败({e})，改用确定性预测兜底")
            action,_ = model.predict(state, deterministic=True)
            groups = [[int(action[0]),int(action[1]),int(action[2])]]

        # 不足时补齐。原来是把最后一注反复复制，12注会出现大量重复，很难看；
        # 改成从各位Top3之外按概率顺延取候选，凑不满就少给几注，绝不重复填充。
        if not groups:
            action,_ = model.predict(state, deterministic=True)
            groups.append([int(action[0]),int(action[1]),int(action[2])])
        if len(groups) < D3_N_BETS:
            try:
                _seen2 = {tuple(g) for g in groups}
                _wide = [np.argsort(p)[::-1][:5] for p in pos_probs]   # 放宽到每位Top5
                _cand = sorted(
                    (([int(b),int(s),int(g)], float(pos_probs[0][b]*pos_probs[1][s]*pos_probs[2][g]))
                     for b in _wide[0] for s in _wide[1] for g in _wide[2]),
                    key=lambda x: -x[1])
                for c, _ in _cand:
                    if len(groups) >= D3_N_BETS: break
                    if tuple(c) not in _seen2:
                        _seen2.add(tuple(c)); groups.append(c)
            except Exception:
                pass
    pred = groups[0] if groups else None  # 兼容旧字段：主推荐仍取第一注（联合概率最高的组合）

    # 记录本次训练时的期数，供下次运行判断是否有新数据
    save_last_trained_n('3d', len(records))

    return {'games_tested':total,'match_distribution':match_dist,
            'avg_match_digits':avg_match,'exact_hit_rate_pct':exact_hit_rate,
            'ppo_pred':pred,'ppo_groups':groups,
            'pos_candidates':pos_candidates,   # 每位Top3候选及其概率，供前端展示
            'fixed_eval':fixed_eval,           # 固定评测集成绩（考题不变，跨天可比）
            'is_first_train':is_new,
            'note':f'PPO给出百/十/个位各3个候选，{len(groups)}注采用"轮转+择优"确保每个候选都参与组合（避免联合概率导致某位被单一数字垄断），近{total}期平均命中{avg_match}位，全中率{exact_hit_rate}%（随机基准0.1%）'}

# ══════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════
print(f"\n{'#'*55}\nPPO 强化学习 每日增量微调  {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'#'*55}")

raw = gh_raw('history.json')
if not raw: print("失败"); sys.exit(1)
history = json.loads(raw)

# 读取 prediction.json 取ML概率向量（RL状态的一部分）
# 加载归一化基准：必须在任何 build_state 之前，保证输入口径与首训时一致
load_seg_scales()

raw_ml = gh_raw('prediction.json')
ml_preds = {}
if raw_ml:
    try: ml_preds = json.loads(raw_ml).get('predictions', {})
    except Exception: pass

# 读取深度学习的预测结果（LSTM+TFM集成），与传统ML概率一起注入RL状态。
# 之前RL对这份数据零引用，DL训练出的7组预测完全没被用上。
raw_dl = gh_raw('dl_lstm_tfm.json')
dl_preds = {}
if raw_dl:
    try:
        _dlj = json.loads(raw_dl)
        dl_preds = _dlj.get('results', _dlj) or {}
        print(f"✓ 已读取深度学习预测 dl_lstm_tfm.json（覆盖游戏: {list(dl_preds.keys())}）")
    except Exception as e:
        print(f"! 解析 dl_lstm_tfm.json 失败: {e}，本次RL状态将不含DL预测")
else:
    print("! 未读取到 dl_lstm_tfm.json，本次RL状态将不含DL预测"
          "（首次运行或DL周训练尚未产出时属正常）")

# 读取上一次的 dl_rl.json，双色球非开奖日跳过训练时用来沿用完整结果
# （保持字段结构跟正常训练完全一致，HTML渲染逻辑不用感知任何变化）
raw_prev_rl = gh_raw('dl_rl.json')
prev_rl_results = {}
if raw_prev_rl:
    try: prev_rl_results = json.loads(raw_prev_rl).get('results', {})
    except Exception: pass

os.makedirs(RL_LOCAL_DIR, exist_ok=True)
rl_results = {}

for game, run_fn in [('3d', run_3d_daily), ('kl8', run_kl8_daily), ('ssq', run_ssq_daily)]:
    records = history.get(game, [])
    if not isinstance(records,list) or len(records)<65:
        print(f"\n{game}: 数据不足，跳过"); continue
    ml_pred = ml_preds.get(game, {})
    try:
        # 三个游戏统一传入上次结果，无新数据时沿用，避免重复训练造成过拟合
        rl_results[game] = run_fn(records, ml_pred, prev_rl_results.get(game), dl_preds)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"{game} 失败: {e}")

# 推送RL模型到Kaggle Dataset
save_seg_scales()   # 基准随模型一起持久化，下次运行沿用同一口径
print(f"\n{'='*50}\n保存PPO模型…\n{'='*50}")
push_rl_dataset()

# ── 写入独立文件 dl_rl.json（不再读取/合并 prediction.json，速度更快）──
out = {
    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'method': 'PPO强化学习（每日增量微调）',
    'state_composition': '原始特征 + 传统ML概率 + 深度学习概率 + LSTM隐层 + Transformer特征 + 遗漏向量',
    'results': rl_results,
}
out_json = json.dumps(out, ensure_ascii=False, indent=2)

if not GH_TOKEN:
    print("\n[DRY RUN] 未配置 GH_TOKEN")
else:
    print("\n推送 dl_rl.json…")
    gh_put('dl_rl.json', out_json, f"PPO每日微调 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("✓ 完成")

print(f"\n✅ 全部完成！{datetime.now().strftime('%Y-%m-%d %H:%M')}")
