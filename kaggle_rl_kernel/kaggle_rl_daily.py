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
import os, json, sys, time, warnings, base64, urllib.request, random, shutil, subprocess
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
WINDOW = 50; SEQ_LEN = 20

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
    """从挂载的 Dataset 加载本周训练好的LSTM/TFM权重"""
    meta_path = f'{DL_MOUNTED}/{game}_meta.json'
    if not os.path.exists(meta_path):
        print(f"  ! 找不到 {game} 的LSTM/TFM权重（先运行 kaggle_lstm_tfm.py 并挂载 {DL_DATASET_SLUG}）")
        return None, None, None
    with open(meta_path) as f: meta = json.load(f)
    # 特征维度校验：若当前特征函数产出维度与保存时不一致（特征工程改了），
    # 直接跳过旧权重，避免运行到forward()时矩阵形状不匹配而崩溃
    if current_feat_dim is not None and meta.get('feat_dim') != current_feat_dim:
        print(f"  ! {game} 的LSTM/TFM权重特征维度({meta.get('feat_dim')})与当前特征工程({current_feat_dim})不一致")
        print(f"    请先重新运行 kaggle_lstm_tfm.py 生成新权重，本次跳过LSTM/TFM隐层")
        return None, None, None
    lstm = LSTMEncoder(meta['feat_dim'], hidden_dim=meta['hidden_dim'], output_dim=meta['n_classes'])
    lstm.load_state_dict(torch.load(f'{DL_MOUNTED}/{game}_lstm.pt', map_location='cpu'))
    lstm.eval()
    tfm = TransformerEncoder(meta['feat_dim'], d_model=meta['d_model'], output_dim=meta['n_classes'])
    tfm.load_state_dict(torch.load(f'{DL_MOUNTED}/{game}_tfm.pt', map_location='cpu'))
    tfm.eval()
    return lstm, tfm, meta

def compute_hidden(model, records, feat_fn, idx, seq_len=SEQ_LEN):
    """计算指定期数idx对应的隐层状态（单次调用，仅用于最后一期推荐）"""
    seq = []
    for j in range(idx-seq_len, idx):
        feat = feat_fn(records, j)
        if feat is None: return None
        seq.append(list(feat.values()))
    x = torch.FloatTensor([seq])
    with torch.no_grad():
        _, h = model(x, return_hidden=True)
    return h.numpy()[0]


def precompute_hidden_all(records, feat_fn, model, seq_len=SEQ_LEN, batch_size=256):
    """
    批量一次性计算所有期数的LSTM/TFM隐层状态（关键性能优化）
    返回 (hidden_array, idx_to_row字典)，训练循环里按idx查表O(1)，不再每步重算
    """
    if model is None:
        return None, {}
    X, idxs = [], []
    for idx in range(seq_len, len(records)):
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
    model.eval()
    hs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[i:i+batch_size])
            _, h = model(xb, return_hidden=True)
            hs.append(h.numpy())
    hs = np.vstack(hs)
    idx_to_row = {idx:i for i,idx in enumerate(idxs)}
    return hs, idx_to_row


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


def cond_aware_picks(scores, n_pick, n_bets, conds, feat_fn, pool_mult=2.4, core_ratio=0.34, cand_n=300):
    """
    条件感知选号：在RL球分数的基础上，叠加"整体特征是否符合ML预测"的校验。

    ── 为什么需要 ──
    RL的状态向量里注入了ML的全部7个目标（和值区间/连号/AC值/组型等），
    神经网络训练时确实看到了这些信息，但推荐生成时只用了它输出的"每个球的分数"，
    模型对"这一注整体该长什么样"的判断在选号环节完全没被检验——
    比如状态里知道该走高和值，但选出的6个球加起来到底是高是低，没人管。
    这里补上这个校验：先按球分生成一批多样化候选，再按符合条件数择优。

    conds:   {目标名: (预测值, 置信度0-1)}
    feat_fn: 传入一注号码，返回 {目标名: 实际特征值}
    """
    order = np.argsort(scores)[::-1]
    pool_size = min(len(scores), max(n_pick + 4, int(round(n_pick * pool_mult))))
    pool = [int(order[i]) + 1 for i in range(pool_size)]
    core_n = max(1, min(n_pick - 1, int(round(n_pick * core_ratio))))
    core, rest = pool[:core_n], pool[core_n:]
    smax = float(np.max(scores)) or 1.0

    # 用轮转起点+不同步长生成一批多样化候选（确定性，不引入随机）
    cands, seen = [], set()
    for start in range(len(rest)):
        for step in (1, 2, 3):
            sel, i, guard = list(core), start, 0
            while len(sel) < n_pick and guard < len(rest) * 3:
                c = rest[i % len(rest)]; i += step; guard += 1
                if c not in sel: sel.append(c)
            if len(sel) == n_pick:
                key = tuple(sorted(sel))
                if key not in seen: seen.add(key); cands.append(sorted(sel))
            if len(cands) >= cand_n: break
        if len(cands) >= cand_n: break

    scored = []
    for c in cands:
        f = feat_fn(c)
        cond_score = sum(w for k, (v, w) in conds.items() if f.get(k) == v)
        pos_score = float(sum(scores[n-1] for n in c)) / (n_pick * abs(smax) + 1e-9)
        scored.append((c, cond_score, pos_score))
    scored.sort(key=lambda x: (-x[1], -x[2]))

    bets, used = [], set()
    for c, cs, ps in scored:
        if len(bets) >= n_bets: break
        if tuple(c) not in used: used.add(tuple(c)); bets.append(c)
    while len(bets) < n_bets and cands:
        for c in cands:
            if len(bets) >= n_bets: break
            if tuple(c) not in used: used.add(tuple(c)); bets.append(c)
        break
    return bets, sorted(core), sorted(pool)


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

def save_last_trained_n(game, n):
    """记录本次训练用到的数据期数，随RL_LOCAL_DIR一起推送到Dataset持久化"""
    os.makedirs(RL_LOCAL_DIR, exist_ok=True)
    try:
        with open(f'{RL_LOCAL_DIR}/{game}_last_trained_n.json', 'w') as f:
            json.dump({'n': n}, f)
    except Exception as e:
        print(f"  ! 记录{game}训练期数失败: {e}")

def carry_over_result(game, prev_result, cur_n, last_n, reason):
    """无新数据时，沿用上次的完整结果（字段结构保持一致，前端渲染无需感知变化）"""
    print(f"  {game} 无新开奖数据（当前{cur_n}期，上次训练时已是{last_n}期），"
          f"跳过训练避免重复数据过拟合")
    save_last_trained_n(game, cur_n)
    if prev_result:
        carried = dict(prev_result)
        carried['skipped'] = True
        carried['carried_over'] = True
        carried['note'] = (carried.get('note','') or '') + f'（{reason}，以上为上次训练结果，本次未重新训练）'
        return carried
    return {'skipped': True, 'games_tested': 0, 'reason': reason,
            'note': f'{reason}，且未找到上次训练结果可沿用（可能是首次运行）'}

def normalize_state_segments(*segments):
    """
    分段独立归一化，替代"整个向量除以自身最大值"的错误做法。
    问题根源：raw特征里像"号码总和均值"这类聚合量级在几百到近千，
    而遗漏值(0-3)、ML概率(0-1)量级很小，如果整个向量共用一个全局最大值做归一化，
    遗漏和ML概率信号会被压缩到接近0，模型实际上"看不到"这些真正能区分号码好坏的关键信息，
    只能学到一些跟具体选哪个球无关的全局统计偏向，导致策略跟状态基本脱钩、
    收敛到一个固定的、看似随意的偏好（这次表现为一直偏向大号）。
    修复：每一段各自独立按自己的最大值缩放到[-1,1]附近，再拼接，
    确保任何一段都不会因为量级差异淹没其它段的信号。
    """
    normed = []
    for seg in segments:
        seg = np.asarray(seg, dtype=np.float32)
        if seg.size == 0:
            normed.append(seg)
            continue
        m = np.abs(seg).max()
        normed.append(seg / (m + 1e-8) if m > 0 else seg)
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
                 tfm_hidden, tfm_idx2row, omit_arr, freq_arr, train_n=KL8_TRAIN_N):
        super().__init__()
        self.records=records; self.feat_fn=feat_fn; self.ml_vec=ml_vec
        self.lstm_hidden=lstm_hidden; self.lstm_idx2row=lstm_idx2row
        self.tfm_hidden=tfm_hidden;   self.tfm_idx2row=tfm_idx2row
        self.omit_arr=omit_arr; self.freq_arr=freq_arr
        self.train_n=train_n
        self.start=SEQ_LEN+30; self.idx=self.start
        sample=feat_fn(records,self.start); feat_dim=len(sample)
        lstm_dim = lstm_hidden.shape[1] if lstm_hidden is not None else 0
        tfm_dim  = tfm_hidden.shape[1]  if tfm_hidden  is not None else 0
        omit_dim = 80; freq_dim = 80
        self.state_dim = feat_dim+len(ml_vec)+lstm_dim+tfm_dim+omit_dim+freq_dim
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

        state = normalize_state_segments(raw,self.ml_vec,lh,th,om,fr)
        # 逐球信号（遗漏+频率，对应state末尾160维）额外加权，让网络有更强动力真正依赖它们
        state = state.copy()
        state[-160:] *= self.PERBALL_WEIGHT
        return np.clip(state, -5, 5)

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
        terminated=(self.idx>=len(self.records)-1)
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
                 tfm_hidden, tfm_idx2row, omit_arr, red_pick_n=SSQ_RED_PICK_N):
        super().__init__()
        self.records=records; self.feat_fn=feat_fn; self.ml_vec=ml_vec
        self.lstm_hidden=lstm_hidden; self.lstm_idx2row=lstm_idx2row
        self.tfm_hidden=tfm_hidden;   self.tfm_idx2row=tfm_idx2row
        self.omit_arr=omit_arr
        self.red_pick_n=red_pick_n
        self.start=SEQ_LEN+30; self.idx=self.start
        sample=feat_fn(records,self.start); feat_dim=len(sample)
        lstm_dim = lstm_hidden.shape[1] if lstm_hidden is not None else 0
        tfm_dim  = tfm_hidden.shape[1]  if tfm_hidden  is not None else 0
        omit_dim = 49   # 33红球+16蓝球遗漏值，已含每个号码的差异化信息
        self.state_dim = feat_dim+len(ml_vec)+lstm_dim+tfm_dim+omit_dim
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

        return normalize_state_segments(raw,self.ml_vec,lh,th,om)

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
        terminated=(self.idx>=len(self.records)-1)
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
                 tfm_hidden, tfm_idx2row, omit_arr):
        super().__init__()
        self.records=records; self.feat_fn=feat_fn; self.ml_vec=ml_vec
        self.lstm_hidden=lstm_hidden; self.lstm_idx2row=lstm_idx2row
        self.tfm_hidden=tfm_hidden;   self.tfm_idx2row=tfm_idx2row
        self.omit_arr=omit_arr
        self.start=SEQ_LEN+5; self.idx=self.start
        sample=feat_fn(records,self.start); feat_dim=len(sample)
        lstm_dim = lstm_hidden.shape[1] if lstm_hidden is not None else 0
        tfm_dim  = tfm_hidden.shape[1]  if tfm_hidden  is not None else 0
        omit_dim = 30
        self.state_dim = feat_dim+len(ml_vec)+lstm_dim+tfm_dim+omit_dim
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
        return normalize_state_segments(raw,self.ml_vec,lh,th,om)

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
        terminated=(self.idx>=len(self.records)-1)
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

def save_ppo(model, game):
    os.makedirs(RL_LOCAL_DIR, exist_ok=True)
    model.save(f'{RL_LOCAL_DIR}/{game}_ppo')
    print(f"  ✓ PPO模型已保存到本地: {RL_LOCAL_DIR}/{game}_ppo.zip")

def push_rl_dataset():
    """把RL_LOCAL_DIR整体推送到Kaggle Dataset"""
    try:
        meta = {"title":"Fucai RL Cache","id":RL_DATASET_ID,"licenses":[{"name":"CC0-1.0"}]}
        with open(f'{RL_LOCAL_DIR}/dataset-metadata.json','w') as f: json.dump(meta,f)
        env = os.environ.copy(); env['KAGGLE_API_TOKEN']=KAGGLE_TOKEN
        for cmd in [
            ['kaggle','datasets','version','-p',RL_LOCAL_DIR,'-m',f'daily-{date.today()}','--dir-mode','tar'],
            ['kaggle','datasets','create','-p',RL_LOCAL_DIR,'--dir-mode','tar'],
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
def run_kl8_daily(records, ml_pred, prev_result=None):
    print(f"\n{'='*50}\n快乐8 PPO 每日增量微调（全号码打分排序，{len(records)}期）\n{'='*50}")

    # 新数据检测：快乐8虽然每天开奖，但手动重复触发时数据是完全相同的，
    # 反复训练会让模型对同一批数据过拟合，这里直接跳过
    last_trained_n = get_last_trained_n('kl8')
    if len(records) <= last_trained_n:
        return carry_over_result('快乐8', prev_result, len(records), last_trained_n,
                                 '本次运行无新开奖数据（可能是当日已训练过或重复手动触发）')

    ml_vec = extract_ml_prob_vec(ml_pred, 'kl8')
    _cur_feat_dim = len(fkl8(records, len(records)-1) or {})
    lstm, tfm, meta = load_lstm_tfm('kl8', current_feat_dim=_cur_feat_dim)

    print("  批量预计算 LSTM/TFM 隐层状态…")
    t0 = time.time()
    lstm_hidden, lstm_idx2row = precompute_hidden_all(records, fkl8, lstm)
    tfm_hidden,  tfm_idx2row  = precompute_hidden_all(records, fkl8, tfm)
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [LSTM隐层] {'✓已加载' if lstm_hidden is not None else '✗未加载（回退为0向量）'}  [TFM隐层] {'✓已加载' if tfm_hidden is not None else '✗未加载（回退为0向量）'}")

    print("  批量预计算遗漏向量…")
    t0 = time.time()
    omit_arr = precompute_omission_kl8(records)
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [遗漏向量] ✓已加载（80维，覆盖全部号码）")

    print("  批量预计算近30期频率向量…")
    t0 = time.time()
    freq_arr = precompute_freq_kl8(records, window=30)
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [频率向量] ✓已加载（80维，第二个逐球差异化信号）")

    print(f"  [状态向量组成] 原始统计特征{_cur_feat_dim}维 + 传统ML概率{len(ml_vec)}维 + LSTM隐层 + TFM隐层 + 遗漏80维 + 频率80维（逐球信号×2加权）")

    def make_env():
        return IntegratedKL8Env(records, fkl8, ml_vec,
                                lstm_hidden, lstm_idx2row, tfm_hidden, tfm_idx2row, omit_arr, freq_arr)
    vec_env = make_vec_env(make_env, n_envs=4)

    model = load_ppo('kl8')
    is_new = model is None
    t0 = time.time()
    if not is_new:
        try:
            model.set_env(vec_env)
        except Exception as e:
            print(f"  ! 旧PPO模型与当前环境结构不兼容（{e}），改为全新训练")
            model = None; is_new = True
    if is_new:
        print("  首次训练（20万步，全80球连续打分排序，兼顾全覆盖与可学习性）…")
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=512, batch_size=128,
                    n_epochs=10, gamma=0.95, gae_lambda=0.95, clip_range=0.2, ent_coef=0.05,
                    verbose=0, device='cpu')
        model.learn(total_timesteps=200000, progress_bar=False)
    else:
        print("  增量微调（2万步，基于最新数据）…")
        model.learn(total_timesteps=20000, reset_num_timesteps=False, progress_bar=False)
    print(f"    PPO训练完成，耗时 {time.time()-t0:.1f}s")

    save_ppo(model, 'kl8')

    def build_state(idx):
        feat = fkl8(records, idx)
        if feat is None: return None
        raw = np.array(list(feat.values()),dtype=np.float32)
        lh = lstm_hidden[lstm_idx2row[idx]] if (lstm_hidden is not None and idx in lstm_idx2row) else np.zeros(lstm_hidden.shape[1] if lstm_hidden is not None else 0,dtype=np.float32)
        th = tfm_hidden[tfm_idx2row[idx]]   if (tfm_hidden  is not None and idx in tfm_idx2row)  else np.zeros(tfm_hidden.shape[1] if tfm_hidden is not None else 0,dtype=np.float32)
        om = omit_arr[idx]
        fr = freq_arr[idx]
        state = normalize_state_segments(raw,ml_vec,lh,th,om,fr)
        state = state.copy()
        state[-160:] *= IntegratedKL8Env.PERBALL_WEIGHT   # 跟训练环境保持一致的逐球信号加权
        return np.clip(state, -5, 5)

    # 回测：同一次预测，同时评估选四/五/六/九/十全部玩法（几乎零额外开销，只是截取不同长度TopN）
    start = max(SEQ_LEN+30, len(records)-30)
    play_sizes = [4,5,6,9,10]
    net_by_size = {n: 0.0 for n in play_sizes}
    hit_by_size = {n: 0.0 for n in play_sizes}
    games=0
    for idx in range(start, len(records)-1):
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
    idx = len(records)-1
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
            if _kl8_conds:
                _b, _c, _p = cond_aware_picks(base_action, _n, _cnt, _kl8_conds, _kl8_cond_feats)
            else:
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
            print(f"  [ML条件校验] 共{len(_kl8_conds)}条预测条件，选六3注平均符合{_cavg:.1f}条")

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
            'backtest_by_play':backtest_by_play,   # 选四/五/六/九/十 各玩法回测对比
            'best_play_n':best_play_n,             # 回测表现最好的玩法（仅供参考，不代表未来）
            'ref_info':ref_info,            # 参考信息：遗漏/频率/ML预测，仅供理解RL判断依据，不影响排序
            'is_first_train':is_new,
            'note':f'以RL自身综合判断为主排序（已融合ML/DL/遗漏/频率/走势特征），选六净收益{avg_net}元/期，遗漏/频率/ML预测仅作参考展示'}


def run_ssq_daily(records, ml_pred, prev_result=None):
    print(f"\n{'='*50}\n双色球 PPO 每日增量微调（红球33全量打分+蓝球，{len(records)}期）\n{'='*50}")

    # 开奖日感知：双色球只在周二/四/日开奖，其余4天没有新数据；
    # 手动重复触发时也会命中这个检查，避免同一批数据被反复训练导致过拟合
    last_trained_n = get_last_trained_n('ssq')
    if len(records) <= last_trained_n:
        return carry_over_result('双色球', prev_result, len(records), last_trained_n,
                                 '双色球周二/四/日开奖，本次运行无新开奖数据')

    ml_vec = extract_ml_prob_vec(ml_pred, 'ssq')
    _cur_feat_dim = len(fssq(records, len(records)-1) or {})
    lstm, tfm, meta = load_lstm_tfm('ssq', current_feat_dim=_cur_feat_dim)

    print("  批量预计算 LSTM/TFM 隐层状态…")
    t0 = time.time()
    lstm_hidden, lstm_idx2row = precompute_hidden_all(records, fssq, lstm)
    tfm_hidden,  tfm_idx2row  = precompute_hidden_all(records, fssq, tfm)
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [LSTM隐层] {'✓已加载' if lstm_hidden is not None else '✗未加载（回退为0向量）'}  [TFM隐层] {'✓已加载' if tfm_hidden is not None else '✗未加载（回退为0向量）'}")

    print("  批量预计算遗漏向量…")
    t0 = time.time()
    omit_arr = precompute_omission_ssq(records)
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [遗漏向量] ✓已加载（49维：33红球+16蓝球）")

    print(f"  [状态向量组成] 原始统计特征{_cur_feat_dim}维 + 传统ML概率{len(ml_vec)}维 + LSTM隐层 + TFM隐层 + 遗漏49维")

    def make_env():
        return IntegratedSSQEnv(records, fssq, ml_vec,
                                lstm_hidden, lstm_idx2row, tfm_hidden, tfm_idx2row, omit_arr)
    vec_env = make_vec_env(make_env, n_envs=4)

    model = load_ppo('ssq')
    is_new = model is None
    t0 = time.time()
    if not is_new:
        try:
            model.set_env(vec_env)
        except Exception as e:
            print(f"  ! 旧PPO模型与当前环境结构不兼容（{e}），改为全新训练")
            model = None; is_new = True
    if is_new:
        print("  首次训练（15万步，红球33全量打分+蓝球联合优化）…")
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=512, batch_size=128,
                    n_epochs=10, gamma=0.95, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
                    verbose=0, device='cpu')
        model.learn(total_timesteps=150000, progress_bar=False)
    else:
        print("  增量微调（1.5万步）…")
        model.learn(total_timesteps=15000, reset_num_timesteps=False, progress_bar=False)
    print(f"    PPO训练完成，耗时 {time.time()-t0:.1f}s")

    save_ppo(model, 'ssq')

    def build_state(idx):
        feat = fssq(records, idx)
        if feat is None: return None
        raw = np.array(list(feat.values()),dtype=np.float32)
        lh = lstm_hidden[lstm_idx2row[idx]] if (lstm_hidden is not None and idx in lstm_idx2row) else np.zeros(lstm_hidden.shape[1] if lstm_hidden is not None else 0,dtype=np.float32)
        th = tfm_hidden[tfm_idx2row[idx]]   if (tfm_hidden  is not None and idx in tfm_idx2row)  else np.zeros(tfm_hidden.shape[1] if tfm_hidden is not None else 0,dtype=np.float32)
        om = omit_arr[idx]
        return normalize_state_segments(raw,ml_vec,lh,th,om)

    # 回测最近30期：红球命中数分布 + 蓝球命中率
    start=max(SEQ_LEN+30, len(records)-30)
    total=0; blue_correct=0; red_hit_dist={0:0,1:0,2:0,3:0,4:0,5:0,6:0}
    for idx in range(start, len(records)-1):
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
    print(f"  回测（近{total}期）：红球平均命中{avg_red_hit}个  蓝球准确率{blue_acc}%（随机基准6.25%）")

    # 今日推荐：以RL自己的判断为主，红球排序滑动窗口切分成6注，遗漏/ML预测仅作参考展示
    idx = len(records)-1
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

        if _ssq_conds:
            red_top6, red_core, red_pool = cond_aware_picks(red_scores, 6, 6, _ssq_conds, _ssq_cond_feats)
            _cavg = sum(len([1 for k,(v,w) in _ssq_conds.items()
                             if _ssq_cond_feats(b).get(k)==v]) for b in red_top6) / max(len(red_top6),1)
            print(f"  [ML条件校验] 共{len(_ssq_conds)}条预测条件，6注平均符合{_cavg:.1f}条")
        else:
            red_top6, red_core, red_pool = diverse_picks(red_scores, 6, 6)
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
            'ppo_groups':groups,'ref_info':ref_info,
            'red_core':red_core_info,'red_pool':red_pool_info,
            'is_first_train':is_new,
            'note':f'以RL自身综合判断为主排序（已融合ML/DL/遗漏/走势特征），红球平均命中{avg_red_hit}个，蓝球准确率{blue_acc}%，遗漏/ML预测仅作参考展示'}


def run_3d_daily(records, ml_pred, prev_result=None):
    print(f"\n{'='*50}\n福彩3D PPO 每日增量微调（{len(records)}期）\n{'='*50}")

    # 新数据检测：避免重复手动触发时拿完全相同的数据反复训练导致过拟合
    last_trained_n = get_last_trained_n('3d')
    if len(records) <= last_trained_n:
        return carry_over_result('福彩3D', prev_result, len(records), last_trained_n,
                                 '本次运行无新开奖数据（可能是当日已训练过或重复手动触发）')

    ml_vec = extract_ml_prob_vec(ml_pred, '3d')
    _cur_feat_dim = len(f3d(records, len(records)-1) or {})
    lstm, tfm, meta = load_lstm_tfm('3d', current_feat_dim=_cur_feat_dim)

    print("  批量预计算 LSTM/TFM 隐层状态…")
    t0 = time.time()
    lstm_hidden, lstm_idx2row = precompute_hidden_all(records, f3d, lstm)
    tfm_hidden,  tfm_idx2row  = precompute_hidden_all(records, f3d, tfm)
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [LSTM隐层] {'✓已加载' if lstm_hidden is not None else '✗未加载（回退为0向量）'}  [TFM隐层] {'✓已加载' if tfm_hidden is not None else '✗未加载（回退为0向量）'}")

    print("  批量预计算遗漏向量…")
    t0 = time.time()
    omit_arr = precompute_omission_3d(records)
    print(f"    完成，耗时 {time.time()-t0:.1f}s  [遗漏向量] ✓已加载（30维：百十个位各10个数字）")

    print(f"  [状态向量组成] 原始统计特征{_cur_feat_dim}维 + 传统ML概率{len(ml_vec)}维 + LSTM隐层 + TFM隐层 + 遗漏30维")

    def make_env():
        return Integrated3DEnv(records, f3d, ml_vec,
                               lstm_hidden, lstm_idx2row, tfm_hidden, tfm_idx2row, omit_arr)
    vec_env = make_vec_env(make_env, n_envs=4)

    model = load_ppo('3d')
    is_new = model is None
    t0 = time.time()
    if not is_new:
        try:
            model.set_env(vec_env)
        except Exception as e:
            print(f"  ! 旧PPO模型与当前环境结构不兼容（{e}），改为全新训练")
            model = None; is_new = True
    if is_new:
        print("  首次训练（10万步，MultiDiscrete([10,10,10])共1000种组合）…")
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=256, batch_size=64,
                    n_epochs=8, gamma=0.9, gae_lambda=0.9, clip_range=0.2, ent_coef=0.03,
                    verbose=0, device='cpu')
        model.learn(total_timesteps=100000, progress_bar=False)
    else:
        print("  增量微调（1万步）…")
        model.learn(total_timesteps=10000, reset_num_timesteps=False, progress_bar=False)
    print(f"    PPO训练完成，耗时 {time.time()-t0:.1f}s")

    save_ppo(model, '3d')

    def build_state(idx):
        feat = f3d(records, idx)
        if feat is None: return None
        raw = np.array(list(feat.values()),dtype=np.float32)
        lh = lstm_hidden[lstm_idx2row[idx]] if (lstm_hidden is not None and idx in lstm_idx2row) else np.zeros(lstm_hidden.shape[1] if lstm_hidden is not None else 0,dtype=np.float32)
        th = tfm_hidden[tfm_idx2row[idx]]   if (tfm_hidden  is not None and idx in tfm_idx2row)  else np.zeros(tfm_hidden.shape[1] if tfm_hidden is not None else 0,dtype=np.float32)
        om = omit_arr[idx]
        return normalize_state_segments(raw,ml_vec,lh,th,om)

    # 回测最近30期：统计位命中数分布 + 全中次数
    start=max(SEQ_LEN+5, len(records)-30); total=0
    match_dist={0:0,1:0,2:0,3:0}
    for idx in range(start, len(records)-1):
        state = build_state(idx)
        if state is None: continue
        action,_ = model.predict(state, deterministic=True)
        pred=[int(action[0]),int(action[1]),int(action[2])]
        actual=records[idx]['digits']
        m = sum(1 for i in range(3) if pred[i]==actual[i])
        match_dist[m]+=1; total+=1
    exact_hit_rate = round(match_dist[3]/total*100,2) if total else 0
    avg_match = round(sum(k*v for k,v in match_dist.items())/total,2) if total else 0

    idx=len(records)-1; state=build_state(idx)
    groups=[]; pos_candidates=[]
    if state is not None:
        # 明确提取百/十/个位各自的完整概率分布（而非随机采样撞运气），
        # 用联合概率排序生成6注真正的次优组合，能说清楚"这是第几优的组合"
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
            # 后3注：从27种组合里补足，优先选符合ML条件多的（同分再看联合概率）
            all27 = sorted(
                (([b, s, g], pb*ps*pg)
                 for b, pb in top3[0] for s, ps in top3[1] for g, pg in top3[2]),
                key=lambda x: (-sum(w for k,(v,w) in _d3_conds.items() if _d3_feats(x[0]).get(k)==v),
                               -x[1]))
            for c, pr in all27:
                if len(picked) >= 6: break
                if tuple(c) not in seen:
                    picked.append((c, pr)); seen.add(tuple(c))
            picked.sort(key=lambda x: -x[1])
            groups = [c for c, _ in picked]
            if _d3_conds:
                _cavg = sum(len([1 for k,(v,w) in _d3_conds.items()
                                 if _d3_feats(g).get(k)==v]) for g in groups) / max(len(groups),1)
                print(f"  [ML条件校验] 共{len(_d3_conds)}条预测条件，6注平均符合{_cavg:.1f}条")
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
        except Exception as e:
            print(f"  ! 提取概率分布失败({e})，改用确定性预测兜底")
            action,_ = model.predict(state, deterministic=True)
            groups = [[int(action[0]),int(action[1]),int(action[2])]]

        # 不足6注时（比如候选池不够8种或提取失败），用确定性预测补齐
        while len(groups)<6:
            if not groups:
                action,_ = model.predict(state, deterministic=True)
                groups.append([int(action[0]),int(action[1]),int(action[2])])
            else:
                groups.append(groups[-1])
    pred = groups[0] if groups else None  # 兼容旧字段：主推荐仍取第一注（联合概率最高的组合）

    # 记录本次训练时的期数，供下次运行判断是否有新数据
    save_last_trained_n('3d', len(records))

    return {'games_tested':total,'match_distribution':match_dist,
            'avg_match_digits':avg_match,'exact_hit_rate_pct':exact_hit_rate,
            'ppo_pred':pred,'ppo_groups':groups,
            'pos_candidates':pos_candidates,   # 每位Top3候选及其概率，供前端展示
            'is_first_train':is_new,
            'note':f'PPO给出百/十/个位各3个候选，6注采用"轮转+择优"确保每个候选都参与组合（避免联合概率导致某位被单一数字垄断），近{total}期平均命中{avg_match}位，全中率{exact_hit_rate}%（随机基准0.1%）'}

# ══════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════
print(f"\n{'#'*55}\nPPO 强化学习 每日增量微调  {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'#'*55}")

raw = gh_raw('history.json')
if not raw: print("失败"); sys.exit(1)
history = json.loads(raw)

# 读取 prediction.json 取ML概率向量（RL状态的一部分）
raw_ml = gh_raw('prediction.json')
ml_preds = {}
if raw_ml:
    try: ml_preds = json.loads(raw_ml).get('predictions', {})
    except Exception: pass

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
        rl_results[game] = run_fn(records, ml_pred, prev_rl_results.get(game))
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"{game} 失败: {e}")

# 推送RL模型到Kaggle Dataset
print(f"\n{'='*50}\n保存PPO模型…\n{'='*50}")
push_rl_dataset()

# ── 写入独立文件 dl_rl.json（不再读取/合并 prediction.json，速度更快）──
out = {
    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'method': 'PPO强化学习（每日增量微调）',
    'state_composition': '原始特征 + ML概率向量 + LSTM隐层 + Transformer特征 + 遗漏向量',
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
