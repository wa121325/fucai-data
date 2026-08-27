"""
福彩 LSTM+Transformer 深度学习训练 — 每周运行一次
kaggle_lstm_tfm.py

功能：
  1. 全量训练 LSTM + Transformer（时序特征提取）
  2. 保存模型权重 + 各期隐层状态到 Kaggle Dataset "fucai-dl-cache"
     （供每天运行的 kaggle_rl_daily.py 加载使用）
  3. 更新 prediction.json 的 dl_result.lstm_tfm 字段

Kaggle Secrets: GH_TOKEN, GH_REPO, KAGGLE_TOKEN
Kaggle 设置: GPU + Internet 开启
"""
import os, json, sys, time, warnings, base64, urllib.request, random, shutil, subprocess
from datetime import datetime, date
from collections import Counter
warnings.filterwarnings('ignore')

_secrets_client = None
_secrets_client_ready = False

def _get_secrets_client(retries=5, delay=4):
    """
    只创建一次 UserSecretsClient 实例并复用。
    每次 UserSecretsClient() 都要重新握手，是导致偶发连接失败的根源，
    改成全局只连一次、所有 Secret 共用这个连接，大幅降低失败率。
    """
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
    """
    从挂载的私有Dataset读取secrets.json（绕过Kaggle Secrets的API推送限制）
    根本原因：kaggle kernels push（API方式）无法传递Kaggle Secrets，这是Kaggle官方已知限制
    （GitHub issue Kaggle/kaggle-cli#582），Dataset挂载则不受此限制影响
    """
    try:
        with open(SECRETS_DATASET_MOUNT) as f:
            return json.load(f)
    except Exception:
        return {}

def get_secret(name, retries=3, delay=3):
    global _dataset_secrets
    # 方式1：交互式Kaggle Secrets（手动Save&Run有效，API push时通常无效）
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

    # 方式2：挂载的私有Dataset secrets.json（API push场景下的正确方式）
    if _dataset_secrets is None:
        _dataset_secrets = _load_secrets_from_dataset()
    if name in _dataset_secrets and _dataset_secrets[name]:
        print(f"  [Secret] {name} 从 fucai-secrets Dataset 读取成功")
        return _dataset_secrets[name]

    # 方式3：环境变量兜底
    return os.environ.get(name, '')

# ── 把你的 Token 填在这里（Kaggle Secrets 不稳定时的兜底）──
_HARDCODED_GH_TOKEN = ''  # 不要在这里写Token！写了会被GitHub自动吊销，必须用Kaggle Secrets      # ← 新的 GitHub Token
_HARDCODED_GH_REPO  = 'wa121325/fucai-data'
_HARDCODED_KAGGLE_TOKEN = 'KGAT_0847d8a3c8619a4db2ff2c7c3e9e824f'

GH_TOKEN = get_secret('GH_TOKEN') or get_secret('gh_token') or _HARDCODED_GH_TOKEN
GH_REPO  = get_secret('GH_REPO')  or get_secret('gh_repo')  or _HARDCODED_GH_REPO
KAGGLE_TOKEN = get_secret('KAGGLE_TOKEN') or get_secret('kaggle_token') or _HARDCODED_KAGGLE_TOKEN
print(f"GitHub: {GH_REPO}  GH_TOKEN: {'✓('+str(len(GH_TOKEN))+')' if GH_TOKEN else '✗'}")

try:
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    def _cuda_actually_works():
        """有些Kaggle环境torch.cuda.is_available()=True但实际算子跑不了，做个真实测试"""
        if not torch.cuda.is_available():
            return False
        try:
            x = torch.randn(4, 4, device='cuda')
            _ = (x @ x).sum().item()
            return True
        except Exception as e:
            print(f"  CUDA测试失败，自动降级到CPU: {e}")
            return False

    DEVICE = torch.device('cuda' if _cuda_actually_works() else 'cpu')
    print(f"PyTorch ✓ 设备:{DEVICE}")
except ImportError:
    print("PyTorch ✗"); sys.exit(1)

import numpy as np

DATASET_SLUG = 'fucai-dl-cache'
DATASET_ID   = f'megskfdbbskeb/{DATASET_SLUG}'
LOCAL_DIR    = '/kaggle/working/dl_cache'
MOUNTED_DIR  = f'/kaggle/input/{DATASET_SLUG}'
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
    req = urllib.request.Request(url, headers={'Cache-Control':'no-cache','User-Agent':'lstm-bot'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r: return r.read().decode('utf-8')
    except Exception as e: print(f"  gh_raw({path}) 失败: {e}"); return None

def gh_put(path, content_str, message):
    url = f'https://api.github.com/repos/{GH_REPO}/contents/{path}'
    sha = None
    try:
        req = urllib.request.Request(url, headers={'Authorization':f'token {GH_TOKEN}',
            'Accept':'application/vnd.github.v3+json','User-Agent':'lstm-bot'})
        with urllib.request.urlopen(req, timeout=15) as r: sha = json.loads(r.read()).get('sha')
    except Exception: pass
    data = {'message':message,'branch':'main','content':base64.b64encode(content_str.encode()).decode()}
    if sha: data['sha'] = sha
    req = urllib.request.Request(url, data=json.dumps(data).encode(), method='PUT', headers={
        'Authorization':f'token {GH_TOKEN}','Content-Type':'application/json',
        'Accept':'application/vnd.github.v3+json','User-Agent':'lstm-bot'})
    with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())

# ══════════════════════════════════════════════════════
#  特征工程
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


def build_seq_dataset(records, feat_fn, target_fn, seq_len=SEQ_LEN):
    X_list, y_list = [], []
    for i in range(seq_len, len(records)):
        seq = []; valid = True
        for j in range(i-seq_len, i):
            feat = feat_fn(records, j)
            if feat is None: valid=False; break
            seq.append(list(feat.values()))
        if not valid: continue
        tgt = target_fn(records[i])
        if tgt is None: continue
        X_list.append(seq); y_list.append(tgt)
    if not X_list: return None, None
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

# ══════════════════════════════════════════════════════
#  模型
# ══════════════════════════════════════════════════════
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=10, dropout=0.3):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim,64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64,output_dim))
    def forward(self, x, return_hidden=False):
        out,(h_n,_) = self.lstm(x)
        last = self.norm(out[:,-1,:])
        logits = self.head(last)
        return (logits,last) if return_hidden else logits

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, output_dim=10, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(d_model,32), nn.GELU(), nn.Linear(32,output_dim))
    def forward(self, x, return_hidden=False):
        x = self.proj(x); x = self.transformer(x)
        pooled = self.pool(x.transpose(1,2)).squeeze(-1)
        logits = self.head(pooled)
        return (logits,pooled) if return_hidden else logits

def build_predict_seq(records, feat_fn, seq_len=SEQ_LEN):
    """
    构造"预测下一期"用的输入序列：取最新的 seq_len 期特征。

    ── 为什么需要单独构造 ──
    build_seq_dataset 的最后一条样本 X[-1]，特征取的是 records[N-1-seq_len : N-1]，
    对应的答案是 records[N-1]——也就是【最后一期，已经开出来了】。
    如果直接拿 X[-1] 去预测，模型输出的是对已知结果的"预测"，
    表现为推荐号码与最新开奖高度重合，看起来很准，实际毫无预测价值。
    这里改成取 records[N-seq_len : N]（含最新一期），
    模型输出的才是对下一期（尚未开奖）的预测。
    """
    if len(records) < seq_len: return None
    seq = []
    for j in range(len(records)-seq_len, len(records)):
        feat = feat_fn(records, j)
        if feat is None: return None
        seq.append(list(feat.values()))
    return np.array([seq], dtype=np.float32)


def train_encoder(model_ctor, X, y, epochs=60, lr=5e-4, batch_size=32, holdout_n=50,
                   warm_start_path=None, warm_start_epochs=10, warm_start_lr=1e-4,
                   warm_start_window=250, predict_X=None):
    """
    分两阶段训练，避免"用训练数据本身当回测题"造成虚假高准确率：
    （神经网络训练60轮后几乎能背下训练集，若直接拿训练集本身算准确率，
     报出99%+纯属正常的过拟合记忆现象，跟有没有学到真实规律毫无关系）

    1) 回测模型：只用 X[:-holdout_n] 训练，在模型真正没见过的 X[-holdout_n:] 上评估准确率
       ⚠️ 这个模型必须从头训练，不能热启动——如果加载"用过全部历史数据（含当前holdout部分）
       训练出的旧权重"去初始化，等于让回测模型提前偷看了它要被考的题目，回测准确率会失真。
    2) 生产模型：优先热启动微调，用于提取隐层状态（供RL使用）和预测下一期
       热启动时只用最近 warm_start_window 期数据（滑动窗口增量），而非全部历史：
       - 只用"今天新增的1条数据"微调 → 梯度被单个样本主导，会把历史学到的规律冲掉（灾难性遗忘）
       - 每次都用全部几千期从头学 → 太慢，也没必要（上次的权重已经吸收了大部分历史信息）
       - 折中：用最近一段窗口（有一定样本量支撑梯度方向，不会被单样本带偏，
         同时训练量远小于全部历史，且天然更侧重近期统计规律）
       没有旧权重可用时（首次运行/特征维度变了），才回退到全部历史数据训练。

    model_ctor: 无参construct函数，每次调用返回一个全新的未训练模型实例
    warm_start_path: 若提供且文件存在，生产模型会从这份权重继续训练（增量微调）
    """
    n = len(X)
    holdout_n = min(holdout_n, max(5, n//5))   # 数据量小时按比例缩减holdout，避免训练集过小
    split = max(1, n - holdout_n)

    def _train_one(m, Xtr, ytr, ep, learning_rate=lr):
        m = m.to(DEVICE)
        Xt = torch.FloatTensor(Xtr).to(DEVICE); yt = torch.LongTensor(ytr).to(DEVICE)
        loader = DataLoader(TensorDataset(Xt,yt), batch_size, shuffle=True)
        opt = optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(ep,1))
        crit = nn.CrossEntropyLoss()
        m.train()
        for e in range(ep):
            for xb,yb in loader:
                opt.zero_grad(); loss=crit(m(xb),yb); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
            sch.step()
        return m

    # ── 1) 回测模型：必须从头训练，不能加载旧权重（避免评估时偷看未来数据）──
    bt_model = _train_one(model_ctor(), X[:split], y[:split], epochs)
    bt_model.eval()
    with torch.no_grad():
        Xte = torch.FloatTensor(X[split:]).to(DEVICE)
        preds = bt_model(Xte).argmax(dim=1).cpu().numpy()
    y_holdout = y[split:]
    acc = round(float((preds==y_holdout).mean())*100,1) if len(y_holdout)>0 else 0.0

    # 基线：训练集里出现最多的那一类，用来判断准确率是不是只是"蒙对"
    from collections import Counter as Ctr
    y_train = y[:split]
    if len(y_train)>0 and len(y_holdout)>0:
        majority = Ctr(y_train.tolist()).most_common(1)[0][0]
        baseline_acc = round(float((y_holdout==majority).mean())*100,1)
    else:
        baseline_acc = 0.0

    # ── 2) 生产模型：优先"滑动窗口热启动微调"，找不到旧权重才全量训练 ──
    is_warm_start = False
    prod_new = model_ctor()
    if warm_start_path and os.path.exists(warm_start_path):
        try:
            prod_new.load_state_dict(torch.load(warm_start_path, map_location='cpu'))
            is_warm_start = True
            win = min(warm_start_window, n)
            print(f"      ✓ 加载上次训练权重，滑动窗口热启动微调（近{win}期，{warm_start_epochs}轮，lr={warm_start_lr}）")
        except Exception as e:
            print(f"      ! 加载旧权重失败（{e}），改为全量训练")
    if is_warm_start:
        win = min(warm_start_window, n)
        prod_model = _train_one(prod_new, X[-win:], y[-win:], warm_start_epochs, learning_rate=warm_start_lr)
    else:
        prod_model = _train_one(prod_new, X, y, epochs)
    prod_model.eval()
    hidden_states=[]
    with torch.no_grad():
        Xt_full = torch.FloatTensor(X).to(DEVICE)
        for i in range(0,len(Xt_full),batch_size):
            xb=Xt_full[i:i+batch_size]; _,h=prod_model(xb,return_hidden=True)
            hidden_states.append(h.cpu().numpy())
    hidden_states=np.vstack(hidden_states)
    with torch.no_grad():
        # 用专门构造的"含最新一期"的序列做预测，而不是 Xt_full[-1]。
        # 后者对应的答案是最后一期（已开奖），拿它预测等于复述已知结果。
        if predict_X is not None:
            _px = torch.FloatTensor(predict_X).to(DEVICE)
            logits, last_h = prod_model(_px, return_hidden=True)
        else:
            logits, last_h = prod_model(Xt_full[-1:], return_hidden=True)
        probs=torch.softmax(logits,dim=1)[0].cpu().numpy()

    return prod_model, hidden_states, last_h.cpu().numpy()[0], probs, acc, baseline_acc, is_warm_start

# ══════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════
print(f"\n{'#'*55}\nLSTM+Transformer 每周训练  {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'#'*55}")

print("\n读取 history.json…")
raw = gh_raw('history.json')
if not raw: print("失败"); sys.exit(1)
history = json.loads(raw)

os.makedirs(LOCAL_DIR, exist_ok=True)
dl_results = {}

def _3d_group_type(digits):
    b,s,g = digits
    is_triplet = (b==s==g)
    is_group3  = (b==s or s==g or b==g) and not is_triplet
    return 0 if is_triplet else (1 if is_group3 else 2)   # 0=豹子 1=组三 2=组六

def _3d_span_grp(digits):
    span = max(digits) - min(digits)
    return 0 if span<=3 else(1 if span<=6 else 2)

def _3d_road_dom(digits):
    road = [d%3 for d in digits]
    return max(set(road), key=road.count)   # 三位数字里012路哪个占多数

def _3d_arith(digits):
    s3 = sorted(digits)
    return int((s3[1]-s3[0])==(s3[2]-s3[1]) and s3[2]-s3[0]>0)

def _ssq_consec(red):
    sred = sorted(red)
    return sum(1 for i in range(len(sred)-1) if sred[i+1]-sred[i]==1)

def _ssq_ac_grp(red):
    sred=sorted(red); diffs=set()
    for i in range(len(sred)):
        for j in range(i+1,len(sred)): diffs.add(sred[j]-sred[i])
    ac=len(diffs)-(len(sred)-1)
    return 0 if ac<=2 else(1 if ac<=5 else 2)

def _ssq_zone_dom(red):
    z1=sum(1 for x in red if x<=11); z2=sum(1 for x in red if 12<=x<=22); z3=sum(1 for x in red if x>=23)
    return int(np.argmax([z1,z2,z3]))

def _ssq_gap_grp(red):
    sred=sorted(red)
    mg=max(sred[i+1]-sred[i] for i in range(len(sred)-1)) if len(sred)>1 else 0
    return 0 if mg<=5 else(1 if mg<=10 else 2)

def _kl8_big_grp(nums):
    big=sum(1 for x in nums if x>40)
    return 0 if big<9 else(1 if big<=11 else 2)

def _kl8_five_dom(nums):
    five=[sum(1 for x in nums if lo<=x<=hi) for lo,hi in [(1,16),(17,32),(33,48),(49,64),(65,80)]]
    return int(np.argmax(five))

def _kl8_consec_grp(nums):
    sn=sorted(nums); cg=0; inc=False
    for i in range(len(sn)-1):
        if sn[i+1]-sn[i]==1:
            if not inc: cg+=1; inc=True
        else: inc=False
    return 0 if cg==0 else(1 if cg<=2 else 2)

def _kl8_range_grp(nums):
    sn=sorted(nums); rng=sn[-1]-sn[0]
    return 0 if rng<60 else(1 if rng<70 else 2)

configs = {
    '3d':  (f3d,  {'sum_grp':lambda r:0 if sum(r['digits'])<=9 else(1 if sum(r['digits'])<=17 else 2),
                    'group_type':lambda r:_3d_group_type(r['digits']),
                    'odd':lambda r:sum(1 for x in r['digits'] if x%2!=0),
                    'big':lambda r:sum(1 for x in r['digits'] if x>=5),
                    'span_grp':lambda r:_3d_span_grp(r['digits']),
                    'road_dom':lambda r:_3d_road_dom(r['digits']),
                    'arith':lambda r:_3d_arith(r['digits'])}),
    'ssq': (fssq, {'odd':lambda r:sum(1 for x in r['red'] if x%2!=0),
                    'sum_grp':lambda r:0 if sum(r['red'])<70 else(1 if sum(r['red'])<100 else 2),
                    'ac_grp':lambda r:_ssq_ac_grp(r['red']),
                    'red_zone_dom':lambda r:_ssq_zone_dom(r['red']),
                    'gap_grp':lambda r:_ssq_gap_grp(r['red']),
                    'big':lambda r:sum(1 for x in r['red'] if x>16),
                    'consec':lambda r:_ssq_consec(r['red'])}),
    'kl8': (fkl8, {'odd_grp':lambda r:0 if sum(1 for x in r['numbers'] if x%2!=0)<9 else(1 if sum(1 for x in r['numbers'] if x%2!=0)<=11 else 2),
                    'zone_dom':lambda r:int(np.argmax([sum(1 for x in r['numbers'] if lo<=x<=hi) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]])),
                    'tot_grp':lambda r:0 if sum(r['numbers'])<640 else(1 if sum(r['numbers'])<820 else 2),
                    'big_grp':lambda r:_kl8_big_grp(r['numbers']),
                    'five_dom':lambda r:_kl8_five_dom(r['numbers']),
                    'consec_grp':lambda r:_kl8_consec_grp(r['numbers']),
                    'range_grp':lambda r:_kl8_range_grp(r['numbers'])}),
}

for game, (feat_fn, targets) in configs.items():
    records = history.get(game, [])
    if not isinstance(records,list) or len(records)<65:
        print(f"\n{game}: 数据不足，跳过"); continue
    print(f"\n{'='*50}\n{game}（{len(records)}期）\n{'='*50}")

    game_results = {}
    lstm_hidden_all=None; tfm_hidden_all=None

    for tname, tgt_fn in targets.items():
        print(f"\n  [{tname}]")
        X,y = build_seq_dataset(records, feat_fn, tgt_fn)
        if X is None or len(X)<40: continue
        nc=len(set(y.tolist())); fd=X.shape[2]
        # 预测下一期专用序列（含最新一期），避免用 X[-1] 去"预测"已开出的最后一期
        predict_X = build_predict_seq(records, feat_fn)

        # 每个目标都检查自己的权重能否热启动（之前只有第一个目标能，其余永远从零训练）
        lstm_warm_path = tfm_warm_path = None
        prev_meta_path = f'{MOUNTED_DIR}/{game}_{tname}_meta.json'
        if os.path.exists(prev_meta_path):
            try:
                with open(prev_meta_path) as f: prev_meta = json.load(f)
                if prev_meta.get('feat_dim') == fd and prev_meta.get('n_classes') == nc:
                    lstm_warm_path = f'{MOUNTED_DIR}/{game}_{tname}_lstm.pt'
                    tfm_warm_path  = f'{MOUNTED_DIR}/{game}_{tname}_tfm.pt'
                else:
                    print(f"    ! 上次权重维度({prev_meta.get('feat_dim')},{prev_meta.get('n_classes')})与当前({fd},{nc})不一致，改为全量训练")
            except Exception as e:
                print(f"    ! 读取上次meta失败({e})，改为全量训练")
        elif os.path.exists(f'{MOUNTED_DIR}/{game}_meta.json') and lstm_hidden_all is None:
            # 兼容旧版本：旧Dataset只有 {game}_meta.json（第一个目标的），
            # 首次升级时让第一个目标仍能热启动，不至于全部从零开始
            try:
                with open(f'{MOUNTED_DIR}/{game}_meta.json') as f: prev_meta = json.load(f)
                if prev_meta.get('feat_dim') == fd and prev_meta.get('n_classes') == nc:
                    lstm_warm_path = f'{MOUNTED_DIR}/{game}_lstm.pt'
                    tfm_warm_path  = f'{MOUNTED_DIR}/{game}_tfm.pt'
                    print(f"    (沿用旧版权重文件名热启动)")
            except Exception: pass

        lstm_m, lstm_h, _, lstm_p, lstm_acc, lstm_baseline, lstm_warm = train_encoder(
            lambda: LSTMEncoder(fd, hidden_dim=64, output_dim=nc), X, y, epochs=20, predict_X=predict_X,
            warm_start_path=lstm_warm_path)
        print(f"    LSTM 准确率: {lstm_acc}%（基线{lstm_baseline}%，提升{round(lstm_acc-lstm_baseline,1)}%）{'[热启动微调]' if lstm_warm else '[全量训练]'}")

        tfm_m, tfm_h, _, tfm_p, tfm_acc, tfm_baseline, tfm_warm = train_encoder(
            lambda: TransformerEncoder(fd, d_model=32, nhead=4, output_dim=nc), X, y, epochs=20, predict_X=predict_X,
            warm_start_path=tfm_warm_path)
        print(f"    TFM  准确率: {tfm_acc}%（基线{tfm_baseline}%，提升{round(tfm_acc-tfm_baseline,1)}%）{'[热启动微调]' if tfm_warm else '[全量训练]'}")

        # 每个目标的权重都保存，供下次热启动微调。
        # 之前只存第一个目标（if lstm_hidden_all is None 那个判断），
        # 另外6个训练完就丢，下周从零再来，等于6/7的算力白烧、永远学不到东西。
        torch.save(lstm_m.state_dict(), f'{LOCAL_DIR}/{game}_{tname}_lstm.pt')
        torch.save(tfm_m.state_dict(),  f'{LOCAL_DIR}/{game}_{tname}_tfm.pt')
        with open(f'{LOCAL_DIR}/{game}_{tname}_meta.json','w') as f:
            json.dump({'feat_dim':fd,'n_classes':nc,'hidden_dim':64,'d_model':32,'seq_len':SEQ_LEN}, f)

        if lstm_hidden_all is None:
            lstm_hidden_all = lstm_h; tfm_hidden_all = tfm_h
            # 第一个目标额外存一份固定名字的副本：RL每天要读它的隐层状态，
            # 用固定文件名RL才找得到（不必关心第一个目标叫什么）
            torch.save(lstm_m.state_dict(), f'{LOCAL_DIR}/{game}_lstm.pt')
            torch.save(tfm_m.state_dict(),  f'{LOCAL_DIR}/{game}_tfm.pt')
            np.save(f'{LOCAL_DIR}/{game}_lstm_hidden.npy', lstm_h)
            np.save(f'{LOCAL_DIR}/{game}_tfm_hidden.npy',  tfm_h)
            meta = {'feat_dim':fd,'n_classes':nc,'hidden_dim':64,'d_model':32,'seq_len':SEQ_LEN}
            with open(f'{LOCAL_DIR}/{game}_meta.json','w') as f: json.dump(meta,f)

        ens = lstm_p*0.6 + tfm_p*0.4
        classes = sorted(set(y.tolist()))
        # 蓝球训练时做了 -1 偏移（1-16 → 0-15分类），这里显示前必须还原回真实号码，
        # 否则会显示"预测值0"这种不存在的蓝球编号，造成误解
        offset = 1 if tname == 'blue' else 0
        pred_class = classes[int(np.argmax(ens))]
        game_results[tname] = {
            'lstm_acc':lstm_acc, 'tfm_acc':tfm_acc,
            'lstm_baseline':lstm_baseline, 'tfm_baseline':tfm_baseline,
            'ensemble_pred': int(pred_class) + offset,
            'confidence': round(float(max(ens))*100,1),
            'probs': {str(int(c)+offset):round(float(p)*100,1) for c,p in zip(classes,ens)},
        }

    dl_results[game] = game_results

# ── 保存到 Kaggle Dataset ──────────────────────────────
print(f"\n{'='*50}\n保存 LSTM/TFM 权重到 Kaggle Dataset…\n{'='*50}")
try:
    meta_ds = {"title":"Fucai DL Cache","id":DATASET_ID,"licenses":[{"name":"CC0-1.0"}]}
    with open(f'{LOCAL_DIR}/dataset-metadata.json','w') as f: json.dump(meta_ds,f)

    env = os.environ.copy(); env['KAGGLE_API_TOKEN'] = KAGGLE_TOKEN
    ok = False
    for cmd in [
        ['kaggle','datasets','version','-p',LOCAL_DIR,'-m',f'weekly-{date.today()}','--dir-mode','tar'],
        ['kaggle','datasets','create','-p',LOCAL_DIR,'--dir-mode','tar'],
    ]:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        if r.returncode==0:
            print(f"  ✓ 已保存到 {DATASET_ID}"); ok=True; break
        else:
            print(f"  [{cmd[1]} {cmd[2]}] rc={r.returncode}  {r.stderr[:150]}")
    if not ok:
        print("  ! 保存失败，请手动创建Dataset后重跑")
except Exception as e:
    print(f"  ! 异常: {e}")

# ── 写入独立文件 dl_lstm_tfm.json（不再读取/合并 prediction.json，速度更快）──
out = {
    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'models': 'LSTM + Transformer（每周全量训练）',
    'seq_len': SEQ_LEN, 'device': str(DEVICE),
    'results': dl_results,
    'note': '权重已存入 Kaggle Dataset，供每日 PPO 强化学习加载使用。',
}

out_json = json.dumps(out, ensure_ascii=False, indent=2)
if not GH_TOKEN:
    print("\n[DRY RUN] 未配置 GH_TOKEN")
else:
    print("\n推送 dl_lstm_tfm.json…")
    gh_put('dl_lstm_tfm.json', out_json, f"LSTM+TFM每周训练 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("✓ 完成")

print(f"\n✅ 全部完成！{datetime.now().strftime('%Y-%m-%d %H:%M')}")
