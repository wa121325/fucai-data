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
# 模型结构/前向计算的版本号。v2 = 多任务共享主干 + Transformer位置编码 + 输入标准化。
# 任何会改变"同样权重算出什么"的改动都要加1：热启动会拒绝版本不同的旧权重，
# kaggle_rl_daily.py 加载时也会核对，防止新旧结构混用却不报错、悄悄算出错的隐层。
ARCH_VERSION = 2

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


def build_predict_seq(records, feat_fn, seq_len=SEQ_LEN):
    """
    构造"预测下一期"用的输入序列：取最新的 seq_len 期特征。

    ── 为什么必须单独构造 ──
    build_seq_dataset 的最后一条样本 X[-1]，特征取的是 records[N-1-seq_len : N-1]，
    对应答案是 records[N-1]——也就是【最后一期，已经开出来了】。
    如果直接拿 X[-1] 去做"预测"（下面 train_encoder 原来的做法），
    模型输出的是对已知结果的"预测"，看起来很准，实际毫无预测价值
    （表现为推荐号码与最新一期开奖高度重合）。
    这里改成取 records[N-seq_len : N]（含最新一期），
    模型输出的才是对下一期（尚未开奖）的真实预测。
    """
    if len(records) < seq_len: return None
    seq = []
    for j in range(len(records)-seq_len, len(records)):
        feat = feat_fn(records, j)
        if feat is None: return None
        seq.append(list(feat.values()))
    return np.array([seq], dtype=np.float32)


def fit_feature_norm(X, n_fit):
    """
    用【训练可用区】(前n_fit个样本，不含RL保留的末段)算每个特征的均值/标准差。
    原来输入特征完全没标准化：快乐8里有13个特征量级超过100(号码总和类，最大约900)，
    跟0/1二值特征混在一起直接喂LSTM，输入门/遗忘门被大数值顶到饱和，
    网络基本只"看得见"那几个大数值特征。
    统计量只用可用区算，保留区的数据不参与，不泄漏。
    """
    fd = X.shape[2]
    flat = X[:max(1, n_fit)].reshape(-1, fd).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)    # 常数特征不除以0，标准化后恒为0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_feature_norm(X, mean, std):
    """⚠️ kaggle_rl_daily.py 里有同样的变换(precompute_hidden_all)，必须保持一致。"""
    return np.clip((X - mean) / std, -5.0, 5.0).astype(np.float32)


def build_seq_dataset_multi(records, feat_fn, tgt_fns, seq_len=SEQ_LEN):
    """
    一次性把【全部目标】的标签都算出来，共用同一个X——不用像以前那样每个目标各自
    重新扫一遍历史特征(那样7个目标要扫7遍，现在10个目标也只扫1遍，顺带更快)。
    tgt_fns: {目标名: 单条record->标签 的函数}
    返回 X, {目标名: y数组}, 目标名列表(保序，后面按这个顺序建多头模型)
    """
    tnames = list(tgt_fns.keys())
    X_list = []
    y_lists = {n: [] for n in tnames}
    # 特征缓存：第j期的特征只取决于records[:j]，跟它出现在哪个序列窗口里无关。
    # 原来每个样本都把窗口里20期的特征重算一遍，同一个j被重复计算20次
    # (3D约17万次调用、白耗2分钟)，缓存之后每个j只算一次，结果完全相同。
    _fc = {}
    def _feat(j):
        if j not in _fc:
            f = feat_fn(records, j)
            _fc[j] = None if f is None else list(f.values())
        return _fc[j]
    for i in range(seq_len, len(records)):
        seq = []; valid = True
        for j in range(i-seq_len, i):
            fv = _feat(j)
            if fv is None: valid=False; break
            seq.append(fv)
        if not valid: continue
        row = {}
        ok = True
        for n in tnames:
            t = tgt_fns[n](records[i])
            if t is None: ok=False; break
            row[n] = t
        if not ok: continue
        X_list.append(seq)
        for n in tnames: y_lists[n].append(row[n])
    if not X_list: return None, None, tnames
    X = np.array(X_list, dtype=np.float32)
    Y = {n: np.array(y_lists[n], dtype=np.int64) for n in tnames}
    return X, Y, tnames

# ══════════════════════════════════════════════════════
#  模型
# ══════════════════════════════════════════════════════
class LSTMEncoder(nn.Module):
    """
    多任务版本：一条LSTM主干 + 每个目标一个独立分类头。

    ── 为什么要改成这样 ──
    原来的结构是"一个主干配一个分类头"，全部7(现在10)个目标各自独立训练一整套
    LSTM+Transformer，但保存给RL用的隐层只取【排在字典第一个的目标】那一份，
    其余目标训练完、打印完准确率就直接丢弃——白白浪费了大部分训练算力，
    保存下来的隐层也只吸收了一个目标的信息。
    现在改成全部目标共享同一条LSTM主干，各自只有最后的分类头是独立的：
    每次反向传播时，主干的参数会被全部目标的梯度共同塑造，
    最终这一份隐层表示(self.norm的输出)是"见过全部目标"的，不再只是第一个目标的产物。
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dims=(10,), dropout=0.3):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0)
        self.norm = nn.LayerNorm(hidden_dim)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim,64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64,od))
            for od in output_dims])
    def forward(self, x, return_hidden=False):
        out,(h_n,_) = self.lstm(x)
        last = self.norm(out[:,-1,:])
        logits_list = [head(last) for head in self.heads]
        return (logits_list,last) if return_hidden else logits_list

def _sinusoidal_pe(seq_len, d_model, device):
    """
    正弦位置编码(固定公式，没有可训练参数，不进state_dict)。
    原来的Transformer完全没有位置信息：自注意力本身对顺序不敏感，后面又是平均池化，
    整个模型对"这20期的先后顺序"是完全无感的——打乱顺序输出一模一样，等于把时间序列
    当成了一袋无序的样本，TFM这条路形同虚设地丢掉了"序列"这个信息。
    ⚠️ kaggle_rl_daily.py 里有一份同样的函数和同样的加法，必须保持一致。
    """
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
                    * (-np.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
    return pe.unsqueeze(0)


class TransformerEncoder(nn.Module):
    """多任务版本，道理同LSTMEncoder：共享Transformer主干，每个目标一个独立分类头。"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, output_dims=(10,), dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model,32), nn.GELU(), nn.Linear(32,od))
            for od in output_dims])
    def forward(self, x, return_hidden=False):
        x = self.proj(x)
        x = x + _sinusoidal_pe(x.size(1), x.size(2), x.device)   # 加位置编码，让模型知道时间先后
        x = self.transformer(x)
        pooled = self.pool(x.transpose(1,2)).squeeze(-1)
        logits_list = [head(pooled) for head in self.heads]
        return (logits_list,pooled) if return_hidden else logits_list

def holdout_size(n_records):
    """
    留出期数，公式必须跟 kaggle_rl_daily.py 里的 holdout_size 完全一致——

    这是两个文件之间的一个隐性契约：RL把这段末期数据当成"训练时完全没碰过的
    样本外区域"来评估自己、挑权重；如果DL自己训练/验证时用到了同一段数据
    （哪怕只是用来决定"该在哪一轮停"这种模型选择，不是直接梯度更新），
    RL这边的"holdout"就不再是真正意义上的样本外了——因为这段数据的
    LSTM/TFM隐层特征，早就被一个"见过这段数据表现"的DL模型影响过。
    """
    return max(80, min(300, int(n_records * 0.10)))


def train_encoder(model_ctor, X, Y_list, epochs=60, lr=5e-4, batch_size=32, holdout_n=None,
                   warm_start_path=None, warm_start_epochs=10, warm_start_lr=1e-4,
                   warm_start_window=250, predict_X=None, n_records=None):
    """
    多任务版本：Y_list 是跟 model_ctor 建出来的多头模型【顺序对齐】的标签数组列表
    （比如3个目标就传3个数组），共享同一条LSTM/Transformer主干同时训练全部目标。

    ── 为什么要多任务 ──
    以前是每个目标单独训一整套模型，只有第一个目标的隐层被保存下来给RL用，
    其余目标训完就丢——白费算力，隐层也只吸收了一个目标的信息。
    现在改成一次训练同时喂全部目标的梯度，共享主干的隐层表示会被全部目标共同塑造，
    保存下来给RL用的隐层因此能反映更全面的统计信号。

    分两阶段训练，避免"用训练数据本身当回测题"造成虚假高准确率(道理不变，只是
    现在准确率/基线都是【每个目标一个数】的列表，不再是单个数字)：
    1) 回测模型：只用 X[:-holdout_n] 训练，在模型真正没见过的 X[-holdout_n:] 上评估准确率
       ⚠️ 这个模型必须从头训练，不能热启动——道理同旧版，避免偷看未来数据。
    2) 生产模型：优先热启动微调(滑动窗口)，用于提取隐层状态(供RL使用)和预测下一期，
       同样要避开RL保留的holdout区间，不参与训练/早停验证决策。

    model_ctor: 无参construct函数，每次调用返回一个全新的未训练【多头】模型实例
    warm_start_path: 若提供且文件存在，生产模型会从这份权重继续训练（增量微调）
    """
    n = len(X)
    n_targets = len(Y_list)
    # 回测评估集：原来固定50个样本，二分类准确率的标准误约±7个百分点，
    # "比基线高/低几个点"全是噪声，白白多训了一整个回测模型。改成跟RL保留区
    # 同口径(80~300期)，样本量至少翻倍到数倍，至少能看出大一点的差距。
    # RL保留区必须按【期数】算(跟RL的holdout_size(len(records))同参数)，
    # 原来按样本数n算：n=期数-25，在10%比例生效的区间(约800~3000期)会比RL少几期，
    # 等于DL的训练/早停验证偷用了RL保留区开头的几期标签。
    _n_rec = n_records if n_records is not None else n + SEQ_LEN
    if holdout_n is None:
        holdout_n = holdout_size(_n_rec)
    holdout_n = min(holdout_n, max(5, n//5))
    split = max(1, n - holdout_n)

    def _train_one(m, Xtr, ytr_list, ep, learning_rate=lr, Xval=None, yval_list=None,
                   patience=8, warmup=5, tag=''):
        """
        训练单个多头模型，带早停+保留最佳权重。早停用的"准确率"是全部目标头的平均值——
        单个目标的波动不会误导早停判断，只有整体都变差才会真的停。

        Xval/yval_list 必须是从【训练数据内部】切出来的，不能是外部最终报告用的holdout，
        道理跟旧版一致：holdout一旦参与了"该停在哪一轮"的决策，评估就不再干净。
        """
        m = m.to(DEVICE)
        Xt = torch.FloatTensor(Xtr).to(DEVICE)
        yts = [torch.LongTensor(y).to(DEVICE) for y in ytr_list]
        loader = DataLoader(TensorDataset(Xt, *yts), batch_size, shuffle=True)
        opt = optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(ep,1))
        crit = nn.CrossEntropyLoss()

        has_val = Xval is not None and len(Xval) > 0
        if has_val:
            Xv = torch.FloatTensor(Xval).to(DEVICE)
            yvs = [torch.LongTensor(y).to(DEVICE) for y in yval_list]
        best_loss, best_acc, best_state, no_imp, stopped, best_ep = float('inf'), -1.0, None, 0, ep, 0

        for e in range(ep):
            m.train()
            for batch in loader:
                xb, ybs = batch[0], batch[1:]
                opt.zero_grad()
                logits_list = m(xb)
                # 全部目标头的交叉熵相加作为总loss，一次反向传播同时更新共享主干+全部分类头
                loss = sum(crit(lg, yb) for lg, yb in zip(logits_list, ybs))
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
            sch.step()
            if not has_val: continue
            m.eval()
            with torch.no_grad():
                logits_list = m(Xv)
                accs = [(lg.argmax(dim=1)==yv).float().mean().item() for lg,yv in zip(logits_list,yvs)]
                acc = float(np.mean(accs))
                # 早停改看验证【损失】(全部目标交叉熵的平均)，不再看准确率：
                # 这些目标大多接近随机，准确率只在几个百分点内抖动，"哪一轮准确率最高"
                # 基本是在挑噪声，挑出来的"最佳轮"还会虚高；损失是连续的、平滑得多，
                # 而且对"模型开始背训练集"(验证损失掉头向上)灵敏——正好顺带压低
                # 隐层里的样本内记忆(见函数末尾的样本内/外对比)。
                vloss = float(np.mean([crit(lg, yv).item() for lg, yv in zip(logits_list, yvs)]))
            if vloss < best_loss - 1e-6:
                best_loss, best_acc, best_ep = vloss, acc, e+1
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
                no_imp = 0
            elif e >= warmup:
                no_imp += 1
                if no_imp >= patience:
                    stopped = e+1; break
        if has_val and best_state is not None:
            m.load_state_dict(best_state)
            if tag:
                print(f"      [{tag}] 跑到第{stopped}/{ep}轮，保留的是第{best_ep}轮的权重，验证损失{best_loss:.4f}"
                      f"（该轮平均准确率{best_acc*100:.1f}%，全部{n_targets}个目标）")
        return m

    # ── 1) 回测模型：必须从头训练，不能加载旧权重 ──
    _inner = max(1, int(split * 0.15))
    _bt_tr_end = max(1, split - _inner)
    ytr_bt = [y[:_bt_tr_end] for y in Y_list]
    yval_bt = [y[_bt_tr_end:split] for y in Y_list]
    bt_model = _train_one(model_ctor(), X[:_bt_tr_end], ytr_bt, epochs,
                          Xval=X[_bt_tr_end:split], yval_list=yval_bt, tag='回测模型')
    bt_model.eval()
    with torch.no_grad():
        Xte = torch.FloatTensor(X[split:]).to(DEVICE)
        preds_list = [lg.argmax(dim=1).cpu().numpy() for lg in bt_model(Xte)]
    y_holdout_list = [y[split:] for y in Y_list]
    acc_list = [round(float((p==yh).mean())*100,1) if len(yh)>0 else 0.0
                for p, yh in zip(preds_list, y_holdout_list)]

    # 基线：每个目标各自在训练集里出现最多的那一类
    from collections import Counter as Ctr
    baseline_list = []
    for i in range(n_targets):
        y_train = Y_list[i][:split]; y_hold = y_holdout_list[i]
        if len(y_train)>0 and len(y_hold)>0:
            majority = Ctr(y_train.tolist()).most_common(1)[0][0]
            baseline_list.append(round(float((y_hold==majority).mean())*100,1))
        else:
            baseline_list.append(0.0)

    # ── 2) 生产模型：优先"滑动窗口热启动微调"，找不到旧权重才全量训练 ──
    # 同样要避开RL保留的末尾_rl_reserved期，道理跟旧版完全一致，只是切片对象
    # 从单个y数组变成Y_list里的每一个都要同步切。
    _rl_reserved = holdout_size(_n_rec)
    n_usable = max(1, n - _rl_reserved)
    is_warm_start = False
    prod_new = model_ctor()
    if warm_start_path and os.path.exists(warm_start_path):
        try:
            prod_new.load_state_dict(torch.load(warm_start_path, map_location='cpu'))
            is_warm_start = True
            win = min(warm_start_window, n_usable)
            print(f"      ✓ 加载上次训练权重，滑动窗口热启动微调（可用区内近{win}期，"
                  f"{warm_start_epochs}轮，lr={warm_start_lr}；已避开末尾{_rl_reserved}期RL保留区）")
        except Exception as e:
            print(f"      ! 加载旧权重失败（{e}），改为全量训练")
    if is_warm_start:
        win = min(warm_start_window, n_usable)
        _v = max(1, int(win * 0.15)); _tr = max(1, win - _v)
        Xw = X[n_usable-win:n_usable]
        yw_list = [y[n_usable-win:n_usable] for y in Y_list]
        ytr_w = [y[:_tr] for y in yw_list]; yval_w = [y[_tr:] for y in yw_list]
        prod_model = _train_one(prod_new, Xw[:_tr], ytr_w, warm_start_epochs, learning_rate=warm_start_lr,
                                Xval=Xw[_tr:], yval_list=yval_w, patience=4, warmup=2, tag='生产模型(微调)')
    else:
        _v = max(1, int(n_usable * 0.15)); _tr = max(1, n_usable - _v)
        ytr_p = [y[:_tr] for y in Y_list]; yval_p = [y[_tr:n_usable] for y in Y_list]
        prod_model = _train_one(prod_new, X[:_tr], ytr_p, epochs,
                                Xval=X[_tr:n_usable], yval_list=yval_p, tag='生产模型')
    prod_model.eval()

    # ── 诊断：生产模型在"训练过的区间"和"RL保留区"上的平均准确率对比 ──
    # 给RL用的隐层是对【全部历史】逐期算出来的：训练区那一段是模型"见过答案"的样本内输出，
    # 保留区/今天是没见过的样本外输出。如果样本内准确率明显高于样本外，说明隐层里
    # 夹带了"记住了训练样本答案"的成分——RL在训练区上学到的"信LSTM/TFM隐层"，
    # 到了保留区和现场预测就不成立了。这个差距就是衡量它的直接指标。
    try:
        with torch.no_grad():
            def _mean_acc(Xa, ya_list):
                if len(Xa) == 0: return float('nan')
                lg = prod_model(torch.FloatTensor(Xa).to(DEVICE))
                return float(np.mean([(l.argmax(dim=1).cpu().numpy() == ya).mean()
                                      for l, ya in zip(lg, ya_list)]))
            _k = min(n_usable, 2000)
            a_in  = _mean_acc(X[n_usable-_k:n_usable], [y[n_usable-_k:n_usable] for y in Y_list])
            a_out = _mean_acc(X[n_usable:], [y[n_usable:] for y in Y_list])
        print(f"      [样本内外对比] 平均准确率: 训练区近{_k}期 {a_in*100:.1f}%  vs  RL保留区{n-n_usable}期 {a_out*100:.1f}%"
              f"（差{(a_in-a_out)*100:+.1f}个点；差距大=隐层里有对训练样本的记忆）")
    except Exception as _e:
        print(f"      [样本内外对比] 计算失败: {_e}")
    hidden_states=[]
    with torch.no_grad():
        Xt_full = torch.FloatTensor(X).to(DEVICE)
        for i in range(0,len(Xt_full),batch_size):
            xb=Xt_full[i:i+batch_size]; _,h=prod_model(xb,return_hidden=True)
            hidden_states.append(h.cpu().numpy())
    hidden_states=np.vstack(hidden_states)
    with torch.no_grad():
        # 用专门构造的"含最新一期"的序列做预测，而不是 Xt_full[-1]。
        # 后者对应的答案就是最后一期（已开奖），拿它预测等于复述已知结果。
        if predict_X is not None:
            _px = torch.FloatTensor(predict_X).to(DEVICE)
            logits_list, last_h = prod_model(_px, return_hidden=True)
        else:
            logits_list, last_h = prod_model(Xt_full[-1:], return_hidden=True)
        probs_list = [torch.softmax(lg,dim=1)[0].cpu().numpy() for lg in logits_list]

    # ── 样本外对数损失所需的原始材料：生产模型在RL保留区(训练和早停都没碰过)上的概率 ──
    # 准确率(argmax是否猜对)在这些接近随机的目标上全是噪声；对数损失看的是"给真实结果的概率"，
    # 连续、灵敏，且可以跟"均匀随机"和"只看历史频率"两条基线直接比较。
    holdout_diag = None
    try:
        with torch.no_grad():
            if n - n_usable > 0:
                _lg = prod_model(torch.FloatTensor(X[n_usable:]).to(DEVICE))
                holdout_diag = {
                    'probs': [torch.softmax(l, dim=1).cpu().numpy() for l in _lg],
                    'y': [y[n_usable:] for y in Y_list],
                    # 频率基线：只用训练可用区的标签频率（拉普拉斯平滑），代表"完全不看特征"的最佳常数预测
                    'freq': [(np.bincount(y[:n_usable], minlength=l.shape[1])[:l.shape[1]] + 1.0)
                             / (len(y[:n_usable]) + l.shape[1]) for y, l in zip(Y_list, _lg)],
                }
    except Exception as _e:
        print(f"      [样本外对数损失] 取材失败: {_e}")

    return prod_model, hidden_states, last_h.cpu().numpy()[0], probs_list, acc_list, baseline_list, is_warm_start, holdout_diag

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

# 下面9个新目标(质合比/和尾/同尾号等)的分类边界，跟kaggle_fucai.py里的tgt3d/tgtssq/tgtkl8
# 完全一致(那边已经用20万次模拟核实过分布均衡)，这里独立手写一份，逻辑必须保持同步。
def _3d_prime_cnt(digits): return sum(1 for x in digits if x in {2,3,5,7})
def _ssq_prime_grp(red):
    n = sum(1 for x in red if x in _PRIMES)
    return 0 if n<=1 else (1 if n==2 else 2)
def _ssq_same_tail(red):
    tc = Counter(x%10 for x in red)
    return min(sum(c*(c-1)//2 for c in tc.values()), 3)
def _kl8_prime_grp(nums):
    n = sum(1 for x in nums if x in _PRIMES)
    return 0 if n<=4 else (1 if n<=6 else 2)
def _kl8_ac_grp(nums):
    sn = sorted(nums); diffs=set()
    for i in range(len(sn)):
        for j in range(i+1,len(sn)): diffs.add(sn[j]-sn[i])
    ac = len(diffs)-(len(sn)-1)
    return 0 if ac<=44 else (1 if ac<=49 else 2)
def _kl8_same_tail_grp(nums):
    tc = Counter(x%10 for x in nums)
    st = sum(c*(c-1)//2 for c in tc.values())
    return 0 if st<=15 else (1 if st<=18 else 2)

configs = {
    '3d':  (f3d,  {'sum_grp':lambda r:0 if sum(r['digits'])<=9 else(1 if sum(r['digits'])<=17 else 2),
                    'group_type':lambda r:_3d_group_type(r['digits']),
                    'odd':lambda r:sum(1 for x in r['digits'] if x%2!=0),
                    'big':lambda r:sum(1 for x in r['digits'] if x>=5),
                    'span_grp':lambda r:_3d_span_grp(r['digits']),
                    'road_dom':lambda r:_3d_road_dom(r['digits']),
                    'arith':lambda r:_3d_arith(r['digits']),
                    'prime_cnt':lambda r:_3d_prime_cnt(r['digits']),
                    'sum_tail':lambda r:sum(r['digits'])%10,
                    'span_odd':lambda r:(max(r['digits'])-min(r['digits']))%2}),
    'ssq': (fssq, {'odd':lambda r:sum(1 for x in r['red'] if x%2!=0),
                    'sum_grp':lambda r:0 if sum(r['red'])<70 else(1 if sum(r['red'])<100 else 2),
                    'ac_grp':lambda r:_ssq_ac_grp(r['red']),
                    'red_zone_dom':lambda r:_ssq_zone_dom(r['red']),
                    'gap_grp':lambda r:_ssq_gap_grp(r['red']),
                    'big':lambda r:sum(1 for x in r['red'] if x>16),
                    'consec':lambda r:_ssq_consec(r['red']),
                    'prime_grp':lambda r:_ssq_prime_grp(r['red']),
                    'sum_tail':lambda r:sum(r['red'])%10,
                    'same_tail':lambda r:_ssq_same_tail(r['red'])}),
    'kl8': (fkl8, {'odd_grp':lambda r:0 if sum(1 for x in r['numbers'] if x%2!=0)<9 else(1 if sum(1 for x in r['numbers'] if x%2!=0)<=11 else 2),
                    'zone_dom':lambda r:int(np.argmax([sum(1 for x in r['numbers'] if lo<=x<=hi) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]])),
                    'tot_grp':lambda r:0 if sum(r['numbers'])<640 else(1 if sum(r['numbers'])<820 else 2),
                    'big_grp':lambda r:_kl8_big_grp(r['numbers']),
                    'five_dom':lambda r:_kl8_five_dom(r['numbers']),
                    'consec_grp':lambda r:_kl8_consec_grp(r['numbers']),
                    'range_grp':lambda r:_kl8_range_grp(r['numbers']),
                    'prime_grp':lambda r:_kl8_prime_grp(r['numbers']),
                    'ac_grp':lambda r:_kl8_ac_grp(r['numbers']),
                    'same_tail_grp':lambda r:_kl8_same_tail_grp(r['numbers'])}),
}

for game, (feat_fn, targets) in configs.items():
    records = history.get(game, [])
    if not isinstance(records,list) or len(records)<65:
        print(f"\n{game}: 数据不足，跳过"); continue
    print(f"\n{'='*50}\n{game}（{len(records)}期）\n{'='*50}")

    # 一次性把全部目标的标签都算出来，共用同一个X——不再是每个目标各自扫一遍历史、
    # 各自训一整套LSTM+Transformer(只有第一个的隐层被保留、其余训完就丢)。
    tnames = list(targets.keys())
    X, Y_dict, tnames = build_seq_dataset_multi(records, feat_fn, targets)
    if X is None or len(X)<40:
        print(f"  数据不足，跳过{game}"); continue
    Y_list = [Y_dict[n] for n in tnames]
    # 分类头的输出维度必须是 max(标签)+1，不能是"出现过几种标签"：
    # 交叉熵把标签直接当下标用(要求落在 0..nc-1)。原来用len(set(标签))，
    # 只要某个类别历史上一次都没出现过就会错位——比如快乐8的consec_grp，类别0只占0.14%，
    # 两千期里约3~6%的概率一次不出现，标签变成{1,2}、输出维度只有2，
    # 训练时标签2直接越界崩溃。用max+1，没出现过的类别只是概率很低，不会崩。
    nc_list = [int(Y_dict[n].max()) + 1 for n in tnames]
    fd = X.shape[2]
    predict_X = build_predict_seq(records, feat_fn)

    # 输入特征标准化：统计量只用训练可用区算(不含RL保留区)，并存进meta，
    # RL那边算隐层时必须套同一个变换，否则LSTM/TFM看到的输入尺度跟训练时对不上。
    _n_usable = max(1, len(X) - holdout_size(len(records)))
    feat_mean, feat_std = fit_feature_norm(X, _n_usable)
    X = apply_feature_norm(X, feat_mean, feat_std)
    if predict_X is not None:
        predict_X = apply_feature_norm(predict_X, feat_mean, feat_std)
    print(f"  {len(tnames)}个目标共享同一条主干训练: {tnames}")
    print(f"  各目标分类数: {dict(zip(tnames, nc_list))}")

    # 热启动检查：多头模型的"维度"现在不只是feat_dim，还要看目标结构(个数+每个的分类数)
    # 是否跟上次完全一致——任何一个目标的分类数变了(比如加了新目标)，多头模型的
    # 某个分类头维度就对不上，必须放弃热启动、改为全量训练。
    lstm_warm_path = tfm_warm_path = None
    prev_meta_path = f'{MOUNTED_DIR}/{game}_meta.json'
    if os.path.exists(prev_meta_path):
        try:
            with open(prev_meta_path) as f: prev_meta = json.load(f)
            if prev_meta.get('feat_dim')==fd and prev_meta.get('target_names')==tnames \
               and prev_meta.get('n_classes_list')==nc_list \
               and prev_meta.get('arch_version')==ARCH_VERSION:
                lstm_warm_path = f'{MOUNTED_DIR}/{game}_lstm.pt'
                tfm_warm_path  = f'{MOUNTED_DIR}/{game}_tfm.pt'
            else:
                print(f"    ! 上次权重的目标结构/维度/模型版本与当前不一致(比如新增了目标、加了位置编码)，改为全量训练")
        except Exception as e:
            print(f"    ! 读取上次meta失败({e})，改为全量训练")

    lstm_m, lstm_h, _, lstm_p_list, lstm_acc_list, lstm_baseline_list, lstm_warm, lstm_hd = train_encoder(
        lambda: LSTMEncoder(fd, hidden_dim=64, output_dims=nc_list), X, Y_list, epochs=20,
        warm_start_path=lstm_warm_path, predict_X=predict_X, n_records=len(records))
    for i, tname in enumerate(tnames):
        print(f"    [{tname}] LSTM 准确率: {lstm_acc_list[i]}%（基线{lstm_baseline_list[i]}%，"
              f"提升{round(lstm_acc_list[i]-lstm_baseline_list[i],1)}%）")
    print(f"    LSTM 主干 {'[热启动微调]' if lstm_warm else '[全量训练]'}")

    tfm_m, tfm_h, _, tfm_p_list, tfm_acc_list, tfm_baseline_list, tfm_warm, tfm_hd = train_encoder(
        lambda: TransformerEncoder(fd, d_model=32, nhead=4, output_dims=nc_list), X, Y_list, epochs=20,
        warm_start_path=tfm_warm_path, predict_X=predict_X, n_records=len(records))
    for i, tname in enumerate(tnames):
        print(f"    [{tname}] TFM  准确率: {tfm_acc_list[i]}%（基线{tfm_baseline_list[i]}%，"
              f"提升{round(tfm_acc_list[i]-tfm_baseline_list[i],1)}%）")
    print(f"    TFM  主干 {'[热启动微调]' if tfm_warm else '[全量训练]'}")

    # ══ 样本外对数损失对比：这一版 DL 到底有没有学到东西 ══
    # 比较对象（都在RL保留区上算，模型训练/早停都没碰过这段）：
    #   均匀随机 = ln(类别数)；历史频率 = 只用训练区标签频率的常数预测；LSTM / TFM / 两者按0.6:0.4融合。
    # 数值越低越好。关键看"融合 对比 历史频率"的差：>0 才说明比不看特征的常数预测强；
    # z = 差的均值 / 标准误（逐期配对），|z|<2 基本等同于没有差别。
    # 想比较新旧版本：对同一份history分别跑两个版本的脚本，看这张表里的"融合"那一列谁更低。
    holdout_report = {}
    try:
        if lstm_hd and tfm_hd and len(lstm_hd['y'][0]) > 0:
            def _ll_each(p, y): return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.0))
            nh = len(lstm_hd['y'][0])
            print(f"    [样本外对数损失] RL保留区{nh}期（越低越好；差>0表示比\"历史频率\"强）")
            print(f"      {'目标':<14}{'均匀':>8}{'频率':>8}{'LSTM':>8}{'TFM':>8}{'融合':>8}{'融合-频率':>10}{'z':>7}")
            _gains = []
            for i, tname in enumerate(tnames):
                y = lstm_hd['y'][i]
                nc_i = lstm_hd['probs'][i].shape[1]
                pf = np.tile(lstm_hd['freq'][i], (len(y), 1))
                pe = 0.6 * lstm_hd['probs'][i] + 0.4 * tfm_hd['probs'][i]
                l_f, l_l, l_t, l_e = (_ll_each(p, y) for p in (pf, lstm_hd['probs'][i], tfm_hd['probs'][i], pe))
                d = l_f - l_e                      # 逐期：频率基线损失 - 融合损失（>0 表示融合更好）
                z = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)) if len(d) > 2 else 0.0
                _gains.append(float(d.mean()))
                holdout_report[tname] = {
                    'n': int(nh), 'uniform': round(float(np.log(nc_i)), 4), 'freq': round(float(l_f.mean()), 4),
                    'lstm': round(float(l_l.mean()), 4), 'tfm': round(float(l_t.mean()), 4),
                    'ensemble': round(float(l_e.mean()), 4), 'ens_minus_freq': round(float(d.mean()), 4), 'z': round(z, 2)}
                print(f"      {tname:<14}{np.log(nc_i):>8.4f}{l_f.mean():>8.4f}{l_l.mean():>8.4f}{l_t.mean():>8.4f}"
                      f"{l_e.mean():>8.4f}{d.mean():>+10.4f}{z:>+7.1f}")
            _n_pos = sum(1 for g in _gains if g > 0)
            _n_sig = sum(1 for t in holdout_report.values() if t['z'] >= 2)
            print(f"      小结：{len(_gains)}个目标里 融合比频率强的 {_n_pos} 个，其中 z≥2(显著) {_n_sig} 个；"
                  f"平均差 {np.mean(_gains):+.4f}。")
            print(f"      解读：显著的个数≈0 且平均差≈0 → 这版DL没有学到超出历史频率的东西（彩票接近随机时这是正常结果）；"
                  f"{len(_gains)}个目标里碰巧有1个 z≥2 属于多重比较下的正常偶然。")
    except Exception as _e:
        print(f"    [样本外对数损失] 计算失败: {_e}")

    # 保存权重（供每日RL加载，也供下次本脚本运行时热启动微调）——
    # 现在只有一套主干权重需要存，不再是"存第一个、丢其余六个"
    torch.save(lstm_m.state_dict(), f'{LOCAL_DIR}/{game}_lstm.pt')
    torch.save(tfm_m.state_dict(),  f'{LOCAL_DIR}/{game}_tfm.pt')
    np.save(f'{LOCAL_DIR}/{game}_lstm_hidden.npy', lstm_h)
    np.save(f'{LOCAL_DIR}/{game}_tfm_hidden.npy',  tfm_h)
    meta = {'feat_dim':fd, 'target_names':tnames, 'n_classes_list':nc_list,
            'hidden_dim':64, 'd_model':32, 'seq_len':SEQ_LEN,
            'arch_version':ARCH_VERSION,        # 模型结构版本：结构/前向变了就加1，RL据此拒绝加载不匹配的旧权重
            'feat_mean':[float(v) for v in feat_mean],   # 输入标准化统计量，RL算隐层时必须套同一个变换
            'feat_std':[float(v) for v in feat_std]}
    with open(f'{LOCAL_DIR}/{game}_meta.json','w') as f: json.dump(meta,f)

    game_results = {}
    for i, tname in enumerate(tnames):
        ens = lstm_p_list[i]*0.6 + tfm_p_list[i]*0.4
        # 分类头第c个输出就对应标签c(输出维度是max+1)，不再用"出现过的类别排序后的下标"
        # 去反查标签——那种写法在类别不连续时会把预测标成错误的类别。
        classes = list(range(len(ens)))
        # 蓝球训练时做了 -1 偏移（1-16 → 0-15分类），这里显示前必须还原回真实号码，
        # 否则会显示"预测值0"这种不存在的蓝球编号，造成误解
        offset = 1 if tname == 'blue' else 0
        pred_class = classes[int(np.argmax(ens))]
        game_results[tname] = {
            'lstm_acc':lstm_acc_list[i], 'tfm_acc':tfm_acc_list[i],
            'lstm_baseline':lstm_baseline_list[i], 'tfm_baseline':tfm_baseline_list[i],
            'ensemble_pred': int(pred_class) + offset,
            'confidence': round(float(max(ens))*100,1),
            'probs': {str(int(c)+offset):round(float(p)*100,1) for c,p in zip(classes,ens)},
            'holdout_logloss': holdout_report.get(tname),   # 样本外对数损失对比（均匀/频率/LSTM/TFM/融合），None=没算出来
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
