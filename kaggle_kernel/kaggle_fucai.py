"""
福彩全能脚本 - Kaggle Notebook 版
功能：抓数据 + ML训练 + 推送所有结果到 GitHub
在 Kaggle 上运行，把结果推回 GitHub 仓库

Kaggle Secrets 配置：
  GH_TOKEN  : GitHub Personal Access Token (repo权限)
  GH_REPO   : 仓库名，如 wa121325/fucai-data
"""
import os, json, sys, re, time, warnings, base64, urllib.request
from datetime import datetime, date
from collections import Counter, defaultdict
import random
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
#  读取 Kaggle Secrets
# ══════════════════════════════════════════════════════
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
                val = client.get_secret(name)
                if val:
                    print(f"  [Secret] {name} 读取成功（kaggle_secrets，第{attempt+1}次）")
                    return val
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

    # 方式3：环境变量
    val = os.environ.get(name, '')
    if val:
        print(f"  [Secret] {name} 读取成功（环境变量）")
        return val

    print(f"  [Secret] {name} 未找到")
    return ''

# ── 读取 Secrets（兜底：直接写死，确保推送不失败）──────────
# ── 直接硬编码，不依赖 Secrets 服务 ──────────────────────
# 如果 Kaggle Secrets 服务不可用，直接用下面的值
# 请把新生成的 GitHub Token 替换到这里
_HARDCODED_TOKEN = ''  # 不要在这里写Token！写了会被GitHub自动吊销，必须用Kaggle Secrets
_HARDCODED_REPO  = 'wa121325/fucai-data'

GH_TOKEN = get_secret('GH_TOKEN') or get_secret('gh_token') or _HARDCODED_TOKEN
GH_REPO  = get_secret('GH_REPO')  or get_secret('gh_repo')  or _HARDCODED_REPO

print(f"GitHub 仓库: {GH_REPO}")
print(f"GH_TOKEN 已配置: {'是（长度'+str(len(GH_TOKEN))+'）' if GH_TOKEN else '否（将跳过推送）'}")

# ══════════════════════════════════════════════════════
#  GitHub API 工具
# ══════════════════════════════════════════════════════
def gh_raw(path):
    """读取 GitHub raw 文件"""
    url = f'https://raw.githubusercontent.com/{GH_REPO}/main/{path}?t={int(time.time())}'
    req = urllib.request.Request(url, headers={'Cache-Control':'no-cache','User-Agent':'kaggle-bot'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode('utf-8')
    except Exception:
        return None

def gh_get_sha(path):
    url = f'https://api.github.com/repos/{GH_REPO}/contents/{path}'
    req = urllib.request.Request(url, headers={
        'Authorization': f'token {GH_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'kaggle-bot'
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read()).get('sha')
    except Exception:
        return None

def gh_put(path, content_str, message):
    sha = gh_get_sha(path)
    url = f'https://api.github.com/repos/{GH_REPO}/contents/{path}'
    data = {'message': message, 'branch': 'main',
            'content': base64.b64encode(content_str.encode('utf-8')).decode()}
    if sha: data['sha'] = sha
    body = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=body, method='PUT', headers={
        'Authorization': f'token {GH_TOKEN}',
        'Content-Type': 'application/json',
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'kaggle-bot'
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

# ══════════════════════════════════════════════════════
#  抓取开奖数据（17500.cn）
# ══════════════════════════════════════════════════════
import urllib.parse
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/plain, */*',
    'Connection': 'keep-alive',
}
URL_MAP = {
    'ssq': 'http://data.17500.cn/ssq_asc.txt',
    '3d':  'http://data.17500.cn/3d_asc.txt',
    'qlc': 'http://data.17500.cn/7lc_asc.txt',
    'kl8': 'http://data.17500.cn/kl8_asc.txt',
}

def parse_line(game, line):
    parts = line.split()
    if len(parts) < 4: return None
    qihao = parts[0]
    if not re.match(r'^\d{6,}$', qihao): return None
    m = re.search(r'(\d{4}-\d{2}-\d{2})', line)
    if not m: return None
    date_str = m.group(1)
    raw = [int(p) for p in parts[1:] if re.match(r'^\d+$', p) and p != qihao and len(p) <= 3]
    try:
        if game == 'ssq' and len(raw) >= 7:
            return {'qihao': qihao, 'date': date_str, 'red': sorted(raw[:6]), 'blue': raw[6]}
        elif game == '3d' and len(raw) >= 3:
            return {'qihao': qihao, 'date': date_str, 'digits': raw[:3]}
        elif game == 'qlc' and len(raw) >= 8:
            return {'qihao': qihao, 'date': date_str, 'numbers': sorted(raw[:7]), 'special': raw[7]}
        elif game == 'kl8' and len(raw) >= 20:
            return {'qihao': qihao, 'date': date_str, 'numbers': sorted(raw[:20])}
    except Exception:
        pass
    return None

def fetch_game_data(game):
    url = URL_MAP[game]
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                text = r.read().decode('utf-8', errors='ignore')
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if len(lines) < 10: raise ValueError(f"内容太少:{len(lines)}行")
            seen, records = set(), []
            for line in lines:
                rec = parse_line(game, line)
                if rec and rec['qihao'] not in seen:
                    seen.add(rec['qihao']); records.append(rec)
            return records
        except Exception as e:
            print(f"  [{game}] 第{attempt+1}次失败: {e}")
            if attempt < 2: time.sleep(5)
    return []

def crawl_all():
    print(f"\n{'='*50}")
    print(f"抓取开奖数据  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    # 读取已有history（增量合并）
    existing = {}
    raw = gh_raw('history.json')
    if raw:
        try:
            existing = json.loads(raw)
            for g in ['ssq','3d','qlc','kl8']:
                if isinstance(existing.get(g), list):
                    print(f"  已有 {g}: {len(existing[g])}期")
        except Exception:
            pass

    games = [('ssq','双色球'), ('3d','福彩3D'), ('qlc','七乐彩'), ('kl8','快乐8')]
    latest_out = {'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    history_out = {'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ok = 0

    for game, name in games:
        print(f"\n── {name} ──")
        all_records = fetch_game_data(game)
        if not all_records:
            print(f"  ✗ 抓取失败，使用已有数据")
            if isinstance(existing.get(game), list) and existing[game]:
                history_out[game] = existing[game]
                latest_out[game] = existing[game][-1]
            continue
        # 增量合并
        old = existing.get(game, [])
        old_set = {r['qihao'] for r in old} if isinstance(old, list) else set()
        new_ones = [r for r in all_records if r['qihao'] not in old_set]
        merged = (old if isinstance(old, list) else []) + new_ones
        merged.sort(key=lambda x: x['qihao'])
        history_out[game] = merged
        latest_out[game] = all_records[-1]
        print(f"  ✓ 全量{len(all_records)}期，新增{len(new_ones)}期，累计{len(merged)}期")
        print(f"    最新: {all_records[-1]['qihao']} / {all_records[-1]['date']}")
        ok += 1

    return latest_out, history_out, ok

# ══════════════════════════════════════════════════════
#  ML 训练（与 lottery_ml.py v3 完全一致）
# ══════════════════════════════════════════════════════
try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    HAS_SKL = True
except ImportError:
    HAS_SKL = False; print("缺少 sklearn")

try:
    import xgboost as xgb; HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb; HAS_LGB = True
except ImportError:
    HAS_LGB = False

print(f"\n依赖: sklearn={'✓' if HAS_SKL else '✗'}  xgb={'✓' if HAS_XGB else '✗'}  lgb={'✓' if HAS_LGB else '✗'}")

WINDOW = 50

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

def _feat_window(w):
    return w


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


def build_dataset(records, feat_fn, tgt_fn, keys):
    """
    正确的时序对齐：
    X[i] = 第i期的特征（用第i期及之前历史计算，不含第i+1期任何信息）
    y[i] = 第i+1期的目标值（模型要预测的未来值）

    训练样本：X[i] → y[i]  即"知道第i期及历史，预测第i+1期"
    预测样本：X[n-1]（最新一期特征）→ 预测真正未开奖的下一期
    """
    X, Y = [], {k:[] for k in keys}
    n = len(records)
    for i in range(n - 1):
        # 特征：用第i期（含）之前的数据，idx=i+1表示窗口截止到第i期
        feat = feat_fn(records, i + 1)
        if feat is None:
            continue
        # 目标：第i+1期的真实值（未来，模型未见过）
        tgt = tgt_fn(records[i + 1])
        # ★关键检查：特征里绝对不能包含第i+1期的号码
        # feat_fn(records, i+1) 内部用 records[max(0,i+1-WINDOW):i+1]
        # 即窗口是 [i+1-WINDOW, i+1)，最后一条是records[i]，正确！
        X.append(list(feat.values()))
        for k in keys:
            Y[k].append(tgt[k])

    # 预测用特征：用全量数据（窗口截止到最新一期records[n-1]）
    # feat_fn(records, n) 内部用 records[max(0,n-WINDOW):n]，最后一条是records[n-1]
    last_feat = feat_fn(records, n)
    if last_feat is None:
        last_feat = feat_fn(records, n - 1)
    last_X = np.array([list(last_feat.values())], dtype=float) if last_feat else None

    names = []
    for _i in range(10, min(len(records), 100)):
        _f = feat_fn(records, _i)
        if _f is not None:
            names = list(_f.keys())
            break
    return np.array(X, dtype=float), Y, names, last_X

def make_models():
    ms={}
    if HAS_SKL: ms['rf']=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=3,random_state=42,n_jobs=-1)
    if HAS_XGB: ms['xgb']=xgb.XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=42,eval_metric='mlogloss',verbosity=0)
    if HAS_LGB: ms['lgb']=lgb.LGBMClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,num_leaves=31,random_state=42,verbose=-1)
    return ms


# ══════════════════════════════════════════════════════
#  模型缓存（Kaggle Dataset 持久化）
# ══════════════════════════════════════════════════════
import pickle, os, subprocess

# ── Kaggle Dataset 作为模型持久存储 ────────────────────
# Dataset slug（你需要先手动在Kaggle创建一个私有dataset）
DATASET_SLUG  = 'fucai-model-cache'
DATASET_ID    = f'megskfdbbskeb/{DATASET_SLUG}'
LOCAL_CACHE   = '/kaggle/working/models_cache.pkl'
DATASET_DIR   = '/kaggle/working/cache_upload/'
# Kaggle Notebook 挂载路径：Add Data → 搜索 fucai-model-cache → 挂载
MOUNTED_CACHE = f'/kaggle/input/{DATASET_SLUG}/models_cache.pkl'
RETRAIN_EVERY = 20  # 新增超过20期才重新全量训练

def load_cache():
    """优先从挂载的Dataset读取模型缓存"""
    for path in [LOCAL_CACHE, MOUNTED_CACHE]:
        if os.path.exists(path):
            try:
                with open(path,'rb') as f: c=pickle.load(f)
                print(f"✓ 模型缓存已加载：{path}（{len(c)}个目标，节省大量训练时间）")
                import shutil
                shutil.copy(path, LOCAL_CACHE)  # 确保本地有一份
                return c
            except Exception as e:
                print(f"  缓存读取失败({path}): {e}")
    print("  未找到缓存 → 首次全量训练")
    return {}

def save_cache_to_dataset(cache):
    """
    保存模型缓存到 Kaggle Dataset（持久存储）
    用 Kaggle API 直接上传，自动处理 创建/更新
    """
    import shutil, base64, urllib.parse

    try:
        # 序列化缓存
        with open(LOCAL_CACHE,'wb') as f: pickle.dump(cache,f)
        size_mb = os.path.getsize(LOCAL_CACHE)/1024/1024
        print(f"  模型缓存大小: {size_mb:.1f}MB")

        kgat = get_secret('KAGGLE_TOKEN') or get_secret('kaggle_token') or 'KGAT_0847d8a3c8619a4db2ff2c7c3e9e824f'
        if not kgat:
            print("  ! 未配置 KAGGLE_TOKEN，跳过缓存保存")
            return

        headers_api = {
            'Authorization': f'Bearer {kgat}',
            'Content-Type': 'application/json',
            'User-Agent': 'kaggle-bot/1.0',
        }

        # ── 1. 确保 Dataset 存在 ──────────────────────────
        check_url = f'https://www.kaggle.com/api/v1/datasets/{DATASET_ID}'
        req = urllib.request.Request(check_url, headers=headers_api)
        dataset_exists = False
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                dataset_exists = r.status == 200
        except Exception:
            dataset_exists = False

        if not dataset_exists:
            print("  Dataset 不存在，正在创建…")
            create_body = json.dumps({
                "ownerSlug": DATASET_ID.split('/')[0],
                "slug": DATASET_SLUG,
                "title": "Fucai Model Cache",
                "licenseName": "CC0-1.0",
                "isPrivate": True,
                "files": [{
                    "token": "placeholder",
                    "description": "model cache"
                }]
            }).encode()
            req2 = urllib.request.Request(
                'https://www.kaggle.com/api/v1/datasets/create/new',
                data=create_body, method='POST', headers=headers_api
            )
            try:
                with urllib.request.urlopen(req2, timeout=30) as r:
                    print(f"  Dataset 创建: HTTP {r.status}")
                    dataset_exists = True
            except Exception as e:
                print(f"  Dataset 创建失败: {e}")

        # ── 2. 用 CLI 推送（最可靠的方式）──────────────────
        os.makedirs(DATASET_DIR, exist_ok=True)
        shutil.copy(LOCAL_CACHE, os.path.join(DATASET_DIR, 'models_cache.pkl'))
        meta = {"title":"Fucai Model Cache","id":DATASET_ID,"licenses":[{"name":"CC0-1.0"}]}
        with open(os.path.join(DATASET_DIR,'dataset-metadata.json'),'w') as f:
            json.dump(meta,f)

        env = os.environ.copy()
        env['KAGGLE_API_TOKEN'] = kgat

        # 先试 version（更新），失败了再试 create
        for cmd in [
            ['kaggle','datasets','version','-p',DATASET_DIR,'-m',f'daily-{date.today()}','--dir-mode','tar'],
            ['kaggle','datasets','create', '-p',DATASET_DIR,'--dir-mode','tar'],
        ]:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
            out = result.stdout.strip(); err = result.stderr.strip()
            if result.returncode == 0:
                print(f"  ✓ 模型缓存已保存到 Kaggle Dataset: {DATASET_ID}")
                print(f"    请确认 Notebook 已挂载此 Dataset（Add Data → {DATASET_SLUG}）")
                return
            else:
                print(f"  [{' '.join(cmd[1:3])}] returncode={result.returncode}")
                if out: print(f"    stdout: {out[:150]}")
                if err: print(f"    stderr: {err[:150]}")

        print("  ! 两种方式均失败，本次缓存未持久化（不影响今日预测结果）")
        print(f"    手动操作：在 Kaggle 创建名为 [{DATASET_SLUG}] 的私有 Dataset 后重跑")

    except Exception as e:
        import traceback
        print(f"  ! 缓存保存异常: {e}")
        traceback.print_exc()

model_cache = load_cache()

# ══════════════════════════════════════════════════════
#  训练函数（增量智能：有缓存→跳过重训，只预测）
# ══════════════════════════════════════════════════════

def train_target(X, y, feat_names, last_X, tname):
    """
    增量训练策略：
    - 有缓存且新数据 < RETRAIN_EVERY：直接用缓存模型预测，只做最近10期回测
    - 无缓存或新数据 ≥ RETRAIN_EVERY：全量训练一次，回测最近50期（不重复训练）
    """
    from sklearn.preprocessing import LabelEncoder
    from collections import Counter as Ctr

    n=len(X); MIN=60
    if n<MIN+5: return None

    le=LabelEncoder(); y_enc=le.fit_transform(y)
    classes_enc = le.classes_.tolist()

    def decode(val):
        return int(le.inverse_transform([val])[0])

    cached = model_cache.get(tname, {})
    cached_n = cached.get('trained_n', 0)
    new_data  = n - cached_n
    has_cache = bool(cached.get('models'))

    # 特征维度校验：若特征工程改了（当前维度与缓存时不一致），
    # 缓存模型对新特征向量会预测出错甚至崩溃，强制作废缓存重新全量训练
    # 注意：旧缓存（本次修复之前保存的）没有 feat_count 字段，视为"维度不明"同样作废，
    # 不能因为读不到就当作兼容——那样会直接用旧维度模型预测新维度特征导致报错
    cached_feat_n = cached.get('feat_count')
    cur_feat_n = X.shape[1] if hasattr(X, 'shape') else len(feat_names)
    if has_cache and cached_feat_n != cur_feat_n:
        print(f"    [{tname}] 特征维度不匹配或未知（缓存{cached_feat_n} → 当前{cur_feat_n}），缓存作废，强制全量重训")
        has_cache = False

    if has_cache and new_data < RETRAIN_EVERY:
        # ── 增量模式：直接用缓存模型 ──
        print(f"    [{tname}] 缓存模式（缓存{cached_n}期，新增{new_data}期，跳过重训）")
        final = cached['models']
        acc   = cached.get('accuracy', {})
        fi    = cached.get('feature_importance', [])
        # 回测起点必须是缓存模型训练截止点 cached_n，而不是简单的"最近10期"，
        # 否则如果 new_data<10，回测窗口会往回覆盖到模型训练时就见过的数据，
        # 又是拿训练集当考题、准确率虚高。只在模型真正没见过的新增数据上评估才有意义。
        bt_start = max(MIN, cached_n)
    else:
        # ── 全量训练：分两组模型 ──
        # 1) 回测专用模型：只用 bt_start 之前的数据训练，在"没见过"的后续50期上真实评估
        #    （修复：之前是用全部数据训练后再"回测"最近50期，等于拿训练集本身当考题，
        #     准确率虚高到100%只是模型记住了训练数据，不代表任何真实预测能力）
        # 2) 生产模型：用全部数据训练，专门用于预测真正的下一期（这个环节本来就该用全部数据，没问题）
        print(f"    [{tname}] 全量训练（{n}期）…")
        bt_start = max(MIN, n-50)

        bt_models={}
        for mname,m in make_models().items():
            try: m.fit(X[:bt_start], y_enc[:bt_start]); bt_models[mname]=m
            except Exception as e: print(f"      {mname}(回测)失败:{e}")

        final={}
        for mname,m in make_models().items():
            try: m.fit(X,y_enc); final[mname]=m
            except Exception as e: print(f"      {mname}(生产)失败:{e}")
        if not final: return None
        acc={}; fi=[]

    # ── 回测（用 bt_models：只在训练时未见过的数据上评估，才是真实的样本外准确率）──
    bt_eval_models = bt_models if (not has_cache or new_data >= RETRAIN_EVERY) else final
    all_true=[]; preds_by={k:[] for k in bt_eval_models}
    for end in range(bt_start, n):
        Xte = X[end:end+1]; yte = y_enc[end:end+1]
        for mname,m in bt_eval_models.items():
            try: preds_by[mname].extend(m.predict(Xte).tolist())
            except: preds_by[mname].extend([int(y_enc[end-1])])
        all_true.extend(yte.tolist())

    if all_true:
        for mname,ps in preds_by.items():
            if len(ps)==len(all_true):
                acc[mname]=round(accuracy_score(all_true,ps)*100,1)
        ens=[]
        for i in range(len(all_true)):
            vs=[preds_by[m][i] for m in preds_by if len(preds_by[m])>i]
            ens.append(Ctr(vs).most_common(1)[0][0] if vs else all_true[i])
        acc['ensemble']=round(accuracy_score(all_true,ens)*100,1)

        # 基线准确率：永远预测"训练集里出现最多的那一类"能拿多少分。
        # 如果模型准确率跟基线差不多甚至更低，说明模型没有学到真实规律，
        # 只是分组本身不均衡导致"蒙对"的概率就很高（比如某一档占了70%的历史样本）。
        train_boundary = bt_start if (not has_cache or new_data >= RETRAIN_EVERY) else cached_n
        train_y = y_enc[:train_boundary]
        if len(train_y) > 0:
            majority = Ctr(train_y.tolist()).most_common(1)[0][0]
            acc['baseline'] = round(accuracy_score(all_true, [majority]*len(all_true))*100, 1)
            acc['lift_over_baseline'] = round(acc['ensemble'] - acc['baseline'], 1)

    if not has_cache or new_data >= RETRAIN_EVERY:
        if 'rf' in final and hasattr(final['rf'],'feature_importances_'):
            imp=final['rf'].feature_importances_
            fi=[{'name':str(k),'score':round(float(v),4)} for k,v in sorted(zip(feat_names,imp),key=lambda x:-x[1])[:5]]
        # 更新缓存
        model_cache[tname]={'models':final,'trained_n':n,'accuracy':acc,'feature_importance':fi,'feat_count':cur_feat_n}

    # ── 用 last_X 预测真正下一期 ──
    px = last_X if last_X is not None else X[-1:]
    all_probs=[]; classes=None
    for m in final.values():
        try:
            p=m.predict_proba(px)[0]; all_probs.append(p)
            if classes is None: classes=m.classes_.tolist()
        except: pass
    if not all_probs or classes is None: return None

    avg_prob  = np.mean(all_probs,axis=0)
    pred_enc  = classes[int(np.argmax(avg_prob))]
    pred_cls  = decode(pred_enc)
    orig_probs= {str(decode(c)):round(float(p)*100,1) for c,p in zip(classes,avg_prob)}

    bt_detail=[]
    for i in range(max(0,len(all_true)-10),len(all_true)):
        row={'true':decode(all_true[i])}
        for mname,ps in preds_by.items():
            if len(ps)>i: row[f'pred_{mname}']=decode(ps[i])
        row['hit']=int(row['true']==row.get('pred_rf',row.get('pred_xgb',-1)))
        bt_detail.append(row)

    return {'target':tname,'data_used':n,'backtest_periods':len(all_true),
            'accuracy':acc,'feature_importance':fi,
            'prediction':{'value':pred_cls,
                          'confidence':round(float(max(avg_prob))*100,1),
                          'probs':orig_probs},
            'bt_detail':bt_detail}


def tgt3d(r):
    b,s,g=r['digits']; sm=b+s+g
    is_triplet = (b==s==g)
    is_group3  = (b==s or s==g or b==g) and not is_triplet
    group_type = 0 if is_triplet else (1 if is_group3 else 2)   # 0=豹子 1=组三 2=组六（真实分布：组六≈72% 组三≈27% 豹子≈1%）
    span = max(b,s,g) - min(b,s,g)
    road = [b%3, s%3, g%3]
    road_dom = max(set(road), key=road.count)   # 三位数字里012路哪个占多数
    sorted3 = sorted([b,s,g])
    is_arith = int((sorted3[1]-sorted3[0])==(sorted3[2]-sorted3[1]) and sorted3[2]-sorted3[0]>0)
    return {'sum_grp':0 if sm<=9 else(1 if sm<=17 else 2),
            'odd':sum(1 for x in [b,s,g] if x%2!=0),
            'group_type':group_type,
            'big':sum(1 for x in [b,s,g] if x>=5),
            'span_grp':0 if span<=3 else(1 if span<=6 else 2),
            'road_dom':road_dom,
            'arith':is_arith}
def tgtssq(r):
    red = sorted(r['red'])
    sm = sum(red)
    # AC值
    diffs=set()
    for i in range(len(red)):
        for j in range(i+1,len(red)): diffs.add(red[j]-red[i])
    ac = len(diffs)-(len(red)-1)
    # 三区主力区
    z1=sum(1 for x in red if x<=11); z2=sum(1 for x in red if 12<=x<=22); z3=sum(1 for x in red if x>=23)
    zone_dom = int(np.argmax([z1,z2,z3]))
    # 最大间距区间
    max_gap = max(red[i+1]-red[i] for i in range(len(red)-1)) if len(red)>1 else 0
    # 连号
    consec = sum(1 for i in range(len(red)-1) if red[i+1]-red[i]==1)
    return {'odd':sum(1 for x in r['red'] if x%2!=0),
            'sum_grp':0 if sm<70 else(1 if sm<100 else 2),
            'ac_grp':0 if ac<=2 else(1 if ac<=5 else 2),
            'red_zone_dom':zone_dom,
            'gap_grp':0 if max_gap<=5 else(1 if max_gap<=10 else 2),
            'big':sum(1 for x in r['red'] if x>16),
            'consec':consec}
def tgtkl8(r):
    nums = sorted(r['numbers'])
    odd=sum(1 for x in r['numbers'] if x%2!=0)
    big=sum(1 for x in r['numbers'] if x>40)
    zn=[sum(1 for x in r['numbers'] if lo<=x<=hi) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]]
    five=[sum(1 for x in r['numbers'] if lo<=x<=hi) for lo,hi in [(1,16),(17,32),(33,48),(49,64),(65,80)]]
    tt=sum(r['numbers'])
    cg=0; inc=False
    for i in range(len(nums)-1):
        if nums[i+1]-nums[i]==1:
            if not inc: cg+=1; inc=True
        else: inc=False
    rng = nums[-1]-nums[0]
    return {'odd_grp':0 if odd<9 else(1 if odd<=11 else 2),
            'zone_dom':int(np.argmax(zn)),
            'tot_grp':0 if tt<640 else(1 if tt<820 else 2),
            'big_grp':0 if big<9 else(1 if big<=11 else 2),
            'five_dom':int(np.argmax(five)),
            'consec_grp':0 if cg==0 else(1 if cg<=2 else 2),
            'range_grp':0 if rng<60 else(1 if rng<70 else 2)}

# 目标变量（单条记录）
def markov3d(records):
    trans=[defaultdict(Counter) for _ in range(3)]
    for i in range(1,len(records)):
        pv=records[i-1]['digits']; cv=records[i]['digits']
        for p in range(3): trans[p][pv[p]][cv[p]]+=1
    last=records[-1]['digits']; out=[]
    for p in range(3):
        probs=trans[p][last[p]]; tot=sum(probs.values()) or 1
        out.append({'pos':p,'from':last[p],'top3':[[int(k),round(v/tot*100,1)] for k,v in sorted(probs.items(),key=lambda x:-x[1])[:3]]})
    return out
def markov_blue(records):
    trans=defaultdict(Counter)
    for i in range(1,len(records)): trans[records[i-1]['blue']][records[i]['blue']]+=1
    last=records[-1]['blue']; probs=trans[last]; tot=sum(probs.values()) or 1
    return [[int(k),round(v/tot*100,1)] for k,v in sorted(probs.items(),key=lambda x:-x[1])[:5]]

# 遗漏
def omit3d(records):
    out=[]
    for pos in range(3):
        om={}
        for d in range(10):
            for i in range(len(records)-1,-1,-1):
                if records[i]['digits'][pos]==d: om[d]=len(records)-1-i; break
            else: om[d]=len(records)
        out.append({'pos':pos,'omission':om,'overdue':[d for d,v in om.items() if v>10]})
    return out
def omitssq(records):
    om_r={}
    for n in range(1,34):
        for i in range(len(records)-1,-1,-1):
            if n in records[i]['red']: om_r[n]=len(records)-1-i; break
        else: om_r[n]=len(records)
    om_b={}
    for n in range(1,17):
        for i in range(len(records)-1,-1,-1):
            if records[i]['blue']==n: om_b[n]=len(records)-1-i; break
        else: om_b[n]=len(records)
    avg_r=len(records)*6/33; avg_b=len(records)/16
    return {'red_omit':om_r,'red_overdue':[n for n,v in om_r.items() if v>avg_r*1.5],
            'blue_omit':om_b,'blue_overdue':[n for n,v in om_b.items() if v>avg_b*1.5]}
def omitkl8(records):
    om={}
    for n in range(1,81):
        for i in range(len(records)-1,-1,-1):
            if n in records[i]['numbers']: om[n]=len(records)-1-i; break
        else: om[n]=len(records)
    avg=len(records)*20/80
    return {'omit':om,'overdue':sorted([n for n,v in om.items() if v>avg*1.5],key=lambda n:om[n],reverse=True)[:20],'avg':round(avg,1)}

# 推荐
def rec3d(records, ml, mk, om):
    """
    3D推荐6注：综合ML概率+马尔可夫+遗漏三路投票
    各位取概率前5候选，加权随机生成6组不重复号码
    """
    score = [{} for _ in range(3)]

    # ML概率贡献（权重45%）
    for pi, pname in enumerate(['bai','shi','ge']):
        m = ml.get(pname, {})
        probs = m.get('prediction', {}).get('probs', {}) if m else {}
        for k, v in probs.items():
            score[pi][int(k)] = score[pi].get(int(k), 0) + float(v) * 0.45

    # 马尔可夫转移贡献（权重35%）
    for mk_item in (mk or []):
        pos = mk_item['pos']
        for val, prob in mk_item['top3']:
            score[pos][val] = score[pos].get(val, 0) + prob * 0.35

    # 遗漏贡献（权重20%）：遗漏越久加分越多
    for om_item in (om or []):
        pos = om_item['pos']
        om_dict = om_item.get('omission', {})
        avg = om_item.get('avg', 10)
        for d, cnt in om_dict.items():
            if cnt > avg:
                bonus = min((cnt - avg) / avg * 8, 15)  # 上限15分
                score[pos][int(d)] = score[pos].get(int(d), 0) + bonus * 0.20

    # 各位TOP3候选（带权重）
    tops = [sorted(score[p].items(), key=lambda x: -x[1])[:3] for p in range(3)]
    candidates = [[int(v) for v, _ in t] for t in tops]

    # 生成6注：轮转+择优（与强化学习脚本同一思路，不再用加权随机抽样）
    # 前3注让每位的第1/2/3候选各当一次主角，保证模型看好的候选都能出场；
    # 后3注从27种组合里按综合权重择优补足。
    # 之所以不用纯"权重最高的Top6"：三位权重相乘时，某位第1名只要略高于第2名，
    # 优势会被乘法放大，导致6注里那一位被同一个数字垄断，其余候选全被浪费。
    groups, seen = [], set()
    for i in range(3):
        combo = [int(tops[p][i][0]) if i < len(tops[p]) else int(tops[p][0][0]) for p in range(3)]
        if tuple(combo) not in seen:
            seen.add(tuple(combo)); groups.append(combo)

    from itertools import product
    scored_all = []
    for bi, (b, wb) in enumerate(tops[0]):
        for si, (s, ws) in enumerate(tops[1]):
            for gi, (g, wg) in enumerate(tops[2]):
                scored_all.append(([int(b), int(s), int(g)],
                                   max(wb, .01) * max(ws, .01) * max(wg, .01)))
    scored_all.sort(key=lambda x: -x[1])
    for combo, _ in scored_all:
        if len(groups) >= 6: break
        if tuple(combo) not in seen:
            seen.add(tuple(combo)); groups.append(combo)
    # 极端情况下候选不足，用各位首选补齐
    while len(groups) < 6 and candidates[0]:
        groups.append([candidates[0][0], candidates[1][0], candidates[2][0]])

    sm_m = ml.get('sum_grp', {})
    sm_pred = sm_m.get('prediction', {}).get('value', 1) if sm_m else 1
    sm_label = {0:'小(0-9)', 1:'中(10-17)', 2:'大(18-27)'}.get(sm_pred, '中(10-17)')
    mk_hint = [f"{'百十个'[it['pos']]}位→可能{'、'.join(str(v) for v,_ in it['top3'][:3])}" for it in (mk or [])]

    # ── 衍生玩法推荐：组选3/6、和值大小、和值奇偶、跨度 ──
    # 全部从已有的group_type/sum_grp/odd/span_grp预测推算，不需要额外训练目标
    derived = {}

    # 组选类型：直接用group_type预测
    gt_m = ml.get('group_type', {})
    if gt_m:
        gt_pred = gt_m.get('prediction', {}).get('value', 2)
        gt_conf = gt_m.get('prediction', {}).get('confidence', 0)
        gt_label = {0:'豹子（三同号）', 1:'组三', 2:'组六'}.get(gt_pred, '组六')
        bet_type = '直选' if gt_pred==0 else (f'组选3' if gt_pred==1 else '组选6')
        derived['group_bet'] = {'name':'组选类型', 'pred':gt_label, 'confidence':gt_conf,
                                 'suggest':f'建议投注：{bet_type}'}

    # 和值大小：用sum_grp的概率分布估算P(大)，阈值≈13.5（sum>=14为大）
    if sm_m:
        sm_probs = sm_m.get('prediction', {}).get('probs', {})
        p_low = float(sm_probs.get('0', 0))/100.0
        p_mid = float(sm_probs.get('1', 0))/100.0
        p_high = float(sm_probs.get('2', 0))/100.0
        p_big = p_high + 0.5*p_mid   # 中档大约一半落在大、一半落在小
        big_label = '大' if p_big >= 0.5 else '小'
        derived['sum_big_small'] = {'name':'和值大小', 'pred':big_label,
                                     'confidence':round(max(p_big,1-p_big)*100,1),
                                     'suggest':f'建议投注：和值{big_label}（估计概率{round(max(p_big,1-p_big)*100,1)}%）'}

    # 和值奇偶：数学恒等关系——奇数个数的奇偶性 = 和值的奇偶性（偶数不影响奇偶，每个奇数贡献1）
    odd_m = ml.get('odd', {})
    if odd_m:
        odd_pred = odd_m.get('prediction', {}).get('value', 1)
        odd_conf = odd_m.get('prediction', {}).get('confidence', 0)
        sum_parity = '奇' if odd_pred % 2 == 1 else '偶'
        derived['sum_odd_even'] = {'name':'和值奇偶', 'pred':sum_parity, 'confidence':odd_conf,
                                    'suggest':f'建议投注：和值{sum_parity}（由奇数个数预测={odd_pred}推算，数学恒等关系，非独立预测）'}

    # 跨度：直接用span_grp
    span_m = ml.get('span_grp', {})
    if span_m:
        span_pred = span_m.get('prediction', {}).get('value', 1)
        span_conf = span_m.get('prediction', {}).get('confidence', 0)
        span_label = {0:'小跨度(0-3)', 1:'中跨度(4-6)', 2:'大跨度(7-9)'}.get(span_pred, '中跨度(4-6)')
        derived['span_bet'] = {'name':'跨度', 'pred':span_label, 'confidence':span_conf,
                                'suggest':f'建议投注：{span_label}'}

    return {
        'groups': groups[:6],
        'pos_candidates': candidates,
        'sum_pred': sm_label,
        'markov_hint': mk_hint,
        'overdue': [it.get('overdue', []) for it in (om or [])],
        'derived_plays': derived,
        'note': '综合ML概率+马尔可夫转移+遗漏分析三路投票，预测下一期特征生成参考号码，仅供娱乐。'
    }


def recssq(records, ml, om):
    """
    双色球推荐6注：
    - 蓝球：从ML概率前5候选中加权选取，每注蓝球不同
    - 红球：三区均衡（1-11/12-22/23-33各2个），结合热号+遗漏+ML奇偶预测
    """
    freq30 = Counter(n for r in records[-30:] for n in r['red'])
    hot = [x[0] for x in freq30.most_common(15)]
    overdue_r = om.get('red_overdue', []) if om else []
    overdue_b = om.get('blue_overdue', []) if om else []

    # ── 蓝球评分 ──
    # 注意：ML的训练目标里已不含 'blue'（单个球号接近均匀噪声，训练意义不大），
    # 所以这里不能再指望 ml.get('blue')，否则永远落到兜底的"近30期最高频前5"，
    # 而该统计新增一期通常只让5个数字内部换序、集合不变，表现为推荐长期纹丝不动。
    # 改为对16个蓝球逐个评分：热度 + 遗漏 + 近期趋势，三者都会随新开奖而变化。
    blue_scores_ml = {}
    _b_recent30 = [r['blue'] for r in records[-30:]]
    _b_recent10 = [r['blue'] for r in records[-10:]]
    _bf30, _bf10 = Counter(_b_recent30), Counter(_b_recent10)
    _bmax30 = max(_bf30.values()) if _bf30 else 1
    _bmax10 = max(_bf10.values()) if _bf10 else 1
    _b_last = {}
    for i, b in enumerate(_b_recent30): _b_last[b] = i
    for b in range(1, 17):
        s  = 0.30 * (_bf30.get(b, 0) / _bmax30)                     # 中期热度
        s += 0.30 * (_bf10.get(b, 0) / _bmax10)                     # 近期趋势(权重同等，让新开奖更快体现)
        omit = len(_b_recent30) - 1 - _b_last[b] if b in _b_last else len(_b_recent30)
        s += 0.40 * min(omit / max(len(_b_recent30), 1), 1.0)       # 遗漏回补
        blue_scores_ml[b] = s
    blue_top = [b for b, _ in sorted(blue_scores_ml.items(), key=lambda x: -x[1])[:5]]

    # 奇偶预测
    odd_m = ml.get('odd', {})
    odd_pred = odd_m.get('prediction', {}).get('value', 3) if odd_m else 3

    # 和值区间预测
    sm_m = ml.get('sum_grp', {})
    sm_label = {0:'低(<70)', 1:'中(70-99)', 2:'高(≥100)'}.get(
        sm_m.get('prediction', {}).get('value', 1) if sm_m else 1, '中(70-99)')
    sm_val = sm_m.get('prediction', {}).get('value', 1) if sm_m else 1

    # 给33个红球逐个打分（ML三区概率+热度+遗漏），让每个号码都有依据
    ssq_scores = score_balls(
        33, records, ml,
        zone_defs=[(1,11),(12,22),(23,33)], zone_key='red_zone_dom',
        get_nums=lambda r: r['red'])
    base_bets, red_core, red_pool = picks_from_scores(ssq_scores, 6, 6)

    def _tune(picked):
        """
        按ML的奇偶/和值预测做定向微调：不满足时用候选池里分数最高的合适球替换，
        而不是像以前那样"整注推倒重新随机生成"——那样号码全靠碰运气。
        """
        picked = list(picked)
        ranked = [n for n, _ in sorted(ssq_scores.items(), key=lambda x: -x[1])]
        # 奇偶调整
        for _ in range(6):
            cur_odd = sum(1 for n in picked if n % 2 != 0)
            if abs(cur_odd - odd_pred) <= 1: break
            need_odd = cur_odd < odd_pred
            out = next((n for n in reversed(picked) if (n % 2 != 0) != need_odd), None)
            inn = next((n for n in ranked if n not in picked and (n % 2 != 0) == need_odd), None)
            if out is None or inn is None: break
            picked[picked.index(out)] = inn
        # 和值调整（0=低<70, 2=高>=100）
        for _ in range(6):
            cur_sum = sum(picked)
            if sm_val == 0 and cur_sum < 100: break
            if sm_val == 2 and cur_sum >= 70: break
            if sm_val == 1: break
            need_small = (sm_val == 0)
            out = max(picked) if need_small else min(picked)
            inn = next((n for n in ranked if n not in picked and
                        (n < out if need_small else n > out)), None)
            if inn is None: break
            picked[picked.index(out)] = inn
        return sorted(picked)

    groups, seen = [], set()
    for i, bet in enumerate(base_bets):
        picked = _tune(bet)
        key = tuple(picked)
        if key in seen:   # 极少数情况下微调后撞车，用原始未微调的那注
            picked = sorted(bet); key = tuple(picked)
            if key in seen: continue
        seen.add(key)
        blue = blue_top[len(groups) % len(blue_top)] if blue_top else 1
        groups.append({'red': picked, 'blue': int(blue)})

    return {
        'groups': groups,
        'blue_recommend': [int(x) for x in blue_top[:5]],
        'hot_red': [int(x) for x in hot[:12]],
        'overdue_red': [int(x) for x in overdue_r[:8]],
        'overdue_blue': [int(x) for x in overdue_b[:4]],
        'odd_pred': int(odd_pred),
        'sum_pred': sm_label,
        'note': '综合ML预测+遗漏分析+三区均衡选号，按奇偶/和值约束生成6注参考，仅供娱乐。'
    }


def score_balls(pool_max, records, ml, zone_defs, zone_key, five_defs=None, five_key=None,
                get_nums=lambda r: r['numbers'], recent_n=30):
    """
    给号池里每个球算一个综合分数，让推荐的每个号码都有可追溯的依据。

    ── 为什么要有这个 ──
    原来的做法是"确定各区选几个 → random.sample 随机抽"，
    即模型只决定了"哪个区多选几个"，具体抽中哪个球完全由随机数决定，
    号码本身讲不出理由。现在改成每个球都有分数，按分数选。

    分数构成（都是有明确含义的真实信号）：
      · ML区间预测概率：球所在区间被模型看好的程度
      · ML五行段预测概率：同上（快乐8专用，其它游戏传None跳过）
      · 近期热度：近recent_n期出现次数（归一化）
      · 遗漏程度：多久没出现（归一化），与热度互补
    """
    freq = Counter(n for r in records[-recent_n:] for n in get_nums(r))
    last_seen = {}
    for i, r in enumerate(records[-recent_n:]):
        for n in get_nums(r): last_seen[n] = i
    total = len(records[-recent_n:])

    zone_probs_raw = (ml.get(zone_key, {}) or {}).get('prediction', {}).get('probs', {})
    zone_w = [float(zone_probs_raw.get(str(i), 100.0/len(zone_defs))) for i in range(len(zone_defs))]
    zmax = max(zone_w) or 1.0

    if five_defs and five_key:
        five_raw = (ml.get(five_key, {}) or {}).get('prediction', {}).get('probs', {})
        five_w = [float(five_raw.get(str(i), 100.0/len(five_defs))) for i in range(len(five_defs))]
        fmax = max(five_w) or 1.0
    else:
        five_w = None

    fmax_cnt = max(freq.values()) if freq else 1
    scores = {}
    for n in range(1, pool_max + 1):
        s = 0.0
        # 权重配比说明：区间/五行预测提供"方向性"，但不能占比过高，
        # 否则会把号码全部压到单一区间里（实测占40%时18个号码100%挤在同一区），
        # 反而失去分布合理性。热度和遗漏提供号码个体差异，权重给足。
        for zi, (lo, hi) in enumerate(zone_defs):
            if lo <= n <= hi: s += 0.22 * (zone_w[zi] / zmax); break
        if five_w:
            for fi, (lo, hi) in enumerate(five_defs):
                if lo <= n <= hi: s += 0.13 * (five_w[fi] / fmax); break
        s += 0.35 * (freq.get(n, 0) / fmax_cnt)                       # 热度
        omit = total - 1 - last_seen[n] if n in last_seen else total  # 遗漏
        s += 0.30 * min(omit / max(total, 1), 1.0)
        scores[n] = s
    return scores


def picks_from_scores(scores, n_pick, n_bets, pool_mult=2.2, core_ratio=0.34):
    """
    按分数选号：胆码 + 拖码轮转（与强化学习脚本同一套思路）。
    分数最高的少数球作为胆码进每一注，其余候选轮转填充，
    既保证号码有依据（全部按分数排序而来），又让各注之间有实质差异。
    """
    order = [n for n, _ in sorted(scores.items(), key=lambda x: -x[1])]
    pool_size = min(len(order), max(n_pick + 2, int(round(n_pick * pool_mult))))
    pool = order[:pool_size]
    core_n = max(1, min(n_pick - 1, int(round(n_pick * core_ratio))))
    core, rest = pool[:core_n], pool[core_n:]
    bets, ri = [], 0
    for _ in range(n_bets):
        sel = list(core); guard = 0
        while len(sel) < n_pick and rest and guard < len(rest) * 3:
            c = rest[ri % len(rest)]; ri += 1; guard += 1
            if c not in sel: sel.append(c)
        oi = 0
        while len(sel) < n_pick and oi < len(order):
            if order[oi] not in sel: sel.append(order[oi])
            oi += 1
        bets.append(sorted(sel))
    return bets, sorted(core), sorted(pool)


def reckl8(records, ml, om):
    """
    快乐8多玩法推荐：
    选四3注/选五3注/选五复式1注/选六3注/选九2注/选十1注（共13注）
    """
    from math import factorial
    def comb(n, k): return factorial(n) // (factorial(k) * factorial(n-k)) if k <= n else 0

    freq30 = Counter(n for r in records[-30:] for n in r['numbers'])
    freq10 = Counter(n for r in records[-10:] for n in r['numbers'])
    hot    = [x[0] for x in freq30.most_common(20)]
    overdue = om.get('overdue', []) if om else []

    # ML区间预测
    zone_m    = ml.get('zone_dom', {})
    zone_pred = zone_m.get('prediction', {}).get('value', 1) if zone_m else 1
    zone_name = {0:'1-20区', 1:'21-40区', 2:'41-60区', 3:'61-80区'}.get(zone_pred, '21-40区')

    # 区间概率：决定各区多选几个
    zone_probs_raw = zone_m.get('prediction', {}).get('probs', {}) if zone_m else {}
    zone_weights = [float(zone_probs_raw.get(str(i), 25)) for i in range(4)]

    tot_m     = ml.get('tot_grp', {})
    tot_label = {0:'低(<640)', 1:'中(640-819)', 2:'高(≥820)'}.get(
        tot_m.get('prediction', {}).get('value', 1) if tot_m else 1, '中(640-819)')

    # 五行分布ML预测：给预测出的主力段额外加权候选
    five_m = ml.get('five_dom', {})
    five_pred = five_m.get('prediction', {}).get('value', 2) if five_m else 2
    five_ranges = [(1,16),(17,32),(33,48),(49,64),(65,80)]
    five_name = ['1-16','17-32','33-48','49-64','65-80'][five_pred if 0<=five_pred<=4 else 2]
    five_lo, five_hi = five_ranges[five_pred if 0<=five_pred<=4 else 2]
    five_bonus = [n for n in range(five_lo, five_hi+1) if freq30.get(n,0)>0][:6]

    big_m = ml.get('big_grp', {})
    big_pred = big_m.get('prediction', {}).get('value', 1) if big_m else 1
    big_label = {0:'少(<9个)',1:'中(9-11个)',2:'多(≥12个)'}.get(big_pred, '中(9-11个)')

    consec_m = ml.get('consec_grp', {})
    consec_label = {0:'无连续',1:'少量连续(1-2组)',2:'较多连续(≥3组)'}.get(
        consec_m.get('prediction',{}).get('value',1) if consec_m else 1, '少量连续(1-2组)')

    range_m = ml.get('range_grp', {})
    range_label = {0:'集中(极差<60)',1:'适中(60-69)',2:'分散(≥70)'}.get(
        range_m.get('prediction',{}).get('value',1) if range_m else 1, '适中(60-69)')

    # 注：原来这里有个"热号+遗漏+随机补齐"的候选池，供 random.sample 抽号用。
    # 现在选号改为按 score_balls 打分排序，该候选池已无用，一并移除。

    # 给80个球逐个打分（ML区间概率+五行概率+热度+遗漏），号码从此有据可查
    kl8_scores = score_balls(
        80, records, ml,
        zone_defs=[(1,20),(21,40),(41,60),(61,80)], zone_key='zone_dom',
        five_defs=[(1,16),(17,32),(33,48),(49,64),(65,80)], five_key='five_dom',
        get_nums=lambda r: r['numbers'])
    _kl8_bets = {}
    _kl8_core = {}
    for _n, _cnt in [(4,3), (5,3), (6,3), (8,1), (9,2), (10,1)]:
        _b, _c, _p = picks_from_scores(kl8_scores, _n, _cnt)
        _kl8_bets[_n], _kl8_core[_n] = _b, _c

    def pick_balanced(n, idx=0):
        """取该玩法第 idx+1 注（已按分数选出，各注间由胆码+拖码轮转保证差异）"""
        bets = _kl8_bets.get(n, [])
        if idx < len(bets): return bets[idx]
        # 兜底：索引越界时按序错开取号，而不是每次都返回同一批分数最高的球
        ranked = [x for x, _ in sorted(kl8_scores.items(), key=lambda t: -t[1])]
        off = idx % max(1, len(ranked) - n)
        return sorted(ranked[off:off + n])

    plays = {
        'xuan4':    {'name':'选四',     'balls':4,  'tip':'4球全中，赔率最高',
                     'groups': [pick_balanced(4, i) for i in range(3)]},
        'xuan5':    {'name':'选五',     'balls':5,  'tip':'5球，赔率与命中率均衡',
                     'groups': [pick_balanced(5, i) for i in range(3)]},
        'xuan5_fu': {'name':'选五复式', 'balls':5,  'tip':f'8球覆盖C(8,5)={comb(8,5)}注',
                     'groups': [pick_balanced(8, 0)]},
        'xuan6':    {'name':'选六',     'balls':6,  'tip':'6球，主流推荐玩法',
                     'groups': [pick_balanced(6, i) for i in range(3)]},
        'xuan9':    {'name':'选九',     'balls':9,  'tip':'9球，高赔率搏奖',
                     'groups': [pick_balanced(9, i) for i in range(2)]},
        'xuan10':   {'name':'选十',     'balls':10, 'tip':'10球，最高赔率',
                     'groups': [pick_balanced(10, 0)]},
    }

    return {
        'plays':              plays,
        'zone_dominant_pred': zone_name,
        'total_range_pred':   tot_label,
        'five_dominant_pred': five_name,
        'big_count_pred':     big_label,
        'consec_pred':        consec_label,
        'range_pred':         range_label,
        'hot_nums':  [int(x) for x in hot[:15]],
        'cold_nums': [int(x) for x in [x[0] for x in freq30.most_common()[-10:]]],
        'overdue':   [int(x) for x in overdue[:10]],
        'note': '综合ML区间预测(区间/五行/大数/连续/极差)+遗漏分析+区间均衡，选四3注/选五3注/复式1注/选六3注/选九2注/选十1注，仅供娱乐。'
    }

# 回测
def daily_backtest(history, prev_pred):
    report={'date':str(date.today()),'games':{}}
    for game in ['3d','ssq','kl8']:
        recs=history.get(game,[])
        if not isinstance(recs,list) or not recs or game not in (prev_pred or {}): continue
        latest=recs[-1]; rec=prev_pred[game].get('recommendation',{})
        # 显式记录：这份推荐是什么时候生成的、实际对比的是哪一期开奖
        # （避免"回测日期"这种模糊标签让人无法验证预测和开奖到底对不对得上）
        pred_generated_at = prev_pred[game].get('updated_at','未知')
        actual_qihao = latest.get('qihao','—')
        actual_date  = latest.get('date','—')
        meta = {'prediction_generated_at':pred_generated_at,
                'actual_qihao':actual_qihao,'actual_date':actual_date}
        if game=='3d':
            actual=latest.get('digits',[]); grps=rec.get('groups',[])
            hits=[g for g in grps if g==actual]
            part=[g for g in grps if sum(1 for i,v in enumerate(g) if i<len(actual) and v==actual[i])>=2]
            report['games'][game]={**meta,'actual':actual,'hit_count':len(hits),'partial_count':len(part)}
        elif game=='ssq':
            ar=sorted(latest.get('red',[])); ab=latest.get('blue',0); grps=rec.get('groups',[])
            res=[{'red_hit':len(set(g.get('red',[]))&set(ar)),'blue_hit':int(g.get('blue')==ab)} for g in grps]
            report['games'][game]={**meta,'actual_red':ar,'actual_blue':ab,'group_results':res,
                                   'best_red_hit':max((x['red_hit'] for x in res),default=0)}
        elif game=='kl8':
            actual=set(latest.get('numbers',[])); plays=rec.get('plays',{})
            play_results={}
            for pk,pd in plays.items():
                grps=pd.get('groups',[]); balls=pd.get('balls',0); name=pd.get('name',pk)
                gr=[{'hit':len(actual&set(g if isinstance(g,list) else [])),'balls':balls,'won':len(actual&set(g if isinstance(g,list) else []))==balls} for g in grps]
                play_results[pk]={'name':name,'balls':balls,'groups':gr,'any_won':any(x['won'] for x in gr),'best_hit':max((x['hit'] for x in gr),default=0)}
            report['games'][game]={**meta,'actual':sorted(actual),'play_results':play_results,'best_hit':max((pr['best_hit'] for pr in play_results.values()),default=0)}
    return report

def run_ml(history):
    print(f"\n{'='*50}")
    print(f"ML训练  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    prev_pred={}
    raw=gh_raw('prediction.json')
    if raw:
        try: prev_pred=json.loads(raw).get('predictions',{})
        except Exception: pass

    bt=daily_backtest(history, prev_pred)

    cfg=[
        ('3d',  f3d,   tgt3d,  ['sum_grp','odd','group_type','big','span_grp','road_dom','arith']),
        ('ssq', fssq,  tgtssq, ['odd','sum_grp','ac_grp','red_zone_dom','gap_grp','big','consec']),
        ('kl8', fkl8,  tgtkl8, ['odd_grp','zone_dom','tot_grp','big_grp','five_dom','consec_grp','range_grp']),
    ]
    predictions={}
    for game,feat_fn,tgt_fn,tkeys in cfg:
        records=history.get(game,[])
        if not isinstance(records,list) or len(records)<65:
            print(f"\n── {game}: 数据不足({len(records) if isinstance(records,list) else 0}期)，跳过"); continue
        print(f"\n── {game}: {len(records)}期数据")
        X, Y, names, last_X = build_dataset(records, feat_fn, tgt_fn, tkeys)
        ml_res={'data_count':len(records),'updated_at':datetime.now().strftime('%Y-%m-%d %H:%M'),'models':{}}
        for tname in tkeys:
            y = np.array(Y[tname])
            r=train_target(X, y, names, last_X, f"{game}_{tname}")
            if r:
                ml_res['models'][tname]=r
                acc=r.get('accuracy',{}); pred=r.get('prediction',{})
                print(f"    集成{acc.get('ensemble','—')}%  预测下一期→{pred.get('value','?')}(置信{pred.get('confidence','?')}%)")
        if game=='3d':
            mk=markov3d(records); om=omit3d(records)
            rec=rec3d(records,ml_res['models'],mk,om)
            ai_ctx={'game':'福彩3D','data_count':len(records),'latest_date':records[-1]['date'],
                    'markov_hint':rec.get('markov_hint',[]),'overdue':rec.get('overdue',[]),
                    'sum_pred':rec.get('sum_pred',''),'recommend_groups':rec.get('groups',[])}
        elif game=='ssq':
            mk=markov_blue(records); om=omitssq(records)
            rec=recssq(records,ml_res['models'],om)
            ac_m = ml_res['models'].get('ac_grp',{})
            ac_label = {0:'低(≤2)',1:'中(3-5)',2:'高(≥6)'}.get(ac_m.get('prediction',{}).get('value',1) if ac_m else 1,'中(3-5)')
            zd_m = ml_res['models'].get('red_zone_dom',{})
            zd_label = {0:'一区(1-11)',1:'二区(12-22)',2:'三区(23-33)'}.get(zd_m.get('prediction',{}).get('value',1) if zd_m else 1,'—')
            gap_m = ml_res['models'].get('gap_grp',{})
            gap_label = {0:'小(≤5)',1:'中(6-10)',2:'大(≥11)'}.get(gap_m.get('prediction',{}).get('value',1) if gap_m else 1,'—')
            ai_ctx={'game':'双色球','data_count':len(records),'latest_date':records[-1]['date'],
                    'blue_recommend':rec.get('blue_recommend',[]),'overdue_red':rec.get('overdue_red',[]),
                    'overdue_blue':rec.get('overdue_blue',[]),'odd_pred':rec.get('odd_pred',''),'sum_pred':rec.get('sum_pred',''),
                    'ac_pred':ac_label,'red_zone_dom_pred':zd_label,'max_gap_pred':gap_label}
        else:
            mk=None; om=omitkl8(records)
            rec=reckl8(records,ml_res['models'],om)
            ai_ctx={'game':'快乐8','data_count':len(records),'latest_date':records[-1]['date'],
                    'zone_pred':rec.get('zone_dominant_pred',''),'total_pred':rec.get('total_range_pred',''),
                    'five_pred':rec.get('five_dominant_pred',''),'big_pred':rec.get('big_count_pred',''),
                    'consec_pred':rec.get('consec_pred',''),'range_pred':rec.get('range_pred',''),
                    'overdue':rec.get('overdue',[]),'hot_nums':rec.get('hot_nums',[])}
        predictions[game]={**ml_res,'recommendation':rec,'ai_context':ai_ctx}
        print(f"  ✓ {game} 完成")

    # 保存模型缓存到 Kaggle Dataset（下次运行直接读取，跳过全量训练）
    print("\n保存模型缓存到 Kaggle Dataset…")
    save_cache_to_dataset(model_cache)

    return predictions, bt

# ══════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════
print(f"\n{'#'*55}")
print(f"福彩全能脚本 Kaggle版  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'#'*55}")

# 1. 抓数据
latest_out, history_out, crawl_ok = crawl_all()

# 2. ML训练
predictions, bt = run_ml(history_out)

# 3. 推送到 GitHub
ts = datetime.now().strftime('%Y-%m-%d %H:%M')
msg = f"Kaggle 自动更新 {ts}"

if not GH_TOKEN or not GH_REPO:
    print("\n[DRY RUN] 未配置 GH_TOKEN/GH_REPO，跳过推送")
    print("prediction.json 预览（前300字）:")
    out = {'updated_at':ts,'source':'kaggle','predictions':predictions,'backtest':bt}
    print(json.dumps(out,ensure_ascii=False)[:300])
else:
    print(f"\n{'='*50}")
    print("推送文件到 GitHub…")
    # latest.json
    gh_put('latest.json', json.dumps(latest_out,ensure_ascii=False,indent=2), msg)
    print("  ✓ latest.json")
    # history.json（不缩进节省体积）
    gh_put('history.json', json.dumps(history_out,ensure_ascii=False), msg)
    print("  ✓ history.json")
    # prediction.json —— 基础ML结果（RF/XGB/LGB/马尔可夫/遗漏），保持原样不动
    # 深度学习(LSTM/Transformer)和强化学习(PPO)结果分别写在独立文件 dl_lstm_tfm.json / dl_rl.json 里
    pred_out={'updated_at':ts,'source':'kaggle',
              'models_used':{'rf':HAS_SKL,'xgb':HAS_XGB,'lgb':HAS_LGB,'ensemble':True,'markov':True,'omission':True},
              'predictions':predictions,'backtest':bt,
              'disclaimer':'彩票开奖具有完全随机性，ML预测仅为数据统计演示，仅供娱乐参考，请理性购彩。'}
    gh_put('prediction.json', json.dumps(pred_out,ensure_ascii=False,indent=2), msg)
    print("  ✓ prediction.json")
    print(f"\n✅ 全部完成！{ts}")
