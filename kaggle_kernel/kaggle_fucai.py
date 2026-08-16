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

    if has_cache and new_data < RETRAIN_EVERY:
        # ── 增量模式：直接用缓存模型 ──
        print(f"    [{tname}] 缓存模式（缓存{cached_n}期，新增{new_data}期，跳过重训）")
        final = cached['models']
        acc   = cached.get('accuracy', {})
        fi    = cached.get('feature_importance', [])
        # 只回测最新10期
        bt_start = max(MIN, n-10)
    else:
        # ── 全量训练（只训练一次！）──
        print(f"    [{tname}] 全量训练（{n}期）…")
        final={}
        for mname,m in make_models().items():
            try: m.fit(X,y_enc); final[mname]=m
            except Exception as e: print(f"      {mname}失败:{e}")
        if not final: return None

        # 回测：用已训练好的模型滚动预测（不重复fit！）
        # 只看模型对最近50期的预测表现
        bt_start = max(MIN, n-50)
        acc={}; fi=[]

    # ── 回测（用已有 final 模型，不重新训练）──
    all_true=[]; preds_by={k:[] for k in final}
    for end in range(bt_start, n):
        Xte = X[end:end+1]; yte = y_enc[end:end+1]
        for mname,m in final.items():
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

    if not has_cache or new_data >= RETRAIN_EVERY:
        if 'rf' in final and hasattr(final['rf'],'feature_importances_'):
            imp=final['rf'].feature_importances_
            fi=[{'name':str(k),'score':round(float(v),4)} for k,v in sorted(zip(feat_names,imp),key=lambda x:-x[1])[:5]]
        # 更新缓存
        model_cache[tname]={'models':final,'trained_n':n,'accuracy':acc,'feature_importance':fi}

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


def tgt3d(r): b,s,g=r['digits']; sm=b+s+g; return {'bai':b,'shi':s,'ge':g,'sum_grp':0 if sm<=9 else(1 if sm<=17 else 2),'odd':sum(1 for x in [b,s,g] if x%2!=0)}
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
    return {'blue':r['blue'],'odd':sum(1 for x in r['red'] if x%2!=0),
            'sum_grp':0 if sm<70 else(1 if sm<100 else 2),
            'ac_grp':0 if ac<=2 else(1 if ac<=5 else 2),
            'red_zone_dom':zone_dom,
            'gap_grp':0 if max_gap<=5 else(1 if max_gap<=10 else 2)}
def tgtkl8(r):
    odd=sum(1 for x in r['numbers'] if x%2!=0)
    zn=[sum(1 for x in r['numbers'] if lo<=x<=hi) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]]
    tt=sum(r['numbers'])
    return {'odd_grp':0 if odd<9 else(1 if odd<=11 else 2),'zone_dom':int(np.argmax(zn)),'tot_grp':0 if tt<640 else(1 if tt<820 else 2)}

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

    # 各位TOP5候选（带权重）
    tops = [sorted(score[p].items(), key=lambda x: -x[1])[:5] for p in range(3)]
    candidates = [[int(v) for v, _ in t] for t in tops]

    # 生成6注，确保不完全重复
    random.seed(int(date.today().strftime('%Y%m%d')))
    groups = []
    seen = set()
    attempts = 0
    while len(groups) < 6 and attempts < 200:
        attempts += 1
        combo = []
        for pos in range(3):
            pool = [v for v, _ in tops[pos]] or list(range(10))
            weights = [max(s, 0.1) for _, s in tops[pos]]
            tot_w = sum(weights)
            rv = random.uniform(0, tot_w)
            cum = 0; chosen = pool[0]
            for v, w in zip(pool, weights):
                cum += w
                if rv <= cum: chosen = v; break
            combo.append(int(chosen))
        key = tuple(combo)
        if key not in seen:
            seen.add(key)
            groups.append(combo)

    # 如果加权随机不够6注，用候选补充
    from itertools import product
    for b, s, g in product(candidates[0], candidates[1], candidates[2]):
        if len(groups) >= 6: break
        if (b, s, g) not in seen:
            seen.add((b, s, g)); groups.append([b, s, g])

    sm_m = ml.get('sum_grp', {})
    sm_pred = sm_m.get('prediction', {}).get('value', 1) if sm_m else 1
    sm_label = {0:'小(0-9)', 1:'中(10-17)', 2:'大(18-27)'}.get(sm_pred, '中(10-17)')
    mk_hint = [f"{'百十个'[it['pos']]}位→可能{'、'.join(str(v) for v,_ in it['top3'][:3])}" for it in (mk or [])]

    return {
        'groups': groups[:6],
        'pos_candidates': candidates,
        'sum_pred': sm_label,
        'markov_hint': mk_hint,
        'overdue': [it.get('overdue', []) for it in (om or [])],
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

    # 蓝球：ML概率前5
    blue_m = ml.get('blue', {})
    blue_probs = blue_m.get('prediction', {}).get('probs', {}) if blue_m else {}
    blue_top = [int(k) for k, v in sorted(blue_probs.items(), key=lambda x: -x[1])[:5]]
    if not blue_top:
        blue_top = [x[0] for x in Counter(r['blue'] for r in records[-30:]).most_common(5)]

    # 奇偶预测
    odd_m = ml.get('odd', {})
    odd_pred = odd_m.get('prediction', {}).get('value', 3) if odd_m else 3

    # 和值区间预测
    sm_m = ml.get('sum_grp', {})
    sm_label = {0:'低(<70)', 1:'中(70-99)', 2:'高(≥100)'}.get(
        sm_m.get('prediction', {}).get('value', 1) if sm_m else 1, '中(70-99)')
    sm_val = sm_m.get('prediction', {}).get('value', 1) if sm_m else 1

    # 候选红球：热号+遗漏号混合
    pool = list(set(hot[:12] + overdue_r[:6]))
    if len(pool) < 20:
        pool += random.sample([n for n in range(1, 34) if n not in pool], 20 - len(pool))

    random.seed(int(date.today().strftime('%Y%m%d')))
    groups = []
    seen = set()
    attempts = 0

    while len(groups) < 6 and attempts < 300:
        attempts += 1
        picked = []
        # 三区各选2个，确保分布均衡
        for lo, hi in [(1, 11), (12, 22), (23, 33)]:
            zp = [n for n in pool if lo <= n <= hi] or list(range(lo, hi+1))
            cnt_needed = 2
            chosen_z = random.sample(zp, min(cnt_needed, len(zp)))
            picked += chosen_z

        # 如果不足6个，从池里补
        if len(picked) < 6:
            rem = [n for n in pool if n not in picked]
            if not rem: rem = [n for n in range(1, 34) if n not in picked]
            picked += random.sample(rem, 6 - len(picked))

        picked = sorted(picked[:6])

        # 按奇偶预测微调：如果奇数个数偏差太大，尝试换一个
        cur_odd = sum(1 for n in picked if n % 2 != 0)
        if abs(cur_odd - odd_pred) > 2:
            continue  # 重新生成

        # 按和值区间微调
        cur_sum = sum(picked)
        if sm_val == 0 and cur_sum >= 100: continue
        if sm_val == 2 and cur_sum < 70: continue

        key = tuple(picked)
        if key not in seen:
            seen.add(key)
            blue = blue_top[len(groups) % len(blue_top)]
            groups.append({'red': picked, 'blue': int(blue)})

    # 不够6注则放宽限制补充
    while len(groups) < 6:
        picked = sorted(random.sample(pool if len(pool) >= 6 else list(range(1, 34)), 6))
        key = tuple(picked)
        if key not in seen:
            seen.add(key)
            blue = blue_top[len(groups) % len(blue_top)]
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

    # 候选池：热号+遗漏+近10期热号
    pool = list(set([x[0] for x in freq10.most_common(12)] + hot[:18] + overdue[:10]))
    if len(pool) < 35:
        pool += random.sample([n for n in range(1, 81) if n not in pool], 35 - len(pool))

    def pick_balanced(n, seed_add=0):
        """按ML区间概率权重选n个球，四区均有覆盖"""
        random.seed(int(date.today().strftime('%Y%m%d')) + seed_add)
        res = []
        zone_ranges = [(1, 20), (21, 40), (41, 60), (61, 80)]
        # 按权重分配各区数量
        total_w = sum(zone_weights)
        alloc = [max(1, round(n * w / total_w)) for w in zone_weights]
        # 调整使总数等于n
        while sum(alloc) > n: alloc[alloc.index(max(alloc))] -= 1
        while sum(alloc) < n: alloc[alloc.index(min(alloc))] += 1
        for (lo, hi), take in zip(zone_ranges, alloc):
            zp = [x for x in pool if lo <= x <= hi] or list(range(lo, hi+1))
            take = min(take, len(zp), n - len(res))
            if take > 0: res += random.sample(zp, take)
        # 不足则补
        while len(res) < n:
            ex = [x for x in range(1, 81) if x not in res]
            if ex: res.append(random.choice(ex))
            else: break
        return sorted(res[:n])

    plays = {
        'xuan4':    {'name':'选四',     'balls':4,  'tip':'4球全中，赔率最高',
                     'groups': [pick_balanced(4, i) for i in range(3)]},
        'xuan5':    {'name':'选五',     'balls':5,  'tip':'5球，赔率与命中率均衡',
                     'groups': [pick_balanced(5, 100+i) for i in range(3)]},
        'xuan5_fu': {'name':'选五复式', 'balls':5,  'tip':f'8球覆盖C(8,5)={comb(8,5)}注',
                     'groups': [pick_balanced(8, 200)]},
        'xuan6':    {'name':'选六',     'balls':6,  'tip':'6球，主流推荐玩法',
                     'groups': [pick_balanced(6, 300+i) for i in range(3)]},
        'xuan9':    {'name':'选九',     'balls':9,  'tip':'9球，高赔率搏奖',
                     'groups': [pick_balanced(9, 400+i) for i in range(2)]},
        'xuan10':   {'name':'选十',     'balls':10, 'tip':'10球，最高赔率',
                     'groups': [pick_balanced(10, 500)]},
    }

    return {
        'plays':              plays,
        'zone_dominant_pred': zone_name,
        'total_range_pred':   tot_label,
        'hot_nums':  [int(x) for x in hot[:15]],
        'cold_nums': [int(x) for x in [x[0] for x in freq30.most_common()[-10:]]],
        'overdue':   [int(x) for x in overdue[:10]],
        'note': '综合ML区间预测+遗漏分析+区间均衡，选四3注/选五3注/复式1注/选六3注/选九2注/选十1注，仅供娱乐。'
    }

# 回测
def daily_backtest(history, prev_pred):
    report={'date':str(date.today()),'games':{}}
    for game in ['3d','ssq','kl8']:
        recs=history.get(game,[])
        if not isinstance(recs,list) or not recs or game not in (prev_pred or {}): continue
        latest=recs[-1]; rec=prev_pred[game].get('recommendation',{})
        if game=='3d':
            actual=latest.get('digits',[]); grps=rec.get('groups',[])
            hits=[g for g in grps if g==actual]
            part=[g for g in grps if sum(1 for i,v in enumerate(g) if i<len(actual) and v==actual[i])>=2]
            report['games'][game]={'actual':actual,'hit_count':len(hits),'partial_count':len(part)}
        elif game=='ssq':
            ar=sorted(latest.get('red',[])); ab=latest.get('blue',0); grps=rec.get('groups',[])
            res=[{'red_hit':len(set(g.get('red',[]))&set(ar)),'blue_hit':int(g.get('blue')==ab)} for g in grps]
            report['games'][game]={'actual_red':ar,'actual_blue':ab,'group_results':res,
                                   'best_red_hit':max((x['red_hit'] for x in res),default=0)}
        elif game=='kl8':
            actual=set(latest.get('numbers',[])); plays=rec.get('plays',{})
            play_results={}
            for pk,pd in plays.items():
                grps=pd.get('groups',[]); balls=pd.get('balls',0); name=pd.get('name',pk)
                gr=[{'hit':len(actual&set(g if isinstance(g,list) else [])),'balls':balls,'won':len(actual&set(g if isinstance(g,list) else []))==balls} for g in grps]
                play_results[pk]={'name':name,'balls':balls,'groups':gr,'any_won':any(x['won'] for x in gr),'best_hit':max((x['hit'] for x in gr),default=0)}
            report['games'][game]={'actual':sorted(actual),'play_results':play_results,'best_hit':max((pr['best_hit'] for pr in play_results.values()),default=0)}
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
        ('3d',  f3d,   tgt3d,  ['bai','shi','ge','sum_grp','odd']),
        ('ssq', fssq,  tgtssq, ['blue','odd','sum_grp','ac_grp','red_zone_dom','gap_grp']),
        ('kl8', fkl8,  tgtkl8, ['odd_grp','zone_dom','tot_grp']),
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
