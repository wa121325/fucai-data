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


def load_lstm_tfm(game):
    """从挂载的 Dataset 加载本周训练好的LSTM/TFM权重"""
    meta_path = f'{DL_MOUNTED}/{game}_meta.json'
    if not os.path.exists(meta_path):
        print(f"  ! 找不到 {game} 的LSTM/TFM权重（先运行 kaggle_lstm_tfm.py 并挂载 {DL_DATASET_SLUG}）")
        return None, None, None
    with open(meta_path) as f: meta = json.load(f)
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


def build_kl8_candidate_pool(records, idx, pool_size=30):
    """
    构建快乐8候选号码池（用于PPO打分排序，而非直接在80个球里选组合）
    池子 = 近30期热号 + 遗漏较久的号码，动态跟随当前idx
    """
    window = records[max(0,idx-30):idx]
    freq = Counter(n for r in window for n in r['numbers'])
    hot = [x[0] for x in freq.most_common(20)]
    cold = [x[0] for x in freq.most_common()[-15:]] if freq else []
    pool = list(dict.fromkeys(hot + cold))  # 去重保序
    if len(pool) < pool_size:
        remain = [n for n in range(1,81) if n not in pool]
        pool += remain[:pool_size-len(pool)]
    return sorted(pool[:pool_size])


def build_ssq_red_pool(records, idx, pool_size=18):
    """
    构建双色球红球候选池（33个红球里选一批候选，PPO对候选池打分排序取Top6）
    """
    window = records[max(0,idx-30):idx]
    freq = Counter(n for r in window for n in r['red'])
    hot = [x[0] for x in freq.most_common(12)]
    cold = [x[0] for x in freq.most_common()[-8:]] if freq else []
    pool = list(dict.fromkeys(hot + cold))
    if len(pool) < pool_size:
        remain = [n for n in range(1,34) if n not in pool]
        pool += remain[:pool_size-len(pool)]
    return sorted(pool[:pool_size])

# ══════════════════════════════════════════════════════
#  ML概率向量 + 遗漏向量
# ══════════════════════════════════════════════════════
def extract_ml_prob_vec(ml_pred, game):
    vec = []
    models_data = ml_pred.get('models', {})
    if game=='3d': tk=['bai','shi','ge','sum_grp','odd']; nc=[10,10,10,3,4]
    elif game=='ssq': tk=['blue','odd','sum_grp','ac_grp','red_zone_dom','gap_grp']; nc=[16,7,3,3,3,3]
    else: tk=['odd_grp','zone_dom','tot_grp']; nc=[3,4,3]
    for tkey,n in zip(tk,nc):
        m = models_data.get(tkey,{}); probs = m.get('prediction',{}).get('probs',{})
        vec.extend([float(probs.get(str(i),0.0))/100.0 for i in range(n)])
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
#  集成环境
# ══════════════════════════════════════════════════════
KL8_POOL_SIZE = 30   # 候选池大小
KL8_TRAIN_N   = 6    # 训练时用"选六"作为奖励标准，推荐时对同一个排序取不同TopN即可覆盖所有玩法

class IntegratedKL8Env(gym.Env):
    """
    快乐8环境 v2：候选池打分排序
    动作空间从 MultiBinary(80)（2^80种组合，几乎学不出东西）
    改成 Box(pool_size)（对候选池中每个号码打一个连续分数），
    取分数最高的N个作为选号，这是标准的排序学习问题，PPO容易收敛。
    """
    metadata={'render_modes':[]}
    def __init__(self, records, feat_fn, ml_vec, lstm_hidden, lstm_idx2row,
                 tfm_hidden, tfm_idx2row, omit_arr, pool_size=KL8_POOL_SIZE, train_n=KL8_TRAIN_N):
        super().__init__()
        self.records=records; self.feat_fn=feat_fn; self.ml_vec=ml_vec
        self.lstm_hidden=lstm_hidden; self.lstm_idx2row=lstm_idx2row
        self.tfm_hidden=tfm_hidden;   self.tfm_idx2row=tfm_idx2row
        self.omit_arr=omit_arr
        self.pool_size=pool_size; self.train_n=train_n
        self.start=SEQ_LEN+30; self.idx=self.start   # 需要更长历史构建候选池
        sample=feat_fn(records,self.start); feat_dim=len(sample)
        lstm_dim = lstm_hidden.shape[1] if lstm_hidden is not None else 0
        tfm_dim  = tfm_hidden.shape[1]  if tfm_hidden  is not None else 0
        omit_dim = 80
        # 状态里额外加入候选池号码的遗漏值，帮助Agent区分候选池内部差异
        self.state_dim = feat_dim+len(ml_vec)+lstm_dim+tfm_dim+omit_dim+pool_size
        self.observation_space = spaces.Box(low=-5.,high=5.,shape=(self.state_dim,),dtype=np.float32)
        self.action_space = spaces.Box(low=-1.,high=1.,shape=(pool_size,),dtype=np.float32)
        self._pool = None

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

        self._pool = build_kl8_candidate_pool(self.records, self.idx, self.pool_size)
        pool_omit = np.array([self.omit_arr[self.idx][n-1] for n in self._pool], dtype=np.float32) \
                    if self.omit_arr is not None else np.zeros(self.pool_size,dtype=np.float32)

        state = np.concatenate([raw,self.ml_vec,lh,th,om,pool_omit]).astype(np.float32)
        return np.clip(state/(np.abs(state).max()+1e-8),-5,5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.idx=self.start
        return self._state(), {}

    def step(self, action):
        pool = self._pool if self._pool is not None else build_kl8_candidate_pool(self.records, self.idx, self.pool_size)
        # 取分数最高的 train_n 个作为本期选号
        top_idx = np.argsort(action)[-self.train_n:]
        selected = sorted([pool[i] for i in top_idx])

        actual=set(self.records[self.idx]['numbers'])
        hit=len(actual&set(selected)); n_sel=len(selected)
        net=calc_payout(n_sel,hit); reward=net/(TICKET_PRICE*100)
        self.idx+=1
        terminated=(self.idx>=len(self.records)-1)
        obs=self._state() if not terminated else np.zeros(self.state_dim,dtype=np.float32)
        return obs, reward, terminated, False, {'hit':hit,'n_sel':n_sel,'net':net}


SSQ_RED_POOL_SIZE = 18   # 红球候选池大小
SSQ_RED_PICK_N    = 6    # 双色球固定选6个红球

class IntegratedSSQEnv(gym.Env):
    """
    双色球环境 v2：红球候选池打分排序 + 蓝球打分，两者融合进同一个Box动作空间
    动作向量 = [红球候选池18个分数, 蓝球16个分数]，共34维
    - 红球：候选池中分数最高的6个作为选号
    - 蓝球：16个分数里argmax作为选号
    奖励按双色球真实奖级结构分级，同时激励红球和蓝球命中。
    """
    metadata={'render_modes':[]}
    def __init__(self, records, feat_fn, ml_vec, lstm_hidden, lstm_idx2row,
                 tfm_hidden, tfm_idx2row, omit_arr,
                 red_pool_size=SSQ_RED_POOL_SIZE, red_pick_n=SSQ_RED_PICK_N):
        super().__init__()
        self.records=records; self.feat_fn=feat_fn; self.ml_vec=ml_vec
        self.lstm_hidden=lstm_hidden; self.lstm_idx2row=lstm_idx2row
        self.tfm_hidden=tfm_hidden;   self.tfm_idx2row=tfm_idx2row
        self.omit_arr=omit_arr
        self.red_pool_size=red_pool_size; self.red_pick_n=red_pick_n
        self.start=SEQ_LEN+30; self.idx=self.start
        sample=feat_fn(records,self.start); feat_dim=len(sample)
        lstm_dim = lstm_hidden.shape[1] if lstm_hidden is not None else 0
        tfm_dim  = tfm_hidden.shape[1]  if tfm_hidden  is not None else 0
        omit_dim = 49
        # 状态额外加入红球候选池各号码的遗漏值
        self.state_dim = feat_dim+len(ml_vec)+lstm_dim+tfm_dim+omit_dim+red_pool_size
        self.observation_space = spaces.Box(low=-5.,high=5.,shape=(self.state_dim,),dtype=np.float32)
        self.action_space = spaces.Box(low=-1.,high=1.,shape=(red_pool_size+16,),dtype=np.float32)
        self._red_pool = None

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

        self._red_pool = build_ssq_red_pool(self.records, self.idx, self.red_pool_size)
        pool_omit = np.array([self.omit_arr[self.idx][n-1] for n in self._red_pool], dtype=np.float32) \
                    if self.omit_arr is not None else np.zeros(self.red_pool_size,dtype=np.float32)

        state = np.concatenate([raw,self.ml_vec,lh,th,om,pool_omit]).astype(np.float32)
        return np.clip(state/(np.abs(state).max()+1e-8),-5,5)

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
        red_pool = self._red_pool if self._red_pool is not None else build_ssq_red_pool(self.records, self.idx, self.red_pool_size)
        red_scores  = action[:self.red_pool_size]
        blue_scores = action[self.red_pool_size:]

        top_idx = np.argsort(red_scores)[-self.red_pick_n:]
        red_selected = sorted([red_pool[i] for i in top_idx])
        blue_pred = int(np.argmax(blue_scores)) + 1

        actual_red = set(self.records[self.idx]['red'])
        actual_blue = self.records[self.idx]['blue']
        red_hit = len(actual_red & set(red_selected))
        blue_hit = int(blue_pred == actual_blue)
        reward = self._tier_reward(red_hit, blue_hit)

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
        state = np.concatenate([raw,self.ml_vec,lh,th,om]).astype(np.float32)
        return np.clip(state/(np.abs(state).max()+1e-8),-5,5)

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
def run_kl8_daily(records, ml_pred):
    print(f"\n{'='*50}\n快乐8 PPO 每日增量微调（候选池打分排序，{len(records)}期）\n{'='*50}")
    ml_vec = extract_ml_prob_vec(ml_pred, 'kl8')
    lstm, tfm, meta = load_lstm_tfm('kl8')

    print("  批量预计算 LSTM/TFM 隐层状态…")
    t0 = time.time()
    lstm_hidden, lstm_idx2row = precompute_hidden_all(records, fkl8, lstm)
    tfm_hidden,  tfm_idx2row  = precompute_hidden_all(records, fkl8, tfm)
    print(f"    完成，耗时 {time.time()-t0:.1f}s")

    print("  批量预计算遗漏向量…")
    t0 = time.time()
    omit_arr = precompute_omission_kl8(records)
    print(f"    完成，耗时 {time.time()-t0:.1f}s")

    def make_env():
        return IntegratedKL8Env(records, fkl8, ml_vec,
                                lstm_hidden, lstm_idx2row, tfm_hidden, tfm_idx2row, omit_arr)
    vec_env = make_vec_env(make_env, n_envs=4)

    model = load_ppo('kl8')
    is_new = model is None
    t0 = time.time()
    if is_new:
        print("  首次训练（20万步，候选池打分排序问题，学习难度远低于原MultiBinary(80)方案）…")
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=512, batch_size=128,
                    n_epochs=10, gamma=0.95, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
                    verbose=0, device='cpu')
        model.learn(total_timesteps=200000, progress_bar=False)
    else:
        model.set_env(vec_env)
        print("  增量微调（2万步，基于最新数据）…")
        model.learn(total_timesteps=20000, reset_num_timesteps=False, progress_bar=False)
    print(f"    PPO训练完成，耗时 {time.time()-t0:.1f}s")

    save_ppo(model, 'kl8')

    def build_state_and_pool(idx):
        feat = fkl8(records, idx)
        if feat is None: return None, None
        raw = np.array(list(feat.values()),dtype=np.float32)
        lh = lstm_hidden[lstm_idx2row[idx]] if (lstm_hidden is not None and idx in lstm_idx2row) else np.zeros(lstm_hidden.shape[1] if lstm_hidden is not None else 0,dtype=np.float32)
        th = tfm_hidden[tfm_idx2row[idx]]   if (tfm_hidden  is not None and idx in tfm_idx2row)  else np.zeros(tfm_hidden.shape[1] if tfm_hidden is not None else 0,dtype=np.float32)
        om = omit_arr[idx]
        pool = build_kl8_candidate_pool(records, idx, KL8_POOL_SIZE)
        pool_omit = np.array([omit_arr[idx][n-1] for n in pool], dtype=np.float32)
        state = np.concatenate([raw,ml_vec,lh,th,om,pool_omit]).astype(np.float32)
        state = np.clip(state/(np.abs(state).max()+1e-8),-5,5)
        return state, pool

    # 回测：按训练时的N=6（选六）标准评估
    start = max(SEQ_LEN+30, len(records)-30)
    total_net=0; games=0
    for idx in range(start, len(records)-1):
        state, pool = build_state_and_pool(idx)
        if state is None: continue
        action,_ = model.predict(state, deterministic=True)
        top_idx = np.argsort(action)[-KL8_TRAIN_N:]
        selected = sorted([pool[i] for i in top_idx])
        actual=set(records[idx]['numbers'])
        net = calc_payout(len(selected), len(actual&set(selected)))
        total_net+=net; games+=1
    avg_net = round(total_net/games,2) if games else 0
    print(f"  回测（近{games}期，选六标准）平均净收益: {avg_net}元/期")

    # 今日推荐：用同一套打分对候选池排序，取不同TopN覆盖各玩法
    idx = len(records)-1
    state, pool = build_state_and_pool(idx)
    ranked = []
    if state is not None:
        action,_ = model.predict(state, deterministic=True)
        order = np.argsort(action)[::-1]   # 分数从高到低排序
        ranked = [pool[i] for i in order]

    def top_n(n): return sorted(ranked[:n]) if ranked else []

    picks_by_n = {n: top_n(n) for n in [4,5,6,9,10]}

    return {'avg_net_per_game':avg_net,'games_tested':games,
            'ppo_selected':picks_by_n[6],   # 兼容旧字段，默认给选六结果
            'ppo_ranked_pool':ranked,       # 完整排序结果，前端可自行截取TopN
            'picks_by_n':picks_by_n,        # 各玩法直接可用的推荐
            'is_first_train':is_new,
            'note':f'PPO候选池打分排序（池大小{KL8_POOL_SIZE}），回测净收益{avg_net}元/期（选六标准）'}


def run_ssq_daily(records, ml_pred):
    print(f"\n{'='*50}\n双色球 PPO 每日增量微调（红球候选池打分+蓝球，{len(records)}期）\n{'='*50}")
    ml_vec = extract_ml_prob_vec(ml_pred, 'ssq')
    lstm, tfm, meta = load_lstm_tfm('ssq')

    print("  批量预计算 LSTM/TFM 隐层状态…")
    t0 = time.time()
    lstm_hidden, lstm_idx2row = precompute_hidden_all(records, fssq, lstm)
    tfm_hidden,  tfm_idx2row  = precompute_hidden_all(records, fssq, tfm)
    print(f"    完成，耗时 {time.time()-t0:.1f}s")

    print("  批量预计算遗漏向量…")
    t0 = time.time()
    omit_arr = precompute_omission_ssq(records)
    print(f"    完成，耗时 {time.time()-t0:.1f}s")

    def make_env():
        return IntegratedSSQEnv(records, fssq, ml_vec,
                                lstm_hidden, lstm_idx2row, tfm_hidden, tfm_idx2row, omit_arr)
    vec_env = make_vec_env(make_env, n_envs=4)

    model = load_ppo('ssq')
    is_new = model is None
    t0 = time.time()
    if is_new:
        print("  首次训练（15万步，红球候选池排序+蓝球联合优化）…")
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=512, batch_size=128,
                    n_epochs=10, gamma=0.95, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
                    verbose=0, device='cpu')
        model.learn(total_timesteps=150000, progress_bar=False)
    else:
        model.set_env(vec_env)
        print("  增量微调（1.5万步）…")
        model.learn(total_timesteps=15000, reset_num_timesteps=False, progress_bar=False)
    print(f"    PPO训练完成，耗时 {time.time()-t0:.1f}s")

    save_ppo(model, 'ssq')

    def build_state_and_pool(idx):
        feat = fssq(records, idx)
        if feat is None: return None, None
        raw = np.array(list(feat.values()),dtype=np.float32)
        lh = lstm_hidden[lstm_idx2row[idx]] if (lstm_hidden is not None and idx in lstm_idx2row) else np.zeros(lstm_hidden.shape[1] if lstm_hidden is not None else 0,dtype=np.float32)
        th = tfm_hidden[tfm_idx2row[idx]]   if (tfm_hidden  is not None and idx in tfm_idx2row)  else np.zeros(tfm_hidden.shape[1] if tfm_hidden is not None else 0,dtype=np.float32)
        om = omit_arr[idx]
        red_pool = build_ssq_red_pool(records, idx, SSQ_RED_POOL_SIZE)
        pool_omit = np.array([omit_arr[idx][n-1] for n in red_pool], dtype=np.float32)
        state = np.concatenate([raw,ml_vec,lh,th,om,pool_omit]).astype(np.float32)
        state = np.clip(state/(np.abs(state).max()+1e-8),-5,5)
        return state, red_pool

    # 回测最近30期：红球命中数分布 + 蓝球命中率 + 综合中奖统计
    start=max(SEQ_LEN+30, len(records)-30)
    total=0; blue_correct=0; red_hit_dist={0:0,1:0,2:0,3:0,4:0,5:0,6:0}; any_prize=0
    for idx in range(start, len(records)-1):
        state, red_pool = build_state_and_pool(idx)
        if state is None: continue
        action,_ = model.predict(state, deterministic=True)
        red_scores = action[:SSQ_RED_POOL_SIZE]; blue_scores = action[SSQ_RED_POOL_SIZE:]
        top_idx = np.argsort(red_scores)[-SSQ_RED_PICK_N:]
        red_selected = set(red_pool[i] for i in top_idx)
        blue_pred = int(np.argmax(blue_scores))+1

        actual_red = set(records[idx]['red']); actual_blue = records[idx]['blue']
        rh = len(actual_red & red_selected); bh = int(blue_pred==actual_blue)
        red_hit_dist[rh]+=1
        if bh: blue_correct+=1
        if rh>=3 or (rh>=3 and bh) or bh: any_prize+=1  # 简化：≥3红或中蓝球算"有奖级"
        total+=1

    blue_acc = round(blue_correct/total*100,1) if total else 0
    avg_red_hit = round(sum(k*v for k,v in red_hit_dist.items())/total,2) if total else 0
    print(f"  回测（近{total}期）：红球平均命中{avg_red_hit}个  蓝球准确率{blue_acc}%（随机基准6.25%）")

    # 今日推荐
    idx = len(records)-1
    state, red_pool = build_state_and_pool(idx)
    red_selected=[]; blue_pred=None
    if state is not None:
        action,_ = model.predict(state, deterministic=True)
        red_scores = action[:SSQ_RED_POOL_SIZE]; blue_scores = action[SSQ_RED_POOL_SIZE:]
        top_idx = np.argsort(red_scores)[-SSQ_RED_PICK_N:]
        red_selected = sorted([red_pool[i] for i in top_idx])
        blue_pred = int(np.argmax(blue_scores))+1

    return {'blue_acc_pct':blue_acc,'games_tested':total,
            'avg_red_hit':avg_red_hit,'red_hit_distribution':red_hit_dist,
            'ppo_red_selected':red_selected,'ppo_blue_pred':blue_pred,
            'is_first_train':is_new,
            'note':f'PPO红球候选池打分排序(池{SSQ_RED_POOL_SIZE}个)+蓝球联合优化，红球平均命中{avg_red_hit}个，蓝球准确率{blue_acc}%'}


def run_3d_daily(records, ml_pred):
    print(f"\n{'='*50}\n福彩3D PPO 每日增量微调（{len(records)}期）\n{'='*50}")
    ml_vec = extract_ml_prob_vec(ml_pred, '3d')
    lstm, tfm, meta = load_lstm_tfm('3d')

    print("  批量预计算 LSTM/TFM 隐层状态…")
    t0 = time.time()
    lstm_hidden, lstm_idx2row = precompute_hidden_all(records, f3d, lstm)
    tfm_hidden,  tfm_idx2row  = precompute_hidden_all(records, f3d, tfm)
    print(f"    完成，耗时 {time.time()-t0:.1f}s")

    print("  批量预计算遗漏向量…")
    t0 = time.time()
    omit_arr = precompute_omission_3d(records)
    print(f"    完成，耗时 {time.time()-t0:.1f}s")

    def make_env():
        return Integrated3DEnv(records, f3d, ml_vec,
                               lstm_hidden, lstm_idx2row, tfm_hidden, tfm_idx2row, omit_arr)
    vec_env = make_vec_env(make_env, n_envs=4)

    model = load_ppo('3d')
    is_new = model is None
    t0 = time.time()
    if is_new:
        print("  首次训练（10万步，MultiDiscrete([10,10,10])共1000种组合）…")
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=256, batch_size=64,
                    n_epochs=8, gamma=0.9, gae_lambda=0.9, clip_range=0.2, ent_coef=0.03,
                    verbose=0, device='cpu')
        model.learn(total_timesteps=100000, progress_bar=False)
    else:
        model.set_env(vec_env)
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
        state = np.concatenate([raw,ml_vec,lh,th,om]).astype(np.float32)
        return np.clip(state/(np.abs(state).max()+1e-8),-5,5)

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

    idx=len(records)-1; state=build_state(idx); pred=None
    if state is not None:
        action,_ = model.predict(state, deterministic=True)
        pred=[int(action[0]),int(action[1]),int(action[2])]

    return {'games_tested':total,'match_distribution':match_dist,
            'avg_match_digits':avg_match,'exact_hit_rate_pct':exact_hit_rate,
            'ppo_pred':pred,'is_first_train':is_new,
            'note':f'PPO直接预测百十个位组合，近{total}期平均命中{avg_match}位，全中率{exact_hit_rate}%（随机基准0.1%）'}

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

os.makedirs(RL_LOCAL_DIR, exist_ok=True)
rl_results = {}

for game, run_fn in [('3d', run_3d_daily), ('kl8', run_kl8_daily), ('ssq', run_ssq_daily)]:
    records = history.get(game, [])
    if not isinstance(records,list) or len(records)<65:
        print(f"\n{game}: 数据不足，跳过"); continue
    ml_pred = ml_preds.get(game, {})
    try:
        rl_results[game] = run_fn(records, ml_pred)
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
