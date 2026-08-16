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

def get_secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        v = UserSecretsClient().get_secret(name)
        if v: return v
    except Exception: pass
    return os.environ.get(name, '')

_HARDCODED_GH_TOKEN = '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'      # ← 新的 GitHub Token
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
    w = records[max(0,idx-WINDOW):idx]
    if len(w)<5: return None
    d=w[-1]['digits']; b,s,g=d; sm=b+s+g; sp=max(d)-min(d)
    prev=w[-2]['digits'] if len(w)>=2 else d
    rep=sum(1 for i in range(3) if prev[i]==d[i])
    s3=sorted(d); arith=int((s3[1]-s3[0])==(s3[2]-s3[1]) and s3[2]-s3[0]>0)
    f={'sum':sm,'tail':sm%10,'span':sp,'odd':sum(1 for x in d if x%2!=0),
       'big':sum(1 for x in d if x>=5),'r0':d[0]%3,'r1':d[1]%3,'r2':d[2]%3,
       'b':b,'s':s,'g':g,'gbs':abs(b-s),'gsg':abs(s-g),
       'grp':0 if b==s==g else(1 if(b==s or s==g or b==g)else 2),'rep':rep,'arith':arith}
    for ws,sfx in [(5,'5'),(10,'10'),(20,'20'),(WINDOW,'W')]:
        chunk=w[-ws:]; sms=[sum(x['digits']) for x in chunk]; sps=[max(x['digits'])-min(x['digits']) for x in chunk]
        f[f'sm{sfx}']=float(np.mean(sms)); f[f'ss{sfx}']=float(np.std(sms)) if len(sms)>1 else 0.0
        f[f'sp{sfx}']=float(np.mean(sps))
        for ci,cn in enumerate(['b','s','g']):
            vals=[x['digits'][ci] for x in chunk]; f[f'{cn}m{sfx}']=float(np.mean(vals))
    tr3=[sum(x['digits']) for x in w[-3:]] if len(w)>=3 else [sm]
    f['trend']=1 if tr3[-1]>tr3[-2] else(-1 if tr3[-1]<tr3[-2] else 0)
    return f

def fssq(records, idx):
    w=records[max(0,idx-WINDOW):idx]
    if len(w)<5: return None
    r=w[-1]; red=sorted(r['red']); bl=r['blue']
    sm=sum(red); odd=sum(1 for x in red if x%2!=0); big=sum(1 for x in red if x>16)
    csc=sum(1 for i in range(len(red)-1) if red[i+1]-red[i]==1)
    df=set()
    for i in range(len(red)):
        for j in range(i+1,len(red)): df.add(red[j]-red[i])
    ac=len(df)-(len(red)-1)
    z1=sum(1 for x in red if x<=11); z2=sum(1 for x in red if 12<=x<=22); z3=sum(1 for x in red if x>=23)
    mg=max(red[i+1]-red[i] for i in range(len(red)-1)) if len(red)>1 else 0
    f={'sm':sm,'odd':odd,'big':big,'consec':csc,'ac':ac,'z1':z1,'z2':z2,'z3':z3,'mg':mg,
       'bl':bl,'bl_odd':bl%2,'bl_big':int(bl>=9),'rmax':red[-1],'rmin':red[0],'rsp':red[-1]-red[0]}
    for ws,sfx in [(5,'5'),(10,'10'),(20,'20'),(WINDOW,'W')]:
        chunk=w[-ws:]; sms=[sum(x['red']) for x in chunk]
        bls=[x['blue'] for x in chunk]; odds=[sum(1 for n in x['red'] if n%2!=0) for x in chunk]
        f[f'sm{sfx}']=float(np.mean(sms)); f[f'ss{sfx}']=float(np.std(sms)) if len(sms)>1 else 0.0
        f[f'bl{sfx}']=float(np.mean(bls)); f[f'od{sfx}']=float(np.mean(odds))
    cnt=Counter(n for x in w for n in x['red'])
    f['hz1']=sum(cnt.get(n,0) for n in range(1,12)); f['hz2']=sum(cnt.get(n,0) for n in range(12,23)); f['hz3']=sum(cnt.get(n,0) for n in range(23,34))
    tr3=[sum(x['red']) for x in w[-3:]] if len(w)>=3 else [sm]
    f['trend']=1 if tr3[-1]>tr3[-2] else(-1 if tr3[-1]<tr3[-2] else 0)
    return f

def fkl8(records, idx):
    w=records[max(0,idx-WINDOW):idx]
    if len(w)<5: return None
    r=w[-1]; nums=sorted(r['numbers']); tot=sum(nums)
    odd=sum(1 for x in nums if x%2!=0); big=sum(1 for x in nums if x>40)
    zn=[sum(1 for x in nums if lo<=x<=hi) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]]
    fv=[sum(1 for x in nums if lo<=x<=hi) for lo,hi in [(1,16),(17,32),(33,48),(49,64),(65,80)]]
    cg=0; inc=False
    for i in range(len(nums)-1):
        if nums[i+1]-nums[i]==1:
            if not inc: cg+=1; inc=True
        else: inc=False
    f={'tot':tot,'odd':odd,'big':big,'mn':nums[0],'mx':nums[-1],
       'z1':zn[0],'z2':zn[1],'z3':zn[2],'z4':zn[3],'f1':fv[0],'f2':fv[1],'f3':fv[2],'f4':fv[3],'f5':fv[4],'cg':cg}
    for ws,sfx in [(5,'5'),(10,'10'),(20,'20'),(WINDOW,'W')]:
        chunk=w[-ws:]; tots=[sum(x['numbers']) for x in chunk]
        f[f'tm{sfx}']=float(np.mean(tots)); f[f'ts{sfx}']=float(np.std(tots)) if len(tots)>1 else 0.0
        for zi,(lo,hi) in enumerate([(1,20),(21,40),(41,60),(61,80)]):
            zv=[sum(1 for n in x['numbers'] if lo<=n<=hi) for x in chunk]
            f[f'z{zi+1}m{sfx}']=float(np.mean(zv))
    cnt=Counter(n for x in w for n in x['numbers'])
    for zi,(lo,hi) in enumerate([(1,20),(21,40),(41,60),(61,80)]):
        f[f'hz{zi+1}']=sum(cnt.get(n,0) for n in range(lo,hi+1))
    tr3=[sum(x['numbers']) for x in w[-3:]] if len(w)>=3 else [tot]
    f['trend']=1 if tr3[-1]>tr3[-2] else(-1 if tr3[-1]<tr3[-2] else 0)
    return f

FEAT_FNS = {'3d':f3d,'ssq':fssq,'kl8':fkl8}

# ══════════════════════════════════════════════════════
#  LSTM/TFM 结构（需与训练时一致，用于加载权重）
# ══════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════
#  ML概率向量 + 遗漏向量
# ══════════════════════════════════════════════════════
def extract_ml_prob_vec(ml_pred, game):
    vec = []
    models_data = ml_pred.get('models', {})
    if game=='3d': tk=['bai','shi','ge','sum_grp','odd']; nc=[10,10,10,3,4]
    elif game=='ssq': tk=['blue','odd','sum_grp']; nc=[16,7,3]
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
class IntegratedKL8Env(gym.Env):
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
        omit_dim = 80
        self.state_dim = feat_dim+len(ml_vec)+lstm_dim+tfm_dim+omit_dim
        self.observation_space = spaces.Box(low=-5.,high=5.,shape=(self.state_dim,),dtype=np.float32)
        self.action_space = spaces.MultiBinary(80)

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
        state = np.concatenate([raw,self.ml_vec,lh,th,om]).astype(np.float32)
        return np.clip(state/(np.abs(state).max()+1e-8),-5,5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.idx=self.start
        return self._state(), {}

    def step(self, action):
        selected=[i+1 for i in range(80) if action[i]]
        if len(selected)<4: selected=list(range(1,5))
        if len(selected)>10: selected=selected[:10]
        actual=set(self.records[self.idx]['numbers'])
        hit=len(actual&set(selected)); n_sel=len(selected)
        net=calc_payout(n_sel,hit); reward=net/(TICKET_PRICE*100)
        self.idx+=1
        terminated=(self.idx>=len(self.records)-1)
        obs=self._state() if not terminated else np.zeros(self.state_dim,dtype=np.float32)
        return obs, reward, terminated, False, {'hit':hit,'n_sel':n_sel,'net':net}


class IntegratedSSQEnv(gym.Env):
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
        omit_dim = 49
        self.state_dim = feat_dim+len(ml_vec)+lstm_dim+tfm_dim+omit_dim
        self.observation_space = spaces.Box(low=-5.,high=5.,shape=(self.state_dim,),dtype=np.float32)
        self.action_space = spaces.Discrete(16)

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
        state = np.concatenate([raw,self.ml_vec,lh,th,om]).astype(np.float32)
        return np.clip(state/(np.abs(state).max()+1e-8),-5,5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.idx=self.start
        return self._state(), {}

    def step(self, action):
        blue_pred=action+1; actual=self.records[self.idx]['blue']
        reward = 10.0 if blue_pred==actual else -1.0
        self.idx+=1
        terminated=(self.idx>=len(self.records)-1)
        obs=self._state() if not terminated else np.zeros(self.state_dim,dtype=np.float32)
        return obs, reward, terminated, False, {'pred':blue_pred,'actual':actual}

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
    print(f"\n{'='*50}\n快乐8 PPO 每日增量微调（{len(records)}期）\n{'='*50}")
    ml_vec = extract_ml_prob_vec(ml_pred, 'kl8')
    lstm, tfm, meta = load_lstm_tfm('kl8')

    print("  批量预计算 LSTM/TFM 隐层状态…")
    t0 = time.time()
    lstm_hidden, lstm_idx2row = precompute_hidden_all(records, fkl8, lstm)
    tfm_hidden,  tfm_idx2row  = precompute_hidden_all(records, fkl8, tfm)
    print(f"    完成，耗时 {time.time()-t0:.1f}s（LSTM:{lstm_hidden.shape if lstm_hidden is not None else None} TFM:{tfm_hidden.shape if tfm_hidden is not None else None}）")

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
        print("  首次训练（3万步）…")
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=256, batch_size=64,
                    n_epochs=8, gamma=0.95, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                    verbose=0, device='cpu')
        model.learn(total_timesteps=30000, progress_bar=False)
    else:
        model.set_env(vec_env)
        print("  增量微调（5000步，基于最新数据）…")
        model.learn(total_timesteps=5000, reset_num_timesteps=False, progress_bar=False)
    print(f"    PPO训练完成，耗时 {time.time()-t0:.1f}s")

    save_ppo(model, 'kl8')

    def build_state(idx):
        feat = fkl8(records, idx)
        if feat is None: return None
        raw = np.array(list(feat.values()),dtype=np.float32)
        lh = lstm_hidden[lstm_idx2row[idx]] if (lstm_hidden is not None and idx in lstm_idx2row) else np.zeros(lstm_hidden.shape[1] if lstm_hidden is not None else 0,dtype=np.float32)
        th = tfm_hidden[tfm_idx2row[idx]]   if (tfm_hidden  is not None and idx in tfm_idx2row)  else np.zeros(tfm_hidden.shape[1] if tfm_hidden is not None else 0,dtype=np.float32)
        om = omit_arr[idx]
        state = np.concatenate([raw,ml_vec,lh,th,om]).astype(np.float32)
        return np.clip(state/(np.abs(state).max()+1e-8),-5,5)

    # 回测最近30期
    start = max(SEQ_LEN+5, len(records)-30)
    total_net=0; games=0
    for idx in range(start, len(records)-1):
        state = build_state(idx)
        if state is None: continue
        action,_ = model.predict(state, deterministic=True)
        selected=[i+1 for i in range(80) if action[i]]
        if len(selected)<4: selected=list(range(1,5))
        if len(selected)>10: selected=selected[:10]
        actual=set(records[idx]['numbers'])
        net = calc_payout(len(selected), len(actual&set(selected)))
        total_net+=net; games+=1
    avg_net = round(total_net/games,2) if games else 0
    print(f"  回测（近{games}期）平均净收益: {avg_net}元/期")

    # 今日推荐
    idx = len(records)-1
    state = build_state(idx)
    selected = []
    if state is not None:
        action,_ = model.predict(state, deterministic=True)
        selected = sorted([i+1 for i in range(80) if action[i]])
        if len(selected)<4: selected=list(range(1,5))
        if len(selected)>10: selected=selected[:10]

    return {'avg_net_per_game':avg_net,'games_tested':games,'ppo_selected':selected,
            'is_first_train':is_new,'note':f'PPO每日增量微调，回测净收益{avg_net}元/期'}


def run_ssq_daily(records, ml_pred):
    print(f"\n{'='*50}\n双色球 PPO 每日增量微调（{len(records)}期）\n{'='*50}")
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
        print("  首次训练（2万步）…")
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=256, batch_size=64,
                    n_epochs=8, gamma=0.95, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                    verbose=0, device='cpu')
        model.learn(total_timesteps=20000, progress_bar=False)
    else:
        model.set_env(vec_env)
        print("  增量微调（3000步）…")
        model.learn(total_timesteps=3000, reset_num_timesteps=False, progress_bar=False)
    print(f"    PPO训练完成，耗时 {time.time()-t0:.1f}s")

    save_ppo(model, 'ssq')

    def build_state(idx):
        feat = fssq(records, idx)
        if feat is None: return None
        raw = np.array(list(feat.values()),dtype=np.float32)
        lh = lstm_hidden[lstm_idx2row[idx]] if (lstm_hidden is not None and idx in lstm_idx2row) else np.zeros(lstm_hidden.shape[1] if lstm_hidden is not None else 0,dtype=np.float32)
        th = tfm_hidden[tfm_idx2row[idx]]   if (tfm_hidden  is not None and idx in tfm_idx2row)  else np.zeros(tfm_hidden.shape[1] if tfm_hidden is not None else 0,dtype=np.float32)
        om = omit_arr[idx]
        state = np.concatenate([raw,ml_vec,lh,th,om]).astype(np.float32)
        return np.clip(state/(np.abs(state).max()+1e-8),-5,5)

    start=max(SEQ_LEN+5, len(records)-30); correct=0; total=0
    for idx in range(start, len(records)-1):
        state = build_state(idx)
        if state is None: continue
        action,_ = model.predict(state, deterministic=True)
        if action+1==records[idx]['blue']: correct+=1
        total+=1
    blue_acc = round(correct/total*100,1) if total else 0
    print(f"  蓝球回测准确率（近{total}期）: {blue_acc}%（随机基准6.25%）")

    idx=len(records)-1
    state = build_state(idx)
    blue_pred=None
    if state is not None:
        action,_ = model.predict(state, deterministic=True)
        blue_pred = int(action)+1

    return {'blue_acc_pct':blue_acc,'games_tested':total,'ppo_blue_pred':blue_pred,
            'is_first_train':is_new,'note':f'PPO每日增量微调，蓝球回测准确率{blue_acc}%'}

# ══════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════
print(f"\n{'#'*55}\nPPO 强化学习 每日增量微调  {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'#'*55}")

raw = gh_raw('history.json')
if not raw: print("失败"); sys.exit(1)
history = json.loads(raw)

raw_pred = gh_raw('prediction.json')
existing = json.loads(raw_pred) if raw_pred else {}
ml_preds = existing.get('predictions', {})

os.makedirs(RL_LOCAL_DIR, exist_ok=True)
rl_results = {}

for game, run_fn in [('kl8', run_kl8_daily), ('ssq', run_ssq_daily)]:
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

# 更新 prediction.json
existing.setdefault('dl_result', {})
existing['dl_result']['rl'] = {
    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'method': 'PPO强化学习（每日增量微调）',
    'state_composition': '原始特征 + ML概率向量 + LSTM隐层 + Transformer特征 + 遗漏向量',
    'results': rl_results,
}
pred_json = json.dumps(existing, ensure_ascii=False, indent=2)

if not GH_TOKEN:
    print("\n[DRY RUN] 未配置 GH_TOKEN")
else:
    print("\n推送 prediction.json…")
    gh_put('prediction.json', pred_json, f"PPO每日微调 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("✓ 完成")

print(f"\n✅ 全部完成！{datetime.now().strftime('%Y-%m-%d %H:%M')}")
