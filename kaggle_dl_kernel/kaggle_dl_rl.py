"""
福彩集成预测系统 v2 — 分层决策架构
kaggle_dl_rl.py

架构：
  第一层：传统ML概率输出（从 prediction.json 读取）
  第二层：LSTM + Transformer 序列特征提取（输出隐层状态）
  第三层：PPO Agent（状态=ML概率+LSTM隐层+原始特征，统一决策）

奖励函数：按快乐8真实赔率计算期望收益率
每周手动运行一次，结果写入 prediction.json 的 dl_result 字段

Kaggle Secrets: GH_TOKEN, GH_REPO
Kaggle 设置: GPU加速 + Internet开启
"""

import os, json, sys, time, math, warnings, base64, urllib.request, random
from datetime import datetime, date
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
#  环境初始化
# ══════════════════════════════════════════════════════
def get_secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        v = UserSecretsClient().get_secret(name)
        if v: return v
    except Exception: pass
    return os.environ.get(name, '')

GH_TOKEN = get_secret('GH_TOKEN')
GH_REPO  = get_secret('GH_REPO') or 'wa121325/fucai-data'
print(f"GitHub: {GH_REPO}")

# PyTorch
try:
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HAS_TORCH = True
    print(f"PyTorch ✓  设备: {DEVICE}")
except ImportError:
    HAS_TORCH = False; print("PyTorch ✗")

# stable-baselines3
try:
    import subprocess
    subprocess.run(['pip','install','stable-baselines3','gymnasium','-q'],
                   capture_output=True, timeout=180)
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True; print("SB3 ✓")
except Exception as e:
    HAS_SB3 = False; print(f"SB3 ✗: {e}")

import numpy as np

# ══════════════════════════════════════════════════════
#  GitHub 工具
# ══════════════════════════════════════════════════════
def gh_raw(path):
    url = f'https://raw.githubusercontent.com/{GH_REPO}/main/{path}?t={int(time.time())}'
    req = urllib.request.Request(url, headers={'Cache-Control':'no-cache','User-Agent':'dl-bot'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode('utf-8')
    except Exception as e:
        print(f"  gh_raw({path}) 失败: {e}"); return None

def gh_put(path, content_str, message):
    url = f'https://api.github.com/repos/{GH_REPO}/contents/{path}'
    sha = None
    try:
        req = urllib.request.Request(url, headers={
            'Authorization':f'token {GH_TOKEN}','Accept':'application/vnd.github.v3+json','User-Agent':'dl-bot'})
        with urllib.request.urlopen(req, timeout=15) as r:
            sha = json.loads(r.read()).get('sha')
    except Exception: pass
    data = {'message':message,'branch':'main',
            'content':base64.b64encode(content_str.encode()).decode()}
    if sha: data['sha'] = sha
    req = urllib.request.Request(url, data=json.dumps(data).encode(), method='PUT', headers={
        'Authorization':f'token {GH_TOKEN}','Content-Type':'application/json',
        'Accept':'application/vnd.github.v3+json','User-Agent':'dl-bot'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

# ══════════════════════════════════════════════════════
#  读取数据
# ══════════════════════════════════════════════════════
print("\n读取 history.json…")
raw = gh_raw('history.json')
if not raw: print("读取失败"); sys.exit(1)
history = json.loads(raw)

print("读取 prediction.json（获取ML概率向量）…")
existing_pred = {}
raw_pred = gh_raw('prediction.json')
if raw_pred:
    try: existing_pred = json.loads(raw_pred)
    except Exception: pass

for g in ['ssq','3d','kl8']:
    recs = history.get(g,[])
    ml   = existing_pred.get('predictions',{}).get(g,{})
    print(f"  {g}: {len(recs) if isinstance(recs,list) else 0}期历史  ML模型数:{len(ml.get('models',{}))}")

# ══════════════════════════════════════════════════════
#  特征工程（与 kaggle_fucai.py 一致）
# ══════════════════════════════════════════════════════
WINDOW  = 50
SEQ_LEN = 20

def _window_stats(w, key_fn):
    vals = [key_fn(x) for x in w]
    return float(np.mean(vals)), float(np.std(vals)) if len(vals)>1 else 0.0

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
       'grp':0 if b==s==g else(1 if(b==s or s==g or b==g)else 2),
       'rep':rep,'arith':arith}
    for ws,sfx in [(5,'5'),(10,'10'),(20,'20'),(WINDOW,'W')]:
        chunk=w[-ws:]; sms=[sum(x['digits']) for x in chunk]
        sps=[max(x['digits'])-min(x['digits']) for x in chunk]
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
    f['hz1']=sum(cnt.get(n,0) for n in range(1,12))
    f['hz2']=sum(cnt.get(n,0) for n in range(12,23))
    f['hz3']=sum(cnt.get(n,0) for n in range(23,34))
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
       'z1':zn[0],'z2':zn[1],'z3':zn[2],'z4':zn[3],
       'f1':fv[0],'f2':fv[1],'f3':fv[2],'f4':fv[3],'f5':fv[4],'cg':cg}
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

# ══════════════════════════════════════════════════════
#  从 prediction.json 提取ML概率向量
# ══════════════════════════════════════════════════════
def extract_ml_prob_vec(ml_pred, game):
    """
    把 prediction.json 里各目标的概率分布展平成向量
    用作 RL 状态的一部分
    """
    vec = []
    models_data = ml_pred.get('models', {})

    if game == '3d':
        target_keys = ['bai','shi','ge','sum_grp','odd']
        n_classes   = [10, 10, 10, 3, 4]
    elif game == 'ssq':
        target_keys = ['blue','odd','sum_grp']
        n_classes   = [16, 7, 3]
    else:  # kl8
        target_keys = ['odd_grp','zone_dom','tot_grp']
        n_classes   = [3, 4, 3]

    for tkey, nc in zip(target_keys, n_classes):
        m = models_data.get(tkey, {})
        probs = m.get('prediction', {}).get('probs', {})
        # 展平为定长向量（按类别顺序）
        prob_vec = [float(probs.get(str(i), 0.0))/100.0 for i in range(nc)]
        vec.extend(prob_vec)

    return np.array(vec, dtype=np.float32)

# ══════════════════════════════════════════════════════
#  遗漏向量（归一化）
# ══════════════════════════════════════════════════════
def omission_vec_3d(records, idx):
    """各位各号码的遗漏归一化，返回30维向量"""
    w = records[:idx]
    vec = []
    for pos in range(3):
        for d in range(10):
            for i in range(len(w)-1,-1,-1):
                if w[i]['digits'][pos]==d:
                    vec.append((len(w)-1-i)/10.0); break
            else:
                vec.append(1.0)
    return np.array(vec, dtype=np.float32)

def omission_vec_kl8(records, idx):
    """80个号码的遗漏归一化，返回80维向量"""
    w = records[:idx]
    avg = max(len(w)*20/80, 1)
    vec = []
    for n in range(1,81):
        for i in range(len(w)-1,-1,-1):
            if n in w[i]['numbers']:
                vec.append((len(w)-1-i)/avg); break
        else:
            vec.append(2.0)
    return np.clip(vec, 0, 3).astype(np.float32)

def omission_vec_ssq(records, idx):
    """红球33个+蓝球16个的遗漏，返回49维向量"""
    w = records[:idx]
    avg_r = max(len(w)*6/33, 1); avg_b = max(len(w)/16, 1)
    vec = []
    for n in range(1,34):
        for i in range(len(w)-1,-1,-1):
            if n in w[i]['red']:
                vec.append((len(w)-1-i)/avg_r); break
        else: vec.append(2.0)
    for n in range(1,17):
        for i in range(len(w)-1,-1,-1):
            if w[i]['blue']==n:
                vec.append((len(w)-1-i)/avg_b); break
        else: vec.append(2.0)
    return np.clip(vec, 0, 3).astype(np.float32)

# ══════════════════════════════════════════════════════
#  深度学习模型（LSTM + Transformer）
#  关键改动：输出隐层状态，供 RL 使用
# ══════════════════════════════════════════════════════
class LSTMEncoder(nn.Module):
    """
    LSTM 编码器
    forward() 返回 (logits, hidden_state)
    hidden_state 用作 RL 的状态输入
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 output_dim=10, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers>1 else 0)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, return_hidden=False):
        out, (h_n, _) = self.lstm(x)
        last = self.norm(out[:, -1, :])
        logits = self.head(last)
        if return_hidden:
            return logits, last   # last 就是隐层状态
        return logits


class TransformerEncoder(nn.Module):
    """
    Transformer 编码器
    forward() 返回 (logits, pooled_state)
    """
    def __init__(self, input_dim, d_model=64, nhead=4,
                 num_layers=2, output_dim=10, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, output_dim))

    def forward(self, x, return_hidden=False):
        x = self.proj(x)
        x = self.transformer(x)
        pooled = self.pool(x.transpose(1,2)).squeeze(-1)
        logits = self.head(pooled)
        if return_hidden:
            return logits, pooled
        return logits


def build_seq_dataset(records, feat_fn, target_fn, seq_len=SEQ_LEN):
    X_list, y_list = [], []
    for i in range(seq_len, len(records)):
        seq = []
        valid = True
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


def train_encoder(model, X, y, epochs=60, lr=5e-4, batch_size=32):
    """训练编码器，返回训练好的模型和最终隐层状态序列"""
    model = model.to(DEVICE)
    X_t = torch.FloatTensor(X).to(DEVICE)
    y_t = torch.LongTensor(y).to(DEVICE)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        if (ep+1) % 20 == 0:
            print(f"      ep{ep+1}/{epochs} loss={loss.item():.4f}")

    # 提取所有样本的隐层状态（用于构建 RL 状态）
    model.eval()
    hidden_states = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i+batch_size]
            _, h = model(xb, return_hidden=True)
            hidden_states.append(h.cpu().numpy())
    hidden_states = np.vstack(hidden_states)

    # 回测准确率
    model.eval()
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).cpu().numpy()
    acc = round(float((preds==y).mean())*100, 1)

    # 预测下一期
    with torch.no_grad():
        logits, last_h = model(X_t[-1:], return_hidden=True)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    return model, hidden_states, last_h.cpu().numpy()[0], probs, acc


# ══════════════════════════════════════════════════════
#  快乐8真实赔率表
# ══════════════════════════════════════════════════════
KL8_PAYOUT = {
    (1,1):2,
    (2,2):10, (2,1):0,
    (3,3):30, (3,2):0,
    (4,4):100,(4,3):3,  (4,2):1,
    (5,5):200,(5,4):8,  (5,3):1,
    (6,6):1000,(6,5):20,(6,4):2,(6,3):1,
    (7,7):2000,(7,6):50,(7,5):4,(7,4):1,
    (8,8):5000,(8,7):100,(8,6):8,(8,5):1,
    (9,9):10000,(9,8):300,(9,7):20,(9,6):2,
    (10,10):18000,(10,9):600,(10,8):30,(10,7):3,(10,6):1,
}
TICKET_PRICE = 2.0

def calc_payout(n_selected, n_hit):
    gross = KL8_PAYOUT.get((n_selected, n_hit), 0) * TICKET_PRICE
    return gross - TICKET_PRICE   # 净收益

# ══════════════════════════════════════════════════════
#  集成环境（核心：状态 = ML概率 + LSTM隐层 + 遗漏向量 + 原始特征）
# ══════════════════════════════════════════════════════
class IntegratedKL8Env(gym.Env):
    """
    快乐8集成环境
    状态空间 = concat[
        原始特征向量,        # fkl8 输出
        ML概率向量,          # RF/XGB/LGB 各目标概率展平
        LSTM隐层状态,        # 128维
        Transformer池化状态, # 64维
        遗漏归一化向量,       # 80维
    ]
    动作空间 = MultiBinary(80)，选哪些号码
    奖励     = 按真实赔率计算净收益率
    """
    metadata = {'render_modes': []}

    def __init__(self, records, feat_fn, ml_prob_vec,
                 lstm_hidden_all, tfm_hidden_all, omit_fn,
                 seq_offset=SEQ_LEN):
        super().__init__()
        self.records        = records
        self.feat_fn        = feat_fn
        self.ml_prob_vec    = ml_prob_vec          # 固定向量（当前期预测）
        self.lstm_hidden    = lstm_hidden_all      # (N, 128)
        self.tfm_hidden     = tfm_hidden_all       # (N, 64)
        self.omit_fn        = omit_fn
        self.seq_offset     = seq_offset
        self.start          = seq_offset + 5
        self.idx            = self.start

        # 计算状态维度
        sample_feat = feat_fn(records, self.start)
        feat_dim    = len(sample_feat)
        ml_dim      = len(ml_prob_vec)
        lstm_dim    = lstm_hidden_all.shape[1] if lstm_hidden_all is not None else 0
        tfm_dim     = tfm_hidden_all.shape[1]  if tfm_hidden_all is not None else 0
        omit_sample = omit_fn(records, self.start)
        omit_dim    = len(omit_sample)

        self.state_dim = feat_dim + ml_dim + lstm_dim + tfm_dim + omit_dim
        print(f"    状态维度: 原始{feat_dim} + ML概率{ml_dim} + LSTM{lstm_dim} + TFM{tfm_dim} + 遗漏{omit_dim} = {self.state_dim}")

        self.observation_space = spaces.Box(
            low=-5., high=5., shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.MultiBinary(80)

    def _build_state(self):
        idx = self.idx
        # 原始特征
        feat = self.feat_fn(self.records, idx)
        if feat is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        raw = np.array(list(feat.values()), dtype=np.float32)

        # ML概率（固定，代表"当前期前的ML预测"）
        ml  = self.ml_prob_vec

        # LSTM/TFM 隐层（按时间步索引）
        hi  = idx - self.seq_offset   # 对应的隐层索引
        hi  = max(0, min(hi, len(self.lstm_hidden)-1)) if self.lstm_hidden is not None else 0
        lstm_h = self.lstm_hidden[hi]  if self.lstm_hidden is not None else np.array([])
        tfm_h  = self.tfm_hidden[hi]   if self.tfm_hidden  is not None else np.array([])

        # 遗漏向量
        omit = self.omit_fn(self.records, idx)

        # 拼接 + 归一化裁剪
        state = np.concatenate([raw, ml, lstm_h, tfm_h, omit]).astype(np.float32)
        state = np.clip(state / (np.abs(state).max() + 1e-8), -5, 5)
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = self.start
        return self._build_state(), {}

    def step(self, action):
        selected = [i+1 for i in range(80) if action[i]]
        # 强制选号数 4-10
        if len(selected) < 4: selected = list(range(1,5))
        if len(selected) > 10: selected = selected[:10]

        actual = set(self.records[self.idx]['numbers'])
        hit    = len(actual & set(selected))
        n_sel  = len(selected)

        # 真实赔率奖励
        net = calc_payout(n_sel, hit)
        reward = net / (TICKET_PRICE * 100)   # 归一化到合理范围

        self.idx += 1
        terminated = (self.idx >= len(self.records) - 1)
        truncated  = False
        obs = self._build_state() if not terminated else np.zeros(self.state_dim, dtype=np.float32)
        return obs, reward, terminated, truncated, {'hit':hit,'n_sel':n_sel,'net':net}


class IntegratedSSQEnv(gym.Env):
    """
    双色球集成环境
    动作：蓝球选择（1-16）
    奖励：蓝球命中 +10，不中 -1
    """
    metadata = {'render_modes': []}

    def __init__(self, records, feat_fn, ml_prob_vec,
                 lstm_hidden_all, tfm_hidden_all, omit_fn):
        super().__init__()
        self.records     = records
        self.feat_fn     = feat_fn
        self.ml_prob_vec = ml_prob_vec
        self.lstm_hidden = lstm_hidden_all
        self.tfm_hidden  = tfm_hidden_all
        self.omit_fn     = omit_fn
        self.start       = SEQ_LEN + 5
        self.idx         = self.start

        sample_feat = feat_fn(records, self.start)
        feat_dim    = len(sample_feat)
        ml_dim      = len(ml_prob_vec)
        lstm_dim    = lstm_hidden_all.shape[1] if lstm_hidden_all is not None else 0
        tfm_dim     = tfm_hidden_all.shape[1]  if tfm_hidden_all is not None else 0
        omit_sample = omit_fn(records, self.start)
        omit_dim    = len(omit_sample)
        self.state_dim = feat_dim+ml_dim+lstm_dim+tfm_dim+omit_dim

        self.observation_space = spaces.Box(low=-5.,high=5.,shape=(self.state_dim,),dtype=np.float32)
        self.action_space = spaces.Discrete(16)  # 选1-16中的一个蓝球

    def _build_state(self):
        feat = self.feat_fn(self.records, self.idx)
        if feat is None: return np.zeros(self.state_dim, dtype=np.float32)
        raw  = np.array(list(feat.values()), dtype=np.float32)
        ml   = self.ml_prob_vec
        hi   = max(0, min(self.idx-SEQ_LEN, len(self.lstm_hidden)-1)) if self.lstm_hidden is not None else 0
        lstm_h = self.lstm_hidden[hi] if self.lstm_hidden is not None else np.array([])
        tfm_h  = self.tfm_hidden[hi]  if self.tfm_hidden  is not None else np.array([])
        omit   = self.omit_fn(self.records, self.idx)
        state  = np.concatenate([raw,ml,lstm_h,tfm_h,omit]).astype(np.float32)
        return np.clip(state/(np.abs(state).max()+1e-8),-5,5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = self.start
        return self._build_state(), {}

    def step(self, action):
        blue_pred  = action + 1          # 动作0-15 → 蓝球1-16
        actual_blue = self.records[self.idx]['blue']
        reward = 10.0 if blue_pred==actual_blue else -1.0
        self.idx += 1
        terminated = (self.idx >= len(self.records)-1)
        obs = self._build_state() if not terminated else np.zeros(self.state_dim,dtype=np.float32)
        return obs, reward, terminated, False, {'pred':blue_pred,'actual':actual_blue}


# ══════════════════════════════════════════════════════
#  PPO训练 + 回测
# ══════════════════════════════════════════════════════
class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._ep_reward = 0

    def _on_step(self):
        for i, done in enumerate(self.locals.get('dones',[])):
            self._ep_reward += self.locals['rewards'][i]
            if done:
                self.episode_rewards.append(self._ep_reward)
                self._ep_reward = 0
        return True


def train_ppo(env_fn, timesteps=100000, n_envs=4, policy='MlpPolicy'):
    vec_env = make_vec_env(env_fn, n_envs=n_envs)
    model   = PPO(policy, vec_env,
                  learning_rate=3e-4, n_steps=512, batch_size=128,
                  n_epochs=10, gamma=0.95, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.01,
                  verbose=0, device='cpu')
    cb = RewardLogger()
    print(f"    PPO 训练 {timesteps} 步，{n_envs} 并行环境…")
    model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
    avg_r = np.mean(cb.episode_rewards[-20:]) if cb.episode_rewards else 0
    print(f"    PPO 完成  最近20轮平均奖励: {avg_r:.3f}")
    return model


def backtest_ppo_kl8(model, records, feat_fn, ml_prob_vec,
                     lstm_hidden, tfm_hidden, omit_fn, last_n=100):
    """回测PPO在最近last_n期的表现"""
    start = max(SEQ_LEN+5, len(records)-last_n)
    total_net = 0; games = 0; hits_dist = defaultdict(int)
    for idx in range(start, len(records)-1):
        feat = feat_fn(records, idx)
        if feat is None: continue
        raw = np.array(list(feat.values()), dtype=np.float32)
        hi  = max(0, min(idx-SEQ_LEN, len(lstm_hidden)-1)) if lstm_hidden is not None else 0
        lh  = lstm_hidden[hi] if lstm_hidden is not None else np.array([])
        th  = tfm_hidden[hi]  if tfm_hidden  is not None else np.array([])
        om  = omit_fn(records, idx)
        state = np.clip(np.concatenate([raw,ml_prob_vec,lh,th,om])/(np.abs(np.concatenate([raw,ml_prob_vec,lh,th,om])).max()+1e-8),-5,5).astype(np.float32)
        action, _ = model.predict(state, deterministic=True)
        selected = [i+1 for i in range(80) if action[i]]
        if not selected: selected=[1]
        if len(selected)>10: selected=selected[:10]
        if len(selected)<4:  selected=list(range(1,5))
        actual = set(records[idx]['numbers'])
        hit    = len(actual & set(selected))
        net    = calc_payout(len(selected), hit)
        total_net += net; games += 1
        hits_dist[hit] += 1
    avg_net = round(total_net/games, 2) if games else 0
    win_rate = round(sum(v for k,v in hits_dist.items() if calc_payout(6,k)>0)/games*100,1) if games else 0
    return {'avg_net_per_game': avg_net, 'win_rate_pct': win_rate,
            'games_tested': games, 'hits_distribution': dict(hits_dist)}


# ══════════════════════════════════════════════════════
#  各彩种完整流程
# ══════════════════════════════════════════════════════
def run_3d(records, ml_pred):
    print(f"\n{'='*50}")
    print(f"福彩3D 集成训练（{len(records)}期）")
    print(f"{'='*50}")
    if not HAS_TORCH: return None

    ml_vec = extract_ml_prob_vec(ml_pred, '3d')
    print(f"  ML概率向量维度: {len(ml_vec)}")

    # 目标：百位（代表性，其他同理）
    targets = {
        'bai':     lambda r: r['digits'][0],
        'sum_grp': lambda r: 0 if sum(r['digits'])<=9 else(1 if sum(r['digits'])<=17 else 2),
    }

    lstm_hidden_all = None; tfm_hidden_all = None
    dl_target_results = {}

    for tname, tgt_fn in targets.items():
        print(f"\n  [LSTM+TFM] 目标: {tname}")
        X, y = build_seq_dataset(records, f3d, tgt_fn)
        if X is None or len(X)<40: continue
        nc = len(set(y.tolist())); fd = X.shape[2]

        lstm = LSTMEncoder(fd, hidden_dim=64, output_dim=nc)
        lstm_m, lstm_h, last_lstm_h, lstm_prob, lstm_acc = train_encoder(lstm, X, y, epochs=50)
        print(f"    LSTM 准确率: {lstm_acc}%")

        tfm = TransformerEncoder(fd, d_model=32, nhead=4, output_dim=nc)
        tfm_m, tfm_h, last_tfm_h, tfm_prob, tfm_acc = train_encoder(tfm, X, y, epochs=50)
        print(f"    TFM  准确率: {tfm_acc}%")

        if lstm_hidden_all is None:
            lstm_hidden_all = lstm_h
            tfm_hidden_all  = tfm_h

        ens_prob = lstm_prob*0.6 + tfm_prob*0.4
        classes  = sorted(set(y.tolist()))
        pred_val = classes[int(np.argmax(ens_prob))]
        dl_target_results[tname] = {
            'lstm_acc':lstm_acc,'tfm_acc':tfm_acc,
            'ensemble_pred':int(pred_val),
            'confidence':round(float(max(ens_prob))*100,1),
            'probs':{str(c):round(float(p)*100,1) for c,p in zip(classes,ens_prob)},
        }

    # 推荐
    score = [{} for _ in range(3)]
    for pi,pname in enumerate(['bai','shi','ge']):
        m = ml_pred.get('models',{}).get(pname,{})
        for k,v in m.get('prediction',{}).get('probs',{}).items():
            score[pi][int(k)]=score[pi].get(int(k),0)+float(v)*0.5
        dl_m = dl_target_results.get(pname,{})
        for k,v in dl_m.get('probs',{}).items():
            score[pi][int(k)]=score[pi].get(int(k),0)+float(v)*0.5
    tops=[sorted(score[p].items(),key=lambda x:-x[1])[:5] for p in range(3)]
    random.seed(int(date.today().strftime('%Y%m%d'))+100)
    groups=[]
    for i in range(6):
        combo=[]
        for pos in range(3):
            pool=[v for v,_ in tops[pos]] or list(range(10))
            weights=[max(s,0.1) for _,s in tops[pos]]
            tot=sum(weights); rv=random.uniform(0,tot); cum=0; chosen=pool[0]
            for v,w in zip(pool,weights):
                cum+=w
                if rv<=cum: chosen=v; break
            combo.append(int(chosen))
        groups.append(combo)

    return {
        'game':'3d','data_count':len(records),
        'model_results':dl_target_results,
        'recommendation':{'groups':groups,'note':'ML+LSTM+TFM三层集成，仅供娱乐参考。'},
        'rl_result':None,   # 3D暂不做RL
    }


def run_ssq(records, ml_pred):
    print(f"\n{'='*50}")
    print(f"双色球 集成训练（{len(records)}期）")
    print(f"{'='*50}")
    if not HAS_TORCH or not HAS_SB3: return None

    ml_vec = extract_ml_prob_vec(ml_pred, 'ssq')
    targets = {
        'blue':    lambda r: r['blue']-1,   # 0-15
        'sum_grp': lambda r: 0 if sum(r['red'])<70 else(1 if sum(r['red'])<100 else 2),
    }
    lstm_h_all=None; tfm_h_all=None; dl_results={}

    for tname, tgt_fn in targets.items():
        print(f"\n  [LSTM+TFM] {tname}")
        X,y = build_seq_dataset(records, fssq, tgt_fn)
        if X is None or len(X)<40: continue
        nc=len(set(y.tolist())); fd=X.shape[2]
        lstm=LSTMEncoder(fd,hidden_dim=64,output_dim=nc)
        lstm_m,lstm_h,_,lstm_p,lstm_acc=train_encoder(lstm,X,y,epochs=50)
        tfm=TransformerEncoder(fd,d_model=32,nhead=4,output_dim=nc)
        tfm_m,tfm_h,_,tfm_p,tfm_acc=train_encoder(tfm,X,y,epochs=50)
        if lstm_h_all is None: lstm_h_all=lstm_h; tfm_h_all=tfm_h
        ens=lstm_p*0.6+tfm_p*0.4; classes=sorted(set(y.tolist()))
        dl_results[tname]={'lstm_acc':lstm_acc,'tfm_acc':tfm_acc,
            'ensemble_pred':int(classes[np.argmax(ens)]),
            'confidence':round(float(max(ens))*100,1),
            'probs':{str(c):round(float(p)*100,1) for c,p in zip(classes,ens)}}
        print(f"    LSTM={lstm_acc}% TFM={tfm_acc}%")

    # PPO 蓝球策略
    print("\n  [PPO] 蓝球选号策略训练…")
    last_h = lstm_h_all[-1:] if lstm_h_all is not None else None

    def make_ssq_env():
        return IntegratedSSQEnv(records, fssq, ml_vec,
                                lstm_h_all, tfm_h_all, omission_vec_ssq)
    ppo = train_ppo(make_ssq_env, timesteps=60000, n_envs=4)

    # 蓝球回测
    correct=0; total=0
    for idx in range(SEQ_LEN+5, min(len(records)-1, SEQ_LEN+105)):
        feat=fssq(records,idx)
        if feat is None: continue
        raw=np.array(list(feat.values()),dtype=np.float32)
        hi=max(0,min(idx-SEQ_LEN,len(lstm_h_all)-1)) if lstm_h_all is not None else 0
        lh=lstm_h_all[hi] if lstm_h_all is not None else np.array([])
        th=tfm_h_all[hi]  if tfm_h_all  is not None else np.array([])
        om=omission_vec_ssq(records,idx)
        state=np.clip(np.concatenate([raw,ml_vec,lh,th,om])/(np.abs(np.concatenate([raw,ml_vec,lh,th,om])).max()+1e-8),-5,5).astype(np.float32)
        action,_=ppo.predict(state,deterministic=True)
        if action+1==records[idx]['blue']: correct+=1
        total+=1
    blue_acc=round(correct/total*100,1) if total else 0
    print(f"    PPO 蓝球准确率（近100期）: {blue_acc}%（随机基准6.25%）")

    # 综合推荐
    blue_ml_probs=ml_pred.get('models',{}).get('blue',{}).get('prediction',{}).get('probs',{})
    blue_dl_probs=dl_results.get('blue',{}).get('probs',{})
    blue_score={}
    for k,v in blue_ml_probs.items(): blue_score[int(k)]=blue_score.get(int(k),0)+float(v)*0.4
    for k,v in blue_dl_probs.items(): blue_score[int(k)]=blue_score.get(int(k),0)+float(v)*0.4
    # PPO预测（最后一期）
    feat=fssq(records,len(records)-1)
    if feat:
        raw=np.array(list(feat.values()),dtype=np.float32)
        hi=max(0,len(lstm_h_all)-1) if lstm_h_all is not None else 0
        lh=lstm_h_all[hi] if lstm_h_all is not None else np.array([])
        th=tfm_h_all[hi]  if tfm_h_all  is not None else np.array([])
        om=omission_vec_ssq(records,len(records)-1)
        state=np.clip(np.concatenate([raw,ml_vec,lh,th,om])/(np.abs(np.concatenate([raw,ml_vec,lh,th,om])).max()+1e-8),-5,5).astype(np.float32)
        ppo_blue,_=ppo.predict(state,deterministic=True)
        blue_score[int(ppo_blue+1)]=blue_score.get(int(ppo_blue+1),0)+20*0.2

    blue_top=[int(k) for k,_ in sorted(blue_score.items(),key=lambda x:-x[1])[:5]]
    freq30=Counter(n for r in records[-30:] for n in r['red'])
    hot=[x[0] for x in freq30.most_common(15)]
    pool=list(set(hot[:12])); random.seed(int(date.today().strftime('%Y%m%d'))+200)
    if len(pool)<18: pool+=random.sample([n for n in range(1,34) if n not in pool],18-len(pool))
    groups=[]
    for i in range(6):
        picked=[]
        for lo,hi_r in [(1,11),(12,22),(23,33)]:
            zp=[n for n in pool if lo<=n<=hi_r] or list(range(lo,hi_r+1))
            picked.append(random.choice(zp))
        rem=[n for n in pool if n not in picked]
        if len(rem)<3: rem=[n for n in range(1,34) if n not in picked]
        picked+=random.sample(rem,3)
        groups.append({'red':sorted(picked),'blue':int(blue_top[i%len(blue_top)])})

    return {
        'game':'ssq','data_count':len(records),
        'model_results':dl_results,
        'rl_result':{'blue_acc_pct':blue_acc,'blue_top':blue_top,
                     'note':f'PPO蓝球策略回测准确率{blue_acc}%（随机基准6.25%）'},
        'recommendation':{'groups':groups,'blue_recommend':blue_top[:3],
                          'note':'ML+LSTM+TFM+PPO四层集成，仅供娱乐参考。'},
    }


def run_kl8(records, ml_pred):
    print(f"\n{'='*50}")
    print(f"快乐8 集成训练（{len(records)}期）")
    print(f"{'='*50}")
    if not HAS_TORCH or not HAS_SB3: return None

    ml_vec = extract_ml_prob_vec(ml_pred, 'kl8')
    targets = {
        'zone_dom': lambda r: int(np.argmax([sum(1 for x in r['numbers'] if lo<=x<=hi) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]])),
        'tot_grp':  lambda r: 0 if sum(r['numbers'])<640 else(1 if sum(r['numbers'])<820 else 2),
    }
    lstm_h_all=None; tfm_h_all=None; dl_results={}

    for tname,tgt_fn in targets.items():
        print(f"\n  [LSTM+TFM] {tname}")
        X,y=build_seq_dataset(records,fkl8,tgt_fn)
        if X is None or len(X)<40: continue
        nc=len(set(y.tolist())); fd=X.shape[2]
        lstm=LSTMEncoder(fd,hidden_dim=64,output_dim=nc)
        lstm_m,lstm_h,_,lstm_p,lstm_acc=train_encoder(lstm,X,y,epochs=50)
        tfm=TransformerEncoder(fd,d_model=32,nhead=4,output_dim=nc)
        tfm_m,tfm_h,_,tfm_p,tfm_acc=train_encoder(tfm,X,y,epochs=50)
        if lstm_h_all is None: lstm_h_all=lstm_h; tfm_h_all=tfm_h
        ens=lstm_p*0.6+tfm_p*0.4; classes=sorted(set(y.tolist()))
        dl_results[tname]={'lstm_acc':lstm_acc,'tfm_acc':tfm_acc,
            'ensemble_pred':int(classes[np.argmax(ens)]),
            'confidence':round(float(max(ens))*100,1),
            'probs':{str(c):round(float(p)*100,1) for c,p in zip(classes,ens)}}
        print(f"    LSTM={lstm_acc}% TFM={tfm_acc}%")

    # PPO 集成环境训练
    print("\n  [PPO] 集成选号策略训练（状态含ML+LSTM+TFM+遗漏）…")
    def make_kl8_env():
        return IntegratedKL8Env(records, fkl8, ml_vec,
                                lstm_h_all, tfm_h_all, omission_vec_kl8)
    ppo = train_ppo(make_kl8_env, timesteps=100000, n_envs=4)

    # 回测
    print("  PPO 回测（近100期）…")
    bt = backtest_ppo_kl8(ppo, records, fkl8, ml_vec,
                          lstm_h_all, tfm_h_all, omission_vec_kl8, last_n=100)
    print(f"  平均净收益: {bt['avg_net_per_game']}元/期  胜率: {bt['win_rate_pct']}%")

    # 生成推荐
    feat=fkl8(records,len(records)-1)
    ppo_selected=[]
    if feat:
        raw=np.array(list(feat.values()),dtype=np.float32)
        hi=max(0,len(lstm_h_all)-1) if lstm_h_all is not None else 0
        lh=lstm_h_all[hi] if lstm_h_all is not None else np.array([])
        th=tfm_h_all[hi]  if tfm_h_all  is not None else np.array([])
        om=omission_vec_kl8(records,len(records)-1)
        state=np.clip(np.concatenate([raw,ml_vec,lh,th,om])/(np.abs(np.concatenate([raw,ml_vec,lh,th,om])).max()+1e-8),-5,5).astype(np.float32)
        action,_=ppo.predict(state,deterministic=True)
        ppo_selected=sorted([i+1 for i in range(80) if action[i]])
        if len(ppo_selected)<4: ppo_selected=list(range(1,5))
        if len(ppo_selected)>10: ppo_selected=ppo_selected[:10]

    from math import factorial
    def comb(n,k): return factorial(n)//(factorial(k)*factorial(n-k)) if k<=n else 0
    freq30=Counter(n for r in records[-30:] for n in r['numbers'])
    hot=[x[0] for x in freq30.most_common(20)]
    pool=list(set(hot[:15])); random.seed(int(date.today().strftime('%Y%m%d'))+300)
    if len(pool)<30: pool+=random.sample([n for n in range(1,81) if n not in pool],30-len(pool))
    def pick(n,sa=0):
        random.seed(int(date.today().strftime('%Y%m%d'))+300+sa); res=[]
        per=max(1,n//4)
        for lo,hi_r in [(1,20),(21,40),(41,60),(61,80)]:
            zp=[x for x in pool if lo<=x<=hi_r] or list(range(lo,hi_r+1))
            take=min(per,len(zp),n-len(res))
            if take>0: res+=random.sample(zp,take)
        while len(res)<n:
            ex=[x for x in range(1,81) if x not in res]
            if ex: res.append(random.choice(ex))
            else: break
        return sorted(res[:n])

    plays={
        'xuan4':    {'name':'选四','balls':4,'tip':'DL区间热号','groups':[pick(4,i) for i in range(3)]},
        'xuan5':    {'name':'选五','balls':5,'tip':'均衡赔率',  'groups':[pick(5,100+i) for i in range(3)]},
        'xuan5_fu': {'name':'选五复式','balls':5,'tip':f'8球C(8,5)={comb(8,5)}注','groups':[pick(8,200)]},
        'xuan6_ppo':{'name':'选六(PPO策略)','balls':len(ppo_selected) if 4<=len(ppo_selected)<=10 else 6,
                     'tip':f'PPO强化学习+真实赔率优化，回测净收益{bt["avg_net_per_game"]}元/期',
                     'groups':[ppo_selected if ppo_selected else pick(6,300)]},
        'xuan6_dl': {'name':'选六(DL推荐)','balls':6,'tip':'LSTM+TFM集成','groups':[pick(6,400+i) for i in range(2)]},
        'xuan9':    {'name':'选九','balls':9,'tip':'高赔率搏奖','groups':[pick(9,500+i) for i in range(2)]},
        'xuan10':   {'name':'选十','balls':10,'tip':'最高赔率', 'groups':[pick(10,600)]},
    }
    zone_name={0:'1-20区',1:'21-40区',2:'41-60区',3:'61-80区'}.get(
        dl_results.get('zone_dom',{}).get('ensemble_pred',1),'21-40区')

    return {
        'game':'kl8','data_count':len(records),
        'model_results':dl_results,
        'rl_result':{**bt,'ppo_selected':ppo_selected,
                     'note':'PPO按真实赔率优化，状态融合ML+LSTM+TFM+遗漏'},
        'recommendation':{'plays':plays,'zone_dominant_pred':zone_name,
                          'hot_nums':[int(x) for x in hot[:15]],
                          'note':'ML+LSTM+TFM+PPO四层分层决策，仅供娱乐参考。'},
    }


# ══════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════
print(f"\n{'#'*55}")
print(f"福彩集成DL+RL系统 v2  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"分层架构: ML概率 + LSTM隐层 + TFM特征 + PPO决策")
print(f"{'#'*55}")

if not HAS_TORCH:
    print("需要 PyTorch，退出"); sys.exit(1)

ml_preds = existing_pred.get('predictions', {})
dl_results = {}

for game, run_fn in [('3d',run_3d),('ssq',run_ssq),('kl8',run_kl8)]:
    records = history.get(game, [])
    if not isinstance(records,list) or len(records)<65:
        print(f"\n{game}: 数据不足，跳过"); continue
    ml_pred = ml_preds.get(game, {})
    try:
        result = run_fn(records, ml_pred)
        if result: dl_results[game] = result
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"{game} 失败: {e}")

# 写入 prediction.json（只更新 dl_result）
existing_pred['dl_result'] = {
    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'architecture': 'ML概率 → LSTM隐层 → TFM特征 → PPO集成决策',
    'reward_fn': '快乐8真实赔率净收益 / 双色球蓝球命中',
    'results': dl_results,
}

pred_json = json.dumps(existing_pred, ensure_ascii=False, indent=2)

if not GH_TOKEN:
    print("\n[DRY RUN] 跳过推送")
    for game, res in dl_results.items():
        rl = res.get('rl_result',{})
        print(f"  {game}: RL={rl}")
else:
    print("\n推送 prediction.json…")
    gh_put('prediction.json', pred_json,
           f"DL+RL集成更新 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("✓ 推送完成")

print(f"\n✅ 完成！{datetime.now().strftime('%Y-%m-%d %H:%M')}")
