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

def train_encoder(model, X, y, epochs=60, lr=5e-4, batch_size=32):
    model = model.to(DEVICE)
    X_t = torch.FloatTensor(X).to(DEVICE); y_t = torch.LongTensor(y).to(DEVICE)
    loader = DataLoader(TensorDataset(X_t,y_t), batch_size, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for xb,yb in loader:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        sch.step()
        if (ep+1)%20==0: print(f"      ep{ep+1}/{epochs} loss={loss.item():.4f}")
    model.eval()
    hidden_states=[]
    with torch.no_grad():
        for i in range(0,len(X_t),batch_size):
            xb=X_t[i:i+batch_size]; _,h=model(xb,return_hidden=True)
            hidden_states.append(h.cpu().numpy())
    hidden_states=np.vstack(hidden_states)
    with torch.no_grad(): preds=model(X_t).argmax(dim=1).cpu().numpy()
    acc=round(float((preds==y).mean())*100,1)
    with torch.no_grad(): logits,last_h=model(X_t[-1:],return_hidden=True); probs=torch.softmax(logits,dim=1)[0].cpu().numpy()
    return model, hidden_states, last_h.cpu().numpy()[0], probs, acc

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

configs = {
    '3d':  (f3d,  {'bai':lambda r:r['digits'][0],'shi':lambda r:r['digits'][1],'ge':lambda r:r['digits'][2],
                    'sum_grp':lambda r:0 if sum(r['digits'])<=9 else(1 if sum(r['digits'])<=17 else 2)}),
    'ssq': (fssq, {'blue':lambda r:r['blue']-1,'odd':lambda r:sum(1 for x in r['red'] if x%2!=0),
                    'sum_grp':lambda r:0 if sum(r['red'])<70 else(1 if sum(r['red'])<100 else 2),
                    'ac_grp':lambda r:_ssq_ac_grp(r['red']),
                    'red_zone_dom':lambda r:_ssq_zone_dom(r['red']),
                    'gap_grp':lambda r:_ssq_gap_grp(r['red'])}),
    'kl8': (fkl8, {'zone_dom':lambda r:int(np.argmax([sum(1 for x in r['numbers'] if lo<=x<=hi) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]])),
                    'tot_grp':lambda r:0 if sum(r['numbers'])<640 else(1 if sum(r['numbers'])<820 else 2)}),
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

        lstm = LSTMEncoder(fd, hidden_dim=64, output_dim=nc)
        lstm_m, lstm_h, _, lstm_p, lstm_acc = train_encoder(lstm, X, y, epochs=50)
        print(f"    LSTM 准确率: {lstm_acc}%")

        tfm = TransformerEncoder(fd, d_model=32, nhead=4, output_dim=nc)
        tfm_m, tfm_h, _, tfm_p, tfm_acc = train_encoder(tfm, X, y, epochs=50)
        print(f"    TFM  准确率: {tfm_acc}%")

        if lstm_hidden_all is None:
            lstm_hidden_all = lstm_h; tfm_hidden_all = tfm_h
            # 保存权重（供每日RL加载，只存主目标模型即可）
            torch.save(lstm_m.state_dict(), f'{LOCAL_DIR}/{game}_lstm.pt')
            torch.save(tfm_m.state_dict(),  f'{LOCAL_DIR}/{game}_tfm.pt')
            np.save(f'{LOCAL_DIR}/{game}_lstm_hidden.npy', lstm_h)
            np.save(f'{LOCAL_DIR}/{game}_tfm_hidden.npy',  tfm_h)
            meta = {'feat_dim':fd,'n_classes':nc,'hidden_dim':64,'d_model':32,'seq_len':SEQ_LEN}
            with open(f'{LOCAL_DIR}/{game}_meta.json','w') as f: json.dump(meta,f)

        ens = lstm_p*0.6 + tfm_p*0.4
        classes = sorted(set(y.tolist()))
        game_results[tname] = {
            'lstm_acc':lstm_acc, 'tfm_acc':tfm_acc,
            'ensemble_pred': int(classes[int(np.argmax(ens))]),
            'confidence': round(float(max(ens))*100,1),
            'probs': {str(c):round(float(p)*100,1) for c,p in zip(classes,ens)},
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
