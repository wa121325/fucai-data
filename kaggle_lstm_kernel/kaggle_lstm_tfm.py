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

def get_secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        v = UserSecretsClient().get_secret(name)
        if v: return v
    except Exception: pass
    return os.environ.get(name, '')

# ── 把你的 Token 填在这里（Kaggle Secrets 不稳定时的兜底）──
_HARDCODED_GH_TOKEN = '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'      # ← 新的 GitHub Token
_HARDCODED_GH_REPO  = 'wa121325/fucai-data'
_HARDCODED_KAGGLE_TOKEN = 'KGAT_0847d8a3c8619a4db2ff2c7c3e9e824f'

GH_TOKEN = get_secret('GH_TOKEN') or get_secret('gh_token') or _HARDCODED_GH_TOKEN
GH_REPO  = get_secret('GH_REPO')  or get_secret('gh_repo')  or _HARDCODED_GH_REPO
KAGGLE_TOKEN = get_secret('KAGGLE_TOKEN') or get_secret('kaggle_token') or _HARDCODED_KAGGLE_TOKEN
print(f"GitHub: {GH_REPO}  GH_TOKEN: {'✓('+str(len(GH_TOKEN))+')' if GH_TOKEN else '✗'}")

try:
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

configs = {
    '3d':  (f3d,  {'bai':lambda r:r['digits'][0],'shi':lambda r:r['digits'][1],'ge':lambda r:r['digits'][2],
                    'sum_grp':lambda r:0 if sum(r['digits'])<=9 else(1 if sum(r['digits'])<=17 else 2)}),
    'ssq': (fssq, {'blue':lambda r:r['blue']-1,'odd':lambda r:sum(1 for x in r['red'] if x%2!=0),
                    'sum_grp':lambda r:0 if sum(r['red'])<70 else(1 if sum(r['red'])<100 else 2)}),
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

# ── 更新 prediction.json ──────────────────────────────
print("\n读取现有 prediction.json…")
existing = {}
raw_pred = gh_raw('prediction.json')
if raw_pred:
    try: existing = json.loads(raw_pred)
    except Exception: pass

existing.setdefault('dl_result', {})
existing['dl_result']['lstm_tfm'] = {
    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'models': 'LSTM + Transformer（每周全量训练）',
    'seq_len': SEQ_LEN, 'device': str(DEVICE),
    'results': dl_results,
    'note': '权重已存入 Kaggle Dataset，供每日 PPO 强化学习加载使用。',
}

pred_json = json.dumps(existing, ensure_ascii=False, indent=2)
if not GH_TOKEN:
    print("\n[DRY RUN] 未配置 GH_TOKEN")
else:
    print("\n推送 prediction.json…")
    gh_put('prediction.json', pred_json, f"LSTM+TFM每周训练 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("✓ 完成")

print(f"\n✅ 全部完成！{datetime.now().strftime('%Y-%m-%d %H:%M')}")
