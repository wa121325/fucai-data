"""
福彩机器学习预测系统 v2
- 全量历史数据训练（不再限制50期）
- 特征窗口仍用近50期滑动
- 新增：LightGBM + LSTM(可选) + 马尔可夫链 + 贝叶斯遗漏 + 集成投票
- 每日自动生成 prediction.json
- AI Gateway 结合：buildPrompt输出结构化数据供AI解读
"""
import json, sys, warnings, math
from datetime import datetime, date
from collections import Counter, defaultdict
import random

warnings.filterwarnings('ignore')
random.seed(42)

# ── 依赖检测 ─────────────────────────────────────────
try:
    import numpy as np
    HAS_NP = True
except ImportError:
    print("请安装: pip install numpy scikit-learn xgboost lightgbm")
    sys.exit(1)

try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    HAS_SKL = True
except ImportError:
    HAS_SKL = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# LSTM 可选（GitHub Actions 不装 tensorflow 节省时间）
HAS_LSTM = False

print(f"依赖: numpy✓  sklearn={'✓' if HAS_SKL else '✗'}  xgboost={'✓' if HAS_XGB else '✗'}  lightgbm={'✓' if HAS_LGB else '✗'}")


# ══════════════════════════════════════════════════════
#  特征工程（窗口最大50期，但输入全量records）
# ══════════════════════════════════════════════════════
WINDOW = 50  # 特征计算窗口

def feat_3d(records, idx):
    window = records[max(0, idx-WINDOW):idx]
    if len(window) < 5:
        return None
    r = window[-1]
    d = r['digits']
    b, s, g = d
    total = b+s+g
    span = max(d)-min(d)
    road = [x%3 for x in d]
    prev = window[-2]['digits'] if len(window)>=2 else d
    repeat = sum(1 for i in range(3) if prev[i]==d[i])
    sorted3 = sorted(d)
    is_arith = (sorted3[1]-sorted3[0])==(sorted3[2]-sorted3[1]) and sorted3[2]-sorted3[0]>0

    feats = {
        'sum':total,'sum_tail':total%10,'span':span,
        'odd':sum(1 for x in d if x%2!=0),
        'big':sum(1 for x in d if x>=5),
        'road0':road.count(0),'road1':road.count(1),'road2':road.count(2),
        'b':b,'s':s,'g':g,
        'gap_bs':abs(b-s),'gap_sg':abs(s-g),
        'group':0 if b==s==g else (1 if (b==s or s==g or b==g) else 2),
        'repeat':repeat,'is_arith':int(is_arith),
    }
    for ws, sfx in [(5,'5'),(10,'10'),(20,'20'),(WINDOW,'50')]:
        w = window[-ws:]
        sums  = [sum(x['digits']) for x in w]
        spans = [max(x['digits'])-min(x['digits']) for x in w]
        feats[f'sum_mean_{sfx}']  = float(np.mean(sums))
        feats[f'sum_std_{sfx}']   = float(np.std(sums)) if len(sums)>1 else 0
        feats[f'span_mean_{sfx}'] = float(np.mean(spans))
        for ci, cn in enumerate(['b','s','g']):
            vals = [x['digits'][ci] for x in w]
            feats[f'{cn}_mean_{sfx}'] = float(np.mean(vals))
    if len(window)>=3:
        s3=[sum(x['digits']) for x in window[-3:]]
        feats['sum_trend'] = 1 if s3[-1]>s3[-2] else (-1 if s3[-1]<s3[-2] else 0)
    else:
        feats['sum_trend'] = 0
    return feats


def feat_ssq(records, idx):
    window = records[max(0,idx-WINDOW):idx]
    if len(window)<5:
        return None
    r = window[-1]
    red = sorted(r['red'])
    blue = r['blue']
    total = sum(red)
    odd = sum(1 for x in red if x%2!=0)
    big = sum(1 for x in red if x>16)
    consec = sum(1 for i in range(len(red)-1) if red[i+1]-red[i]==1)
    diffs = set()
    for i in range(len(red)):
        for j in range(i+1,len(red)):
            diffs.add(red[j]-red[i])
    ac = len(diffs)-(len(red)-1)
    z1=sum(1 for x in red if x<=11)
    z2=sum(1 for x in red if 12<=x<=22)
    z3=sum(1 for x in red if x>=23)
    max_gap=max(red[i+1]-red[i] for i in range(len(red)-1)) if len(red)>1 else 0

    feats={'red_sum':total,'odd':odd,'big':big,'consec':consec,'ac':ac,
           'z1':z1,'z2':z2,'z3':z3,'max_gap':max_gap,
           'blue':blue,'blue_odd':blue%2,'blue_big':int(blue>=9),
           'red_max':red[-1],'red_min':red[0],'red_span':red[-1]-red[0]}

    for ws,sfx in [(5,'5'),(10,'10'),(20,'20'),(WINDOW,'50')]:
        w = window[-ws:]
        sums  = [sum(x['red']) for x in w]
        blues = [x['blue'] for x in w]
        odds  = [sum(1 for n in x['red'] if n%2!=0) for x in w]
        feats[f'sum_mean_{sfx}']  = float(np.mean(sums))
        feats[f'sum_std_{sfx}']   = float(np.std(sums)) if len(sums)>1 else 0
        feats[f'blue_mean_{sfx}'] = float(np.mean(blues))
        feats[f'odd_mean_{sfx}']  = float(np.mean(odds))

    all_red = [n for x in window for n in x['red']]
    cnt = Counter(all_red)
    feats['hot_z1']=sum(cnt.get(n,0) for n in range(1,12))
    feats['hot_z2']=sum(cnt.get(n,0) for n in range(12,23))
    feats['hot_z3']=sum(cnt.get(n,0) for n in range(23,34))
    if len(window)>=3:
        s3=[sum(x['red']) for x in window[-3:]]
        feats['sum_trend']=1 if s3[-1]>s3[-2] else(-1 if s3[-1]<s3[-2] else 0)
    else:
        feats['sum_trend']=0
    return feats


def feat_kl8(records, idx):
    window = records[max(0,idx-WINDOW):idx]
    if len(window)<5:
        return None
    r = window[-1]
    nums = sorted(r['numbers'])
    total = sum(nums)
    odd = sum(1 for x in nums if x%2!=0)
    big = sum(1 for x in nums if x>40)
    zones=[sum(1 for x in nums if lo<=x<=hi) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]]
    five =[sum(1 for x in nums if lo<=x<=hi) for lo,hi in [(1,16),(17,32),(33,48),(49,64),(65,80)]]
    cg=0; inc=False
    for i in range(len(nums)-1):
        if nums[i+1]-nums[i]==1:
            if not inc: cg+=1; inc=True
        else: inc=False

    feats={'total':total,'odd':odd,'big':big,
           'min_n':nums[0],'max_n':nums[-1],
           'z1':zones[0],'z2':zones[1],'z3':zones[2],'z4':zones[3],
           'f1':five[0],'f2':five[1],'f3':five[2],'f4':five[3],'f5':five[4],
           'consec_grp':cg}

    for ws,sfx in [(5,'5'),(10,'10'),(20,'20'),(WINDOW,'50')]:
        w=window[-ws:]
        tots=[sum(x['numbers']) for x in w]
        feats[f'total_mean_{sfx}']=float(np.mean(tots))
        feats[f'total_std_{sfx}']=float(np.std(tots)) if len(tots)>1 else 0
        for zi,(lo,hi) in enumerate([(1,20),(21,40),(41,60),(61,80)]):
            zv=[sum(1 for n in x['numbers'] if lo<=n<=hi) for x in w]
            feats[f'z{zi+1}_mean_{sfx}']=float(np.mean(zv))

    all_n=[n for x in window for n in x['numbers']]
    cnt=Counter(all_n)
    feats['hot_z1']=sum(cnt.get(n,0) for n in range(1,21))
    feats['hot_z2']=sum(cnt.get(n,0) for n in range(21,41))
    feats['hot_z3']=sum(cnt.get(n,0) for n in range(41,61))
    feats['hot_z4']=sum(cnt.get(n,0) for n in range(61,81))
    if len(window)>=3:
        t3=[sum(x['numbers']) for x in window[-3:]]
        feats['total_trend']=1 if t3[-1]>t3[-2] else(-1 if t3[-1]<t3[-2] else 0)
    else:
        feats['total_trend']=0
    return feats


# ══════════════════════════════════════════════════════
#  马尔可夫链（号码转移概率）
# ══════════════════════════════════════════════════════

def markov_3d(records):
    """3D各位的一阶马尔可夫转移矩阵"""
    trans = [defaultdict(Counter) for _ in range(3)]
    for i in range(1, len(records)):
        prev = records[i-1]['digits']
        curr = records[i]['digits']
        for pos in range(3):
            trans[pos][prev[pos]][curr[pos]] += 1
    result = []
    last = records[-1]['digits']
    for pos in range(3):
        probs = trans[pos][last[pos]]
        total = sum(probs.values()) or 1
        top3 = sorted(probs.items(), key=lambda x:-x[1])[:3]
        result.append({'pos':pos,'from':last[pos],
                       'top3':[(int(k),round(v/total*100,1)) for k,v in top3]})
    return result


def markov_ssq_blue(records):
    """双色球蓝球马尔可夫"""
    trans = defaultdict(Counter)
    for i in range(1,len(records)):
        trans[records[i-1]['blue']][records[i]['blue']] += 1
    last_blue = records[-1]['blue']
    probs = trans[last_blue]
    total = sum(probs.values()) or 1
    return [(int(k),round(v/total*100,1)) for k,v in sorted(probs.items(),key=lambda x:-x[1])[:5]]


# ══════════════════════════════════════════════════════
#  贝叶斯遗漏分析
# ══════════════════════════════════════════════════════

def omission_3d(records):
    """3D各位各号码遗漏期数（上次出现到现在）"""
    result = []
    for pos in range(3):
        omit = {}
        for d in range(10):
            for i in range(len(records)-1,-1,-1):
                if records[i]['digits'][pos]==d:
                    omit[d]=len(records)-1-i; break
            else:
                omit[d]=len(records)
        # 超过理论平均遗漏(10期)的号码
        avg=10
        overdue=[d for d,v in omit.items() if v>avg]
        result.append({'pos':pos,'omission':omit,'overdue':overdue,'avg':avg})
    return result


def omission_ssq(records):
    """双色球红球遗漏"""
    omit={}
    for n in range(1,34):
        for i in range(len(records)-1,-1,-1):
            if n in records[i]['red']:
                omit[n]=len(records)-1-i; break
        else:
            omit[n]=len(records)
    avg = len(records)*6/33  # 平均每期6个球，33个号码
    overdue=[n for n,v in omit.items() if v>avg*1.5]
    blue_omit={}
    for n in range(1,17):
        for i in range(len(records)-1,-1,-1):
            if records[i]['blue']==n:
                blue_omit[n]=len(records)-1-i; break
        else:
            blue_omit[n]=len(records)
    return {'red_omit':omit,'red_overdue':overdue,'blue_omit':blue_omit,
            'blue_overdue':[n for n,v in blue_omit.items() if v>len(records)/16*1.5]}


def omission_kl8(records):
    """快乐8号码遗漏"""
    omit={}
    for n in range(1,81):
        for i in range(len(records)-1,-1,-1):
            if n in records[i]['numbers']:
                omit[n]=len(records)-1-i; break
        else:
            omit[n]=len(records)
    avg=len(records)*20/80  # 平均每期20个，80个号码
    overdue=sorted([n for n,v in omit.items() if v>avg*1.5], key=lambda n: omit[n], reverse=True)[:15]
    return {'omit':omit,'overdue':overdue,'avg':round(avg,1)}


# ══════════════════════════════════════════════════════
#  模型集成训练 + 时间序列回测
# ══════════════════════════════════════════════════════

def build_dataset(records, feat_fn):
    X_all, valid_idx = [], []
    for i in range(len(records)):
        f = feat_fn(records, i)
        if f is not None:
            X_all.append(list(f.values()))
            valid_idx.append(i)
    feat_names = list(feat_fn(records, valid_idx[0]).keys()) if valid_idx else []
    return np.array(X_all, dtype=float), valid_idx, feat_names


def make_models():
    """构建所有可用模型"""
    models = {}
    models['rf'] = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        random_state=42, n_jobs=-1)
    if HAS_XGB:
        models['xgb'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='mlogloss', verbosity=0)
    if HAS_LGB:
        models['lgb'] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, random_state=42, verbose=-1)
    return models


def train_and_backtest(X, y, feat_names, target_name, min_train=60):
    """
    时间序列滚动回测（全量数据）
    - min_train: 至少60期才开始预测
    - 每期向前滚动一步
    """
    n = len(X)
    if n < min_train + 10:
        print(f"    {target_name}: 数据不足({n}期)，跳过")
        return None

    model_list = make_models()
    all_true = []
    preds_by_model = {k: [] for k in model_list}

    # 只回测最近100期（避免太慢）
    backtest_start = max(min_train, n-100)

    for end in range(backtest_start, n):
        X_tr, y_tr = X[:end], y[:end]
        X_te, y_te = X[end:end+1], y[end:end+1]
        for mname, m in model_list.items():
            try:
                m.fit(X_tr, y_tr)
                preds_by_model[mname].extend(m.predict(X_te).tolist())
            except Exception:
                preds_by_model[mname].extend([y_tr[-1]])
        all_true.extend(y_te.tolist())

    if not all_true:
        return None

    acc = {}
    for mname, preds in preds_by_model.items():
        if len(preds)==len(all_true):
            acc[mname] = round(accuracy_score(all_true, preds)*100, 1)

    # 集成投票准确率
    from collections import Counter as C
    ensemble_preds = []
    for i in range(len(all_true)):
        votes = C(preds_by_model[m][i] for m in preds_by_model if len(preds_by_model[m])>i)
        ensemble_preds.append(votes.most_common(1)[0][0])
    acc['ensemble'] = round(accuracy_score(all_true, ensemble_preds)*100, 1)

    # 用全量数据重新训练最终模型
    final_models = {}
    for mname, m in make_models().items():
        try:
            m.fit(X, y)
            final_models[mname] = m
        except Exception:
            pass

    # 特征重要性（随机森林）
    feat_imp = []
    if 'rf' in final_models:
        imp = final_models['rf'].feature_importances_
        top5 = sorted(zip(feat_names,imp),key=lambda x:-x[1])[:5]
        feat_imp = [{'name':k,'score':round(float(v),4)} for k,v in top5]

    # 集成预测下一期概率
    last_X = X[-1:].copy()
    all_probs = []
    classes = None
    for m in final_models.values():
        try:
            p = m.predict_proba(last_X)[0]
            all_probs.append(p)
            if classes is None:
                classes = m.classes_.tolist()
        except Exception:
            pass

    if not all_probs or classes is None:
        return None

    avg_prob = np.mean(all_probs, axis=0)
    pred_class = classes[int(np.argmax(avg_prob))]

    # 近10期回测详情
    recent_bt = []
    for i in range(max(0,len(all_true)-10), len(all_true)):
        recent_bt.append({
            'true': int(all_true[i]),
            'pred_ensemble': int(ensemble_preds[i]),
            'pred_rf': int(preds_by_model['rf'][i]) if 'rf' in preds_by_model and len(preds_by_model['rf'])>i else None,
        })

    return {
        'target': target_name,
        'data_used': n,
        'backtest_periods': len(all_true),
        'accuracy': acc,
        'feature_importance': feat_imp,
        'prediction': {
            'value': int(pred_class),
            'confidence': round(float(max(avg_prob))*100, 1),
            'probs': {str(c): round(float(p)*100,1) for c,p in zip(classes,avg_prob)},
        },
        'backtest_latest': recent_bt,
    }


# ══════════════════════════════════════════════════════
#  各彩种完整分析
# ══════════════════════════════════════════════════════

def targets_3d(records):
    t={'bai':[],'shi':[],'ge':[],'sum_group':[],'odd_cnt':[]}
    for r in records:
        d=r['digits']
        t['bai'].append(d[0]); t['shi'].append(d[1]); t['ge'].append(d[2])
        s=sum(d); t['sum_group'].append(0 if s<=9 else(1 if s<=17 else 2))
        t['odd_cnt'].append(sum(1 for x in d if x%2!=0))
    return t

def targets_ssq(records):
    t={'blue':[],'odd_cnt':[],'sum_group':[],'zone_combo':[]}
    for r in records:
        t['blue'].append(r['blue'])
        t['odd_cnt'].append(sum(1 for x in r['red'] if x%2!=0))
        s=sum(r['red']); t['sum_group'].append(0 if s<70 else(1 if s<100 else 2))
        z1=sum(1 for x in r['red'] if x<=11); z3=sum(1 for x in r['red'] if x>=23)
        t['zone_combo'].append(z1*10+z3)
    return t

def targets_kl8(records):
    t={'odd_group':[],'zone_dom':[],'total_group':[]}
    for r in records:
        odd=sum(1 for x in r['numbers'] if x%2!=0)
        t['odd_group'].append(0 if odd<9 else(1 if odd<=11 else 2))
        zones=[sum(1 for x in r['numbers'] if lo<=x<=hi) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]]
        t['zone_dom'].append(int(np.argmax(zones)))
        tt=sum(r['numbers']); t['total_group'].append(0 if tt<640 else(1 if tt<820 else 2))
    return t


def analyze_game(game, records, feat_fn, targets_fn, target_keys):
    print(f"  {game}: 全量{len(records)}期数据，特征窗口{WINDOW}期")
    X, valid_idx, feat_names = build_dataset(records, feat_fn)
    tgts = targets_fn(records)
    result = {'game':game,'data_count':len(records),'window':WINDOW,
              'updated_at':datetime.now().strftime('%Y-%m-%d %H:%M'),'models':{}}
    for tname in target_keys:
        y = np.array([tgts[tname][i] for i in valid_idx])
        print(f"    [{tname}] 训练中（{len(y)}期）…")
        r = train_and_backtest(X, y, feat_names, tname)
        if r:
            result['models'][tname] = r
            best_acc = max(r['accuracy'].values()) if r['accuracy'] else 0
            print(f"      集成准确率: {r['accuracy'].get('ensemble','—')}%  最优: {best_acc}%")
    return result


# ══════════════════════════════════════════════════════
#  推荐号码生成（结合ML + 马尔可夫 + 遗漏）
# ══════════════════════════════════════════════════════

def recommend_3d(records, ml_result, markov, omission):
    """三种方法投票生成3D推荐"""
    score = [{} for _ in range(3)]  # 各位号码得分

    # 1. ML概率贡献
    for pos, pname in enumerate(['bai','shi','ge']):
        m = ml_result.get('models',{}).get(pname,{})
        probs = m.get('prediction',{}).get('probs',{}) if m else {}
        for k,v in probs.items():
            score[pos][int(k)] = score[pos].get(int(k),0) + v*0.5

    # 2. 马尔可夫贡献
    for mk in markov:
        pos = mk['pos']
        for val, prob in mk['top3']:
            score[pos][val] = score[pos].get(val,0) + prob*0.3

    # 3. 遗漏贡献（遗漏久的+分）
    for om in omission:
        pos = om['pos']
        for d in om['overdue']:
            score[pos][d] = score[pos].get(d,0) + 10*0.2

    # 各位TOP5候选
    tops = [sorted(score[pos].items(),key=lambda x:-x[1])[:5] for pos in range(3)]

    # 生成6注推荐
    random.seed(int(datetime.now().strftime('%Y%m%d')))
    groups=[]
    for i in range(6):
        combo=[]
        for pos in range(3):
            pool=[v for v,_ in tops[pos]] or list(range(10))
            # 加权随机选
            weights=[s for _,s in tops[pos]] or [1]*len(pool)
            total_w=sum(weights)
            r_val=random.uniform(0,total_w)
            cum=0
            chosen=pool[0]
            for v,w in zip(pool,weights):
                cum+=w
                if r_val<=cum: chosen=v; break
            combo.append(chosen)
        groups.append(combo)

    sum_m=ml_result.get('models',{}).get('sum_group',{})
    sum_label={0:'小(0-9)',1:'中(10-17)',2:'大(18-27)'}.get(
        sum_m.get('prediction',{}).get('value',1) if sum_m else 1,'中(10-17)')

    return {
        'groups':groups,
        'pos_candidates':[[int(v) for v,_ in t] for t in tops],
        'markov':markov,
        'omission_overdue':[om['overdue'] for om in omission],
        'sum_pred':sum_label,
        'note':'综合ML+马尔可夫转移+遗漏分析三路投票，仅供娱乐参考。',
    }


def recommend_ssq(records, ml_result, omission):
    """双色球推荐：结合ML+遗漏"""
    freq30=Counter(n for r in records[-30:] for n in r['red'])
    hot=[x[0] for x in freq30.most_common(15)]
    overdue_red=omission.get('red_overdue',[])
    overdue_blue=omission.get('blue_overdue',[])

    blue_m=ml_result.get('models',{}).get('blue',{})
    blue_probs=blue_m.get('prediction',{}).get('probs',{}) if blue_m else {}
    blue_top=[int(k) for k,v in sorted(blue_probs.items(),key=lambda x:-x[1])[:5]] or \
             [x[0] for x in Counter(r['blue'] for r in records[-30:]).most_common(5)]

    # 推荐红球候选池（热号+遗漏号混合）
    pool=list(set(hot[:10]+overdue_red[:8]))
    if len(pool)<18:
        pool+=random.sample([n for n in range(1,34) if n not in pool],18-len(pool))

    odd_m=ml_result.get('models',{}).get('odd_cnt',{})
    odd_pred=odd_m.get('prediction',{}).get('value',3) if odd_m else 3
    sum_m=ml_result.get('models',{}).get('sum_group',{})
    sum_pred={0:'低(<70)',1:'中(70-99)',2:'高(≥100)'}.get(
        sum_m.get('prediction',{}).get('value',1) if sum_m else 1,'中(70-99)')

    random.seed(int(datetime.now().strftime('%Y%m%d')))
    groups=[]
    for i in range(6):
        # 三区均衡选红球
        picked=[]
        zones=[(1,11),(12,22),(23,33)]
        for lo,hi in zones:
            zp=[n for n in pool if lo<=n<=hi]
            if not zp: zp=list(range(lo,hi+1))
            picked.append(random.choice(zp))
        # 剩余3个从全池选
        remain=[n for n in pool if n not in picked]
        if len(remain)<3: remain=[n for n in range(1,34) if n not in picked]
        picked+=random.sample(remain,3)
        blue=blue_top[i%len(blue_top)]
        groups.append({'red':sorted(picked),'blue':blue})

    return {
        'groups':groups,
        'blue_recommend':blue_top[:3],
        'hot_red':hot[:12],
        'overdue_red':overdue_red[:8],
        'overdue_blue':overdue_blue[:4],
        'odd_pred':int(odd_pred),
        'sum_pred':sum_pred,
        'note':'综合ML+遗漏分析+三区均衡选号，仅供娱乐参考。',
    }


def recommend_kl8(records, ml_result, omission):
    """快乐8多玩法推荐：结合ML+遗漏"""
    freq30=Counter(n for r in records[-30:] for n in r['numbers'])
    freq10=Counter(n for r in records[-10:] for n in r['numbers'])
    hot=sorted([x[0] for x in freq30.most_common(20)])
    cold=sorted([x[0] for x in freq30.most_common()[-20:]])
    overdue=omission.get('overdue',[])

    zone_m=ml_result.get('models',{}).get('zone_dom',{})
    zone_pred=zone_m.get('prediction',{}).get('value',1) if zone_m else 1
    zone_name={0:'1-20区',1:'21-40区',2:'41-60区',3:'61-80区'}.get(zone_pred,'21-40区')
    total_m=ml_result.get('models',{}).get('total_group',{})
    total_pred={0:'低',1:'中',2:'高'}.get(
        total_m.get('prediction',{}).get('value',1) if total_m else 1,'中')

    # 智能候选池
    pool=list(set(hot[:15]+overdue[:8]+[x[0] for x in freq10.most_common(8)]))
    if len(pool)<30:
        pool+=random.sample([n for n in range(1,81) if n not in pool],30-len(pool))

    def pick_balanced(n, seed_add=0):
        random.seed(int(datetime.now().strftime('%Y%m%d'))+seed_add)
        result=[]
        zone_ranges=[(1,20),(21,40),(41,60),(61,80)]
        per_zone=max(1,n//4)
        for lo,hi in zone_ranges:
            zp=[x for x in pool if lo<=x<=hi]
            if not zp: zp=list(range(lo,hi+1))
            take=min(per_zone,len(zp),n-len(result))
            result+=random.sample(zp,take)
        while len(result)<n:
            extra=[x for x in range(1,81) if x not in result]
            result.append(random.choice(extra))
        return sorted(result[:n])

    def factorial(n):
        r=1
        for i in range(2,n+1): r*=i
        return r

    def comb(n,k):
        if k>n: return 0
        return factorial(n)//(factorial(k)*factorial(n-k))

    plays={
        'xuan4':{'name':'选四','balls':4,'groups':[pick_balanced(4,i) for i in range(3)],
                 'tip':'4球全中，高赔率'},
        'xuan5':{'name':'选五','balls':5,'groups':[pick_balanced(5,100+i) for i in range(3)],
                 'tip':'5球全中，赔率与命中率均衡'},
        'xuan5_fu':{'name':'选五复式','balls':5,'groups':[pick_balanced(8,200)],
                    'tip':f'8球覆盖C(8,5)={comb(8,5)}注五球组合'},
        'xuan6':{'name':'选六','balls':6,'groups':[pick_balanced(6,300+i) for i in range(3)],
                 'tip':'6球全中，主流玩法推荐'},
        'xuan9':{'name':'选九','balls':9,'groups':[pick_balanced(9,400+i) for i in range(2)],
                 'tip':'9球全中，搏高赔率'},
        'xuan10':{'name':'选十','balls':10,'groups':[pick_balanced(10,500)],
                  'tip':'10球全中，最高赔率'},
    }

    return {
        'plays':plays,
        'zone_dominant_pred':zone_name,
        'total_range_pred':total_pred,
        'hot_nums':hot[:15],
        'cold_nums':cold[:10],
        'overdue':overdue[:10],
        'note':'综合ML+遗漏分析+区间均衡选号，按玩法分类推荐，仅供娱乐参考。',
    }


# ══════════════════════════════════════════════════════
#  每日回测（昨日预测 vs 今日实际）
# ══════════════════════════════════════════════════════

def daily_backtest(history, prev_pred):
    report={'date':str(date.today()),'games':{}}
    for game in ['3d','ssq','kl8']:
        records=history.get(game,[])
        if not records or game not in (prev_pred or {}): continue
        latest=records[-1]
        rec=prev_pred[game].get('recommendation',{})

        if game=='3d':
            actual=latest.get('digits',[])
            groups=rec.get('groups',[])
            hits=[g for g in groups if g==actual]
            partial=[g for g in groups if sum(1 for i,v in enumerate(g) if i<len(actual) and v==actual[i])>=2]
            report['games'][game]={'actual':actual,'hit_count':len(hits),'partial_count':len(partial)}

        elif game=='ssq':
            ar=sorted(latest.get('red',[])); ab=latest.get('blue',0)
            groups=rec.get('groups',[])
            results=[{'red_hit':len(set(g.get('red',[]))&set(ar)),'blue_hit':int(g.get('blue')==ab)} for g in groups]
            report['games'][game]={'actual_red':ar,'actual_blue':ab,'group_results':results,
                                   'best_red_hit':max((x['red_hit'] for x in results),default=0)}

        elif game=='kl8':
            actual=set(latest.get('numbers',[]))
            plays=rec.get('plays',{})
            play_results={}
            for pk,pd in plays.items():
                grps=pd.get('groups',[])
                balls=pd.get('balls',0)
                name=pd.get('name',pk)
                grp_res=[]
                for g in grps:
                    nums=g if isinstance(g,list) else g.get('numbers',g)
                    hit=len(actual&set(nums)); won=(hit==balls)
                    grp_res.append({'nums':nums,'hit':hit,'balls':balls,'won':won})
                play_results[pk]={'name':name,'balls':balls,'groups':grp_res,
                                  'any_won':any(x['won'] for x in grp_res),
                                  'best_hit':max((x['hit'] for x in grp_res),default=0)}
            report['games'][game]={'actual':sorted(actual),'play_results':play_results,
                                   'best_hit':max((pr['best_hit'] for pr in play_results.values()),default=0)}
    return report


# ══════════════════════════════════════════════════════
#  AI Gateway 结合：生成结构化解读提示
# ══════════════════════════════════════════════════════

def build_ai_context(game, records, ml_result, markov_data, omission_data, recommendation):
    """生成供AI解读的结构化数据摘要（写入prediction.json）"""
    latest=records[-1]
    n=len(records)

    if game=='3d':
        best_acc={k:v.get('accuracy',{}).get('ensemble',0) for k,v in ml_result.get('models',{}).items()}
        top_model=max(best_acc.items(),key=lambda x:x[1]) if best_acc else ('—',0)
        context={
            'game':'福彩3D','latest_qihao':latest.get('qihao',''),
            'latest_date':latest.get('date',''),'data_count':n,
            'latest_digits':latest.get('digits',[]),
            'ml_top_accuracy':f"{top_model[0]}目标集成准确率{top_model[1]}%",
            'markov_summary':[f"{'百十个'[m['pos']]}位上期{m['from']}→最可能{m['top3'][0][0]}({m['top3'][0][1]}%)" for m in markov_data],
            'overdue_summary':[f"{'百十个'[i]}位遗漏号:{v}" for i,v in enumerate(omission_data['overdue'] if isinstance(omission_data,dict) else [om['overdue'] for om in omission_data])],
            'recommend_groups':recommendation.get('groups',[]),
            'sum_pred':recommendation.get('sum_pred',''),
        }
    elif game=='ssq':
        context={
            'game':'双色球','latest_qihao':latest.get('qihao',''),
            'latest_date':latest.get('date',''),'data_count':n,
            'latest_red':latest.get('red',[]),'latest_blue':latest.get('blue',0),
            'blue_overdue':omission_data.get('blue_overdue',[]),
            'red_overdue':omission_data.get('red_overdue',[])[:8],
            'blue_recommend':recommendation.get('blue_recommend',[]),
            'sum_pred':recommendation.get('sum_pred',''),
            'odd_pred':recommendation.get('odd_pred',''),
        }
    elif game=='kl8':
        context={
            'game':'快乐8','latest_qihao':latest.get('qihao',''),
            'latest_date':latest.get('date',''),'data_count':n,
            'zone_pred':recommendation.get('zone_dominant_pred',''),
            'total_pred':recommendation.get('total_range_pred',''),
            'overdue_top10':omission_data.get('overdue',[])[:10],
            'hot_nums':recommendation.get('hot_nums',[])[:10],
        }
    return context


# ══════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════

def main():
    print(f"\n{'='*50}")
    print(f"福彩ML预测系统 v2  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    # 读取历史数据
    try:
        with open('history.json',encoding='utf-8') as f:
            history=json.load(f)
    except FileNotFoundError:
        print("history.json 不存在，请先运行 lottery_crawler.py"); sys.exit(1)

    # 读取上次预测（用于回测）
    prev_pred={}
    try:
        with open('prediction.json',encoding='utf-8') as f:
            prev_pred=json.load(f).get('predictions',{})
    except Exception:
        pass

    # 回测
    print("\n── 生成昨日回测报告…")
    bt=daily_backtest(history,prev_pred)

    predictions={}
    for game, feat_fn, tgt_fn, tkeys in [
        ('3d',  feat_3d,  targets_3d,  ['bai','shi','ge','sum_group','odd_cnt']),
        ('ssq', feat_ssq, targets_ssq, ['blue','odd_cnt','sum_group']),
        ('kl8', feat_kl8, targets_kl8, ['odd_group','zone_dom','total_group']),
    ]:
        records=history.get(game,[])
        if len(records)<70:
            print(f"\n── {game}: 数据不足({len(records)}期，需≥70)，跳过"); continue

        print(f"\n── 分析 {game}（全量{len(records)}期，窗口{WINDOW}期）…")
        try:
            ml_res=analyze_game(game,records,feat_fn,tgt_fn,tkeys)

            # 马尔可夫 & 遗漏
            if game=='3d':
                mk=markov_3d(records); om=omission_3d(records)
                rec=recommend_3d(records,ml_res,mk,om)
                mk_out=mk; om_out={'overdue':[x['overdue'] for x in om]}
            elif game=='ssq':
                mk=markov_ssq_blue(records); om=omission_ssq(records)
                rec=recommend_ssq(records,ml_res,om)
                mk_out=mk; om_out=om
            else:
                mk=None; om=omission_kl8(records)
                rec=recommend_kl8(records,ml_res,om)
                mk_out=None; om_out=om

            ai_ctx=build_ai_context(game,records,ml_res,mk_out or [],om_out,rec)

            predictions[game]={**ml_res,'recommendation':rec,
                               'markov':mk_out,'omission_summary':om_out,
                               'ai_context':ai_ctx}
            print(f"  ✓ {game} 完成")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ✗ {game} 失败: {e}")

    output={
        'updated_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models_used':{'rf':True,'xgb':HAS_XGB,'lgb':HAS_LGB,'ensemble':True,
                       'markov':True,'omission':True},
        'predictions':predictions,
        'backtest':bt,
        'disclaimer':'彩票开奖结果具有完全随机性，本系统仅为数学统计模型演示，预测结果不具有实际预测意义，请理性购彩，量力而行。',
    }
    with open('prediction.json','w',encoding='utf-8') as f:
        json.dump(output,f,ensure_ascii=False,indent=2)

    print(f"\n✓ 完成！已写入 prediction.json")
    for game,res in predictions.items():
        print(f"  {game}: {res['data_count']}期数据")
        for t,m in res.get('models',{}).items():
            acc=m.get('accuracy',{})
            print(f"    {t}: 集成{acc.get('ensemble','—')}% RF{acc.get('rf','—')}% XGB{acc.get('xgb','—')}% LGB{acc.get('lgb','—')}%")


if __name__=='__main__':
    main()
