"""
福彩机器学习预测系统 v3
核心改进：
1. 全量历史数据训练（不限50期）
2. 增量更新策略：首次完整训练保存模型包，每日只增量微调
3. 预测目标明确为"下一期"（用当期特征预测下期结果）
4. 马尔可夫链 + 遗漏分析 + 集成投票
5. 字段名统一，避免undefined
"""
import json, sys, os, warnings, pickle
from datetime import datetime, date
from collections import Counter, defaultdict
import random

warnings.filterwarnings('ignore')

try:
    import numpy as np
except ImportError:
    print("pip install numpy scikit-learn xgboost lightgbm"); sys.exit(1)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    HAS_SKL = True
except ImportError:
    HAS_SKL = False; print("缺少 scikit-learn")

try:
    import xgboost as xgb; HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb; HAS_LGB = True
except ImportError:
    HAS_LGB = False

MODEL_PKL = 'models.pkl'   # 模型包文件
WINDOW    = 50             # 特征窗口（近50期滑动）
MIN_TRAIN = 60             # 最少训练期数

print(f"sklearn={HAS_SKL} xgb={HAS_XGB} lgb={HAS_LGB}")


# ══════════════════════════════════════════════════════
#  特征工程
#  注意：feat_xxx(records, idx) 用 records[0..idx-1] 作特征，
#        目标变量是 records[idx]，即"预测下一期"
# ══════════════════════════════════════════════════════

def extract_nums(record, game):
    if game == '3d':  return record.get('digits', [0,0,0])
    if game == 'ssq': return record.get('red', []), record.get('blue', 1)
    if game == 'kl8': return record.get('numbers', [])

def feat_3d(records, idx):
    """用 records[0..idx-1] 的历史特征，预测 records[idx] 的结果"""
    if idx < 5: return None
    window = records[max(0, idx-WINDOW):idx]
    last   = window[-1]
    d      = last['digits']
    b, s, g = d
    total = b+s+g
    span  = max(d)-min(d)
    prev  = window[-2]['digits'] if len(window)>=2 else d
    repeat = sum(1 for i in range(3) if prev[i]==d[i])
    sorted3 = sorted(d)
    is_arith = (sorted3[1]-sorted3[0])==(sorted3[2]-sorted3[1]) and span>0

    feats = {
        'sum':total,'sum_tail':total%10,'span':span,
        'odd':sum(1 for x in d if x%2!=0),
        'big':sum(1 for x in d if x>=5),
        'road0':d[0]%3==0,'road1':d[1]%3==0,'road2':d[2]%3==0,
        'b':b,'s':s,'g':g,
        'gap_bs':abs(b-s),'gap_sg':abs(s-g),
        'group':0 if b==s==g else(1 if(b==s or s==g or b==g)else 2),
        'repeat':repeat,'is_arith':int(is_arith),
    }
    for ws,sfx in [(5,'5'),(10,'10'),(20,'20'),(min(WINDOW,idx),'50')]:
        w = window[-ws:]
        sums  = [sum(x['digits']) for x in w]
        spans = [max(x['digits'])-min(x['digits']) for x in w]
        feats[f'sum_mean_{sfx}']  = float(np.mean(sums)) if sums else 0
        feats[f'sum_std_{sfx}']   = float(np.std(sums))  if len(sums)>1 else 0
        feats[f'span_mean_{sfx}'] = float(np.mean(spans)) if spans else 0
        for ci,cn in enumerate(['b','s','g']):
            vals = [x['digits'][ci] for x in w]
            feats[f'{cn}_mean_{sfx}'] = float(np.mean(vals)) if vals else 0
    if len(window)>=3:
        s3=[sum(x['digits']) for x in window[-3:]]
        feats['sum_trend']=1 if s3[-1]>s3[-2] else(-1 if s3[-1]<s3[-2] else 0)
    else:
        feats['sum_trend']=0
    return feats


def feat_ssq(records, idx):
    if idx < 5: return None
    window = records[max(0,idx-WINDOW):idx]
    last = window[-1]
    red = sorted(last['red']); blue = last['blue']
    total=sum(red); odd=sum(1 for x in red if x%2!=0); big=sum(1 for x in red if x>16)
    consec=sum(1 for i in range(len(red)-1) if red[i+1]-red[i]==1)
    diffs=set()
    for i in range(len(red)):
        for j in range(i+1,len(red)): diffs.add(red[j]-red[i])
    ac=len(diffs)-(len(red)-1)
    z1=sum(1 for x in red if x<=11); z2=sum(1 for x in red if 12<=x<=22); z3=sum(1 for x in red if x>=23)
    max_gap=max(red[i+1]-red[i] for i in range(len(red)-1)) if len(red)>1 else 0
    feats={'red_sum':total,'odd':odd,'big':big,'consec':consec,'ac':ac,
           'z1':z1,'z2':z2,'z3':z3,'max_gap':max_gap,
           'blue':blue,'blue_odd':blue%2,'blue_big':int(blue>=9),
           'red_max':red[-1],'red_min':red[0],'red_span':red[-1]-red[0]}
    for ws,sfx in [(5,'5'),(10,'10'),(20,'20'),(min(WINDOW,idx),'50')]:
        w=window[-ws:]
        sums=[sum(x['red']) for x in w]; blues=[x['blue'] for x in w]
        odds=[sum(1 for n in x['red'] if n%2!=0) for x in w]
        feats[f'sum_mean_{sfx}']=float(np.mean(sums)) if sums else 0
        feats[f'sum_std_{sfx}']=float(np.std(sums)) if len(sums)>1 else 0
        feats[f'blue_mean_{sfx}']=float(np.mean(blues)) if blues else 0
        feats[f'odd_mean_{sfx}']=float(np.mean(odds)) if odds else 0
    all_red=[n for x in window for n in x['red']]; cnt=Counter(all_red)
    feats['hot_z1']=sum(cnt.get(n,0) for n in range(1,12))
    feats['hot_z2']=sum(cnt.get(n,0) for n in range(12,23))
    feats['hot_z3']=sum(cnt.get(n,0) for n in range(23,34))
    if len(window)>=3:
        s3=[sum(x['red']) for x in window[-3:]]
        feats['sum_trend']=1 if s3[-1]>s3[-2] else(-1 if s3[-1]<s3[-2] else 0)
    else: feats['sum_trend']=0
    return feats


def feat_kl8(records, idx):
    if idx < 5: return None
    window=records[max(0,idx-WINDOW):idx]
    last=window[-1]; nums=sorted(last['numbers'])
    total=sum(nums); odd=sum(1 for x in nums if x%2!=0); big=sum(1 for x in nums if x>40)
    zones=[sum(1 for x in nums if lo<=x<=hi) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]]
    five=[sum(1 for x in nums if lo<=x<=hi) for lo,hi in [(1,16),(17,32),(33,48),(49,64),(65,80)]]
    cg=0; inc=False
    for i in range(len(nums)-1):
        if nums[i+1]-nums[i]==1:
            if not inc: cg+=1; inc=True
        else: inc=False
    feats={'total':total,'odd':odd,'big':big,'min_n':nums[0],'max_n':nums[-1],
           'z1':zones[0],'z2':zones[1],'z3':zones[2],'z4':zones[3],
           'f1':five[0],'f2':five[1],'f3':five[2],'f4':five[3],'f5':five[4],'consec_grp':cg}
    for ws,sfx in [(5,'5'),(10,'10'),(20,'20'),(min(WINDOW,idx),'50')]:
        w=window[-ws:]
        tots=[sum(x['numbers']) for x in w]
        feats[f'total_mean_{sfx}']=float(np.mean(tots)) if tots else 0
        feats[f'total_std_{sfx}']=float(np.std(tots)) if len(tots)>1 else 0
        for zi,(lo,hi) in enumerate([(1,20),(21,40),(41,60),(61,80)]):
            zv=[sum(1 for n in x['numbers'] if lo<=n<=hi) for x in w]
            feats[f'z{zi+1}_mean_{sfx}']=float(np.mean(zv)) if zv else 0
    all_n=[n for x in window for n in x['numbers']]; cnt=Counter(all_n)
    for zi,(lo,hi) in enumerate([(1,20),(21,40),(41,60),(61,80)]):
        feats[f'hot_z{zi+1}']=sum(cnt.get(n,0) for n in range(lo,hi+1))
    if len(window)>=3:
        t3=[sum(x['numbers']) for x in window[-3:]]
        feats['total_trend']=1 if t3[-1]>t3[-2] else(-1 if t3[-1]<t3[-2] else 0)
    else: feats['total_trend']=0
    return feats


# ══════════════════════════════════════════════════════
#  目标变量：records[idx] 本期实际结果
# ══════════════════════════════════════════════════════

def target_3d(records):
    """目标是本期（由上期特征预测），idx 从1开始（需要至少1期历史）"""
    bai=[]; shi=[]; ge=[]; sum_g=[]; odd_c=[]
    for i in range(1, len(records)):   # i=目标期，特征来自records[0..i-1]
        d = records[i]['digits']
        bai.append(d[0]); shi.append(d[1]); ge.append(d[2])
        s=sum(d); sum_g.append(0 if s<=9 else(1 if s<=17 else 2))
        odd_c.append(sum(1 for x in d if x%2!=0))
    return {'bai':bai,'shi':shi,'ge':ge,'sum_group':sum_g,'odd_cnt':odd_c}

def target_ssq(records):
    blue=[]; odd_c=[]; sum_g=[]
    for i in range(1, len(records)):
        r=records[i]
        blue.append(r['blue'])
        odd_c.append(sum(1 for x in r['red'] if x%2!=0))
        s=sum(r['red']); sum_g.append(0 if s<70 else(1 if s<100 else 2))
    return {'blue':blue,'odd_cnt':odd_c,'sum_group':sum_g}

def target_kl8(records):
    odd_g=[]; zone_d=[]; total_g=[]
    for i in range(1, len(records)):
        r=records[i]
        odd=sum(1 for x in r['numbers'] if x%2!=0)
        odd_g.append(0 if odd<9 else(1 if odd<=11 else 2))
        zones=[sum(1 for x in r['numbers'] if lo<=x<=hi) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]]
        zone_d.append(int(np.argmax(zones)))
        tt=sum(r['numbers']); total_g.append(0 if tt<640 else(1 if tt<820 else 2))
    return {'odd_group':odd_g,'zone_dom':zone_d,'total_group':total_g}


# ══════════════════════════════════════════════════════
#  构建数据集（特征与目标对齐：feat[i]对应target[i]）
# ══════════════════════════════════════════════════════

def build_dataset(records, feat_fn, target_fn):
    """
    records[i] 的特征 = feat_fn(records, i)（用0..i-1历史）
    records[i] 的目标 = target_fn(records)[i-1]（本期实际）
    两者对齐后返回 X, y
    """
    targets = target_fn(records)   # 长度 len(records)-1，对应 records[1..n-1]
    X_list, feat_names = [], None
    valid_target_indices = []      # 记录哪些 i 有对应特征

    for i in range(1, len(records)):   # i 是目标期的索引
        f = feat_fn(records, i)        # 用 records[0..i-1] 的特征
        if f is None:
            continue
        if feat_names is None:
            feat_names = list(f.keys())
        X_list.append(list(f.values()))
        valid_target_indices.append(i-1)   # targets 的下标是 i-1

    if not X_list:
        return None, None, None

    X = np.array(X_list, dtype=float)
    # 对各目标变量提取对应的y
    y_dict = {}
    for tname, tvals in targets.items():
        y_dict[tname] = np.array([tvals[vi] for vi in valid_target_indices])

    return X, y_dict, feat_names or []


# ══════════════════════════════════════════════════════
#  模型工厂
# ══════════════════════════════════════════════════════

def make_models():
    m = {}
    m['rf'] = RandomForestClassifier(n_estimators=200,max_depth=8,
                                     min_samples_leaf=3,random_state=42,n_jobs=-1)
    if HAS_XGB:
        m['xgb'] = xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,
                                      subsample=0.8,colsample_bytree=0.8,
                                      random_state=42,eval_metric='mlogloss',verbosity=0)
    if HAS_LGB:
        m['lgb'] = lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,
                                       num_leaves=31,random_state=42,verbose=-1)
    return m


# ══════════════════════════════════════════════════════
#  训练 + 时间序列回测
# ══════════════════════════════════════════════════════

def train_one_target(X, y, feat_names, tname, saved_models=None):
    """
    全量训练 + 最近100期滚动回测
    saved_models: 上次保存的模型（增量微调）
    """
    n = len(X)
    if n < MIN_TRAIN:
        return None

    model_zoo = saved_models if saved_models else make_models()

    # 回测：用最近min(100,n//5)期做滚动验证
    bt_size = min(100, max(20, n//5))
    bt_start = n - bt_size
    all_true, preds = [], {k:[] for k in model_zoo}

    for end in range(bt_start, n):
        Xtr, ytr = X[:end], y[:end]
        Xte, yte = X[end:end+1], y[end:end+1]
        for mname, model in model_zoo.items():
            try:
                model.fit(Xtr, ytr)
                preds[mname].extend(model.predict(Xte).tolist())
            except Exception:
                preds[mname].extend([int(ytr[-1])])
        all_true.extend(yte.tolist())

    acc = {}
    for mname, ps in preds.items():
        if len(ps)==len(all_true):
            acc[mname] = round(accuracy_score(all_true,ps)*100,1)

    # 集成准确率
    from collections import Counter as C
    ens_preds = [C(preds[m][i] for m in preds if len(preds[m])>i).most_common(1)[0][0]
                 for i in range(len(all_true))]
    acc['ensemble'] = round(accuracy_score(all_true,ens_preds)*100,1)

    # 全量重新训练
    final = {}
    for mname, model in make_models().items():
        try:
            model.fit(X, y)
            final[mname] = model
        except Exception as e:
            print(f"    {mname} 训练失败: {e}")

    # 特征重要性
    feat_imp = []
    if 'rf' in final:
        top5 = sorted(zip(feat_names, final['rf'].feature_importances_),
                      key=lambda x: -x[1])[:5]
        feat_imp = [{'name':k,'score':round(float(v),4)} for k,v in top5]

    # 预测下一期（用最后一期特征）
    last_X = X[-1:]
    all_probs, classes = [], None
    for m in final.values():
        try:
            p = m.predict_proba(last_X)[0]
            all_probs.append(p)
            if classes is None: classes = m.classes_.tolist()
        except Exception: pass

    if not all_probs or classes is None:
        return None

    avg_prob = np.mean(all_probs, axis=0)
    pred_val = classes[int(np.argmax(avg_prob))]

    # 近10期回测明细
    bt_detail = [
        {'true':int(t),'pred':int(p),'hit':int(t==p)}
        for t,p in zip(all_true[-10:], ens_preds[-10:])
    ]

    return {
        'target':    tname,
        'data_used': n,
        'bt_periods': len(all_true),
        'accuracy':  acc,
        'feat_imp':  feat_imp,
        'prediction':{
            'value':      int(pred_val),
            'confidence': round(float(max(avg_prob))*100,1),
            'probs':      {str(c):round(float(p)*100,1) for c,p in zip(classes,avg_prob)},
        },
        'bt_detail': bt_detail,
        'models_obj': final,   # 临时保存，后面存pkl
    }


# ══════════════════════════════════════════════════════
#  马尔可夫链
# ══════════════════════════════════════════════════════

def markov_3d(records):
    trans = [defaultdict(Counter) for _ in range(3)]
    for i in range(1,len(records)):
        p=records[i-1]['digits']; c=records[i]['digits']
        for pos in range(3): trans[pos][p[pos]][c[pos]] += 1
    last=records[-1]['digits']
    result=[]
    pos_names=['百位','十位','个位']
    for pos in range(3):
        probs=trans[pos][last[pos]]; total=sum(probs.values()) or 1
        top3=sorted(probs.items(),key=lambda x:-x[1])[:3]
        result.append({
            'pos_name': pos_names[pos],
            'from':     last[pos],
            'top3':     [{'val':int(k),'prob':round(v/total*100,1)} for k,v in top3]
        })
    return result

def markov_ssq_blue(records):
    trans=defaultdict(Counter)
    for i in range(1,len(records)): trans[records[i-1]['blue']][records[i]['blue']] += 1
    last=records[-1]['blue']; probs=trans[last]; total=sum(probs.values()) or 1
    return [{'val':int(k),'prob':round(v/total*100,1)}
            for k,v in sorted(probs.items(),key=lambda x:-x[1])[:5]]


# ══════════════════════════════════════════════════════
#  遗漏分析
# ══════════════════════════════════════════════════════

def omission_3d(records):
    result=[]
    pos_names=['百位','十位','个位']
    for pos in range(3):
        omit={}
        for d in range(10):
            for i in range(len(records)-1,-1,-1):
                if records[i]['digits'][pos]==d: omit[d]=len(records)-1-i; break
            else: omit[d]=len(records)
        overdue=[d for d,v in omit.items() if v>10]
        result.append({'pos_name':pos_names[pos],'omission':omit,'overdue':overdue})
    return result

def omission_ssq(records):
    red_omit={}
    for n in range(1,34):
        for i in range(len(records)-1,-1,-1):
            if n in records[i]['red']: red_omit[n]=len(records)-1-i; break
        else: red_omit[n]=len(records)
    avg_red=len(records)*6/33
    blue_omit={}
    for n in range(1,17):
        for i in range(len(records)-1,-1,-1):
            if records[i]['blue']==n: blue_omit[n]=len(records)-1-i; break
        else: blue_omit[n]=len(records)
    avg_blue=len(records)/16
    return {
        'red_omit':  {str(k):v for k,v in red_omit.items()},
        'red_overdue': sorted([n for n,v in red_omit.items() if v>avg_red*1.5]),
        'blue_omit':  {str(k):v for k,v in blue_omit.items()},
        'blue_overdue':sorted([n for n,v in blue_omit.items() if v>avg_blue*1.5]),
    }

def omission_kl8(records):
    omit={}
    for n in range(1,81):
        for i in range(len(records)-1,-1,-1):
            if n in records[i]['numbers']: omit[n]=len(records)-1-i; break
        else: omit[n]=len(records)
    avg=len(records)*20/80
    overdue=sorted([n for n,v in omit.items() if v>avg*1.5],key=lambda n:omit[n],reverse=True)
    return {'omit':{str(k):v for k,v in omit.items()},'overdue':overdue[:20],'avg':round(avg,1)}


# ══════════════════════════════════════════════════════
#  推荐号码（预测下一期）
# ══════════════════════════════════════════════════════

def recommend_3d(records, models_result, markov, omission):
    score=[{} for _ in range(3)]
    pos_map={'bai':0,'shi':1,'ge':2}
    for pname,pidx in pos_map.items():
        m=models_result.get(pname)
        if m:
            for k,v in m['prediction']['probs'].items():
                score[pidx][int(k)]=score[pidx].get(int(k),0)+v*0.5
    for mk in markov:
        pos=markov.index(mk)
        for item in mk['top3']:
            v=item['val']; p=item['prob']
            score[pos][v]=score[pos].get(v,0)+p*0.3
    for om in omission:
        pos=omission.index(om)
        for d in om['overdue']:
            score[pos][d]=score[pos].get(d,0)+15*0.2

    tops=[sorted(score[pos].items(),key=lambda x:-x[1])[:5] for pos in range(3)]
    seed=int(datetime.now().strftime('%Y%m%d'))
    groups=[]
    for i in range(6):
        random.seed(seed+i)
        combo=[]
        for pos in range(3):
            pool=[v for v,_ in tops[pos]] or list(range(10))
            weights=[s for _,s in tops[pos]] or [1]*len(pool)
            total_w=sum(weights); r=random.uniform(0,total_w); cum=0
            chosen=pool[0]
            for v,w in zip(pool,weights):
                cum+=w
                if r<=cum: chosen=v; break
            combo.append(chosen)
        groups.append(combo)

    sm=models_result.get('sum_group')
    sum_label={0:'小(0-9)',1:'中(10-17)',2:'大(18-27)'}.get(
        sm['prediction']['value'] if sm else 1,'中(10-17)')
    return {
        'groups':        groups,
        'pos_candidates':[[int(v) for v,_ in t] for t in tops],
        'sum_pred':      sum_label,
        'note':          '综合ML+马尔可夫+遗漏三路投票预测下一期，仅供娱乐参考。',
    }


def recommend_ssq(records, models_result, omission):
    freq30=Counter(n for r in records[-30:] for n in r['red'])
    hot=  [x[0] for x in freq30.most_common(15)]
    overdue_red  = omission.get('red_overdue',[])
    overdue_blue = omission.get('blue_overdue',[])

    bm=models_result.get('blue')
    blue_probs=bm['prediction']['probs'] if bm else {}
    blue_top=[int(k) for k,v in sorted(blue_probs.items(),key=lambda x:-x[1])[:5]] or \
             [x[0] for x in Counter(r['blue'] for r in records[-30:]).most_common(5)]

    om_val=models_result.get('odd_cnt')
    odd_pred=int(om_val['prediction']['value']) if om_val else 3
    sm=models_result.get('sum_group')
    sum_pred={0:'低(<70)',1:'中(70-99)',2:'高(≥100)'}.get(
        sm['prediction']['value'] if sm else 1,'中(70-99)')

    pool=list(set(hot[:10]+overdue_red[:8]))
    if len(pool)<18:
        pool+=random.sample([n for n in range(1,34) if n not in pool],18-len(pool))

    seed=int(datetime.now().strftime('%Y%m%d'))
    groups=[]
    for i in range(6):
        random.seed(seed+i)
        picked=[]
        for lo,hi in [(1,11),(12,22),(23,33)]:
            zp=[n for n in pool if lo<=n<=hi] or list(range(lo,hi+1))
            picked.append(random.choice(zp))
        remain=[n for n in pool if n not in picked]
        if len(remain)<3: remain=[n for n in range(1,34) if n not in picked]
        picked+=random.sample(remain,3)
        groups.append({'red':sorted(picked),'blue':blue_top[i%len(blue_top)]})

    return {
        'groups':          groups,
        'blue_recommend':  blue_top[:3],
        'hot_red':         hot[:12],
        'overdue_red':     overdue_red[:8],
        'overdue_blue':    overdue_blue[:4],
        'odd_pred':        odd_pred,
        'sum_pred':        sum_pred,
        'note':            '综合ML+遗漏+三区均衡预测下一期，仅供娱乐参考。',
    }


def recommend_kl8(records, models_result, omission):
    freq30=Counter(n for r in records[-30:] for n in r['numbers'])
    freq10=Counter(n for r in records[-10:] for n in r['numbers'])
    hot=  sorted([x[0] for x in freq30.most_common(20)])
    cold= sorted([x[0] for x in freq30.most_common()[-20:]])
    overdue=omission.get('overdue',[])

    zm=models_result.get('zone_dom')
    zone_pred=zm['prediction']['value'] if zm else 1
    zone_name={0:'1-20区',1:'21-40区',2:'41-60区',3:'61-80区'}.get(zone_pred,'21-40区')
    tm=models_result.get('total_group')
    total_pred={0:'低(<640)',1:'中(640-819)',2:'高(≥820)'}.get(
        tm['prediction']['value'] if tm else 1,'中(640-819)')

    pool=list(set(hot[:15]+overdue[:8]+[x[0] for x in freq10.most_common(8)]))
    if len(pool)<30:
        pool+=random.sample([n for n in range(1,81) if n not in pool],30-len(pool))

    def comb(n,k):
        if k>n: return 0
        from math import factorial
        return factorial(n)//(factorial(k)*factorial(n-k))

    def pick(n, seed_add=0):
        random.seed(int(datetime.now().strftime('%Y%m%d'))+seed_add)
        res=[]; zones=[(1,20),(21,40),(41,60),(61,80)]
        per=max(1,n//4)
        for lo,hi in zones:
            zp=[x for x in pool if lo<=x<=hi] or list(range(lo,hi+1))
            take=min(per,len(zp),n-len(res))
            res+=random.sample(zp,take)
        while len(res)<n:
            extra=[x for x in range(1,81) if x not in res]
            res.append(random.choice(extra))
        return sorted(res[:n])

    plays={
        'xuan4': {'name':'选四','balls':4,'tip':'4球全中，高赔率',
                  'groups':[pick(4,i) for i in range(3)]},
        'xuan5': {'name':'选五','balls':5,'tip':'5球全中，赔率命中均衡',
                  'groups':[pick(5,100+i) for i in range(3)]},
        'xuan5_fu':{'name':'选五复式','balls':5,'tip':f'8球复式含{comb(8,5)}注五球组合',
                    'groups':[pick(8,200)]},
        'xuan6': {'name':'选六','balls':6,'tip':'主流玩法，推荐',
                  'groups':[pick(6,300+i) for i in range(3)]},
        'xuan9': {'name':'选九','balls':9,'tip':'9球全中，搏高赔率',
                  'groups':[pick(9,400+i) for i in range(2)]},
        'xuan10':{'name':'选十','balls':10,'tip':'最高赔率',
                  'groups':[pick(10,500)]},
    }

    return {
        'plays':             plays,
        'zone_dominant_pred':zone_name,
        'total_range_pred':  total_pred,
        'hot_nums':          hot[:15],
        'cold_nums':         cold[:10],
        'overdue':           overdue[:10],
        'note':              '综合ML+遗漏+区间均衡预测下一期，仅供娱乐参考。',
    }


# ══════════════════════════════════════════════════════
#  每日回测（昨日推荐 vs 今日实际）
# ══════════════════════════════════════════════════════

def daily_backtest(history, prev_pred):
    report={'date':str(date.today()),'games':{}}
    for game in ['3d','ssq','kl8']:
        records=history.get(game,[])
        if not records or game not in (prev_pred or {}): continue
        latest=records[-1]; rec=prev_pred[game].get('recommendation',{})
        if game=='3d':
            actual=latest.get('digits',[])
            groups=rec.get('groups',[])
            hits=[g for g in groups if g==actual]
            partial=[g for g in groups if sum(1 for i,v in enumerate(g) if i<len(actual) and v==actual[i])>=2]
            report['games'][game]={'actual':actual,'hit_count':len(hits),'partial_count':len(partial)}
        elif game=='ssq':
            ar=sorted(latest.get('red',[])); ab=latest.get('blue',0)
            groups=rec.get('groups',[])
            grs=[{'red_hit':len(set(g.get('red',[]))&set(ar)),'blue_hit':int(g.get('blue')==ab)} for g in groups]
            report['games'][game]={'actual_red':ar,'actual_blue':ab,'group_results':grs,
                                   'best_red_hit':max((x['red_hit'] for x in grs),default=0)}
        elif game=='kl8':
            actual=set(latest.get('numbers',[]))
            plays=rec.get('plays',{})
            play_results={}
            for pk,pd in plays.items():
                grps=pd.get('groups',[]); balls=pd.get('balls',0); name=pd.get('name',pk)
                grs=[]
                for g in grps:
                    nums=g if isinstance(g,list) else []
                    hit=len(actual&set(nums)); grs.append({'hit':hit,'balls':balls,'won':hit==balls})
                play_results[pk]={'name':name,'balls':balls,'groups':grs,
                                  'any_won':any(x['won'] for x in grs),
                                  'best_hit':max((x['hit'] for x in grs),default=0)}
            report['games'][game]={'actual':sorted(actual),'play_results':play_results,
                                   'best_hit':max((pr['best_hit'] for pr in play_results.values()),default=0)}
    return report


# ══════════════════════════════════════════════════════
#  AI上下文（供网页AI解读使用）
# ══════════════════════════════════════════════════════

def build_ai_context(game, records, models_result, markov_data, omission_data, rec):
    latest=records[-1]; n=len(records)
    ctx={'game':game,'data_count':n,
         'latest_qihao':latest.get('qihao',''),
         'latest_date':latest.get('date','')}

    # 最优模型准确率
    best={}
    for t,m in models_result.items():
        if m and 'accuracy' in m:
            best[t]=m['accuracy'].get('ensemble',0)
    if best:
        top=max(best.items(),key=lambda x:x[1])
        ctx['ml_best']={'target':top[0],'acc':top[1]}

    if game=='3d':
        ctx['latest_digits']=latest.get('digits',[])
        ctx['markov']=[f"{m['pos_name']}上期{m['from']}→预测{m['top3'][0]['val']}({m['top3'][0]['prob']}%)"
                       for m in (markov_data or [])]
        ctx['overdue']=[f"{o['pos_name']}遗漏号:{o['overdue']}"
                        for o in (omission_data or []) if o.get('overdue')]
        sm=models_result.get('sum_group')
        ctx['sum_pred']=sm['prediction']['value'] if sm else None
        ctx['recommend_groups']=rec.get('groups',[])
    elif game=='ssq':
        ctx['latest_red']=latest.get('red',[]); ctx['latest_blue']=latest.get('blue',0)
        ctx['blue_overdue']=omission_data.get('blue_overdue',[]) if omission_data else []
        ctx['red_overdue']=omission_data.get('red_overdue',[])[:8] if omission_data else []
        ctx['blue_recommend']=rec.get('blue_recommend',[])
        ctx['sum_pred']=rec.get('sum_pred',''); ctx['odd_pred']=rec.get('odd_pred','')
    elif game=='kl8':
        ctx['latest_nums']=latest.get('numbers',[])[:10]
        ctx['zone_pred']=rec.get('zone_dominant_pred','')
        ctx['total_pred']=rec.get('total_range_pred','')
        ctx['overdue']=omission_data.get('overdue',[])[:10] if omission_data else []
        ctx['hot_nums']=rec.get('hot_nums',[])[:10]
    return ctx


# ══════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════

def main():
    print(f"\n{'='*55}")
    print(f"福彩ML预测系统 v3  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"RF={HAS_SKL} XGB={HAS_XGB} LGB={HAS_LGB}")
    print(f"{'='*55}")

    with open('history.json',encoding='utf-8') as f:
        history=json.load(f)

    # 加载上次预测（回测用）
    prev_pred={}
    try:
        with open('prediction.json',encoding='utf-8') as f:
            prev_pred=json.load(f).get('predictions',{})
    except Exception: pass

    print("\n── 昨日回测…")
    bt=daily_backtest(history,prev_pred)

    # 加载已有模型包（增量模式）
    saved_pkg={}
    if os.path.exists(MODEL_PKL):
        try:
            with open(MODEL_PKL,'rb') as f: saved_pkg=pickle.load(f)
            print(f"已加载模型包 {MODEL_PKL}")
        except Exception: saved_pkg={}

    game_configs=[
        ('3d',  feat_3d,  target_3d,  ['bai','shi','ge','sum_group','odd_cnt'],
         markov_3d, omission_3d, recommend_3d),
        ('ssq', feat_ssq, target_ssq, ['blue','odd_cnt','sum_group'],
         markov_ssq_blue, omission_ssq, recommend_ssq),
        ('kl8', feat_kl8, target_kl8, ['odd_group','zone_dom','total_group'],
         None, omission_kl8, recommend_kl8),
    ]

    predictions={}
    new_pkg={}

    for game,feat_fn,tgt_fn,tkeys,mk_fn,om_fn,rec_fn in game_configs:
        records=history.get(game,[])
        if len(records)<MIN_TRAIN+10:
            print(f"\n── {game}: 数据不足({len(records)}期)，跳过"); continue

        print(f"\n── {game}（全量{len(records)}期，窗口{WINDOW}期）…")

        try:
            X,y_dict,feat_names=build_dataset(records,feat_fn,tgt_fn)
            if X is None: print(f"  特征构建失败"); continue

            models_result={}
            new_pkg[game]={}
            for tname in tkeys:
                if tname not in y_dict: continue
                y=y_dict[tname]
                saved=saved_pkg.get(game,{}).get(tname)
                print(f"  [{tname}] n={len(y)} 训练…",end=' ')
                r=train_one_target(X,y,feat_names,tname,saved)
                if r:
                    new_pkg[game][tname]=r.pop('models_obj',{})
                    models_result[tname]=r
                    print(f"集成={r['accuracy'].get('ensemble','—')}% "
                          f"RF={r['accuracy'].get('rf','—')}% "
                          f"XGB={r['accuracy'].get('xgb','—')}% "
                          f"LGB={r['accuracy'].get('lgb','—')}%")
                else:
                    print("失败")

            mk  = mk_fn(records) if mk_fn else None
            om  = om_fn(records)
            rec = rec_fn(records,models_result,*([mk,om] if mk_fn else [om]))
            ctx = build_ai_context(game,records,models_result,mk,om,rec)

            predictions[game]={
                'game':game,'data_count':len(records),
                'updated_at':datetime.now().strftime('%Y-%m-%d %H:%M'),
                'models':     models_result,
                'recommendation': rec,
                'markov':     mk,
                'omission':   om,
                'ai_context': ctx,
            }
            print(f"  ✓ {game} 完成")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ✗ {game}: {e}")

    # 保存模型包
    try:
        with open(MODEL_PKL,'wb') as f: pickle.dump(new_pkg,f)
        print(f"\n✓ 模型包已保存 → {MODEL_PKL}")
    except Exception as e:
        print(f"模型包保存失败: {e}")

    output={
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models_used':{'rf':HAS_SKL,'xgb':HAS_XGB,'lgb':HAS_LGB,
                       'markov':True,'omission':True,'ensemble':True},
        'predictions': predictions,
        'backtest':    bt,
        'disclaimer':  '彩票开奖完全随机，本系统仅为统计模型演示，不具预测意义，请理性购彩。',
    }
    with open('prediction.json','w',encoding='utf-8') as f:
        json.dump(output,f,ensure_ascii=False,indent=2)
    print("✓ prediction.json 已写出")

    for game,res in predictions.items():
        n=res['data_count']
        accs={t:m['accuracy'].get('ensemble','—') for t,m in res.get('models',{}).items() if 'accuracy' in m}
        print(f"  {game}({n}期): {accs}")

if __name__=='__main__':
    main()
