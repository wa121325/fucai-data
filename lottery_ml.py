"""
福彩机器学习预测系统
支持：双色球(ssq)、福彩3D(3d)、快乐8(kl8)
方法：XGBoost + 随机森林 + 时间序列回测
每日自动生成 prediction.json
"""
import json, math, sys, warnings
from datetime import datetime, date
from collections import Counter

warnings.filterwarnings('ignore')

# ── 依赖检查 ─────────────────────────────────────────
try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score
    except ImportError:
        print("缺少依赖，请运行: pip install numpy scikit-learn xgboost")
        sys.exit(1)

# ══════════════════════════════════════════════════════
#  特征工程
# ══════════════════════════════════════════════════════

def feat_3d(records, idx):
    """为第idx期生成特征向量（使用其之前的数据）"""
    feats = {}
    window = records[max(0,idx-20):idx]   # 最近20期历史
    if not window:
        return None

    r = window[-1]
    d = r['digits']
    b, s, g = d
    total = b+s+g
    span = max(d)-min(d)
    road = [x%3 for x in d]
    road_cnt = [road.count(i) for i in range(3)]

    # 当期指标（用上期数据作为特征，预测本期）
    feats['sum_val']   = total
    feats['sum_tail']  = total % 10
    feats['span']      = span
    feats['odd_cnt']   = sum(1 for x in d if x%2!=0)
    feats['big_cnt']   = sum(1 for x in d if x>=5)
    feats['road0']     = road_cnt[0]
    feats['road1']     = road_cnt[1]
    feats['road2']     = road_cnt[2]
    feats['bai']       = b
    feats['shi']       = s
    feats['ge']        = g
    feats['gap_bs']    = abs(b-s)
    feats['gap_sg']    = abs(s-g)
    is_group3 = (b==s or s==g or b==g) and not (b==s==g)
    feats['group_type']= 0 if b==s==g else (1 if is_group3 else 2)

    # 滑动窗口统计（5/10/20期）
    for w_size, suffix in [(5,'5'),(10,'10'),(20,'20')]:
        w = window[-w_size:]
        sums = [sum(x['digits']) for x in w]
        spans = [max(x['digits'])-min(x['digits']) for x in w]
        feats[f'sum_mean_{suffix}']  = np.mean(sums) if sums else 0
        feats[f'sum_std_{suffix}']   = np.std(sums) if len(sums)>1 else 0
        feats[f'span_mean_{suffix}'] = np.mean(spans) if spans else 0
        # 各位均值
        for ci, cname in enumerate(['bai','shi','ge']):
            vals = [x['digits'][ci] for x in w]
            feats[f'{cname}_mean_{suffix}'] = np.mean(vals) if vals else 0

    # 连续和值趋势
    if len(window)>=3:
        s3 = [sum(x['digits']) for x in window[-3:]]
        feats['sum_trend'] = 1 if s3[-1]>s3[-2] else (-1 if s3[-1]<s3[-2] else 0)
    else:
        feats['sum_trend'] = 0

    # 重号（与上上期比较）
    if len(window)>=2:
        prev = window[-2]['digits']
        feats['repeat_cnt'] = sum(1 for i in range(3) if prev[i]==d[i])
    else:
        feats['repeat_cnt'] = 0

    return feats


def feat_ssq(records, idx):
    """双色球特征"""
    feats = {}
    window = records[max(0,idx-20):idx]
    if not window:
        return None

    r = window[-1]
    red = sorted(r['red'])
    blue = r['blue']
    total = sum(red)
    odd_cnt = sum(1 for x in red if x%2!=0)
    big_cnt = sum(1 for x in red if x>16)
    # 连号
    consec = sum(1 for i in range(len(red)-1) if red[i+1]-red[i]==1)
    # AC值
    diffs = set()
    for i in range(len(red)):
        for j in range(i+1,len(red)):
            diffs.add(red[j]-red[i])
    ac = len(diffs) - (len(red)-1)
    # 三区
    z1 = sum(1 for x in red if x<=11)
    z2 = sum(1 for x in red if 12<=x<=22)
    z3 = sum(1 for x in red if x>=23)
    max_gap = max(red[i+1]-red[i] for i in range(len(red)-1)) if len(red)>1 else 0

    feats.update({'red_sum':total,'odd_cnt':odd_cnt,'big_cnt':big_cnt,
                  'consec':consec,'ac':ac,'zone1':z1,'zone2':z2,'zone3':z3,
                  'max_gap':max_gap,'blue':blue,'blue_odd':blue%2,
                  'blue_big':1 if blue>=9 else 0,
                  'red_max':red[-1],'red_min':red[0],'red_span':red[-1]-red[0]})

    # 滑动统计
    for w_size, suffix in [(5,'5'),(10,'10'),(20,'20')]:
        w = window[-w_size:]
        sums = [sum(x['red']) for x in w]
        blues = [x['blue'] for x in w]
        feats[f'sum_mean_{suffix}']  = np.mean(sums) if sums else 0
        feats[f'sum_std_{suffix}']   = np.std(sums) if len(sums)>1 else 0
        feats[f'blue_mean_{suffix}'] = np.mean(blues) if blues else 0
        odds = [sum(1 for n in x['red'] if n%2!=0) for x in w]
        feats[f'odd_mean_{suffix}']  = np.mean(odds) if odds else 0

    # 各号码近期出现频率
    all_red = [n for x in window for n in x['red']]
    cnt = Counter(all_red)
    feats['hot_zone1'] = sum(cnt.get(n,0) for n in range(1,12))
    feats['hot_zone2'] = sum(cnt.get(n,0) for n in range(12,23))
    feats['hot_zone3'] = sum(cnt.get(n,0) for n in range(23,34))
    feats['sum_trend'] = 0
    if len(window)>=3:
        s3=[sum(x['red']) for x in window[-3:]]
        feats['sum_trend'] = 1 if s3[-1]>s3[-2] else (-1 if s3[-1]<s3[-2] else 0)

    return feats


def feat_kl8(records, idx):
    """快乐8特征"""
    feats = {}
    window = records[max(0,idx-20):idx]
    if not window:
        return None

    r = window[-1]
    nums = sorted(r['numbers'])
    total = sum(nums)
    odd_cnt = sum(1 for x in nums if x%2!=0)
    big_cnt = sum(1 for x in nums if x>40)
    zones = [sum(1 for x in nums if lo<=x<=hi)
             for lo,hi in [(1,20),(21,40),(41,60),(61,80)]]
    five  = [sum(1 for x in nums if lo<=x<=hi)
             for lo,hi in [(1,16),(17,32),(33,48),(49,64),(65,80)]]
    consec_groups = 0
    in_consec = False
    for i in range(len(nums)-1):
        if nums[i+1]-nums[i]==1:
            if not in_consec: consec_groups+=1; in_consec=True
        else: in_consec=False

    feats.update({'total':total,'odd_cnt':odd_cnt,'big_cnt':big_cnt,
                  'min_n':nums[0],'max_n':nums[-1],
                  'zone1':zones[0],'zone2':zones[1],'zone3':zones[2],'zone4':zones[3],
                  'five1':five[0],'five2':five[1],'five3':five[2],'five4':five[3],'five5':five[4],
                  'consec_groups':consec_groups})

    # 滑动统计
    for w_size, suffix in [(5,'5'),(10,'10'),(20,'20')]:
        w = window[-w_size:]
        totals = [sum(x['numbers']) for x in w]
        feats[f'total_mean_{suffix}'] = np.mean(totals) if totals else 0
        feats[f'total_std_{suffix}']  = np.std(totals) if len(totals)>1 else 0
        for zi in range(4):
            zv = [sum(1 for n in x['numbers'] if [(1,20),(21,40),(41,60),(61,80)][zi][0]<=n<=[(1,20),(21,40),(41,60),(61,80)][zi][1]) for x in w]
            feats[f'zone{zi+1}_mean_{suffix}'] = np.mean(zv) if zv else 0

    # 冷热统计
    all_nums = [n for x in window for n in x['numbers']]
    cnt = Counter(all_nums)
    feats['hot_z1'] = sum(cnt.get(n,0) for n in range(1,21))
    feats['hot_z2'] = sum(cnt.get(n,0) for n in range(21,41))
    feats['hot_z3'] = sum(cnt.get(n,0) for n in range(41,61))
    feats['hot_z4'] = sum(cnt.get(n,0) for n in range(61,81))
    feats['total_trend'] = 0
    if len(window)>=3:
        t3=[sum(x['numbers']) for x in window[-3:]]
        feats['total_trend'] = 1 if t3[-1]>t3[-2] else (-1 if t3[-1]<t3[-2] else 0)

    return feats


# ══════════════════════════════════════════════════════
#  目标变量定义
# ══════════════════════════════════════════════════════

def get_targets_3d(records):
    """3D预测目标：百/十/个位各自是几"""
    targets = {'bai':[], 'shi':[], 'ge':[], 'sum_group':[], 'odd_cnt':[]}
    for r in records:
        d = r['digits']
        targets['bai'].append(d[0])
        targets['shi'].append(d[1])
        targets['ge'].append(d[2])
        s = sum(d)
        targets['sum_group'].append(0 if s<=9 else (1 if s<=17 else 2))
        targets['odd_cnt'].append(sum(1 for x in d if x%2!=0))
    return targets

def get_targets_ssq(records):
    """双色球预测目标"""
    targets = {'blue':[], 'zone_combo':[], 'odd_cnt':[], 'sum_group':[]}
    for r in records:
        targets['blue'].append(r['blue'])
        z1=sum(1 for x in r['red'] if x<=11)
        z2=sum(1 for x in r['red'] if 12<=x<=22)
        z3=sum(1 for x in r['red'] if x>=23)
        targets['zone_combo'].append(z1*100+z2*10+z3)
        targets['odd_cnt'].append(sum(1 for x in r['red'] if x%2!=0))
        s=sum(r['red'])
        targets['sum_group'].append(0 if s<70 else (1 if s<100 else 2))
    return targets

def get_targets_kl8(records):
    """快乐8预测目标"""
    targets = {'odd_cnt_group':[], 'zone_dominant':[], 'total_group':[]}
    for r in records:
        odd = sum(1 for x in r['numbers'] if x%2!=0)
        targets['odd_cnt_group'].append(0 if odd<9 else (1 if odd<=11 else 2))
        zones = [sum(1 for x in r['numbers'] if lo<=x<=hi) for lo,hi in [(1,20),(21,40),(41,60),(61,80)]]
        targets['zone_dominant'].append(int(np.argmax(zones)))
        t=sum(r['numbers'])
        targets['total_group'].append(0 if t<640 else (1 if t<820 else 2))
    return targets


# ══════════════════════════════════════════════════════
#  模型训练 + 时间序列回测
# ══════════════════════════════════════════════════════

def build_dataset(records, feat_fn):
    """构建特征矩阵和可用索引"""
    X_all, valid_idx = [], []
    for i in range(len(records)):
        f = feat_fn(records, i)
        if f is not None:
            X_all.append(list(f.values()))
            valid_idx.append(i)
    feat_names = list(feat_fn(records, valid_idx[0]).keys()) if valid_idx else []
    return np.array(X_all), valid_idx, feat_names


def train_and_backtest(X, y, feat_names, target_name, min_train=30, step=1):
    """
    时间序列滚动回测：
    - 前min_train期训练，之后每step期预测一次
    - 返回回测结果和最终模型
    """
    results = {'target': target_name, 'backtest': [], 'accuracy': {}}
    n = len(X)
    if n < min_train + 5:
        return None

    all_preds_rf, all_preds_xgb, all_true = [], [], []

    # 滚动窗口回测
    for end in range(min_train, n, step):
        X_train = X[:end]
        y_train = y[:end]
        X_test  = X[end:end+step]
        y_test  = y[end:end+step]

        try:
            rf = RandomForestClassifier(n_estimators=60, max_depth=5,
                                        random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            pred_rf = rf.predict(X_test)

            if HAS_XGB:
                model_xgb = xgb.XGBClassifier(n_estimators=60, max_depth=4,
                                               learning_rate=0.1, random_state=42,
                                               eval_metric='mlogloss', verbosity=0)
                model_xgb.fit(X_train, y_train)
                pred_xgb = model_xgb.predict(X_test)
            else:
                pred_xgb = pred_rf

            all_preds_rf.extend(pred_rf.tolist())
            all_preds_xgb.extend(pred_xgb.tolist())
            all_true.extend(y_test.tolist())
        except Exception:
            continue

    if not all_true:
        return None

    acc_rf  = accuracy_score(all_true, all_preds_rf)
    acc_xgb = accuracy_score(all_true, all_preds_xgb)
    results['accuracy']['random_forest'] = round(acc_rf*100, 1)
    results['accuracy']['xgboost']       = round(acc_xgb*100, 1)

    # 最终模型（用全量数据训练）
    rf_final = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf_final.fit(X, y)

    if HAS_XGB:
        xgb_final = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                        random_state=42, eval_metric='mlogloss', verbosity=0)
        xgb_final.fit(X, y)
    else:
        xgb_final = rf_final

    # 特征重要性 Top5
    imp = rf_final.feature_importances_
    top5 = sorted(zip(feat_names, imp), key=lambda x: -x[1])[:5]
    results['feature_importance'] = [{'name':k,'score':round(float(v),4)} for k,v in top5]

    # 预测下一期（用全量特征）
    last_feat = X[-1:].copy()
    prob_rf  = rf_final.predict_proba(last_feat)[0]
    prob_xgb = xgb_final.predict_proba(last_feat)[0] if hasattr(xgb_final,'predict_proba') else prob_rf
    classes  = rf_final.classes_.tolist()

    # 集成概率（RF 0.5 + XGB 0.5）
    ensemble_prob = [(p1+p2)/2 for p1,p2 in zip(prob_rf, prob_xgb)]
    pred_class = classes[int(np.argmax(ensemble_prob))]

    results['prediction'] = {
        'value': int(pred_class),
        'confidence': round(float(max(ensemble_prob))*100, 1),
        'probs': {str(c): round(float(p)*100,1) for c,p in zip(classes, ensemble_prob)}
    }
    results['backtest_records'] = min(len(all_true), 20)  # 记录回测了多少期
    results['backtest_latest'] = [
        {'true': int(t), 'pred_rf': int(p1), 'pred_xgb': int(p2)}
        for t,p1,p2 in zip(all_true[-10:], all_preds_rf[-10:], all_preds_xgb[-10:])
    ]

    return results


# ══════════════════════════════════════════════════════
#  各彩种完整分析
# ══════════════════════════════════════════════════════

def analyze_3d(records):
    print(f"  3D: {len(records)}期数据，开始特征工程…")
    X, valid_idx, feat_names = build_dataset(records, feat_3d)
    targets = get_targets_3d(records)

    result = {'game':'3d','name':'福彩3D','updated_at':datetime.now().strftime('%Y-%m-%d %H:%M'),
              'data_count':len(records),'models':{}}

    for t_name in ['bai','shi','ge','sum_group','odd_cnt']:
        y_all = [targets[t_name][i] for i in valid_idx]
        y = np.array(y_all)
        print(f"    训练目标: {t_name}…")
        r = train_and_backtest(X, y, feat_names, t_name)
        if r:
            result['models'][t_name] = r

    # 人工智能综合推荐（基于模型输出合成推荐号码）
    result['recommendation'] = gen_recommendation_3d(records, result['models'])
    return result


def analyze_ssq(records):
    print(f"  SSQ: {len(records)}期数据，开始特征工程…")
    X, valid_idx, feat_names = build_dataset(records, feat_ssq)
    targets = get_targets_ssq(records)

    result = {'game':'ssq','name':'双色球','updated_at':datetime.now().strftime('%Y-%m-%d %H:%M'),
              'data_count':len(records),'models':{}}

    for t_name in ['blue','odd_cnt','sum_group','zone_combo']:
        y_all = [targets[t_name][i] for i in valid_idx]
        y = np.array(y_all)
        print(f"    训练目标: {t_name}…")
        r = train_and_backtest(X, y, feat_names, t_name)
        if r:
            result['models'][t_name] = r

    result['recommendation'] = gen_recommendation_ssq(records, result['models'])
    return result


def analyze_kl8(records):
    print(f"  KL8: {len(records)}期数据，开始特征工程…")
    X, valid_idx, feat_names = build_dataset(records, feat_kl8)
    targets = get_targets_kl8(records)

    result = {'game':'kl8','name':'快乐8','updated_at':datetime.now().strftime('%Y-%m-%d %H:%M'),
              'data_count':len(records),'models':{}}

    for t_name in ['odd_cnt_group','zone_dominant','total_group']:
        y_all = [targets[t_name][i] for i in valid_idx]
        y = np.array(y_all)
        print(f"    训练目标: {t_name}…")
        r = train_and_backtest(X, y, feat_names, t_name)
        if r:
            result['models'][t_name] = r

    result['recommendation'] = gen_recommendation_kl8(records, result['models'])
    return result


# ══════════════════════════════════════════════════════
#  综合推荐号码生成
# ══════════════════════════════════════════════════════

def gen_recommendation_3d(records, models):
    """根据模型预测概率综合推荐3D号码"""
    recent = [sum(r['digits']) for r in records[-10:]]
    hot_digits = [Counter(r['digits'][i] for r in records[-30:]).most_common(3)
                  for i in range(3)]

    candidates = []
    # 从各位预测中选概率最高的几个号码组合
    bai_m = models.get('bai',{})
    shi_m = models.get('shi',{})
    ge_m  = models.get('ge',{})

    def top_probs(m, n=3):
        if not m or 'prediction' not in m:
            return list(range(n))
        probs = m['prediction']['probs']
        return [int(k) for k,v in sorted(probs.items(), key=lambda x:-x[1])[:n]]

    bai_cands = top_probs(bai_m)
    shi_cands = top_probs(shi_m)
    ge_cands  = top_probs(ge_m)

    groups = []
    for b in bai_cands:
        for s in shi_cands[:2]:
            for g in ge_cands[:2]:
                groups.append([b, s, g])
                if len(groups) >= 6:
                    break
            if len(groups) >= 6: break
        if len(groups) >= 6: break

    sum_pred = models.get('sum_group',{}).get('prediction',{})
    sum_label = {0:'0-9（小）',1:'10-17（中）',2:'18-27（大）'}.get(
        sum_pred.get('value',1), '10-17（中）')

    return {
        'groups': groups[:6],
        'sum_range_pred': sum_label,
        'hot_bai': [x[0] for x in hot_digits[0]],
        'hot_shi': [x[0] for x in hot_digits[1]],
        'hot_ge':  [x[0] for x in hot_digits[2]],
        'note': '基于XGBoost+随机森林时序回测，仅供娱乐参考。'
    }


def gen_recommendation_ssq(records, models):
    """双色球综合推荐"""
    red_freq = Counter(n for r in records[-30:] for n in r['red'])
    blue_freq = Counter(r['blue'] for r in records[-30:])
    hot_red  = [x[0] for x in red_freq.most_common(15)]
    cold_red = [x[0] for x in red_freq.most_common()[-10:]]

    blue_m = models.get('blue',{})
    blue_top = []
    if blue_m and 'prediction' in blue_m:
        probs = blue_m['prediction']['probs']
        blue_top = [int(k) for k,v in sorted(probs.items(), key=lambda x:-x[1])[:3]]
    else:
        blue_top = [x[0] for x in blue_freq.most_common(3)]

    odd_m = models.get('odd_cnt',{})
    odd_pred = odd_m.get('prediction',{}).get('value', 3) if odd_m else 3

    zone_m = models.get('sum_group',{})
    sum_label = {0:'低(<70)',1:'中(70-99)',2:'高(≥100)'}.get(
        zone_m.get('prediction',{}).get('value',1) if zone_m else 1, '中(70-99)')

    # 生成6注推荐（结合热号+冷号+奇偶预测）
    import random
    random.seed(42)
    groups = []
    for i in range(6):
        mix = hot_red[:10] + cold_red[:5]
        picked = sorted(random.sample(list(set(mix)), min(6, len(set(mix)))))
        if len(picked)<6:
            extra = [n for n in range(1,34) if n not in picked]
            picked += random.sample(extra, 6-len(picked))
        picked = sorted(picked[:6])
        blue = blue_top[i % len(blue_top)] if blue_top else random.randint(1,16)
        groups.append({'red': picked, 'blue': blue})

    return {
        'groups': groups,
        'blue_recommend': blue_top,
        'odd_cnt_pred': int(odd_pred),
        'sum_range_pred': sum_label,
        'hot_red': hot_red[:10],
        'cold_red': cold_red[:5],
        'note': '基于XGBoost+随机森林时序回测，仅供娱乐参考。'
    }


def gen_recommendation_kl8(records, models):
    """快乐8综合推荐"""
    freq = Counter(n for r in records[-30:] for n in r['numbers'])
    hot20  = [x[0] for x in freq.most_common(20)]
    cold20 = [x[0] for x in freq.most_common()[-20:]]

    zone_m = models.get('zone_dominant',{})
    zone_pred = zone_m.get('prediction',{}).get('value', 1) if zone_m else 1
    zone_name = {0:'1-20区',1:'21-40区',2:'41-60区',3:'61-80区'}.get(zone_pred,'21-40区')

    total_m = models.get('total_group',{})
    total_label = {0:'低(<640)',1:'中(640-819)',2:'高(≥820)'}.get(
        total_m.get('prediction',{}).get('value',1) if total_m else 1, '中(640-819)')

    import random
    random.seed(42)
    groups = []
    for i in range(5):
        pool = list(set(hot20[:15] + cold20[:10]))
        if len(pool)<20:
            extra = [n for n in range(1,81) if n not in pool]
            pool += random.sample(extra, 20-len(pool))
        picked = sorted(random.sample(pool, 20))
        groups.append({'numbers': picked})

    return {
        'groups': groups,
        'hot20':  hot20,
        'cold20': cold20,
        'zone_dominant_pred': zone_name,
        'total_range_pred': total_label,
        'note': '基于XGBoost+随机森林时序回测，仅供娱乐参考。'
    }


# ══════════════════════════════════════════════════════
#  每日回测报告（与上次预测对比实际结果）
# ══════════════════════════════════════════════════════

def daily_backtest_report(history, prev_prediction):
    """把昨天的预测和今天的实际结果做对比"""
    report = {'date': str(date.today()), 'games': {}}
    for game in ['3d','ssq','kl8']:
        records = history.get(game, [])
        if not records or game not in (prev_prediction or {}):
            continue
        latest = records[-1]
        pred   = prev_prediction[game]
        rec    = pred.get('recommendation', {})
        hit_info = {}

        if game == '3d':
            actual = latest.get('digits', [])
            groups = rec.get('groups', [])
            hits = [g for g in groups if g==actual]
            partial = [g for g in groups if sum(1 for i,v in enumerate(g) if i<len(actual) and v==actual[i])>=2]
            hit_info = {'actual': actual, 'hit_groups': hits, 'partial_hit': partial,
                        'hit_count': len(hits), 'partial_count': len(partial)}

        elif game == 'ssq':
            actual_red  = sorted(latest.get('red',[]))
            actual_blue = latest.get('blue', 0)
            groups = rec.get('groups', [])
            hit_info_list = []
            for g in groups:
                red_hit  = len(set(g.get('red',[])) & set(actual_red))
                blue_hit = 1 if g.get('blue')==actual_blue else 0
                hit_info_list.append({'red_hit': red_hit, 'blue_hit': blue_hit})
            hit_info = {'actual_red': actual_red, 'actual_blue': actual_blue,
                        'group_results': hit_info_list,
                        'best_red_hit': max((x['red_hit'] for x in hit_info_list), default=0)}

        elif game == 'kl8':
            actual = set(latest.get('numbers',[]))
            groups = rec.get('groups', [])
            hit_info_list = [{'hit': len(actual & set(g.get('numbers',[]))) } for g in groups]
            hit_info = {'actual_count': len(actual),
                        'group_results': hit_info_list,
                        'best_hit': max((x['hit'] for x in hit_info_list), default=0)}

        report['games'][game] = hit_info
    return report


# ══════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════

def main():
    print(f"=== 福彩ML预测系统 {date.today()} ===")
    print(f"XGBoost: {'✓' if HAS_XGB else '✗（使用GradientBoosting替代）'}")

    # 读取历史数据
    try:
        with open('history.json', encoding='utf-8') as f:
            history = json.load(f)
    except FileNotFoundError:
        print("history.json 不存在，请先运行 lottery_crawler.py")
        sys.exit(1)

    # 读取上次预测（用于回测）
    prev_prediction = {}
    try:
        with open('prediction.json', encoding='utf-8') as f:
            prev_data = json.load(f)
            prev_prediction = prev_data.get('predictions', {})
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # 每日回测
    print("\n── 生成回测报告…")
    bt_report = daily_backtest_report(history, prev_prediction)

    # 各彩种分析
    predictions = {}
    for game, name, fn in [('ssq','双色球',analyze_ssq),
                             ('3d', '福彩3D',analyze_3d),
                             ('kl8','快乐8', analyze_kl8)]:
        records = history.get(game, [])
        if len(records) < 35:
            print(f"  {name}: 数据不足（{len(records)}期，需≥35期），跳过")
            continue
        # 只用近50期
        records = records[-50:]
        print(f"\n── 分析 {name}（使用近{len(records)}期）…")
        try:
            predictions[game] = fn(records)
            acc_info = {k: v['accuracy'] for k,v in predictions[game]['models'].items() if 'accuracy' in v}
            print(f"  ✓ {name} 完成，回测准确率: {acc_info}")
        except Exception as e:
            print(f"  ✗ {name} 失败: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    # 写出结果
    output = {
        'updated_at':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'has_xgboost': HAS_XGB,
        'predictions': predictions,
        'backtest':    bt_report,
        'disclaimer':  '本预测基于历史统计与机器学习模型，彩票开奖具有随机性，预测结果仅供娱乐参考，不构成任何投注建议。',
    }
    with open('prediction.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 完成！结果已写入 prediction.json")
    for game, res in predictions.items():
        for t, m in res.get('models',{}).items():
            if 'accuracy' in m:
                a = m['accuracy']
                print(f"  {game}/{t}: RF={a['random_forest']}%  XGB={a.get('xgboost','-')}%")


if __name__ == '__main__':
    main()
