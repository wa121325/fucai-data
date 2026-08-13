"""
福彩开奖数据爬虫
数据源：data.17500.cn
输出：latest.json（最新一期）、history.json（全量历史，ML训练用）

运行模式：
  python lottery_crawler.py          # 日常模式：只抓最新，追加到history
  python lottery_crawler.py --full   # 全量模式：抓全部历史（首次运行时用）
"""
import requests, json, re, sys
from datetime import datetime

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

URL_MAP = {
    'ssq': 'http://data.17500.cn/ssq_asc.txt',
    '3d':  'http://data.17500.cn/3d_asc.txt',
    'qlc': 'http://data.17500.cn/7lc_asc.txt',
    'kl8': 'http://data.17500.cn/kl8_asc.txt',
}

# ── 解析单行 ─────────────────────────────────────────
def parse_line(game, line):
    parts = line.split()
    if len(parts) < 4:
        return None
    qihao = parts[0]
    if not re.match(r'^\d{6,}$', qihao):
        return None
    m = re.search(r'(\d{4}-\d{2}-\d{2})', line)
    if not m:
        return None
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

# ── 解析全文本 ───────────────────────────────────────
def parse_all(game, text):
    """解析txt所有行，返回去重的全量记录列表（旧→新）"""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    seen, records = set(), []
    for line in lines:
        rec = parse_line(game, line)
        if rec and rec['qihao'] not in seen:
            seen.add(rec['qihao'])
            records.append(rec)
    return records  # 已升序（17500是升序文件）

# ── 拉取文本 ─────────────────────────────────────────
def fetch_text(game, name):
    try:
        r = requests.get(URL_MAP[game], headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f'  ✗ {name} 请求失败: {e}', file=sys.stderr)
        return None

# ── 主流程 ───────────────────────────────────────────
def main():
    full_mode = '--full' in sys.argv
    mode_label = '全量历史' if full_mode else '日常增量'
    print(f'福彩开奖数据爬虫 [{mode_label}模式] {datetime.now().strftime("%Y-%m-%d %H:%M")}')

    # 读取已有history（日常模式用于追加）
    existing_history = {}
    if not full_mode:
        try:
            with open('history.json', encoding='utf-8') as f:
                existing_history = json.load(f)
            print('已读取现有 history.json')
        except Exception:
            print('未找到 history.json，将创建新文件')

    games = [('ssq','双色球'), ('3d','福彩3D'), ('qlc','七乐彩'), ('kl8','快乐8')]
    latest_out  = {'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    history_out = {'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ok = 0

    for game, name in games:
        print(f'\n── {name} ──')
        text = fetch_text(game, name)
        if not text:
            continue

        all_records = parse_all(game, text)
        if not all_records:
            print(f'  ✗ {name} 解析失败')
            continue

        latest = all_records[-1]
        print(f'  最新: 期号={latest["qihao"]} 日期={latest["date"]}')
        print(f'  全量记录数: {len(all_records)}期')

        if full_mode:
            # 全量模式：直接用所有记录
            history_records = all_records
        else:
            # 日常模式：合并已有记录 + 新记录，去重
            old = existing_history.get(game, [])
            old_qihaos = {r['qihao'] for r in old}
            new_ones = [r for r in all_records if r['qihao'] not in old_qihaos]
            history_records = old + new_ones
            # 按期号排序
            history_records.sort(key=lambda x: x['qihao'])
            if new_ones:
                print(f'  新增 {len(new_ones)} 期，累计 {len(history_records)} 期')
            else:
                print(f'  无新数据，保持 {len(history_records)} 期')

        latest_out[game]  = latest
        history_out[game] = history_records
        ok += 1
        print(f'  ✓ {name} 完成，history共 {len(history_records)} 期')

    # 写出文件
    with open('latest.json', 'w', encoding='utf-8') as f:
        json.dump(latest_out, f, ensure_ascii=False, indent=2)

    with open('history.json', 'w', encoding='utf-8') as f:
        json.dump(history_out, f, ensure_ascii=False)  # 不缩进，节省体积

    # 统计
    print(f'\n完成：{ok}/4 成功')
    for game, _ in games:
        recs = history_out.get(game, [])
        if recs:
            print(f'  {game}: {len(recs)}期 ({recs[0]["date"]} ~ {recs[-1]["date"]})')

    if ok == 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
