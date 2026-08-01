"""
福彩开奖数据爬虫 - 17500.cn 版
数据源：data.17500.cn 文本存档
输出：latest.json + history.json（追加历史，保留近100期用于走势图）
"""
import requests
import json
import re
import sys
from datetime import datetime

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

URL_MAP = {
    'ssq': 'http://data.17500.cn/ssq_asc.txt',
    '3d':  'http://data.17500.cn/3d_asc.txt',
    'qlc': 'http://data.17500.cn/7lc_asc.txt',
    'kl8': 'http://data.17500.cn/kl8_asc.txt'
}

HISTORY_KEEP = 100   # 每种彩票保留最近多少期

def parse_17500_line(game, text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        raise ValueError("Empty content")

    line = lines[-1]
    print(f'  [原始最后一行] {line}')
    parts = line.split()
    qihao = parts[0]

    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
    if not date_match:
        raise ValueError(f"未找到日期: {line}")
    date_str = date_match.group(1)

    raw_nums = []
    for p in parts[1:]:
        if re.match(r'^\d+$', p) and p != qihao and len(p) <= 3:
            raw_nums.append(int(p))

    print(f'  [解析] 期号={qihao} 日期={date_str} 号码={raw_nums}')

    if game == 'ssq':
        if len(raw_nums) < 7:
            raise ValueError(f"SSQ号码不足7个: {raw_nums}")
        return {'qihao': qihao, 'date': date_str,
                'red': sorted(raw_nums[:6]), 'blue': raw_nums[6]}
    elif game == '3d':
        if len(raw_nums) < 3:
            raise ValueError(f"3D号码不足3个: {raw_nums}")
        return {'qihao': qihao, 'date': date_str, 'digits': raw_nums[:3]}
    elif game == 'qlc':
        if len(raw_nums) < 8:
            raise ValueError(f"QLC号码不足8个: {raw_nums}")
        return {'qihao': qihao, 'date': date_str,
                'numbers': sorted(raw_nums[:7]), 'special': raw_nums[7]}
    elif game == 'kl8':
        if len(raw_nums) < 20:
            raise ValueError(f"KL8号码不足20个: {raw_nums}")
        return {'qihao': qihao, 'date': date_str,
                'numbers': sorted(raw_nums[:20])}

# ── 历史数据（多期）：读取文件里最后N行 ─────────────────────
def parse_17500_history(game, text, n=HISTORY_KEEP):
    """解析txt里最后n期，返回列表（旧→新顺序）"""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    records = []
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        qihao = parts[0]
        if not re.match(r'^\d{6,}$', qihao):
            continue
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
        if not date_match:
            continue
        date_str = date_match.group(1)
        raw_nums = [int(p) for p in parts[1:]
                    if re.match(r'^\d+$', p) and p != qihao and len(p) <= 3]
        try:
            if game == 'ssq' and len(raw_nums) >= 7:
                records.append({'qihao': qihao, 'date': date_str,
                                'red': sorted(raw_nums[:6]), 'blue': raw_nums[6]})
            elif game == '3d' and len(raw_nums) >= 3:
                records.append({'qihao': qihao, 'date': date_str,
                                'digits': raw_nums[:3]})
            elif game == 'qlc' and len(raw_nums) >= 8:
                records.append({'qihao': qihao, 'date': date_str,
                                'numbers': sorted(raw_nums[:7]), 'special': raw_nums[7]})
            elif game == 'kl8' and len(raw_nums) >= 20:
                records.append({'qihao': qihao, 'date': date_str,
                                'numbers': sorted(raw_nums[:20])})
        except Exception:
            continue
    return records[-n:]   # 取最近n期

def fetch_game(game, name):
    try:
        resp = requests.get(URL_MAP[game], headers=HEADERS, timeout=15)
        resp.raise_for_status()
        latest = parse_17500_line(game, resp.text)
        history = parse_17500_history(game, resp.text)
        print(f'  ✓ {name} 期号:{latest["qihao"]} 日期:{latest["date"]} 历史:{len(history)}期')
        return latest, history
    except Exception as e:
        print(f'  ✗ {name} 失败: {e}', file=sys.stderr)
        return None, None

def main():
    print(f'开始抓取 {datetime.now().strftime("%Y-%m-%d")} 最新开奖数据')
    games = [('ssq','双色球'), ('3d','福彩3D'), ('qlc','七乐彩'), ('kl8','快乐8')]

    latest_out  = {'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    history_out = {'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ok = 0

    for game, name in games:
        print(f'\n── {name} ──')
        latest, history = fetch_game(game, name)
        if latest:
            latest_out[game]  = latest
            history_out[game] = history
            ok += 1

    with open('latest.json', 'w', encoding='utf-8') as f:
        json.dump(latest_out, f, ensure_ascii=False, indent=2)

    with open('history.json', 'w', encoding='utf-8') as f:
        json.dump(history_out, f, ensure_ascii=False)   # 不缩进，节省体积

    print(f'\n完成：{ok}/4 成功 → latest.json + history.json')
    if ok == 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
