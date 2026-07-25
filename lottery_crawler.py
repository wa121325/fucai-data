"""
福彩开奖数据爬虫 - 17500.cn 版
数据源：data.17500.cn 文本存档
输出：latest.json (严格保持指定格式)
"""
import requests
import json
import sys
from datetime import datetime

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# --- 17500.cn 数据源配置 ---
URL_MAP = {
    'ssq': 'http://data.17500.cn/ssq_asc.txt',
    '3d': 'http://data.17500.cn/3d_asc.txt',
    'qlc': 'http://data.17500.cn/7lc_asc.txt',
    'kl8': 'http://data.17500.cn/kl8_asc.txt'
}

def parse_17500_line(game, text):
    """解析 17500.cn 的文本格式并映射到指定的 JSON 结构"""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        raise ValueError("Empty content")
    
    parts = lines[-1].split()
    if len(parts) < 3:
        raise ValueError("Invalid format")

    qihao = parts[0]
    date_str = parts[-1]
    raw_nums = [int(x) for x in parts[1:-1] if x.isdigit()]

    if game == 'ssq':
        return {'qihao': qihao, 'date': date_str, 'red': raw_nums[:6], 'blue': raw_nums[6]}
    elif game == '3d':
        return {'qihao': qihao, 'date': date_str, 'digits': raw_nums[:3]}
    elif game == 'qlc':
        return {'qihao': qihao, 'date': date_str, 'numbers': raw_nums[:7], 'special': raw_nums[7]}
    elif game == 'kl8':
        return {'qihao': qihao, 'date': date_str, 'numbers': raw_nums[:20]}
    return None

def fetch_game(game, name):
    """从 17500.cn 抓取数据"""
    try:
        url = URL_MAP[game]
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        result = parse_17500_line(game, resp.text)
        print(f'  ✓ {name} (17500.cn) 期号:{result["qihao"]} 日期:{result["date"]}')
        return result
    except Exception as e:
        print(f'  ✗ {name} 失败: {e}', file=sys.stderr)
        return None

def main():
    print(f'开始抓取 {datetime.now().strftime("%Y-%m-%d")} 最新开奖数据')
    print(f'数据源: 乐彩网 (data.17500.cn)')

    # 严格按此顺序输出
    games = [('ssq','双色球'), ('3d','福彩3D'), ('qlc','七乐彩'), ('kl8','快乐8')]
    result = {'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ok = 0

    for game, name in games:
        data = fetch_game(game, name)
        if data:
            result[game] = data
            ok += 1

    with open('latest.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f'\n完成：{ok}/4 成功 → latest.json')
    if ok == 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
