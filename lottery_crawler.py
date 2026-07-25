import requests
import json
from datetime import datetime
import sys

# --- Configuration ---
# Reordered as requested: SSQ, 3D, QLC, KL8
URL_MAP = {
    'ssq': 'http://data.17500.cn/ssq_asc.txt',
    '3d': 'http://data.17500.cn/3d_asc.txt',
    'qlc': 'http://data.17500.cn/7lc_asc.txt',
    'kl8': 'http://data.17500.cn/kl8_asc.txt'
}

NAMES = {
    'ssq': '双色球', '3d': '福彩3D', 'qlc': '七乐彩', 'kl8': '快乐8'
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def parse_latest_line(txt, key):
    """Parses the 17500.cn format: [Qihao] [Numbers...] [Date]"""
    lines = [l.strip() for l in txt.strip().splitlines() if l.strip()]
    if not lines:
        return None

    last_line = lines[-1]
    parts = last_line.split()

    if len(parts) < 3:
        return None

    qihao = parts[0]
    date_str = parts[-1]
    raw_nums = parts[1:-1]

    return {
        'qihao': qihao,
        'date': date_str,
        'numbers': [int(x) if x.isdigit() else x for x in raw_nums]
    }

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting data fetch for requested games from 17500.cn...")
    final_data = {'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    count = 0

    # Dictionary iteration follows the order of keys defined in URL_MAP (Python 3.7+)
    for key, url in URL_MAP.items():
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                data = parse_latest_line(resp.text, key)
                if data:
                    final_data[key] = data
                    print(f"  ✓ {NAMES[key]}: 期号 {data['qihao']} ({data['date']})")
                    count += 1
                else:
                    print(f"  ! {NAMES[key]}: Parsing failed.")
            else:
                print(f"  ✗ {NAMES[key]}: HTTP Error {resp.status_code} at {url}")
        except Exception as e:
            print(f"  ✗ {NAMES[key]}: Error - {e}")

    with open('latest.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"\nSuccess: {count}/{len(URL_MAP)} updated to latest.json.")

if __name__ == '__main__':
    main()
