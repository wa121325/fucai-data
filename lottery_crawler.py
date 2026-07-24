"""
福彩开奖数据爬虫 - GitHub Actions 版
数据来源：datachart.500.com（与用户提供的新版LotteryDataCrawler.py同源）
抓取：双色球(ssq)、福彩3D(3d)、七乐彩(qlc)、快乐8(kl8) 最新一期
输出：latest.json
"""
import requests
from bs4 import BeautifulSoup
import json, re, sys
from datetime import datetime, date

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}

# 500网各彩种 URL 路径（与新版 LotteryDataCrawler.py 一致）
URL_MAP = {
    'ssq': 'https://datachart.500.com/ssq/history/newinc/history.php',
    '3d':  'https://datachart.500.com/sd/history/inc/history.php',
    'qlc': 'https://datachart.500.com/qlc/history/newinc/history.php',
    'kl8': 'https://datachart.500.com/kl8/history/newinc/history.php',
}

def get_html(url, params=None, timeout=20):
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        r = session.get(url, params=params, timeout=timeout)
        r.encoding = 'gb2312'
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  请求失败 {url}: {e}", file=sys.stderr)
        return None

def find_table(soup):
    """智能找表格：优先 id=tablelist，其次 class=chartTable，最后第一个table"""
    return (soup.find('table', {'id': 'tablelist'})
            or soup.find('table', class_='chartTable')
            or soup.find('table'))

def find_data_rows(table):
    """自动跳过表头行，返回纯数据行列表"""
    rows = table.find_all('tr')
    start = 0
    for i, row in enumerate(rows):
        txt = row.get_text()
        if '期号' in txt or '开奖日' in txt:
            start = i + 1
    data_rows = []
    for row in rows[start:]:
        cols = row.find_all('td')
        if cols:
            data_rows.append([td.get_text(strip=True) for td in cols])
    return data_rows

def get_cell_text(cols, idx):
    return cols[idx] if idx < len(cols) else ''

# ── 双色球 ──────────────────────────────────────────────
def fetch_ssq():
    year = datetime.now().year
    html = get_html(URL_MAP['ssq'], params={'start': f'{year}001', 'end': f'{year}999'})
    if not html:
        return None
    soup = BeautifulSoup(html, 'html.parser')
    table = find_table(soup)
    if not table:
        return None
    rows = find_data_rows(table)
    if not rows:
        return None
    # 最新一期在末尾
    for cols in reversed(rows):
        try:
            qihao = cols[0]
            if not qihao.isdigit():
                continue
            red = [int(cols[i]) for i in range(1, 7)]
            blue = int(cols[7])
            date_str = cols[-1]
            if not re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                continue
            return {'qihao': qihao, 'date': date_str, 'red': red, 'blue': blue}
        except Exception:
            continue
    return None

# ── 福彩3D ──────────────────────────────────────────────
def fetch_3d():
    year = datetime.now().year
    html = get_html(URL_MAP['3d'], params={'start': f'{year}001', 'end': f'{year}999'})
    if not html:
        return None
    soup = BeautifulSoup(html, 'html.parser')
    table = find_table(soup)
    if not table:
        return None
    rows = find_data_rows(table)
    if not rows:
        return None
    for cols in reversed(rows):
        try:
            qihao = cols[0]
            if not qihao.isdigit():
                continue
            # 号码在第2列（索引1），3位数字
            nums_raw = cols[1].replace(' ', '')
            if not re.match(r'^\d{3}$', nums_raw):
                continue
            date_str = cols[-1]
            if not re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                continue
            return {'qihao': qihao, 'date': date_str, 'digits': [int(c) for c in nums_raw]}
        except Exception:
            continue
    return None

# ── 七乐彩 ──────────────────────────────────────────────
def fetch_qlc():
    year = datetime.now().year
    html = get_html(URL_MAP['qlc'], params={'start': f'{year}001', 'end': f'{year}999'})
    if not html:
        return None
    soup = BeautifulSoup(html, 'html.parser')
    table = find_table(soup)
    if not table:
        return None
    rows = find_data_rows(table)
    if not rows:
        return None
    for cols in reversed(rows):
        try:
            qihao = cols[0]
            if not qihao.isdigit():
                continue
            # 七乐彩：7个基本号 + 1个特别号，共8个号码列
            numbers = [int(cols[i]) for i in range(1, 8)]
            special = int(cols[8])
            date_str = cols[-1]
            if not re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                continue
            return {'qihao': qihao, 'date': date_str, 'numbers': numbers, 'special': special}
        except Exception:
            continue
    return None

# ── 快乐8 ──────────────────────────────────────────────
def fetch_kl8():
    year = datetime.now().year
    html = get_html(URL_MAP['kl8'], params={'start': f'{year}001', 'end': f'{year}999'})
    if not html:
        return None
    soup = BeautifulSoup(html, 'html.parser')
    table = find_table(soup)
    if not table:
        return None
    rows = find_data_rows(table)
    if not rows:
        return None
    for cols in reversed(rows):
        try:
            qihao = cols[0]
            if not qihao.isdigit():
                continue
            # 快乐8：20个号码在cols[1..20]
            numbers = [int(cols[i]) for i in range(1, 21)]
            date_str = cols[-1]
            if not re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                continue
            return {'qihao': qihao, 'date': date_str, 'numbers': numbers}
        except Exception:
            continue
    return None

# ── 主流程 ──────────────────────────────────────────────
def main():
    print(f"开始抓取 {date.today()} 最新开奖数据（数据源：datachart.500.com）")
    fetchers = {
        'ssq': ('双色球', fetch_ssq),
        '3d':  ('福彩3D', fetch_3d),
        'qlc': ('七乐彩', fetch_qlc),
        'kl8': ('快乐8',  fetch_kl8),
    }
    result = {'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ok, fail = 0, 0
    for game, (name, fn) in fetchers.items():
        print(f"  抓取 {name}...", end=' ')
        data = fn()
        if data:
            result[game] = data
            print(f"✓  期号:{data['qihao']}  日期:{data['date']}")
            ok += 1
        else:
            print(f"✗ 失败")
            fail += 1

    with open('latest.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n完成：成功 {ok}/4，失败 {fail}/4 → latest.json")
    if ok == 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
