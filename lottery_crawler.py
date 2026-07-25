"""
福彩开奖数据爬虫 - GitHub Actions 版
主数据源：福彩官网 JSON API（cwl.gov.cn）—— GitHub Actions 服务器可直接访问
备用数据源：datachart.500.com HTML（解析后按期号排序取最新）
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
    'Referer': 'https://www.cwl.gov.cn/',
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# ── 主数据源：福彩官网 JSON API ───────────────────────────
CWL_API = 'https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice'

def fetch_cwl_api(game):
    """直接调福彩官网接口，GitHub Actions 服务器无跨域限制"""
    r = SESSION.get(CWL_API, params={
        'name': game, 'issueCount': 1,
        'pageNo': 1, 'pageSize': 1, 'systemType': 'PC'
    }, timeout=15)
    r.raise_for_status()
    data = r.json()
    item = data.get('result', [None])[0]
    if not item:
        raise ValueError('API返回result为空')
    return item

def parse_cwl(game, item):
    """把官网API返回的item转成我们的标准格式"""
    qihao = str(item.get('code', ''))
    date_str = str(item.get('date', ''))[:10]
    red_str = item.get('red', '')
    blue_str = str(item.get('blue', ''))

    if game == 'ssq':
        return {'qihao': qihao, 'date': date_str,
                'red': [int(x) for x in red_str.split(',')],
                'blue': int(blue_str)}
    elif game == '3d':
        digits = [int(x) for x in red_str.split(',')]
        return {'qihao': qihao, 'date': date_str, 'digits': digits[:3]}
    elif game == 'qlc':
        nums = [int(x) for x in red_str.split(',')]
        return {'qihao': qihao, 'date': date_str,
                'numbers': nums[:7], 'special': int(blue_str)}
    elif game == 'kl8':
        return {'qihao': qihao, 'date': date_str,
                'numbers': [int(x) for x in red_str.split(',')]}

# ── 备用数据源：datachart.500.com HTML ───────────────────
URL_500 = {
    'ssq': 'https://datachart.500.com/ssq/history/newinc/history.php',
    '3d':  'https://datachart.500.com/sd/history/inc/history.php',
    'qlc': 'https://datachart.500.com/qlc/history/newinc/history.php',
    'kl8': 'https://datachart.500.com/kl8/history/newinc/history.php',
}

def fetch_500_html(game):
    year = datetime.now().year
    r = SESSION.get(URL_500[game],
                    params={'start': f'{year}001', 'end': f'{year}999'},
                    timeout=20)
    r.encoding = 'gb2312'
    r.raise_for_status()
    return r.text

def parse_500(game, html):
    soup = BeautifulSoup(html, 'html.parser')
    table = (soup.find('table', {'id': 'tablelist'})
             or soup.find('table', class_='chartTable')
             or soup.find('table'))
    if not table:
        raise ValueError('未找到表格')

    # 自动跳过表头行
    rows = table.find_all('tr')
    start = 0
    for i, row in enumerate(rows):
        if '期号' in row.get_text() and ('开奖日期' in row.get_text() or '日期' in row.get_text()):
            start = i + 1
            break

    records = []
    for row in rows[start:]:
        cols = [td.get_text(strip=True) for td in row.find_all('td')]
        if len(cols) < 3:
            continue
        qihao = cols[0]
        if not qihao.isdigit():
            continue
        date_str = cols[-1]
        if not re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            continue
        records.append((int(qihao), date_str, cols))

    if not records:
        raise ValueError('表格无有效数据行')

    # 按期号降序排列，取最新一期（不依赖表格原始顺序）
    records.sort(key=lambda x: x[0], reverse=True)
    qihao_int, date_str, cols = records[0]
    qihao = str(qihao_int)

    try:
        if game == 'ssq':
            red = [int(cols[i]) for i in range(1, 7)]
            blue = int(cols[7])
            return {'qihao': qihao, 'date': date_str, 'red': red, 'blue': blue}
        elif game == '3d':
            # 号码在第2列，3位数字
            nums_raw = cols[1].replace(' ', '')
            if not re.match(r'^\d{3}$', nums_raw):
                raise ValueError(f'3D号码格式错误: {nums_raw}')
            return {'qihao': qihao, 'date': date_str, 'digits': [int(c) for c in nums_raw]}
        elif game == 'qlc':
            numbers = [int(cols[i]) for i in range(1, 8)]
            special = int(cols[8])
            return {'qihao': qihao, 'date': date_str, 'numbers': numbers, 'special': special}
        elif game == 'kl8':
            numbers = [int(cols[i]) for i in range(1, 21)]
            return {'qihao': qihao, 'date': date_str, 'numbers': numbers}
    except Exception as e:
        raise ValueError(f'解析列数据失败: {e}，cols={cols[:5]}')

# ── 每种彩票的抓取入口（主+备用自动切换）────────────────
def fetch_game(game, name):
    # 优先官网API
    try:
        item = fetch_cwl_api(game)
        result = parse_cwl(game, item)
        print(f'  ✓ {name}（官网API）期号:{result["qihao"]} 日期:{result["date"]}')
        return result
    except Exception as e:
        print(f'  ! {name} 官网API失败: {e}，尝试500网备用...', file=sys.stderr)

    # 备用500网HTML
    try:
        html = fetch_500_html(game)
        result = parse_500(game, html)
        print(f'  ✓ {name}（500网备用）期号:{result["qihao"]} 日期:{result["date"]}')
        return result
    except Exception as e:
        print(f'  ✗ {name} 备用也失败: {e}', file=sys.stderr)
        return None

# ── 主流程 ───────────────────────────────────────────────
def main():
    today = date.today()
    print(f'开始抓取 {today} 最新开奖数据')
    print(f'主数据源: 福彩官网JSON API (cwl.gov.cn)')

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
