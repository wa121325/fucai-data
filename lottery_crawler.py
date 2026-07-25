"""
福彩开奖数据爬虫 - GitHub Actions 版 v3
主数据源：福彩官网 JSON API
备用数据源：datachart.500.com
"""
import requests
from bs4 import BeautifulSoup
import json, re, sys
from datetime import datetime, date

SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
})

CWL_API = 'https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice'

# ── 官网API ─────────────────────────────────────────────
def fetch_cwl(game):
    r = SESSION.get(CWL_API, params={
        'name': game, 'issueCount': 1,
        'pageNo': 1, 'pageSize': 1, 'systemType': 'PC'
    }, headers={'Referer': 'https://www.cwl.gov.cn/'}, timeout=15)
    r.raise_for_status()
    data = r.json()
    print(f'  [DEBUG] {game} 官网原始返回: {json.dumps(data, ensure_ascii=False)[:300]}')
    item = (data.get('result') or [None])[0]
    if not item:
        raise ValueError('result为空')

    qihao = str(item.get('code',''))
    date_str = str(item.get('date',''))[:10]
    red_raw = str(item.get('red',''))
    blue_raw = str(item.get('blue',''))

    print(f'  [DEBUG] {game} code={qihao} date={date_str} red={red_raw} blue={blue_raw}')

    if game == 'ssq':
        red = sorted([int(x) for x in red_raw.split(',') if x.strip()])
        return {'qihao':qihao,'date':date_str,'red':red,'blue':int(blue_raw)}
    elif game == '3d':
        digits = [int(x) for x in red_raw.split(',') if x.strip()][:3]
        return {'qihao':qihao,'date':date_str,'digits':digits}
    elif game == 'qlc':
        nums = sorted([int(x) for x in red_raw.split(',') if x.strip()])
        return {'qihao':qihao,'date':date_str,'numbers':nums[:7],'special':int(blue_raw)}
    elif game == 'kl8':
        nums = sorted([int(x) for x in red_raw.split(',') if x.strip()])
        return {'qihao':qihao,'date':date_str,'numbers':nums}

# ── 备用：500网 HTML ──────────────────────────────────────
URL_500 = {
    'ssq': 'https://datachart.500.com/ssq/history/newinc/history.php',
    '3d':  'https://datachart.500.com/sd/history/inc/history.php',
    'qlc': 'https://datachart.500.com/qlc/history/newinc/history.php',
    'kl8': 'https://datachart.500.com/kl8/history/newinc/history.php',
}

def fetch_500(game):
    year = datetime.now().year
    r = SESSION.get(URL_500[game],
                    params={'start':f'{year}001','end':f'{year}999'},
                    timeout=20)
    r.encoding = 'gb2312'
    r.raise_for_status()
    soup = BeautifulSoup(r.text,'html.parser')
    table = (soup.find('table',{'id':'tablelist'})
             or soup.find('table',class_='chartTable')
             or soup.find('table'))
    if not table:
        raise ValueError('未找到表格')

    # 找数据行起始（跳过所有表头）
    rows = table.find_all('tr')
    start = 0
    for i, row in enumerate(rows):
        txt = row.get_text()
        if '期号' in txt and ('日期' in txt or '开奖' in txt):
            start = i + 1
    
    # 调试：打印前3行看结构
    print(f'  [DEBUG] {game} 500网 start={start} 总行={len(rows)}')
    for row in rows[start:start+3]:
        cols = [td.get_text(strip=True) for td in row.find_all('td')]
        print(f'  [DEBUG] {game} 500网数据行: {cols[:8]}')

    records = []
    for row in rows[start:]:
        cols = [td.get_text(strip=True) for td in row.find_all('td')]
        if len(cols) < 3: continue
        qihao = cols[0]
        if not qihao.isdigit(): continue
        date_str = ''
        for c in reversed(cols):
            if re.match(r'\d{4}-\d{2}-\d{2}', c):
                date_str = c; break
        if not date_str: continue
        records.append((int(qihao), date_str, cols))

    if not records:
        raise ValueError('无有效数据行')

    # 按期号排序取最新
    records.sort(key=lambda x: x[0], reverse=True)
    qihao_int, date_str, cols = records[0]
    qihao = str(qihao_int)
    print(f'  [DEBUG] {game} 500网最新行: 期号={qihao} 日期={date_str} cols={cols[:10]}')

    if game == 'ssq':
        red = sorted([int(cols[i]) for i in range(1,7)])
        blue = int(cols[7])
        return {'qihao':qihao,'date':date_str,'red':red,'blue':blue}
    elif game == '3d':
        # 找3位连续数字的列
        for c in cols[1:]:
            c2 = c.replace(' ','')
            if re.match(r'^\d{3}$', c2):
                return {'qihao':qihao,'date':date_str,'digits':[int(x) for x in c2]}
        raise ValueError(f'未找到3D号码列，cols={cols}')
    elif game == 'qlc':
        nums = sorted([int(cols[i]) for i in range(1,8)])
        special = int(cols[8])
        return {'qihao':qihao,'date':date_str,'numbers':nums,'special':special}
    elif game == 'kl8':
        nums = sorted([int(cols[i]) for i in range(1,21)])
        return {'qihao':qihao,'date':date_str,'numbers':nums}

# ── 主流程 ────────────────────────────────────────────────
def fetch_game(game, name):
    try:
        result = fetch_cwl(game)
        print(f'  ✓ {name}（官网API）期号:{result["qihao"]} 日期:{result["date"]}')
        return result
    except Exception as e:
        print(f'  ! {name} 官网API失败: {e}', file=sys.stderr)

    try:
        result = fetch_500(game)
        print(f'  ✓ {name}（500网）期号:{result["qihao"]} 日期:{result["date"]}')
        return result
    except Exception as e:
        print(f'  ✗ {name} 全部失败: {e}', file=sys.stderr)
        return None

def main():
    print(f'抓取日期: {date.today()}')
    games = [('ssq','双色球'),('3d','福彩3D'),('qlc','七乐彩'),('kl8','快乐8')]
    result = {'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ok = 0
    for game, name in games:
        print(f'\n── {name} ──')
        data = fetch_game(game, name)
        if data:
            result[game] = data
            ok += 1

    with open('latest.json','w',encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n\n完成：{ok}/4 → latest.json')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if ok == 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
