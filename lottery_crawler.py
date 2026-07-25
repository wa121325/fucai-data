# 福彩开奖数据爬虫 - 最终修复版 (支持自动定位最新期号)
import requests
from bs4 import BeautifulSoup
import json, re, time
from datetime import datetime

SESSION = requests.Session()
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Referer": "https://datachart.500.com/"
}

def find_date_in_list(tds):
    for item in tds:
        match = re.search(r"\d{4}-\d{2}-\d{2}", item)
        if match: return match.group()
    return "unknown"

def fetch_data(url, min_tds, qihao_pattern):
    try:
        resp = SESSION.get(url, headers=DEFAULT_HEADERS, timeout=10)
        resp.encoding = 'gbk'
        soup = BeautifulSoup(resp.text, "html.parser")
        for tr in soup.find_all("tr"):
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(tds) < min_tds or "统计" in str(tr) or "期号" in str(tr):
                continue
            for i in range(min(6, len(tds))):
                if re.match(qihao_pattern, tds[i]):
                    if any(re.search(r"\d{4}-\d{2}-\d{2}", t) for t in tds):
                        return tds, i
    except Exception as e: print(f"网络请求异常: {e}")
    return None, None

def fetch_ssq():
    res, idx = fetch_data("https://datachart.500.com/ssq/history/newinc/history.php?limit=10", 10, r"^\d{5,8}$")
    if res: return {"qihao": res[idx], "date": find_date_in_list(res), "red": res[idx+1:idx+7], "blue": res[idx+7]}
    raise ValueError("SSQ fail")

def fetch_3d():
    res, idx = fetch_data("https://datachart.500.com/sd/history/inc/history.php?limit=10", 5, r"^\d{7,8}$")
    if res:
        nums = re.findall(r"\d+", res[idx+1])[:3]
        return {"qihao": res[idx], "date": find_date_in_list(res), "digits": nums}
    raise ValueError("3D fail")

def fetch_qlc():
    res, idx = fetch_data("https://datachart.500.com/qlc/history/newinc/history.php?limit=10", 5, r"^\d{5,8}$")
    if res:
        nums = re.findall(r"\b(?:0[1-9]|[12][0-9]|30)\b", " ".join(res[idx:]))
        if len(nums) >= 8: return {"qihao": res[idx], "date": find_date_in_list(res), "basic": nums[:7], "special": nums[7]}
    raise ValueError("QLC fail")

def fetch_kl8():
    # 快乐8: 扫描全表并提取期号最大的行
    try:
        resp = SESSION.get("https://datachart.500.com/kl8/", headers=DEFAULT_HEADERS)
        resp.encoding = 'gbk'
        soup = BeautifulSoup(resp.text, "html.parser")
        all_valid_rows = []
        for tr in soup.find_all("tr"):
            txt = tr.get_text(" ", strip=True)
            nums = re.findall(r"\b(?:0[1-9]|[1-7][0-9]|80)\b", txt)
            if len(nums) >= 20:
                qihao_match = re.search(r"\b\d{7,11}\b", txt)
                if qihao_match:
                    all_valid_rows.append({
                        "qihao": qihao_match.group(),
                        "date": find_date_in_list([txt]),
                        "numbers": nums[:20]
                    })
        if all_valid_rows:
            # 按期号降序排列，取第一个（最新的）
            latest = sorted(all_valid_rows, key=lambda x: x['qihao'], reverse=True)[0]
            if latest['date'] == "unknown": 
                latest['date'] = datetime.now().strftime("%Y-%m-%d")
            return latest
    except: pass
    raise ValueError("KL8 fail")

def main():
    results = {"updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    fetchers = [("ssq", "双色球", fetch_ssq), ("3d", "福彩3D", fetch_3d), ("qlc", "七乐彩", fetch_qlc), ("kl8", "快乐8", fetch_kl8)]
    for code, name, func in fetchers:
        try:
            data = func()
            results[code] = data
            print(f"[成功] {name} | 期号: {data['qihao']} | 日期: {data['date']}")
        except Exception as e: print(f"[失败] {name}: {e}")
        time.sleep(0.5)
    with open("latest.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n数据已保存至 latest.json")

if __name__ == "__main__":
    main()
