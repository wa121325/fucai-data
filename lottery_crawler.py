# 福彩开奖数据爬虫 - 更鲁棒的版本（支持多 URL 备选、编码回退、宽松数字提取与校验）
import requests
from bs4 import BeautifulSoup
import json, re, time
from datetime import datetime

SESSION = requests.Session()
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Referer": "https://datachart.500.com/"
}


def get_soup(url, timeout=10):
    try:
        resp = SESSION.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        # 尝试多种常见编码
        for enc in ('utf-8', 'gbk', 'gb2312', None):
            try:
                if enc:
                    resp.encoding = enc
                soup = BeautifulSoup(resp.text, "html.parser")
                return soup
            except Exception:
                continue
    except Exception as e:
        print(f"请求 {url} 异常: {e}")
    return None


def find_date_in_list(tds):
    for item in tds:
        match = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", item)
        if match:
            return match.group().replace('/', '-')
    return "unknown"


def find_row_with_qihao(soup, qihao_pattern):
    """
    遍历 <tr>，返回第一个包含匹配期号的列文本列表和该列索引（列内可能有 <a> 等）
    """
    if not soup:
        return None, None
    for tr in soup.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) < 2:
            continue
        # 跳过表头或统计行
        raw_tr = str(tr)
        if any(skip in raw_tr for skip in ("统计", "合计", "期号", "选号")):
            continue
        for i, txt in enumerate(tds):
            if re.search(qihao_pattern, txt):
                return tds, i
    return None, None


def extract_numbers_from_row(tds, qihao_idx):
    """
    从期号所在行中提取所有可能的数字（1-2位数），返回格式化为两位的字符串列表
    """
    if not tds or qihao_idx is None:
        return []
    # 合并期号之后的所有列内容作为号码来源（有些页面号码分散在多列）
    tail = " ".join(tds[qihao_idx+1:])
    # 提取所有 1-2 位数字（避免取到年份等多位数字）
    raw_nums = re.findall(r"\b\d{1,2}\b", tail)
    # 格式化并过滤 0 前缀问题
    formatted = []
    for n in raw_nums:
        try:
            iv = int(n)
            if iv >= 0:  # 基本过滤（后续按彩票类型再做范围过滤）
                formatted.append(f"{iv:02d}")
        except:
            continue
    return formatted


def clean_qihao(text):
    m = re.search(r"\d{5,11}", text)
    return m.group() if m else text.strip()


def fetch_ssq():
    # 备选 URL（优先使用 datachart 的历史表）
    urls = [
        "https://datachart.500.com/ssq/history/newinc/history.php?limit=10",
        "https://datachart.500.com/ssq/history/history.shtml"
    ]
    qihao_pattern = r"\d{5,8}"
    for url in urls:
        soup = get_soup(url)
        tds, idx = find_row_with_qihao(soup, qihao_pattern)
        if tds:
            qihao = clean_qihao(tds[idx])
            nums = extract_numbers_from_row(tds, idx)
            # 找到至少 7 个数字（6 红 + 1 蓝）
            if len(nums) >= 7:
                red = nums[:6]
                blue = nums[6]
                date = find_date_in_list(tds)
                return {"qihao": qihao, "date": date, "red": red, "blue": blue, "source": url}
    raise ValueError("SSQ fail")


def fetch_3d():
    urls = [
        "https://datachart.500.com/sd/history/inc/history.php?limit=10",
        "https://datachart.500.com/sd/history/history.shtml"
    ]
    qihao_pattern = r"\d{7,8}"
    for url in urls:
        soup = get_soup(url)
        tds, idx = find_row_with_qihao(soup, qihao_pattern)
        if tds:
            qihao = clean_qihao(tds[idx])
            nums = extract_numbers_from_row(tds, idx)
            # 3D 通常取连续 3 个数字（每个 0-9）
            # 在页面中可能表现为连续三位合并或分列，尝试取第一个含 3 个个位数的段
            if len(nums) >= 3:
                digits = [n[-1] if len(n) == 2 else n for n in nums[:3]]  # 取个位数字字符或直接数字
                date = find_date_in_list(tds)
                return {"qihao": qihao, "date": date, "digits": digits, "source": url}
    raise ValueError("3D fail")


def fetch_qlc():
    urls = [
        "https://datachart.500.com/qlc/history/newinc/history.php?limit=10",
        "https://datachart.500.com/qlc/history/history.shtml"
    ]
    qihao_pattern = r"\d{5,8}"
    for url in urls:
        soup = get_soup(url)
        tds, idx = find_row_with_qihao(soup, qihao_pattern)
        if tds:
            qihao = clean_qihao(tds[idx])
            nums = extract_numbers_from_row(tds, idx)
            # 七乐彩要求 7 个基本 + 1 个特别（共 8），且号码范围 1-30
            nums_int = []
            for s in nums:
                try:
                    v = int(s)
                    if 1 <= v <= 30:
                        nums_int.append(f"{v:02d}")
                except:
                    continue
            if len(nums_int) >= 8:
                basic = nums_int[:7]
                special = nums_int[7]
                date = find_date_in_list(tds)
                return {"qihao": qihao, "date": date, "basic": basic, "special": special, "source": url}
    raise ValueError("QLC fail")


def fetch_kl8():
    # 快乐8需要 20 个号码（1-80）
    urls = [
        "https://datachart.500.com/kl8/",
        "https://datachart.500.com/kl8/history.shtml"
    ]
    qihao_pattern = r"\d{7,11}"
    for url in urls:
        soup = get_soup(url)
        if not soup:
            continue
        all_valid_rows = []
        for tr in soup.find_all("tr"):
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(tds) < 3:
                continue
            raw_tr = str(tr)
            if any(skip in raw_tr for skip in ("统计", "合计", "期号")):
                continue
            # 找到期号列
            q_idx = None
            for i, txt in enumerate(tds):
                if re.search(qihao_pattern, txt):
                    q_idx = i
                    break
            if q_idx is None:
                continue
            qihao = clean_qihao(tds[q_idx])
            # 在期号之后搜集所有数字并过滤 1-80
            nums = extract_numbers_from_row(tds, q_idx)
            valid = []
            for s in nums:
                try:
                    v = int(s)
                    if 1 <= v <= 80:
                        valid.append(f"{v:02d}")
                except:
                    continue
            if len(valid) == 20:
                date = find_date_in_list(tds)
                all_valid_rows.append({"qihao": qihao, "date": date, "numbers": valid, "source": url})
        if all_valid_rows:
            latest = sorted(all_valid_rows, key=lambda x: int(re.sub(r'\D', '', x['qihao'])), reverse=True)[0]
            if latest['date'] == "unknown":
                latest['date'] = datetime.now().strftime("%Y-%m-%d")
            return latest
    raise ValueError("KL8 fail")


def main():
    results = {"updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    fetchers = [
        ("ssq", "双色球", fetch_ssq),
        ("3d", "福彩3D", fetch_3d),
        ("qlc", "七乐彩", fetch_qlc),
        ("kl8", "快乐8", fetch_kl8)
    ]
    for code, name, func in fetchers:
        try:
            data = func()
            results[code] = data
            print(f"[成功] {name} | 期号: {data['qihao']} | 日期: {data['date']} | 来源: {data.get('source')}")
        except Exception as e:
            print(f"[失败] {name}: {e}")
        time.sleep(0.5)
    with open("latest.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n数据已保存至 latest.json")


if __name__ == "__main__":
    main()
