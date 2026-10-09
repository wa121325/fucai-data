#!/usr/bin/env bash
# 推送一个 Kaggle kernel 并等它"真正跑完"。
# 用法: bash ci/run_kernel.sh <kernel> <推送目录> <数据仓库里用来判断新鲜度的文件> [最长等待分钟=60]
#   例:  bash ci/run_kernel.sh megskfdbbskeb/fucai-ml-daily ./kaggle_kernel/ prediction.json 60
#
# 成功条件（两个必须同时成立）：
#   ① Kaggle 状态为 complete；
#   ② 数据仓库里的新鲜度文件，其 updated_at（Kaggle 机器是 UTC）不早于本次触发时刻。
# 为什么要②：push 之后 Kaggle 状态要过一会儿才从上一次的 COMPLETE 切到 RUNNING，
# 只看状态会把"上一轮的完成"误判成"这一轮完成"。
# 失败/超时一律 exit 1，下游步骤不会启动。
# 环境变量：DATA_REPO（如 wa121325/fucai-data）必填；DATA_BASE 可覆盖取文件的地址（测试用）。
set -u
KERNEL="$1"; DIR="$2"; FRESH_FILE="$3"; MAXMIN="${4:-60}"
DATA_BASE="${DATA_BASE:-https://raw.githubusercontent.com/${DATA_REPO}/main}"
START_EPOCH=$(date -u +%s)
echo "[$KERNEL] 触发时刻 $(date -u -d @"$START_EPOCH" +'%H:%M:%S') UTC"

kaggle kernels push -p "$DIR" || { echo "✗ 推送失败: $DIR"; exit 1; }
echo "✓ 已推送，Kaggle 开始运行"

fresh() {
  python3 - "$START_EPOCH" "$DATA_BASE/$FRESH_FILE" <<'PY'
import sys, json, time, urllib.request, calendar
start, url = int(sys.argv[1]), sys.argv[2]
try:
    req = urllib.request.Request(f"{url}?t={int(time.time())}", headers={'Cache-Control': 'no-cache', 'User-Agent': 'ci'})
    ua = json.loads(urllib.request.urlopen(req, timeout=60).read().decode('utf-8')).get('updated_at', '')
    ts = calendar.timegm(time.strptime(ua[:16], '%Y-%m-%d %H:%M'))   # 只取到分钟，不同文件秒的格式不同
    ok = ts >= start - 120
    print(f"    {url.split('/')[-1]} 更新于 {ua} UTC，{'是' if ok else '不是'}本轮产出")
    sys.exit(0 if ok else 1)
except Exception as e:
    print(f"    读取新鲜度文件失败: {e}")
    sys.exit(1)
PY
}

# 阶段1：等 Kaggle 开始新一轮（最多10分钟）。这一阶段读到的 error/complete 多半是上一轮的残留，一律忽略。
SEEN_ACTIVE=0
for i in $(seq 1 20); do
  sleep "${POLL_START:-30}"
  STATUS=$(kaggle kernels status "$KERNEL" 2>&1 | tail -1)
  echo "$(date +'%H:%M:%S') [启动 $i/20] $STATUS"
  if echo "$STATUS" | grep -qiE "running|queued"; then SEEN_ACTIVE=1; break; fi
done

# 阶段2：等完成（最多 MAXMIN 分钟），完成后必须再通过新鲜度检查
for i in $(seq 1 "$MAXMIN"); do
  STATUS=$(kaggle kernels status "$KERNEL" 2>&1 | tail -1)
  echo "$(date +'%H:%M:%S') [运行 $i/$MAXMIN] $STATUS"
  if echo "$STATUS" | grep -qiE "running|queued"; then SEEN_ACTIVE=1; fi
  if echo "$STATUS" | grep -qiE "error|cancel|fail"; then
    # 没见过本轮开始运行、且才刚过阶段1时，看到的 error 可能是上一轮的残留，再等几轮确认
    if [ "$SEEN_ACTIVE" = "1" ] || [ "$i" -ge 5 ]; then echo "✗ Kaggle 运行失败: $KERNEL"; exit 1; fi
  fi
  if echo "$STATUS" | grep -qi "complete"; then
    if fresh; then echo "✓ $KERNEL 运行完成，且本轮结果已写入数据仓库"; exit 0; fi
    echo "  状态是 complete 但结果文件还是旧的（多半是上一轮的完成状态），继续等待…"
  fi
  sleep "${POLL_RUN:-60}"
done
echo "✗ 等待超过 ${MAXMIN} 分钟，$KERNEL 仍未产出本轮结果。按失败处理。"
exit 1
