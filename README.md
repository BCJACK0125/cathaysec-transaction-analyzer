# Cathay Securities Transaction Analyzer

Cathay Securities Transaction Analyzer 是一個以國泰證券對帳單為核心的投資績效分析儀表板。這個版本仿照 Firstrade 專案的結構與輸出，差異在於資料來源改為國泰證券 CSV，並先停用投入成本的計算。

## 專案結構

```text
.
├── .github/workflows/analyze.yml   # GitHub Actions 分析與輸出同步流程
├── data/
│   ├── transactions.csv            # 國泰證券對帳單資料
│   ├── output.json                 # 分析輸出
│   └── report.html                 # 簡易 HTML 報告
├── docs/
│   ├── index.html                  # GitHub Pages 前端
│   └── output.json                 # GitHub Pages 使用的分析資料
├── scripts/
│   ├── analyze.py                  # 主分析流程
│   ├── fifo.py                     # FIFO 損益計算
│   └── health.py                   # 健康度指標
└── README.md
```

## 資料來源

### `data/transactions.csv`

本專案使用國泰證券對帳單 CSV。資料首行可能是說明文字，因此分析程式會自動跳過非表頭的第一行。

預期欄位如下：

| 欄位 | 說明 |
| --- | --- |
| `股名` | 股票或 ETF 名稱，當作 symbol 使用 |
| `日期` | 交易日期 |
| `成交股數` | 股數 |
| `淨收付金額` | 買入為負、賣出為正 |
| `買賣別` | `現買` / `現賣` |
| `成交價` | 成交價 |
| `成本` | 成交成本 |
| `手續費` | 手續費 |
| `交易稅` | 交易稅 |

## 分析邏輯

### FIFO Realized PnL

`scripts/fifo.py` 會依日期排序交易，使用 FIFO 計算已實現損益。

- `現買` 建立庫存 lot。
- `現賣` 依最早 lot 沖銷並產生 realized PnL。
- 若遇到缺少庫存的賣出，會以負庫存 lot 表示 short 或資料不足狀態。

### 股票分割調整

為避免分割前後的股數/成本失真，分析會先調整已知分割事件。目前內建：

- 元大台灣50（0050）：2025-06-18 1 拆 4

分割前的交易會自動將股數乘上 4、成交價除以 4。

### Unrealized PnL

現持有部位的 unrealized PnL 計算：

```text
unrealized_pnl = (market_price - lot_cost) * quantity
```

目前版本會先嘗試使用 TWSE OpenAPI (`STOCK_DAY_ALL`) 的收盤價作為 market price，若無法對照或連線失敗則 fallback 到最後成交價，`price_source` 會標記為 `twse_name` / `twse_code` 或 `last_transaction`。

### 投入成本與報酬率

此版本先停用投入成本計算，因此 `return_pct` 與 Sharpe Ratio 相關指標會顯示為未啟用。

### Sharpe Ratio

Sharpe Ratio 依賴投入成本，目前版本停用。

### Health Score

目前第一版 Health Score 是簡化模型：

```text
profit_factor = average_win / abs(average_loss)
health_score = clamp(50 + profit_factor * 10, 0, 100)
```

## 輸出資料

主分析會產生：

```text
data/output.json
docs/output.json
data/report.html
```

`docs/index.html` 會讀取：

```text
./output.json?v=<timestamp>
```

## 本機執行

建議使用虛擬環境：

```bash
python -m venv .venv
.venv\Scripts\python -m pip install pandas numpy requests
```

執行分析：

```bash
.venv\Scripts\python -B scripts\analyze.py
```

同步到 GitHub Pages 資料目錄：

```bash
copy data\output.json docs\output.json
```

啟動本機網站：

```bash
python -m http.server 8000 --directory docs
```

## GitHub Actions

`.github/workflows/analyze.yml` 會在以下情況執行：

- 手動觸發 `workflow_dispatch`
- push 修改：
	- `data/transactions.csv`
	- `scripts/*.py`
	- `.github/workflows/analyze.yml`

流程會：

1. 安裝 Python 3.12。
2. 安裝 `pandas numpy requests`。
3. 執行 `python scripts/analyze.py`。
4. 將 `data/output.json` 複製到 `docs/output.json`。
5. 自動 commit 分析結果。

## 第一版限制

- 投入成本尚未定義，報酬率與 Sharpe Ratio 暫停。
- 未提供現倉參考檔，因此持倉驗算功能停用。
- 目前價格來源以 TWSE OpenAPI 收盤價為主，失敗時才使用最後成交價。