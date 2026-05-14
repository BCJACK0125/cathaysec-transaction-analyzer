import requests

URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TARGET_ETF = "0050"
VERIFY_TLS = False


def fetch_data(url: str, verify_tls: bool) -> list[dict]:
    if not verify_tls:
        print("Warning: TLS verification disabled (verify=False). Use only for testing.")

    response = requests.get(url, timeout=20, verify=verify_tls)
    response.raise_for_status()

    data = response.json()
    if not isinstance(data, list):
        raise ValueError("Unexpected response shape; expected a list.")
    return data


def find_etf(data: list[dict], target_code: str) -> dict | None:
    for item in data:
        if item.get("Code") == target_code:
            return item
    return None


if __name__ == "__main__":
    try:
        data = fetch_data(URL, VERIFY_TLS)
        etf = find_etf(data, TARGET_ETF)

        if etf:
            print(f"Code: {etf.get('Code')}")
            print(f"Name: {etf.get('Name')}")
            print(f"ClosingPrice: {etf.get('ClosingPrice')}")
        else:
            print(f"ETF {TARGET_ETF} not found.")
    except Exception as exc:
        print("Failed to fetch TWSE data:", type(exc).__name__, exc)
