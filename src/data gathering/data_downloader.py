# Imports
import warnings

import pandas as pd
import yfinance as yf
from curl_cffi import requests
from tqdm.auto import tqdm

# disable warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class ForexDataHandler:
    def __init__(
        self,
        main_currencies=["PLN", "EUR"],
        additional_currencies=["CZK", "HUF", "USD", "CHF", "GBP", "JPY"],
    ):
        self.main_currencies = main_currencies
        self.additional_currencies = additional_currencies
        self.session = requests.Session(impersonate="chrome", timeout=5)

    def download_data(self, cur_1, cur_2, t_period="5d", t_interval="1m"):
        symbol = f"{cur_1}{cur_2}=X"
        data = yf.Ticker(symbol, session=self.session)
        f_data = data.history(period=t_period, interval=t_interval)

        mask = f_data.ne(0).any(axis=0)
        f_data = f_data.loc[:, mask]

        f_data.reset_index(inplace=True)
        f_data.rename(
            columns={
                "Datetime": "timestamp",
                "Open": f"{cur_1}{cur_2}_OPEN",
                "High": f"{cur_1}{cur_2}_HIGH",
                "Low": f"{cur_1}{cur_2}_LOW",
                "Close": f"{cur_1}{cur_2}_CLOSE",
            },
            inplace=True,
        )

        f_data["timestamp"] = (
            f_data["timestamp"].apply(lambda x: x.timestamp()).astype(int)
        )
        f_data.set_index("timestamp", inplace=True)
        return f_data

    def update_forex_data(self, old_data_path, save_path="forex_data.feather"):
        old_data = pd.read_feather(old_data_path)
        old_data.set_index("timestamp", inplace=True)

        forex_data = self.download_data("EUR", "PLN")

        for main in self.main_currencies:
            for add in self.additional_currencies:
                if main == add:
                    continue
                try:
                    temp_data = self.download_data(main, add)
                    forex_data = forex_data.join(temp_data)
                except Exception as e:
                    print(f"Error: {e}, cur1: {main}, cur2:{add}")

        forex_data = pd.concat([forex_data, old_data])

        forex_data.reset_index(inplace=True)
        forex_data.drop_duplicates(subset=["timestamp"], inplace=True)
        forex_data.sort_index(inplace=True)

        forex_data.to_feather(save_path)


class EconomicDataHandler:
    def __init__(
        self,
        start_date: str = "2020-01-01",
        end_date: pd.Timestamp = pd.Timestamp.today().normalize(),
        chunk_days: int = 30,
        save_path: str | None = None,
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = end_date
        self.chunk_delta = pd.Timedelta(days=chunk_days)
        self.save_path = save_path
        self.date_ranges = self._generate_date_ranges()
        self.clean_chunks = []

    def _generate_date_ranges(self):
        # Split the full interval into successive (start, end) pairs
        ranges = []
        curr = self.start_date

        while curr <= self.end_date:
            end = min(curr + self.chunk_delta, self.end_date)
            ranges.append((curr, end))
            curr = curr + self.chunk_delta

        return ranges

    def _fetch_range(self, start: pd.Timestamp, to: pd.Timestamp) -> pd.DataFrame:
        # Download raw events JSON and convert to DataFrame.
        url = "https://economic-calendar.tradingview.com/events"
        params = {"from": start.date().isoformat(), "to": to.date().isoformat()}
        headers = {"Origin": "https://www.tradingview.com"}

        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json().get("result", [])

        return pd.DataFrame(data)

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:

        # Fill missing values in referenceDate and create new "Timestamp" column
        df["referenceDate"] = df["referenceDate"].fillna(df["date"])
        df["referenceDate"] = pd.to_datetime(
            df["referenceDate"], format="mixed", yearfirst=True
        )
        df["timestamp"] = df["referenceDate"].apply(lambda x: x.timestamp()).astype(int)

        # Drop unimportant columns, duplicates and rows without crucial data
        calendar_drop = [
            "id",
            "period",
            "source",
            "ticker",
            "scale",
            "category",
            "actualRaw",
            "previousRaw",
            "forecastRaw",
            "source_url",
        ]
        df.drop(calendar_drop, axis=1, inplace=True)
        df.dropna(subset=["actual"], inplace=True)
        df.drop_duplicates(
            subset=["title", "date", "indicator", "country", "referenceDate", "actual"],
            inplace=True,
            keep="last",
        )

        return df

    def download(self) -> pd.DataFrame:
        # Main entry: fetch all data, clean and save if requested
        for start, to in tqdm(self.date_ranges, desc="Downloading chunks"):
            try:
                raw = self._fetch_range(start, to)
                clean = self._clean_df(raw)
                self.clean_chunks.append(clean)
            except Exception as e:
                print(f"Error fetching {start.date()}→{to.date()}: {e}")

        data = pd.concat(self.clean_chunks, ignore_index=True)
        data.reset_index(inplace=True, drop=True)

        if self.save_path:
            data.to_feather(self.save_path)
        return data
