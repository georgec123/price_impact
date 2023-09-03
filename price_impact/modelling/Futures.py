from __future__ import annotations  # for self reference
import datetime as dt
from pathlib import Path
import os
import pandas as pd
from typing import List
import json
from dateutil.relativedelta import relativedelta


months = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
future_month_lookup = {month: idx+1 for idx, month in enumerate(months)}
month_future_lookup = {v: k for k, v in future_month_lookup.items()}


data_dir = Path(os.path.dirname(__file__))/'..'/'data'

future_chain = pd.read_csv(data_dir/'exchange_open_info.csv')
future_active_months = future_chain[['tick_code', 'future_months']].set_index('tick_code').to_dict()['future_months']
expiry_rules = future_chain[['tick_code', 'expiry_rule']].dropna().set_index('tick_code').to_dict()['expiry_rule']
roll_lookback_weeks = future_chain[['tick_code', 'roll_lookback_weeks']].dropna().set_index('tick_code').to_dict()[
    'roll_lookback_weeks']
ticker_descriptions = future_chain[['tick_code', 'asset_class', 'description',
                                    'FUT_TRADING_HRS', 'bst_offset', 'Ticker']].set_index('tick_code').to_dict(orient='index')


def clean_bbg_ticker(ticker: str):
    return ticker.replace(' A ', ' ').replace('A ', ' ')


def get_trading_window(str_window: str):
    list_window = str_window.split('-')
    start_str, end_str = list_window[0], list_window[-1]

    start_date = dt.datetime.strptime(start_str, "%H:%M")
    end_date = dt.datetime.strptime(end_str, "%H:%M")

    if end_date.hour < start_date.hour:
        end_date += relativedelta(days=1)
    return start_date, end_date


for k in future_active_months.keys():
    future_active_months[k] = json.loads(
        future_active_months[k].replace("'", '"'))


def n_th_dom(year: int, month: int, dow: int, n: int) -> dt.date:
    """
    Find nth day of the month for a specific month year
    ie, 3rd sunday in April 2022 -> n_th_dom(2022, 4, 6, 3)
    Note, weekdays are 0 indexed from monday.

    """
    if n < 1:
        raise ValueError("n must be greater than 0")

    date = dt.date(year=year, month=month, day=1)

    first_dow = date.weekday()
    first_shift = dow-first_dow if dow-first_dow >= 0 else 7 + dow-first_dow

    # find first instance
    first = date + dt.timedelta(days=first_shift)

    # find nth instance
    nth = first + dt.timedelta(days=(n-1)*7)

    return nth


class Spot(str):
    def __new__(cls, spot: str):
        return str.__new__(cls, spot)

    def __init__(self, spot: str):
        self.spot = spot
        self.asset_class = ticker_descriptions[spot]['asset_class']
        self.description = ticker_descriptions[spot]['description']
        self.bst_offset = ticker_descriptions[spot]['bst_offset']
        self.trading_start_time, self.trading_end_time = get_trading_window(ticker_descriptions[spot]['FUT_TRADING_HRS'])
        self.bbg_ticker = clean_bbg_ticker(ticker_descriptions[spot]['Ticker'])
        self.bbg_active_ticker = ticker_descriptions[spot]['Ticker']

    def hour_to_bst_time(self, hour: int) -> dt.time:
        return dt.time((hour-self.bst_offset) % 24, 0, 0)


def date_to_time(date: dt.date) -> dt.datetime:

    datetime = dt.datetime.combine(date, dt.datetime.min.time())
    return datetime


class FutureContract(str):

    def __new__(cls, tick_code):
        return str.__new__(cls, tick_code)

    def __init__(self, tick_code):
        self.tick_code = tick_code
        self.spot = Spot(tick_code[:-3])

        self.expiry_year: int = int('20'+tick_code[-2:])
        self.expiry_month_str: str = tick_code[-3:-2]
        self.expiry_month: int = future_month_lookup[self.expiry_month_str]

        self.expiry_date: dt.datetime = self._get_expiry_date()
        self.active_months: List[str] = future_active_months[self.spot]
        self.roll_lookback_weeks: float = roll_lookback_weeks[self.spot]
        self.backtest_end_date = self.expiry_date - relativedelta(weeks=self.roll_lookback_weeks)

    @classmethod
    def from_spot(cls, spot: str, year: int = 2022):
        last_month = future_active_months[spot][-1]
        year_str = str(year)[-2:]

        return cls(f"{spot}{last_month}{year_str}")

    def copy(self) -> FutureContract:
        return FutureContract(self.tick_code)

    def is_active_contract(self) -> bool:
        return self.expiry_month_str in self.active_months

    def _get_expiry_date(self) -> dt.datetime:

        if self.spot not in expiry_rules:
            # default 3rd friday of the month
            return date_to_time(n_th_dom(self.expiry_year, self.expiry_month, 4, 3))

        accepted_methods = ['last_business_day']

        expiry_method = expiry_rules[self.spot]
        if expiry_method not in accepted_methods:
            raise ValueError(f"Method {expiry_method}")

        if expiry_method == 'last_business_day':
            base_date = dt.date(self.expiry_year, self.expiry_month, 1) + \
                relativedelta(months=1) - dt.timedelta(days=1)

            while base_date.isoweekday() > 4:
                base_date -= dt.timedelta(days=1)

            return date_to_time(base_date)

        return

    def previous_contract(self, n=1) -> FutureContract:

        curr_pos = self.active_months.index(self.expiry_month_str)

        new_year = self.expiry_year-1 if curr_pos == 0 else self.expiry_year
        new_year_str = str(new_year)[-2:]
        new_month_str = self.active_months[curr_pos-1]

        if n == 1:
            return FutureContract(f"{self.spot}{new_month_str}{new_year_str}")
        else:
            return FutureContract(f"{self.spot}{new_month_str}{new_year_str}").previous_contract(n-1)

    def next_contract(self) -> FutureContract:

        curr_pos = self.active_months.index(self.expiry_month_str)

        new_year = self.expiry_year + \
            1 if curr_pos == len(self.active_months)-1 else self.expiry_year
        new_year_str = str(new_year)[-2:]

        # loop to the start of list if we are at the end
        new_month_str = self.active_months[(curr_pos+1) % len(self.active_months)]

        return FutureContract(f"{self.spot}{new_month_str}{new_year_str}")


if __name__ == '__main__':
    print(1)
