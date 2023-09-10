import logging
import os
import re
from datetime import datetime
from typing import Dict, List

import pandas as pd

from src.data.sec_data.utils import (call_api, fetch_10k_url_from_rss,
                                     fetch_ingested_ticker,
                                     load_ticker_cik_mapping, parse_10k)
from src.meta_data import get_meta_data

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# console
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)-8s %(message)s')
stream_handler.setFormatter(formatter)

file_handler = logging.FileHandler('submissio_fetch.log', 'a+')
formatter_with_date = logging.Formatter('%(asctime)s - %(levelname)-8s %(message)s',
                                        '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter_with_date)
file_handler.setLevel(logging.INFO)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


class FetchSubmission:
    """ fetch submission ID for a given CIK
    """
    def __init__(
        self,
        ticker: str,
        ) -> None:
        """constructor
        """
        self.cik_ticker_mapping = self.fetch_ticker_mapping()
        self.ticker = ticker
        self.cik = self.cik_ticker_mapping.get(ticker, None)
        self.save_dir = get_meta_data()['SEC_DIR']

    def fetch_ticker_mapping(self) -> Dict[str, str]:
        """fetch cik to ticker mapping

        Returns:
            Dict[str, str]:  cik to ticker mapping

        """
        raw_data_dir = os.path.dirname(os.path.realpath(__file__))
        raw_data_dir = os.path.dirname(raw_data_dir)
        file_path = os.path.join(raw_data_dir, 'raw_data', 'ticker_cik_mapping.txt')

        return load_ticker_cik_mapping(file_path)


    def make_sub_txt_url(self):
        """"
        make the url of submission main txt for file type and date
        """
        return fetch_10k_url_from_rss(self.cik)

    @staticmethod
    def extract_modified_date(last_modified: str) -> datetime:
        """extract date from string

        Args:
            last_modified (str): input date

        Returns:
            datetime: datetime date
        """
        return datetime.strptime(last_modified[0:10], '%Y-%m-%d')

    def fetch_sub_info(self) -> None:
        """fetch one at a time
        """
        if self.cik is None:
            print(f'cik not found for {self.ticker}')
            return

        sub_txt_url = self.make_sub_txt_url()
        # no 10-K found
        if sub_txt_url is None:
            return

        resp = call_api(sub_txt_url)
        if resp is not None and resp.status_code == 200:
            try:
                landing_txt = resp.content.decode('utf-8')
            except UnicodeDecodeError:
                print(f'utf-8 decode failed for url {sub_txt_url}, use latin-1')
                try:
                    landing_txt = resp.content.decode('latin-1')
                except Exception:
                    print(f'latin-1 decode failed for url {sub_txt_url}')
                    landing_txt = "N/A"
                    return
            except Exception:
                print(f'Unknown decode error failed for url {sub_txt_url}')
                landing_txt = "N/A"
                return

            # submission type
            try:
                submission_type = re.search('CONFORMED SUBMISSION TYPE:(.*)\n', landing_txt).group(1)
            except AttributeError:
                submission_type = 'N/A'

            submission_type = submission_type.strip()
            if submission_type in ['10-K']:
                try:
                    item1 = parse_10k(landing_txt)
                    if len(item1) < 100:
                        return
                except Exception as e:
                    print(f'paser {self.ticker} 10-K failed with {e}')
                    return

                # write to file
                file_path = os.path.join(self.save_dir, f'{self.ticker}.txt')
                # 'w' mode will create the file if it doesn't exist, and truncate if it does
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(item1)
        else:
            logger.warning("API call faild after max retry")

def fetch_10k_item1(tickers: List[str]) -> None:
    """fetch 10-K item 1 for multiple tickers

    Args:
        tickers (List[str]): tickers
    """
    for ticker in tickers:
        fetch_submission = FetchSubmission(ticker)
        fetch_submission.fetch_sub_info()

def main() -> None:
    """ingest company item 1
    """
    raw_data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    russell_3000 = pd.read_csv(
        os.path.join(raw_data_dir, 'raw_data', 'ticker_to_gics.csv'),
        names=['ticker', 'GICS'],
        dtype={'ticker': 'object', 'GICS': 'object'}
        )

    cik_mapping = pd.read_csv(
        os.path.join(raw_data_dir, 'raw_data', 'ticker_cik_mapping.txt'),
        sep='\t',
        names=['ticker', 'cik'],
        dtype={'ticker': 'object', 'cik': 'object'}
        )

    active_tickers = cik_mapping['ticker']
    russell_tickers = [e.lower().replace('/', '-') for e in russell_3000['ticker']]
    target_tickers = set(russell_tickers).intersection(active_tickers)

    ingested_tickers = fetch_ingested_ticker()
    tickers_to_ingest = [
        e.lower().replace('/', '-') for e in target_tickers if e.lower().replace('/', '-') not in ingested_tickers
        ]
    tickers_to_ingest = ['z']
    fetch_10k_item1(tickers_to_ingest)


if __name__ == "__main__":
    main()