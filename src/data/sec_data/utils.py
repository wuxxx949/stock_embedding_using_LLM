import os
import re
import time
from datetime import datetime
from random import uniform
from typing import Callable, Dict, List, Optional

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from src.meta_data import get_meta_data


def make_url(base_url: str , comp: str) -> str:
    """generate url

    Args:
        base_url (str): base URL
        comp (str): URL component

    Returns:
        str: URL
    """
    url = base_url
    # add each component to the base url
    url = url + '/'.join(comp)

    return url


# request related call function
def retry(orig_func: Callable) -> Callable:
    """decorator function to handle bad API call
    """

    def retried_func(*args, **kwargs):
        max_tries = 10
        tries = 0
        while True:
            resp = orig_func(*args, **kwargs)
            if resp.status_code != 200 and tries < max_tries:
                print("call failed with code {}".format(resp.status_code))
                tries += 1
                continue
            break

        return resp

    return retried_func

# TODO: rewrite recursion to loop
# TODO: add logger
# TODO: add longer stop time after error for all thread
@retry
def _call_api(url, call_record, lock, verbose = False, max_per_sec = 5):
    """
    call URL under rate limit

    Arguments:
        url {str} -- URL
        call_record {list} -- [request count, starting time, if long sleep]
        lock {_thread.lock} -- a lock object from multithreading

    Keyword Arguments:
        verbose {bool} -- if print details (default: {False})
        max_per_sec {int} -- max allowed call per second (default: {8})

    Returns:
        [type] -- [description]
    """
    if call_record[2]: # sleep 300 sec if requests.get failed
        print('sleep for 300 sec after call failed\n')
        time.sleep(300)

    headers={"User-Agent": "Mozilla/5.0"}
    lock.acquire()
    call_cnt = call_record[0]
    time_last = (datetime.now() - call_record[1]).total_seconds()
    if verbose:
        print(f'{call_cnt} calls, {time_last} sec \n')
    lock.release()

    # resp = None
    # max_retry = 5
    # try_cnt = 0
    if call_cnt < max_per_sec and time_last < 1:
        try:
            lock.acquire()
            call_record[0] += 1
            lock.release()
            resp = requests.get(url, headers=headers)
            call_record[2] = False  # disable long sleep if call succeeded
            time.sleep(0.1 + uniform(0.02, 0.05))  # sleep so call evenly distributed
        except Exception as e:
            print(f'error requests.get, {e}.\n')
            return _call_api(url, call_record, lock, verbose, max_per_sec)
    elif call_cnt <= max_per_sec and time_last >= 1:
        # reset counter
        lock.acquire()
        call_record[0] = 1
        call_record[1] = datetime.now()
        lock.release()
        try:
            resp = requests.get(url, headers=headers)
            call_record[2] = False
            time.sleep(0.1 + uniform(0.01, 0.03))
        except Exception as e:
            print(f'error requests.get, {e}.\n')
            call_record[2] = True
            return _call_api(url, call_record, lock, verbose, max_per_sec)
    else: # call_cnt == max_per_sec and time_last < 1
        # sleep 0.1 sec until call quota recover
        # time.sleep(max(1 - time_last + 0.001, 0))  # add 0.001 for safe side, sleep only for 1 thread not all
        time.sleep(1)
        # start all over again after sleep
        return _call_api(url, call_record, lock, verbose, max_per_sec)

    return resp

@retry
def call_api(url: str) -> requests.models.Response:
    """
    call URL under rate limit
    """
    headers={"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    time.sleep(0.1)

    return resp

def fetch_10k_url_from_rss(cik: str) -> Optional[str]:
    """fetch most recent 10-k submission for a CIK

    Args:
        cik (str): Central Index Key

    Returns:
        Optional[str]: URL for 10-K or None if not found
    """
    rss_url = f'https://data.sec.gov/rss?cik={cik}&type=10-K&count=40'
    response = call_api(rss_url)
    feed = feedparser.parse(response.text)

    filing_href = None
    for entry in feed.entries:
        if entry['tags'][0]['term'] == '10-K':
            filing_href = entry['filing-href'].replace('-index.htm', '.txt')
            break

    return filing_href


def load_cik_ticker_mapping(path: str) -> Dict[str, str]:
    """fetch cik to ticker mapping

    Args:
        path (str): _description_

    Returns:
        Dict[str, str]: cik as key and ticker as value
    """
    cik_ticker_mapping = pd.read_csv(
        path,
        sep='\t',
        names=['ticker', 'cik'],
        dtype={'cik': 'object', 'ticker': 'object'}
        )

    cik = cik_ticker_mapping['cik']
    ticker = cik_ticker_mapping['ticker']

    return dict(zip(cik, ticker))


def load_ticker_cik_mapping(path: str) -> Dict[str, str]:
    """fetch ticker to cik mapping

    Args:
        path (str): _description_

    Returns:
        Dict[str, str]: cik as key and ticker as value
    """
    cik_ticker_mapping = pd.read_csv(
        path,
        sep='\t',
        names=['ticker', 'cik'],
        dtype={'cik': 'object', 'ticker': 'object'}
        )

    cik = cik_ticker_mapping['cik']
    ticker = cik_ticker_mapping['ticker']

    return dict(zip(ticker, cik))


def fetch_ingested_ticker() -> List[str]:
    """fetch ingested tickers

    Returns:
        List[str]: tickers
    """
    save_dir = get_meta_data()['SEC_DIR']
    files = os.listdir(save_dir)
    tickers = [e.replace('.txt', '') for e in files]

    return tickers


def fetch_10k(raw_content: str) -> str:
    """fetch 10-k section

    Args:
        raw_10k (str): raw text from EDGAR

    Returns:
        str: 10-k raw string
    """
    # Regex to find <DOCUMENT> tags
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')
    # Regex to find <TYPE> tag prceeding any characters, terminating at new line
    type_pattern = re.compile(r'<TYPE>[^\n]+')

    doc_start_is = [x.end() for x in doc_start_pattern.finditer(raw_content)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(raw_content)]

    doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(raw_content)]

    # Create a loop to go through each section type and save only the 10-K section in the dictionary
    for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
        if doc_type == '10-K':
            # content = BeautifulSoup(raw_10k[doc_start: doc_end], 'lxml').get_text(' ')
            content = raw_content[doc_start: doc_end]
            break

    return content


def _parse_10k(raw_10k: str) -> str:
    """fetch Item 1. in 10K submission
        reference: https://gist.github.com/anshoomehra/ead8925ea291e233a5aa2dcaa2dc61b2

    Args:
        raw_10k (str): raw strings
    """
    # Regex to find <DOCUMENT> tags
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')
    # Regex to find <TYPE> tag prceeding any characters, terminating at new line
    type_pattern = re.compile(r'<TYPE>[^\n]+')

    doc_start_is = [x.end() for x in doc_start_pattern.finditer(raw_10k)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(raw_10k)]

    doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(raw_10k)]

    document = {}
    # Create a loop to go through each section type and save only the 10-K section in the dictionary
    for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
        if doc_type == '10-K':
            content = BeautifulSoup(raw_10k[doc_start: doc_end], 'lxml').get_text(' ')
            # rm forward-looking statements potentially in Item 1.
            # document['10-K'] = re.sub(r'Forward-Looking Statements.*?obligation', '', content, count=1, flags=re.IGNORECASE)
            document['10-K'] = content
    # regex = re.compile(r'(>Item(\s|&#160;|&nbsp;)(1|1A)\.{1,1})|(ITEM\s(1|1A))')
    # regex = re.compile(r'(>Item[\s&#160;&nbsp;#xA0;#xa0;]+(1|1A)\.)|(>ITEM[\s&#160;&nbsp;#xA0;#xa0;]+(1|1A)\.)')
    regex = re.compile(r'(?<!“)(i\s?t\s?e\s?m[\s&#160;&nbsp;#xA0;#xa0;]+([1\.A]+)(\.|:|#|&|—))(?!\s“)', re.IGNORECASE)
    # Use finditer to math the regex
    matches = regex.finditer(document['10-K'])
    pos_dat = pd.DataFrame([(x.group(), x.start(), x.end()) for x in matches])

    pos_dat.columns = ['item', 'start', 'end']
    pos_dat['item'] = pos_dat.item.str.lower()

    # Get rid of unnesesary charcters from the dataframe
    pos_dat = pos_dat \
        .replace(['&nbsp;', '&#160;', '&#xa0;'], '', regex=True) \
        .replace('[^a-zA-Z0-9]', '', regex=True) \
        .replace('\.', '', regex=True) \
        .sort_values('start', ascending=True)

    # max item1a in case item 1 mentioned later
    # max_pos = pos_dat.loc[pos_dat['item']=='item1a'].iat[1, 1]
    max_pos = pos_dat.loc[pos_dat['item']=='item1a']['start'].max()

    pos_dat = pos_dat \
        .loc[pos_dat['start'] <= max_pos, :] \
        .drop_duplicates(subset=['item'], keep='last') \
        .set_index('item') \
        .loc[['item1', 'item1a']]

    # print(document['10-K'][(pos_dat['start'].loc['item1']-10): (pos_dat['start'].loc['item1'] + 30)])
    item_1_content = document['10-K'][pos_dat['start'].loc['item1']:pos_dat['start'].loc['item1a']]

    # Apply BeautifulSoup to refine the content
    # item_1_content = BeautifulSoup(item_1_raw, 'lxml').get_text(' ')
    item_1_content = re.sub(r'[^a-zA-Z0-9_\s,\.;:-]', '', item_1_content)

    # rm newline
    item_1_content = re.sub('[\n\xa0]', ' ', item_1_content)

    # rm additional whitespace
    item_1_content = re.sub(r' +', ' ', item_1_content)

    # rm repeated words
    item_1_content = re.sub(r'\b(\w+)\s+\1\b', r'\1', item_1_content)

    # rm item 1. business pattern
    item_1_content = ' '.join(item_1_content.split(' ')[3:])

    return item_1_content


def parse_10k_str_match(raw_10k: str) -> Optional[pd.DataFrame]:
    """find item 1 and item 1a position by string match

    Args:
        raw_10k (str): raw text from EDGAR

    Returns:
        Optional[pd.DataFrame]: df for item position
    """
    raw_10k = BeautifulSoup(raw_10k, 'lxml').get_text(' ')
    regex = re.compile(r'(?<!“)(i\s?t\s?e\s?m[\s&#160;&nbsp;#xA0;#xa0;]+([1\.A]+)(\.|:|#|&|—))(?!\s“)', re.IGNORECASE)
    # Use finditer to math the regex
    matches = regex.finditer(raw_10k)
    pos_dat = pd.DataFrame([(x.group(), x.start(), x.end()) for x in matches])

    pos_dat.columns = ['item', 'start', 'end']
    pos_dat['item'] = pos_dat['item'].str.lower()

    # Get rid of unnesesary charcters from the dataframe
    pos_dat = pos_dat \
        .replace(['&nbsp;', '&#160;', '&#xa0;'], '', regex=True) \
        .replace('[^a-zA-Z0-9]', '', regex=True) \
        .replace('\.', '', regex=True) \
        .sort_values('start', ascending=True)

    # max item1a in case item 1 mentioned later
    max_pos = pos_dat.loc[pos_dat['item']=='item1a']['start'].max()

    pos_dat = pos_dat \
        .loc[pos_dat['start'] <= max_pos, :] \
        .drop_duplicates(subset=['item'], keep='last') \
        .set_index('item') \
        .loc[['item1', 'item1a']]

    return pos_dat


def parse_10k_href(raw_10k: str) -> Optional[pd.DataFrame]:
    """find item 1 and item 1a position by search for href link

    Args:
        raw_10k (str): raw text from EDGAR

    Returns:
        Optional[pd.DataFrame]: df for item position
    """
    regex = re.compile(r'href="#(.*?)"', re.IGNORECASE)
    matches = regex.finditer(raw_10k)
    hrefs = set(['"' + raw_10k[(x.start() + 7): x.end()] for x in matches])

    item_1_patten = r'i\s?t\s?e\s?m\s?1\s?b\s?u\s?s\s?i\s?n\s?e\s?s\s?s'
    item_1a_patten = r'i\s?t\s?e\s?m\s?1\s?a\s?r\s?i\s?s\s?k'
    toc_pattern = 'table of contents'
    pattern = r'<[^>]*>([^<>]+)<[^>]*>'

    pos_lst = []
    for h in hrefs:
        regex = re.compile(h, re.IGNORECASE)
        match = regex.finditer(raw_10k)
        try:
            hloc = [(x.group(), x.end()) for x in match][0][1]
        except IndexError:
            continue
        # text = BeautifulSoup(raw_10k[(hloc - 500): (hloc + 500)]).get_text(' ')
        match = re.search(pattern, raw_10k[(hloc + 1): (hloc + 1000)])
        if match is None:
            continue
        text = match.group(1)
        text = re.sub(r'[^a-zA-Z0-9_ ]', ' ', text).lower()
        # rm additional whitespace
        text = re.sub(r' +', ' ', text)
        if len(re.findall(r'b\s?u\s?s\s?i\s?n\s?e\s?s\s?s', text)) > 0 or len(re.findall(r'i\s?t\s?e\s?m\s?1\s?\b', text)) > 0:
            pos_lst.append(('item1', hloc + 1))
        if len(re.findall(r'r\s?i\s?s\s?k\s?f\s?a\s?c\s?t\s?o\s?r\s?', text))> 0 or len(re.findall(r'i\s?t\s?e\s?m\s?1\s?a', text)) > 0:
            pos_lst.append(('item1a', hloc + 1))
        # if len(re.findall(item_1_patten, text)) > 0 and len(re.findall(item_1a_patten, text)) == 0 and len(re.findall(toc_pattern, text)) == 0:
        #     pos_lst.append(('item1', hloc + 1))
        # if len(re.findall(item_1a_patten, text)) > 0 and len(re.findall(item_1_patten, text)) == 0 and len(re.findall(toc_pattern, text)) == 0:
        #     pos_lst.append(('item1a', hloc + 1))
    if len(pos_lst) == 0:
        return None

    pos_dat = pd.DataFrame(pos_lst)
    pos_dat.columns = ['item', 'start']
    max_pos = pos_dat.loc[pos_dat['item']=='item1a']['start'].min()
    pos_dat = pos_dat \
        .loc[pos_dat['start'] <= max_pos, :] \
        .sort_values('start') \
        .drop_duplicates(subset=['item'], keep='first') \
        .set_index('item')

    if len(pos_dat) < 2:
        return None

    return pos_dat


def parse_10k(raw_content: str) -> str:
    """fetch Item 1. in 10K submission
        reference: https://gist.github.com/anshoomehra/ead8925ea291e233a5aa2dcaa2dc61b2

    Args:
        raw_10k (str): raw strings
    """
    raw_10k = fetch_10k(raw_content)

    # try href first
    pos_dat = parse_10k_href(raw_10k)
    if pos_dat is None:
        pos_dat = parse_10k_str_match(raw_10k)
        raw_10k = BeautifulSoup(raw_10k, 'lxml').get_text(' ')

    # print(document['10-K'][(pos_dat['start'].loc['item1']-10): (pos_dat['start'].loc['item1'] + 30)])
    item_1_content = raw_10k[pos_dat['start'].loc['item1'] - 1 :pos_dat['start'].loc['item1a']]

    # Apply BeautifulSoup to refine the content
    item_1_content = BeautifulSoup(item_1_content, 'lxml').get_text(' ')
    item_1_content = re.sub(r'[^a-zA-Z0-9_\s,\.;:-]', '', item_1_content)

    # rm newline
    item_1_content = re.sub('[\n\xa0]', ' ', item_1_content)

    # rm additional whitespace
    item_1_content = re.sub(r' +', ' ', item_1_content)

    # rm repeated words
    item_1_content = re.sub(r'\b(\w+)\s+\1\b', r'\1', item_1_content)

    # rm item 1. business pattern
    item_1_content = ' '.join(item_1_content.split(' ')[3:])

    return item_1_content