"""
Web Intelligence & NLP
======================

Web scraping module.

This module provides reusable functions for:
- Sending HTTP requests
- Parsing HTML
- Extracting quotes
- Handling pagination
- Scraping complete datasets
- Validating scraped data
- Saving raw data to CSV
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import Response


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

BASE_URL = "https://quotes.toscrape.com/"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

DEFAULT_TIMEOUT = 30
DEFAULT_DELAY = 1.0


# ============================================================
# HTTP REQUEST
# ============================================================

def fetch_page(
    url: str,
    headers: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Response:
    """
    Fetch a web page using an HTTP GET request.

    Parameters
    ----------
    url : str
        Target URL.

    headers : dict, optional
        HTTP request headers.

    timeout : int, default=30
        Maximum request time in seconds.

    Returns
    -------
    requests.Response
        HTTP response object.

    Raises
    ------
    requests.HTTPError
        If the server returns an unsuccessful status code.
    requests.RequestException
        If a network-related error occurs.
    """

    request_headers = headers or DEFAULT_HEADERS

    response = requests.get(
        url,
        headers=request_headers,
        timeout=timeout,
    )

    response.raise_for_status()

    return response


# ============================================================
# HTML PARSING
# ============================================================

def parse_html(html: str) -> BeautifulSoup:
    """
    Convert raw HTML into a BeautifulSoup object.

    Parameters
    ----------
    html : str
        Raw HTML content.

    Returns
    -------
    BeautifulSoup
        Parsed HTML document.
    """

    return BeautifulSoup(html, "html.parser")


# ============================================================
# QUOTE EXTRACTION
# ============================================================

def extract_quotes(
    soup: BeautifulSoup,
    source_url: str,
) -> list[dict]:
    """
    Extract quote information from a parsed Quotes to Scrape page.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed HTML document.

    source_url : str
        URL from which the quotes were extracted.

    Returns
    -------
    list[dict]
        List containing quote text, author, tags, and source URL.
    """

    records = []

    quote_elements = soup.select("div.quote")

    for quote_element in quote_elements:

        text_element = quote_element.select_one("span.text")
        author_element = quote_element.select_one("small.author")

        quote_text = (
            text_element.get_text(strip=True)
            if text_element
            else ""
        )

        author = (
            author_element.get_text(strip=True)
            if author_element
            else ""
        )

        tag_elements = quote_element.select("div.tags a.tag")

        tags = [
            tag.get_text(strip=True)
            for tag in tag_elements
        ]

        records.append(
            {
                "quote_text": quote_text,
                "author": author,
                "tags": ", ".join(tags),
                "source_url": source_url,
            }
        )

    return records


# ============================================================
# NEXT PAGE
# ============================================================

def get_next_page_url(
    soup: BeautifulSoup,
    base_url: str = BASE_URL,
) -> Optional[str]:
    """
    Find the next pagination URL.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed HTML document.

    base_url : str
        Website base URL.

    Returns
    -------
    str or None
        Absolute next-page URL, or None if no next page exists.
    """

    next_link = soup.select_one("li.next a")

    if next_link is None:
        return None

    href = next_link.get("href")

    if not href:
        return None

    return base_url.rstrip("/") + href


# ============================================================
# SINGLE PAGE SCRAPING
# ============================================================

def scrape_page(
    url: str,
    headers: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[list[dict], Optional[str]]:
    """
    Scrape one page.

    Parameters
    ----------
    url : str
        Page URL.

    headers : dict, optional
        HTTP headers.

    timeout : int, default=30
        Request timeout.

    Returns
    -------
    tuple
        Extracted records and next-page URL.
    """

    response = fetch_page(
        url=url,
        headers=headers,
        timeout=timeout,
    )

    soup = parse_html(response.text)

    records = extract_quotes(
        soup=soup,
        source_url=url,
    )

    next_url = get_next_page_url(
        soup=soup,
        base_url=BASE_URL,
    )

    return records, next_url


# ============================================================
# COMPLETE SCRAPER
# ============================================================

def scrape_quotes(
    start_url: str = BASE_URL,
    max_pages: Optional[int] = None,
    headers: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
    delay: float = DEFAULT_DELAY,
) -> pd.DataFrame:
    """
    Scrape quotes across multiple pages.

    Parameters
    ----------
    start_url : str, default=BASE_URL
        Starting page URL.

    max_pages : int or None, default=None
        Maximum number of pages to scrape.
        None means scrape until pagination ends.

    headers : dict, optional
        HTTP request headers.

    timeout : int, default=30
        HTTP request timeout.

    delay : float, default=1.0
        Delay between requests in seconds.

    Returns
    -------
    pandas.DataFrame
        Scraped quote dataset.
    """

    all_records = []

    current_url = start_url
    page_number = 1

    while current_url:

        print(
            f"Scraping page {page_number}: {current_url}"
        )

        try:

            records, next_url = scrape_page(
                url=current_url,
                headers=headers,
                timeout=timeout,
            )

        except requests.RequestException as exc:

            print(
                f"Request failed on page "
                f"{page_number}: {exc}"
            )

            break

        all_records.extend(records)

        if max_pages is not None and page_number >= max_pages:
            break

        current_url = next_url
        page_number += 1

        if current_url:
            time.sleep(delay)

    df = pd.DataFrame(all_records)

    if not df.empty:

        df.insert(
            0,
            "quote_id",
            range(1, len(df) + 1),
        )

    return df


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_scraped_data(
    df: pd.DataFrame,
) -> dict:
    """
    Validate the scraped dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Scraped dataset.

    Returns
    -------
    dict
        Validation results.
    """

    required_columns = [
        "quote_id",
        "quote_text",
        "author",
        "tags",
        "source_url",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    duplicate_rows = int(
        df.duplicated().sum()
    )

    duplicate_quotes = 0

    if "quote_text" in df.columns:
        duplicate_quotes = int(
            df["quote_text"].duplicated().sum()
        )

    missing_values = (
        df.isna()
        .sum()
        .to_dict()
    )

    empty_quotes = 0

    if "quote_text" in df.columns:
        empty_quotes = int(
            df["quote_text"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

    validation_result = {
        "is_empty": df.empty,
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_columns": missing_columns,
        "duplicate_rows": duplicate_rows,
        "duplicate_quotes": duplicate_quotes,
        "empty_quotes": empty_quotes,
        "missing_values": missing_values,
        "is_valid": (
            not df.empty
            and len(missing_columns) == 0
            and empty_quotes == 0
        ),
    }

    return validation_result


# ============================================================
# SAVE DATA
# ============================================================

def save_scraped_data(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save scraped data to CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        Scraped dataset.

    output_path : str or Path
        Destination CSV path.

    Returns
    -------
    pathlib.Path
        Saved file path.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return output_path


# ============================================================
# END-TO-END PIPELINE
# ============================================================

def run_scraping_pipeline(
    output_path: str | Path,
    start_url: str = BASE_URL,
    max_pages: Optional[int] = None,
    headers: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
    delay: float = DEFAULT_DELAY,
) -> pd.DataFrame:
    """
    Run the complete scraping workflow.

    Workflow
    --------
    URL
      ↓
    HTTP Request
      ↓
    HTML Parsing
      ↓
    Quote Extraction
      ↓
    Pagination
      ↓
    Validation
      ↓
    CSV Export

    Parameters
    ----------
    output_path : str or Path
        CSV output location.

    start_url : str
        Starting website URL.

    max_pages : int or None
        Maximum number of pages.

    headers : dict, optional
        HTTP request headers.

    timeout : int
        Request timeout.

    delay : float
        Delay between requests.

    Returns
    -------
    pandas.DataFrame
        Final scraped dataset.

    Raises
    ------
    ValueError
        If validation fails.
    """

    df = scrape_quotes(
        start_url=start_url,
        max_pages=max_pages,
        headers=headers,
        timeout=timeout,
        delay=delay,
    )

    validation = validate_scraped_data(df)

    if not validation["is_valid"]:

        raise ValueError(
            "Scraped data validation failed: "
            f"{validation}"
        )

    saved_path = save_scraped_data(
        df=df,
        output_path=output_path,
    )

    print(
        f"\\nScraping completed successfully."
    )

    print(
        f"Records collected: {len(df)}"
    )

    print(
        f"Saved to: {saved_path}"
    )

    return df


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    output_file = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "scraped_quotes.csv"
    )

    run_scraping_pipeline(
        output_path=output_file,
        start_url=BASE_URL,
        max_pages=None,
        delay=DEFAULT_DELAY,
    )