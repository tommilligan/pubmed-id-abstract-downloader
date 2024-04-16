#!/usr/bin/env python

from typing import Any
import sys
from bs4 import BeautifulSoup
import codecs
import csv
import logging
import math
import time
from http import HTTPStatus

from requests import Session, HTTPError
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

_log = logging.getLogger("downloader")

ABSTRACT_FIELDNAME = "abstract"


def setup_logging() -> None:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)5s %(name)s: %(message)s"))
    logger.handlers = [ch]


def _html_to_abstract(html: str) -> str | None:
    soup = BeautifulSoup(html, features="html.parser")
    abstract = soup.find("div", {"id": "abstract"})
    if abstract is None:
        return None

    return abstract.get_text()


def _fetch_pmid(session: Session, pmid: str) -> str | None:
    response = session.get(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
    try:
        response.raise_for_status()
    except HTTPError:
        _log.exception(f"Failed to fetch PMID {pmid!r}")
        return None
    return response.text


def _get_pmid_abstract(session: Session, pmid: str) -> str:
    html = _fetch_pmid(session=session, pmid=pmid)
    if html is None:
        return "Abstract download failed: could not download page"

    abstract = _html_to_abstract(html)
    if abstract is None:
        _log.warning(f"No abstract found for PMID {pmid!r}")
        return "Abstract download failed: abstract not found on page"

    return abstract


def strip_bom(line):
    return line[1:] if line.startswith(codecs.BOM_UTF8.decode()) else line


def is_power_of_two(n: int) -> bool:
    return math.log(n, 2).is_integer()


def update_row(session: Session, row: dict[str, Any]) -> dict[str, Any]:
    pmid = row["PMID"]
    abstract = _get_pmid_abstract(session=session, pmid=pmid)
    row[ABSTRACT_FIELDNAME] = abstract
    return row


def sane_retry(
    retries: int = 4,
    backoff_factor: float = 0.1,
    additional_status_forcelist: frozenset[int] = frozenset(
        (
            HTTPStatus.INTERNAL_SERVER_ERROR,
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.SERVICE_UNAVAILABLE,
            HTTPStatus.GATEWAY_TIMEOUT,
        )
    ),
    additional_allowed_methods: frozenset[str] = frozenset(),
) -> Retry:
    """A sane by default retry strategy.

    Retries can be added to more calls using the `additional_*` arguments.
    """
    allowed_methods = Retry.DEFAULT_ALLOWED_METHODS | additional_allowed_methods
    status_forcelist = Retry.RETRY_AFTER_STATUS_CODES | additional_status_forcelist

    return Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
    )


def with_http_retry(session: Session, retry: Retry) -> Session:
    """Mount the given retry on the session for HTTP(S) calls."""
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def requests_retry_session() -> Session:
    """Returns a generally useful requests session, with sane retries."""
    return with_http_retry(Session(), sane_retry())


def main() -> None:
    setup_logging()

    start = time.monotonic()
    row_count = 0
    session = requests_retry_session()

    with open(sys.argv[1], "r") as input_handle:
        with open(sys.argv[2], "w") as output_handle:
            reader = csv.DictReader(map(strip_bom, input_handle))
            assert reader.fieldnames is not None
            fieldnames = list(reader.fieldnames)
            fieldnames.append(ABSTRACT_FIELDNAME)
            writer = csv.DictWriter(
                output_handle, fieldnames=fieldnames, dialect="unix"
            )
            writer.writeheader()

            # Could parallelise here, but we get rate limited anyway
            for row_index, updated_row in enumerate(
                update_row(session=session, row=row) for row in reader
            ):
                row_number = row_index + 1
                writer.writerow(updated_row)
                if is_power_of_two(row_number):
                    _log.info(f"Processed row {row_number}")

                row_count = row_number

    end = time.monotonic()
    duration = end - start
    _log.info(f"Download complete, processed {row_count} rows")
    if row_count > 0:
        rate = duration / row_count
        _log.info(f"Process rate: {rate:.3f} s/row")


if __name__ == "__main__":
    main()
