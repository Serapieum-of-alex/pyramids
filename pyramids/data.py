import json
import os
from urllib.request import urlretrieve

import requests
from loguru import logger
from requests.exceptions import HTTPError


def getExampleData(article_id: int = 19991261, directory: str = None):
    """getExampleData.

        retrieve the data required for the examples.

    Parameters
    ----------
    article_id: [int]
        the article id in figshare url link
        >>> url = "https://figshare.com/articles/dataset/Pyramids/19991261"
        >>> article_id = 19991261
    directory: [str]
        the directory where you want to save the data

    Returns
    -------
    None

    Examples
    --------
    to download the netcdf file needed for the netcdf examples
    >>> article_id = 19991261
    >>> dir = "pyramids\examples\data"
    >>> getExampleData(article_id, directory=dir)
    """
    baseurl = "https://api.figshare.com/v2"
    url = f"{baseurl}/articles/{article_id}/files"
    headers = {'Content-Type': 'application/json'}

    data=None
    # binary=False
    method = "GET"
    response = requests.request(method, url, headers=headers, data=data)
    try:
        response.raise_for_status()
        try:
            response_data = json.loads(response.text)
        except ValueError:
            response_data = response.content
    except HTTPError as error:
        print('Caught an HTTPError: {}'.format(error))
        print('Body:\n', response.text)
        raise

    response_data = response_data[0]

    if directory is None:
        directory = os.getcwd()

    # dir = os.path.join(directory, f"figshare_{article_id}/")
    # os.makedirs(dir, exist_ok=True)
    urlretrieve(response_data['download_url'], os.path.join(directory, response_data['name']))
    logger.info(f"{response_data['name']} - has been downloaded successfully")
