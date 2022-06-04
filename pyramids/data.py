import os
import json
from requests.exceptions import HTTPError
from urllib.request import urlretrieve
import requests
from loguru import logger


def get_example_data(article_id: int = 19991261, directory: str = None):
    """get_example_data.

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