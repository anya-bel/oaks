import requests
from bs4 import BeautifulSoup


def parse_link(url):
    session = requests.Session()
    response = session.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text)

    text = " ".join([x.text for x in soup.findAll('p')])
    return text
