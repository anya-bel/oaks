import argparse

from .neural_network_approach.neural_network import RNNclassifier

NAME = 'logreg'


def start(article_url):
    print('Backend pipeline started\nURL:', article_url)
    from .ensemble import Ensembling
    from .parse_url import parse_link

    text = parse_link(article_url)

    predictor = Ensembling(name=NAME)
    prediction = predictor.predict(text)
    print(prediction)
    if NAME == 'logreg':
        results = dict(likelihood=round((prediction[1]), 2) * 100)
    else:
        results = dict(likelihood=round((1 - prediction[1].count(True) / 3), 2) * 100)
    print(results)
    return results


def main(article_url: str):
    return start(article_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply fake news detector to an article by providing its url")
    parser.add_argument("--article_url", type=str, help="link to the article")
    args = parser.parse_args()
    main(args.article_url)
