def convert_to_rdf(text):
    from src.info_extractor import extract_information
    triples = extract_information(text)
    return triples

def check_consistency(triples):
    from src.reasoner import get_accuracy
    accuracy = get_accuracy(triples)
    return accuracy

if __name__ == '__main__':
    
    import argparse
    from util import *

    parser = argparse.ArgumentParser(description='knowledge-based substantiation')
    parser.add_argument('--filename', required=True, metavar='article',
                        help='an article to be checked')
    arg = parser.parse_args()
    filepath = (data_path+'/'+arg.filename)

    triples = convert_to_rdf(filepath)
    
    accuracy = check_consistency(triples)

    print(accuracy)