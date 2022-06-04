from owlready2 import *
import owlready2
from rdflib import Graph, URIRef
from rdflib.namespace import RDFS, SKOS
from util import *
import subprocess
import os.path

## Download and unpack reference ontology
if os.path.isfile(data_path+'/dbpedia.owl') == False:
    print ('no .owl found')
    command = '!wget "http://dief.tools.dbpedia.org/server/ontology/dbpedia.owl" -P $data_path'
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

onto = get_ontology(data_path+"/dbpedia.owl")#.load()

archive_name = data_path+'/instance-types_lang=en_transitive.ttl.bz2'
unpacked_name = data_path+'/instance-types_lang=en_transitive.ttl'

if os.path.isfile(unpacked_name) == False:
    print ('No .ttl found.')
    if os.path.isfile(archive_name) == False:
        # download entities
        print ('downloding archive')
        command = '!wget "https://databus.dbpedia.org/dbpedia/mappings/instance-types/2021.09.01/instance-types_lang=en_transitive.ttl.bz2" -P $data_path'
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    # unpack
    print ('Extracting the .ttl...')
    command = '!bzip2 -d $data_path/instance-types_lang=en_transitive.ttl.bz2'
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

## Download sample RDF (TODO: replace with Text2Rdf)
if os.path.isfile(data_path+'/RDF-Collection.ttl') == False:
    command = '!wget "http://www.iro.umontreal.ca/~lapalme/ift6282/RDF/RDF-Collection.ttl" -P $data_path'
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

## Load (TODO: subsection) of reference KB
#graph = Graph()
#graph.parse(data_path+'/instance-types_lang=en_transitive.ttl', format='ttl')

def get_accuracy(triples):
    ## Merge ontologies
    with onto:
        sync_reasoner_pellet(
            infer_property_values=True, 
            infer_data_property_values = True,
            debug = 0
            )

    try: # catching an inconsistency exception
        with onto:
            sync_reasoner()
    except OwlReadyInconsistentOntologyError:
        print("="*30)
        print("there's at least one inconsistency")
        print("="*30)

    ## check for inconsistent classes
    #print(list(onto.inconsistent_classes()))
    return(len(list(onto.inconsistent_classes())))
