import subprocess
import urllib.parse

txt = '''
Unicorns are real
'''

## TODO split sentences and process individually

try:
    with open('token.txt') as f:
        token = f.readline()
except:
    print('No token found!')

outputfile_name = 'data/try.ttl'

encoded_txt = urllib.parse.quote(txt)
print(encoded_txt)

bashCmd = 'curl -X GET "http://wit.istc.cnr.it/stlab-tools/fred/?text=' + encoded_txt + '&semantic-subgraph=true" -H  "accept: text/turtle" -H  "Authorization: Bearer ' + token + '" >> ' + outputfile_name

process = subprocess.Popen(bashCmd,
                            shell=True,
                            stdout=subprocess.PIPE)

## Extract classes with Sparql
#e.g.
#PREFIX dbr: <http://dbpedia.org/resource/>
#PREFIX dbo: <http://dbpedia.org/ontology/>
#SELECT ?class WHERE {
#  dbr:Horse dbo:class ?class
#}