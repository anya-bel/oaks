<h1 align="center">Online App for Knowledge Substantiation</h1>
<div style="text-align:center"><img width="450" src=https://github.com/e-lubrini/fake-news-detector/blob/main/img/logos/logo_g.png /></div>

## Abstract
The aim of this project is to develop a software that provides users with a probability value reflecting the likelihood of an inputted news article being fake. To achieve this, a multi-modal pipeline was implemented, by ensembling the results from 4 different modules (rule-based, neural network, cross-checking, and knowledge-base algorithm). The app is accessible to the users through a user-friendly GUI at [website-domain-pending.fr](website.com).

## Repo Structure


    ├── app/
    |   ├── backend/
    |   |   ├── cross_checking_approach/
    |   |   ├── neural_network_approach/
    |   |   ├── feature_based_approach/
    |   |   ├── knowledge_base_approach/
    |   |   ├── pipeline.py
    |   |   └── ensemble.py
    |   └── frontend/website/
    |       ├── static/           # styling files (.css, .js, ...)
    |       ├── templates/        # .html pages
    |       ├── run.sh
    |       └── app.py
    ├── cited_articles/           
    ├── presentations/            
    ├── report/                   
    ├── .gitingnore
    ├── LICENSE
    ├── docker-compose.yml
    └── README.md


## Dependencies
See [app/requirements.txt](https://raw.githubusercontent.com/e-lubrini/oaks/main/app/requirements.txt) for full dependency list with versions.

## Installation intructions
From the terminal:
- clone the repository (`git clone https://github.com/e-lubrini/oaks/`)
- `cd` to the _oaks/app/_ folder
- _(recommended)_ create and activate an environment
  (e.g. `python3 -m venv venv; source venv\bin\activate`)
- `pip install -e .`

## Usage guide
### Command line usage
The command `python3 pipeline.py [https://article_url.com]` will output the likelihood of the article being fake, in the form of a percentage. 

### Docker usage
- switch to the root directory (`/oaks`)
- run `docker compose up` in the terminal
- the website will be available at http://127.0.0.1:5000

