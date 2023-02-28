# =============================================================================
# Input Variables
# =============================================================================
STARTING_DOI = '10.1088/1758-5090/ac24dc'
GENERATIONS_BACK = 2

# Modules
import time
import numpy as np
import scholarly

# =============================================================================
# Code
# =============================================================================
start = time.time()

import scholarly

def get_article_data(doi, gen):
    print(f"Retrieving data for article: {doi}")
    article = scholarly.search_pubs(doi)
    article_data = []
    for pub in article:
        pub_data = {'Title': pub.bib['title'],
                    'DOI': pub.bib['doi'],
                    'Authors': pub.bib['author'],
                    'Conclusions': pub.bib['abstract']}
        if 'citedby' in pub.bib:
            pub_data['Citations'] = pub.bib['citedby']
        else:
            pub_data['Citations'] = 0
        pub_data['Generation'] = gen
        article_data.append(pub_data)
    return article_data

def collect_data(doi, num_gen):
    article_list = []
    current_list = [{'DOI': doi, 'Generation': 0}]
    article_list += current_list
    for gen in range(1, num_gen + 1):
        next_list = []
        for article in current_list:
            new_articles = scholarly.search_pubs(article['DOI'])
            for new_article in new_articles:
                new_doi = new_article.bib['doi']
                if not any(d['DOI'] == new_doi for d in article_list + next_list):
                    next_list.append({'DOI': new_doi, 'Generation': gen})
            article_data = get_article_data(article['DOI'], gen)
            article_list += article_data
        current_list = next_list
    return article_list


# Function Call
# article_data = collect_article_data(STARTING_DOI, GENERATIONS_BACK)
article_data = collect_data(STARTING_DOI, GENERATIONS_BACK)


################################### Outputs ###################################


end = time.time()
print("The total runtime of the above code was",(end-start), "seconds")
