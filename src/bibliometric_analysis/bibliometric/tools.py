from os import makedirs
from os.path import abspath, dirname, join, isfile, exists
from collections import Counter
from typing import Literal, Optional

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns
import transformers
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from nltk.corpus import stopwords

import cleaner
import helper

BASE_PATH = abspath(__file__)
ROOT_PATH = dirname(dirname(dirname(dirname(BASE_PATH))))
SAVE_PATH = join(ROOT_PATH, 'saved_files')
if not exists(SAVE_PATH):
    makedirs(SAVE_PATH)
IMAGE_PATH = join(ROOT_PATH, 'images')
if not exists(IMAGE_PATH):
    makedirs(IMAGE_PATH)
MODEL_PATH = join(ROOT_PATH, 'models')
if not exists(MODEL_PATH):
    makedirs(MODEL_PATH)
STOP_WORDS = list(set(stopwords.words('english')))


def get_probe(df: pd.DataFrame) -> None:
    """
    Check basic metadata of a journal.
    Parameters:
        df: Database of a journal.
    """
    docs = len(df)
    dup = len(df.duplicated()[df.duplicated() == True])
    print(f'A Total of {docs} Documents were Found: {docs} Documents and {dup} Duplicates.')
    print('--------------------------------------', end='\n')
    for journal in df['Source title'].unique():
        part = df[df['Source title'] == journal]
        print('Source: '+str(*part['Source'].unique()), sep=',')
        print('Name of Journal: '+journal)
        print('Year of Publication: '+str(min(part['Year']))+'-'+str(max(part['Year'])))
        print('Number of Documents: '+str(len(part['Document Type'])))
        print(part['Document Type'].value_counts())
        print('--------------------------------------', end='\n')


def get_info_bib(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check necessary entries of database of a journal.
    Parameters:
        df: Database of a journal.
    Returns:
        target_df: Necessary entries of database for anlaysis.
    """
    target_df = pd.DataFrame()
    total = len(df)
    target_df['Entries'] = ['Title', 'Authors', 'Affiliations', 'Author Keywords',
                            'Index Keywords', 'Abstract', 'References', 'Year', 'Open Access', 'DOI']
    for entry in target_df['Entries']:
        if entry not in df.columns:
            df[entry] = None
    target_df['Completeness'] = [
        f'{df['Title'].notnull().sum()/total:.2%}',
        f'{df['Authors'].notnull().sum()/total:.2%}',
        f'{df['Affiliations'].notnull().sum()/total:.2%}',
        f'{df['Author Keywords'].notnull().sum()/total:.2%}',
        f'{df['Index Keywords'].notnull().sum()/total:.2%}',
        f'{df['Abstract'][df['Abstract']!='[No abstract available]'].notnull().sum()/total:.2%}',
        f'{df['References'].notnull().sum()/total:.2%}',
        f'{df['Year'].notnull().sum()/total:.2%}',
        f'{df['Open Access'].notnull().sum()/total:.2%}',
        f'{df['DOI'].notnull().sum()/total:.2%}'
    ]
    target_df['Number of Docs'] = [
        df['Title'].notnull().sum(),
        df['Authors'].notnull().sum(),
        df['Affiliations'].notnull().sum(),
        df['Author Keywords'].notnull().sum(),
        df['Index Keywords'].notnull().sum(),
        df['Abstract'][df['Abstract']!='[No abstract available]'].notnull().sum(),
        df['References'].notnull().sum(),
        df['Year'].notnull().sum(),
        df['Open Access'].notnull().sum(),
        df['DOI'].notnull().sum()
    ]
    return target_df


def plot_docs(df: pd.DataFrame, journal_num: Optional[int]=None) -> None:
    """
    Display annual trend of numbers of documents of a journal for the entire period.
    Parameters:
        df: Database of a journal.
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    if len(df['Source title'].unique()) <= len(colors):
        fig, ax= plt.subplots(figsize=(15,4))
        for num, journal in enumerate(df['Source title'].unique()):
            part = df[df['Source title'] == journal]
            year_list = part['Year'].to_list()
            year_dict = dict(Counter(year_list))
            cnt_year_list = sorted(year_dict.items(), key=lambda x : x[0], reverse=False)
            year_list = [i[0] for i in cnt_year_list]
            value_list = [i[-1] for i in cnt_year_list]
            year_df = pd.DataFrame({'Year':year_list, 'Count':value_list})
            ax.plot(year_df['Year'], year_df['Count'], color=colors[num], marker='o', label=journal)
            a, b = np.polyfit(year_df['Year'], year_df['Count'], 1)
            ax.plot(year_df['Year'], a*year_df['Year']+b, color=colors[num], linestyle='--')
            ax.legend(loc='upper left')
        xmin=min(df['Year'].unique())
        xmax=max(df['Year'].unique())
        plt.xlim([xmin,xmax])
        plt.xticks(df['Year'].unique(), rotation=45)
        plt.xlabel('Year of Publications')
        plt.ylabel('Number of Publications')
        plt.grid(True)
        plt.show()
    else:
        print("There are too many journals. The maximum number of journals that can be visualized is 10.")
        print("Input the number of journals in parameter journal_num.")
        print(f"The top {journal_num} journals with the most papers are visualized.")
        counts = df['Source title'].value_counts()
        target_list = counts.nlargest(journal_num).keys()
        target_df = df[df['Source title'].isin(target_list)]
        fig, ax= plt.subplots(figsize=(20,6))
        for num, journal in enumerate(target_df['Source title'].unique()):
            part = df[df['Source title'] == journal]
            year_list = part['Year'].to_list()
            year_dict = dict(Counter(year_list))
            cnt_year_list = sorted(year_dict.items(), key=lambda x : x[0], reverse=False)
            year_list = [i[0] for i in cnt_year_list]
            value_list = [i[-1] for i in cnt_year_list]
            year_df = pd.DataFrame({'Year':year_list, 'Count':value_list})
            ax.plot(year_df['Year'], year_df['Count'], color=colors[num], marker='o', label=journal)
            a, b = np.polyfit(year_df['Year'], year_df['Count'], 1)
            ax.plot(year_df['Year'], a*year_df['Year']+b, color=colors[num], linestyle='--')
            ax.legend(loc='upper left')
        xmin=min(df['Year'].unique())
        xmax=max(df['Year'].unique())
        plt.xlim([xmin,xmax])
        plt.xticks(df['Year'].unique(), rotation=45)
        plt.xlabel('Year of Publications')
        plt.ylabel('Number of Publications')
        plt.grid(True)
        plt.show()


def plot_docs_type(df: pd.DataFrame) -> None:
    """
    Display annual trend of document types of a journal for the entire period.
    Parameters:
        df: Database of a journal.
    """
    if len(df['Source title'].unique()) == 1:
        journal = str(*df['Source title'].unique())
    else:
        journal = 'Annual Trend of Types of Document Types'
    df.groupby(['Year', 'Document Type']).size().unstack().plot(
        kind='bar',
        figsize=(15,5),
        stacked=True,
        title=journal,
        xlabel='Year of Publication',
        ylabel='Number of Publications')
    plt.show()


def plot_open_access(df: pd.DataFrame) -> None:
    """
    Display annual trend of types of open access of a journal for the entire period.
    Parameters:
        df: Database of a journal.
    """
    if len(df['Source title'].unique()) == 1:
        journal = str(*df['Source title'].unique())
    else:
        journal = 'Annual Trend of Types of Open Access'
    result = df.fillna(value={'Open Access':'Non-Open Access'})
    target_df = result[['Open Access', 'Year']]
    target_df.groupby(['Year', 'Open Access']).size().unstack().plot(
        kind='bar',
        figsize=(15,4),
        stacked=True,
        title=journal,
        xlabel='Year of Publication',
        ylabel='Number of Publications'
        )
    plt.legend(loc='best', fontsize='small')


def get_highly_cited_authors(df: pd.DataFrame, top_num: int = 10,
                             min_year: Optional[int] = None, max_year: Optional[int] = None) -> pd.DataFrame:
    """
    Display highly cited authors.
    Parameters:
        df: Database of a journal.
        top_num: Number of high-ranking authors highly cited.
                 If any argument isn't inputted in top_num, the argument "10" is automatically inputted in top_num.
        min_year: Minimum year. If the min_year is "None", the lowest year of database is automatically inputted.
        max_year: Maximum year. If the max_year is "None", the highest year of database is automatically inputted.
    """
    min_year, max_year = helper.get_min_max_year(df, min_year, max_year)
    result = cleaner.clean_authors(df)
    df['Clened author full names'] = result
    target_df = df[['Source title','Clened author full names', 'Title', 'Cited by', 'Year']]
    target_df.columns = ['Journal', 'Author full names', 'Title', 'Cited by', 'Year']
    target_df = target_df[(min_year <= target_df['Year'])&(target_df['Year'] <= max_year)]
    target_df['Period'] = f'{min_year}-{max_year}'
    target_df = target_df[['Journal', 'Period', 'Author full names', 'Title', 'Cited by', 'Year']]
    target_df = target_df.sort_values(by='Cited by', ascending=False)[:top_num]
    return target_df


def plot_top_(df: pd.DataFrame,
              category: Literal['author', 'institution', 'country', 'keyword', 'author keyword', 'index keyword', 'reference'],
              top_num: int = 10, min_year: Optional[int] = None, max_year: Optional[int] = None) -> None:
    """
    Display high-ranking objects for the entire period.
    Parameters:
        df: Database of a journal.
        category: A type of objects (author, institution, country, keyword, author keyword, index keyword and reference).
        top_num: Number of high-ranking objects.
                 If any argument isn't inputted in top_num, the argument "10" is automatically inputted in top_num.
        min_year: Minimum year. If the min_year is "None", the lowest year of the database is automatically inputted.
        max_year: Maximum year. If the max_year is "None", the highest year of the database is automatically inputted.
    """
    min_year, max_year = helper.get_min_max_year(df, min_year, max_year)
    target_df = df[(min_year <= df['Year'])&(df['Year'] <= max_year)]
    result = cleaner.clean_(target_df, category)
    target = Counter(result).most_common(top_num)
    if category == 'author':
        ylabel = 'Authors'
    elif category == 'institution':
        ylabel = 'Institutions'
    elif category == 'country':
        ylabel = 'Countries'
    elif category == 'keyword':
        ylabel = 'Keywords'
    elif category == 'author keyword':
        ylabel = 'Author Keywords'
    elif category == 'index keyword':
        ylabel = 'Index Keywords'
    elif category == 'reference':
        ylabel = 'References'
    if len(df['Source title'].unique()) == 1:
        journal = str(*df['Source title'].unique())
    else:
        journal = f'High-Ranking {ylabel}'
    keys = [k for k, _ in target]
    values = [v for _, v in target]
    plt.barh(keys, values)
    plt.title(f'{journal} {min_year}-{max_year}')
    plt.xlabel('Count')
    plt.ylabel(ylabel)
    plt.show()
    print(f'{journal} {min_year}-{max_year}')
    print(tabulate(target))


def get_undirected_graph_centrality_(df: pd.DataFrame,
                                    category: Literal['author', 'institution', 'country', 'keyword', 'author keyword', 'index keyword', 'reference'],
                                    centrality: Literal['degree', 'betweenness', 'closeness', 'eigenvector'],
                                    top_num: int = 10, min_year: Optional[int] = None, max_year: Optional[int] = None) -> pd.DataFrame:
    """
    Display high-ranking objects with high centrality for the period.
    Parameters:
        df: Database of a journal.
        category: A Type of objects (author, institution, country, keyword, author keyword, index keyword and reference).
        centrality: A type of centralities in undirected graph (Degree, Betweenness, Closeness and Eigenvector Centrality).
        top_num: Number of high-ranking objects.
                 If any argument isn't inputted in top_num, the argument "10" is automatically inputted in top_num.
        min_year: Minimum year. If the min_year is "None", the lowest year of the database is automatically inputted.
        max_year: Maximum year. If the max_year is "None", the highest year of the database is automatically inputted.
    Returns:
        co_df: A database that stores the centrality of concurrent network.
    Raises:
        PowerIterationFailedConvergence: If there are not enough or too many number of nodes and edges, the error occurs.
    """
    try:
        min_year, max_year = helper.get_min_max_year(df, min_year, max_year)
        target_df = df[(min_year <= df['Year'])&(df['Year'] <= max_year)]
        co_df = helper.get_co_(target_df, category, min_year, max_year)
        dgr, btw, cls, egv = helper.get_centrality(co_df)
        if category == 'author':
            column = 'Authors'
        elif category == 'institution':
            column = 'Institutions'
        elif category == 'country':
            column = 'Countries'
        elif category == 'keyword':
            column = 'Keywords'
        elif category == 'author keyword':
            column = 'Author Keywords'
        elif category == 'index keyword':
            column = 'Index Keywords'
        elif category == 'reference':
            column = 'References'
        if len(df['Source title'].unique()) == 1:
            journal = str(*df['Source title'].unique()).lower().replace(' ', '_')
        else:
            journal = f'High-Ranking {column}'
        if centrality == 'degree':
            names = [k for k, _ in dgr.items()]
            values = [v for _, v in dgr.items()]
            df = pd.DataFrame({'Journal':journal, 'Period':f'{min_year}-{max_year}', column:names,'Degree Centrality':values})
            if isfile(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_{category}_{centrality}.csv')):
                print(f'{journal}_{min_year}_{max_year}_{category}_{centrality}.csv is already created.')
                co_df = df.sort_values(by='Degree Centrality', ascending=False)[:top_num]
            else:
                df.to_csv(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_{category}_{centrality}.csv'), sep=';', index=False)
                co_df = df.sort_values(by='Degree Centrality', ascending=False)[:top_num]
        elif centrality == 'betweenness':
            names = [k for k, _ in btw.items()]
            values = [v for _, v in btw.items()]
            df = pd.DataFrame({'Journal':journal, 'Period':f'{min_year}-{max_year}', column:names,'Betweenness Centrality':values})
            if isfile(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_{category}_{centrality}.csv')):
                print(f'{journal}_{min_year}_{max_year}_{category}_{centrality}.csv is already created.')
                co_df = df.sort_values(by='Betweenness Centrality', ascending=False)[:top_num]
            else:
                df.to_csv(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_{category}_{centrality}.csv'), sep=';', index=False)
                co_df = df.sort_values(by='Betweenness Centrality', ascending=False)[:top_num]
        elif centrality == 'closeness':
            names = [k for k, _ in cls.items()]
            values = [v for _, v in cls.items()]
            df = pd.DataFrame({'Journal':journal, 'Period':f'{min_year}-{max_year}', column:names,'Closeness Centrality':values})
            if isfile(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_{category}_{centrality}.csv')):
                print(f'{journal}_{min_year}_{max_year}_{category}_{centrality}.csv is already created.')
                co_df = df.sort_values(by='Closeness Centrality', ascending=False)[:top_num]
            else:
                df.to_csv(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_{category}_{centrality}.csv'), sep=';', index=False)
                co_df = df.sort_values(by='Closeness Centrality', ascending=False)[:top_num]
        elif centrality == 'eigenvector':
            names = [k for k, _ in egv.items()]
            values = [v for _, v in egv.items()]
            df = pd.DataFrame({'Journal':journal, 'Period':f'{min_year}-{max_year}', column:names,'Eigenvector Centrality':values})
            if isfile(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_{category}_{centrality}.csv')):
                print(f'{journal}_{min_year}_{max_year}_{category}_{centrality}.csv is already created.')
                co_df = df.sort_values(by='Eigenvector Centrality', ascending=False)[:top_num]
            else:
                df.to_csv(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_{category}_{centrality}.csv'), sep=';', index=False)
                co_df = df.sort_values(by='Eigenvector Centrality', ascending=False)[:top_num]
        return co_df
    except:
        print('PowerIterationFailedConvergence')
        print('The insufficient or superabundant number of nodes and edges makes meaningful network analysis impossible.')

    
def plot_keywords(df: pd.DataFrame, category: Literal['keyword', 'author keyword', 'index keyword'],
                  top_num: int = 10, min_year: Optional[int] = None, max_year: Optional[int] = None
                  ) -> None:
    """
    Display annual trend of high-ranking keywords by year for the entire period.
    Parameters:
        df: Database of a journal.
        category: A Type of objects (keyword, author keyword, and index keyword).
        top_num: Number of high-ranking objects by year.
                 If any argument isn't inputted in top_num, the argument "10" is automatically inputted in top_num.
        min_year: Minimum year. If the min_year is "None", the lowest year of the database is automatically inputted.
        max_year: Maximum year. If the max_year is "None", the highest year of the database is automatically inputted.
    Returns:
        helfer.get_top_keywords(target_df, merged_df): Database for top keywords with a minimum value of 2 or more by year.
    """
    min_year, max_year = helper.get_min_max_year(df, min_year, max_year)
    target_df = df[(min_year <= df['Year'])&(df['Year'] <= max_year)]
    result = cleaner.clean_keywords(target_df, category)
    if category != 'keyword':
        target_df['Keywords'] = result
    else:
        target_df = result
    target_df = target_df.explode('Keywords', ignore_index=True)
    if category == 'keyword':
        target_df.dropna(subset=['Keywords'], inplace=True)
    elif category == 'author keyword':
        target_df.dropna(subset=['Author Keywords'], inplace=True)
    else: target_df.dropna(subset=['Index Keywords'], inplace=True)
    years = sorted(target_df['Year'].unique())
    merged_df = pd.DataFrame()
    for year in years:
        year_df = target_df[['Year', 'Keywords']][target_df['Year'] == year]
        keywords_list = year_df['Keywords'].to_list()
        keywords = Counter(keywords_list)
        top_keywords = [k for k, v in keywords.items() if v > 1]
        if top_keywords:
            top_df = year_df.loc[year_df['Keywords'].isin(top_keywords[:top_num])]
            others_df = year_df.loc[~year_df['Keywords'].isin(top_keywords[:top_num])]
            others_df.loc[:,'Keywords'] = ''
            temp = pd.concat([top_df, others_df], ignore_index=True)
            merged_df = pd.concat([merged_df, temp], ignore_index=True)
        else:
            year_df.loc[:,'Keywords'] = ''
            merged_df = pd.concat([merged_df, year_df], ignore_index=True)
    ct = pd.crosstab(merged_df['Year'], merged_df['Keywords'])
    cols = ct.columns
    ax = ct.plot(
        kind='bar',
        stacked=True,
        figsize=(15,7),
        legend=False,
        rot=90,
        xlabel='Year of Publication',
        ylabel='Number of Keywords')
    for i, c in enumerate(ax.containers):
        labels = [f'{cols[i]}' if (h := v.get_height()) > 0 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=4)
    return helper.get_top_keywords(merged_df)


def get_directed_graph_centrality_(df: pd.DataFrame,
                                   centrality: Literal['katz', 'pagerank'],
                                   top_num: int = 10, min_year: Optional[int] = None, max_year :Optional[int] = None) -> pd.DataFrame:
    """
    Display high-ranking objects with high centrality for the period.
    Parameters:
        df: Database of a journal.
        centrality: A type of centralities in directed graph (Katz and PageRank Centrality).
        top_num: Number of high-ranking objects.
                 If any argument isn't inputted in top_num, the argument "10" is automatically inputted in top_num.
        min_year: Minimum year. If the min_year is "None", the lowest year of the database is automatically inputted.
        max_year: Maximum year. If the max_year is "None", the highest year of the database is automatically inputted.
    Returns:
        co_df: A database that stores the centrality of concurrent network.
    Raises:
        PowerIterationFailedConvergence: There are not enough nodes and edges.
    """
    try:
        min_year, max_year = helper.get_min_max_year(df, min_year, max_year)
        if len(df['Source title'].unique()) == 1:
            journal = str(*df['Source title'].unique())
        else:
            journal = 'High-Ranking References'
        target_df = pd.DataFrame()
        target_df['source'] = df['Title']
        target_df['target'] = cleaner.clean_omissions_references(df)
        target_df['year'] = df['Year']
        target_df = target_df[(min_year <= target_df.year)&(target_df.year <= max_year)]
        target_df = target_df.explode('target', ignore_index=True)
        target_df = target_df[(target_df['target'] != 'nan') & (target_df['target'].str.len() > 1)]
        g = nx.from_pandas_edgelist(df=target_df, source='source', target='target', create_using=nx.DiGraph())
        if centrality == 'pagerank':
            dict = nx.pagerank(G=g, alpha=0.85)
            keys = [k for k, _ in dict.items()]
            values = [v for _, v in dict.items()]
            df = pd.DataFrame({'Journal':journal, 'Period':f'{min_year}-{max_year}', 'Reference':keys, 'PageRank Centrality':values})
            if isfile(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_{centrality}.csv')):
                print(f'{journal}_{min_year}_{max_year}_{centrality}.csv is already created.')
                co_df = df.sort_values(by='PageRank Centrality', ascending=False)[:top_num]
            else:
                df.to_csv(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_{centrality}.csv'), sep=';', index=False)
                co_df = df.sort_values(by='PageRank Centrality', ascending=False)[:top_num]
        elif centrality == 'katz':
            dict = nx.katz_centrality(G=g)
            keys = [k for k, _ in dict.items()]
            values = [v for _, v in dict.items()]
            df = pd.DataFrame({'Journal':journal, 'Period':f'{min_year}-{max_year}', 'Reference':keys, 'Katz Centrality':values})
            if isfile(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_{centrality}.csv')):
                print(f'{journal}_{min_year}_{max_year}_{centrality}.csv is already created.')
                co_df = df.sort_values(by='Katz Centrality', ascending=False)[:top_num]
            else:
                df.to_csv(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_{centrality}.csv'), sep=';', index=False)
                co_df = df.sort_values(by='Katz Centrality', ascending=False)[:top_num]
        return co_df
    except:
        print('PowerIterationFailedConvergence')
        print('The insufficient number of nodes and edges makes meaningful network analysis impossible.')


def plot_references(df: pd.DataFrame) -> None:
    """
    Display annual trend of average numbers of references per publication of a journal for the entire period.
    Parameters:
        df: Database of a journal.
    """
    if len(df['Source title'].unique()) == 1:
        journal = str(*df['Source title'].unique())
    else:
        journal = 'Annual Trend of Average Numbers of References per Publication'
    df['Cleaned references'] = cleaner.clean_references(df)
    temp_df = df.explode('Cleaned references')
    references_df = temp_df.groupby('Year')['Cleaned references'].count().reset_index().rename(
        columns={'Year':'Year of Publications', 'Cleaned references':'Number of Refereces'})
    num_titles = [i for i in df.groupby(['Year'])['Title'].count()]
    num_titles_df = pd.DataFrame(num_titles, columns=['Number of Titles'])
    target_df = pd.concat([references_df, num_titles_df], axis=1)
    target_df['Average Number of References per Publication'] = target_df['Number of Refereces']/target_df['Number of Titles']
    target_df['Average Number of References per Publication'] = target_df['Average Number of References per Publication'].apply(lambda x : round(x, 2))
    g = sns.barplot(data=target_df, x='Year of Publications', y='Average Number of References per Publication')
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    plt.title(f'{journal}')


def plot_top_authors_of_references(df: pd.DataFrame, delimiter: Optional[Literal[',', None]] = None,
                                   top_num: int = 10,
                                   use_merged_dataframe: Literal[False, True] = False,
                                   min_year: Optional[int] = None, max_year: Optional[int] = None) -> None:
    """
    Display high-ranking authors in references.
    Parameters:
        df: Database of a journal.
        delimiter: Delimiter to distinguish authors.
        top_num: Number of high-ranking authors in references.
                 If any argument isn't inputted in top_num, the argument "10" is automatically inputted in top_num.
        use_merged_dataframe: If you use a merged dataframe where authors are extracted individually through multiple separators, use_merged_dataframe is used. The default is "False".
        min_year: Minimum year. If the min_year is "None", the lowest year of the database is automatically inputted.
        max_year: Maximum year. If the max_year is "None", the highest year of the database is automatically inputted.
    Notes:
        * There are various notations for authors and seperations for elements of a citation accroding to referencing styles or journals.
        * For exemple, "Gwanghun P., Stilometrische Analyse ..., (2024)".
    """
    min_year, max_year = helper.get_min_max_year(df, min_year, max_year)
    if use_merged_dataframe == False:
        if len(df['Source title'].unique()) == 1:
            journal = str(*df['Source title'].unique())
        else:
            journal = 'High-Ranking Authors in References'
        target_df = df[(min_year <= df['Year'])&(df['Year'] <= max_year)]
        target_df = helper.get_author_reference(target_df, delimiter)
    else:
        target_df = df
        journal = 'High-Ranking Authors in References'

    target_df = target_df['Authors of references'][target_df['Authors of references']!='none'].value_counts().reset_index()
    target_df.columns = ['Count', 'Author']
    top_contributors = target_df[:top_num]
    plt.barh(top_contributors['Count'], top_contributors['Author'])
    plt.title(f'{journal} {min_year}-{max_year}')
    plt.xlabel('Count')
    plt.ylabel('Author')
    plt.show()
    print(tabulate(top_contributors))


def plot_age_of_citataions(df: pd.DataFrame, top_age: Optional[int] = 50,
                           min_year: Optional[int] = None, max_year: Optional[int] = None) -> pd.DataFrame:
    """
    Display ages of references by year of publications.
    Parameters:
        df: Database of a journal.
        top_age: Upper limit of difference between years of references and year of publications.
                 If the argument of top_age is negative, entire age values of references are returned.
                 If any argument isn't inputted in top_age, the argument "50" is automatically inputted in top_age.
        min_year: Minimum year. If the min_year is "None", the lowest year of the database is automatically inputted.
        max_year: Maximum year. If the max_year is "None", the highest year of the database is automatically inputted.
    Returns:
        helfer.calculate_age_of_citations(target_age_df): Database including count of citations and median, mean, and standard error of age of citations by year.
    """
    min_year, max_year = helper.get_min_max_year(df, min_year, max_year)
    if len(df['Source title'].unique()) == 1:
        journal = str(*df['Source title'].unique())
    else:
        journal = 'Ages of References by Year of Publications'
    df = helper.get_year_reference(df)
    df = df.explode('Years of references').explode('Years of references')
    df['Year'] = df['Year'].fillna(0).astype(int)
    df['Years of references'] = df['Years of references'].fillna(0).astype(int)
    target_df = df[(min_year <= df['Year'])&(df['Year'] <= max_year)&(df['Years of references']!=0)]
    target_year_list = []
    for year in sorted(target_df['Year'].unique()):
        target_year_df = target_df[target_df['Year'] == year]
        target_year_df['Age of Citations'] = target_year_df['Years of references'].apply(lambda x : year-x)
        target_year_list.append(target_year_df)
    target_age_df = pd.concat(target_year_list, ignore_index=True)
    if top_age < 0:
        g = sns.boxenplot(x='Year', y='Age of Citations', data=target_age_df)
        g.set_xticklabels(g.get_xticklabels(), rotation=90)
        plt.title(f'{journal} {min_year}-{max_year}')
    else:
        target_age_df = target_age_df[target_age_df['Age of Citations'] <= top_age]
        g = sns.boxenplot(x='Year', y='Age of Citations', data=target_age_df)
        g.set_xticklabels(g.get_xticklabels(), rotation=90)
        plt.title(f'{journal} {min_year}-{max_year}')
    return helper.calculate_age_of_citations(target_age_df)


def generate_topic_model():
    """
    Create a BERTopic instance.
    Returns:
        topic_model: A generated Bertopic instance.
    """
    representation_model = KeyBERTInspired()
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
    hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean',
                            cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(ngram_range=(1,2))
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
    topic_model = BERTopic(
        top_n_words=10,
        nr_topics='auto',
        low_memory=True,
        calculate_probabilities=True,       
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        verbose=True
        )
    return topic_model


def fit_topic_model(topic_model, texts: list[str], file_name: str, use_reduce_outlier: Literal[False, True] = False):
    """
    fit a Bertopic instance and return a fitted Bertopic instance -- It is saved.
    Parameters:
        topic_model: Generated topic model.
        texts: A collection of documents to find topics.
        file_name: A source of documents to save a topic model.
        use_reduce_outlier: If number of outliers is a lot, use_reduce_outlier is used. The default is "False".
    Returns:
        topic_model: A fitted BERTopic instance that discovers topics within a collection of documents.
    """
    if use_reduce_outlier == True:
        topics, prob = topic_model.fit_transform(texts)
        new_topics = topic_model.reduce_outliers(texts, topics, strategy='c-tf-idf')
        topic_model.update_topics(texts, topics=new_topics)
        topic_model.save(join(MODEL_PATH, f'{file_name}_topic_model'), serialization='pickle', save_embedding_model=True, save_ctfidf=True)
    else:
        topics, prob = topic_model.fit_transform(texts)
        topic_model.save(join(MODEL_PATH, f'{file_name}_topic_model'), serialization='pickle', save_embedding_model=True, save_ctfidf=True)
    return topic_model


def plot_topics_of_documents(texts: list[str], topic_model) -> None:
    """
    Visualize documents and their topics in 2D.
    Parameters:
        texts: Refined texts based on a topic models.
        topic_model: A fitted BERTopic instance.
    """
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    reduced_embeddings = UMAP(n_neighbors=10, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    return topic_model.visualize_documents(
        texts, reduced_embeddings=reduced_embeddings, hide_document_hover=True, hide_annotations=True,
        title=f'<b>Documents and Topics</b>'
        )


def plot_hierarchical_clustering(texts: list[str], topic_model) -> None:
    """
    Visualize a hierarchical structure of topics.
    Parameters:
        texts: Refined texts based on a topic models.
        topic_model: A fitted BERTopic instance.
    """
    hierarchical_topics = topic_model.hierarchical_topics(texts)
    return topic_model.visualize_hierarchy(
        hierarchical_topics=hierarchical_topics, custom_labels=True, title=f'<b>Hierarchical Clustering</b>'
        )