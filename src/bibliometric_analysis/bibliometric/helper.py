from os import makedirs
from os.path import abspath, dirname, join, exists, isfile
from typing import Literal, Optional
import json
import re

import networkx as nx
import pandas as pd
import numpy as np

import cleaner
import utils

BASE_PATH = abspath(__file__)
ROOT_PATH = dirname(dirname(dirname(dirname(BASE_PATH))))
SAVE_PATH = join(ROOT_PATH, 'saved_files')
if not exists(SAVE_PATH):
    makedirs(SAVE_PATH)


def get_min_max_year(df: pd.DataFrame, min_year: Optional[int], max_year: Optional[int]) -> int:
    """
    Return minimum year and maximum year of database, if min_year or max_year are "None".
    Prameters:
        df: Database of a journal.
        min_year: Minimum year. If the min_year is "None", the lowest year of database is automatically inputted.
        max_year: Maximum year. If the max_year is "None", the highest year of database is automatically inputted.
    """
    if min_year is not None and max_year is not None:
       if max_year < min_year:
            min_year, max_year = max_year, min_year
    if min_year == None:
        min_year = min(df['Year'].unique())
    if max_year == None:
        max_year = max(df['Year'].unique())
    return min_year, max_year


def get_co_(df: pd.DataFrame,
            category: Literal['author', 'institution', 'country', 'author keyword', 'index keyword', 'reference'],
            min_year: int, max_year: int) -> pd.DataFrame:
    """
    Return Database of frequencies of two elements among co-occurring elements within a category.
    Parameters:
        df: Database of a journal.
        category: Type of object (author, institution, country, author keyword, index keyword and reference).
        min_year: Minimum year.
        max_year: Maximum year.
    Returns:
        co_df: Database of frequencies of two elements among co-occurring elements within a category.
    """
    if len(df['Source title'].unique()) == 1:
        journal = str(*df['Source title'].unique()).lower().replace(' ', '_')
    else:
        journal = f'high-ranking_{category}'
    co_res = cleaner.clean_co_(df, category)
    co_res = utils.get_co_occurrence_dict(co_res)
    co_df = pd.DataFrame.from_dict(co_res, orient='index')
    co_list=[]
    for i in range(len(co_df)):
        co_list.append([co_df.index[i][0], co_df.index[i][1], co_df[0][i]])
    co_df = pd.DataFrame(co_list, columns=['source','target','weight'])
    co_df = co_df.sort_values(by=['weight'], ascending=False)
    co_df = co_df.reset_index(drop=True)
    co_df.columns = ['source', 'target', 'weight']
    if category == 'author':
        if isfile(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_co_author_edge_list.csv')):
            print(f'{journal}_{min_year}_{max_year}_co_author_edge_list.csv is already created.')
        else:
            co_df.to_csv(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_co_author_edge_list.csv'), sep=';', index=False)
            print(f'{journal}_{min_year}_{max_year}_co_author_edge_list.csv is created.')
    elif category == 'institution':
        if isfile(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_co_institution_edge_list.csv')):
            print(f'{journal}_{min_year}_{max_year}_co_institution_edge_list.csv is already created.')
        else:
            co_df.to_csv(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_co_institution_edge_list.csv'), sep=';', index=False)
            print(f'{journal}_{min_year}_{max_year}_co_institution_edge_list.csv is created.')
    elif category == 'country':
        if isfile(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_co_country_edge_list.csv')):
            print(f'{journal}_{min_year}_{max_year}_co_country_edge_list.csv is already created.')
        else:
            co_df.to_csv(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_co_country_edge_list.csv'), sep=';', index=False)
            print(f'{journal}_{min_year}_{max_year}_co_country_edge_list.csv is created.')
    elif category == 'keyword':
        if isfile(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_co_keyword_edge_list.csv')):
            print(f'{journal}_{min_year}_{max_year}_co_keyword_edge_list.csv is already created.')
        else:
            co_df.to_csv(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_co_keyword_edge_list.csv'), sep=';', index=False)
            print(f'{journal}_{min_year}_{max_year}_co_keyword_edge_list.csv is created.')
    elif category == 'author keyword':
        if isfile(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_co_author_keyword_edge_list.csv')):
            print(f'{journal}_{min_year}_{max_year}_co_author_keyword_edge_list.csv is already created.')
        else:
            co_df.to_csv(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_co_author_keyword_edge_list.csv'), sep=';', index=False)
            print(f'{journal}_{min_year}_{max_year}_co_author_keyword_edge_list.csv is created.')
    elif category == 'index keyword':
        if isfile(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_co_index_keyword_edge_list.csv')):
            print(f'{journal}_{min_year}_{max_year}_co_index_keyword_edge_list.csv is already created.')
        else:
            co_df.to_csv(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_co_index_keyword_edge_list.csv'), sep=';', index=False)
            print(f'{journal}_{min_year}_{max_year}_co_index_keyword_edge_list.csv is created.')
    elif category == 'reference':
        if isfile(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_co_reference_edge_list.csv')):
            print(f'{journal}_{min_year}_{max_year}_co_reference_edge_list.csv is already created.')
        else:
            co_df.to_csv(join(SAVE_PATH, f'{journal}_{min_year}_{max_year}_co_reference_edge_list.csv'), sep=';', index=False)
            print(f'{journal}_{min_year}_{max_year}_co_reference_edge_list.csv is created.')
    return co_df


def get_centrality(co_df: pd.DataFrame) -> dict[str:float]:
    """
    Return various centralities, e.g. degree, betweenness, closeness and eigenvector centrality.
    Parameters:
        co_df: Database of frequencies of two elements among co-occurring elements within a category.
    Returns:
        dgr: A dictionary of degree centrality of co-occurring elements within a category.
        btw: A dictionary of betweenness centrality of co-occurring elements within a category.
        cls: A dictionary of closeness centrality of co-occurring elements within a category.
        egv: A dictionary of eigenvector centrality of co-occurring elements within a category.
    """
    G = nx.from_pandas_edgelist(co_df, source='source', target='target', edge_attr=True, create_using=nx.Graph(max_iter=10^5))
    dgr = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G)
    cls = nx.closeness_centrality(G)
    egv = nx.eigenvector_centrality(G)
    return dgr, btw, cls, egv


def get_year_reference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return Database including entities, i.e. "Year" and "Years of references" in references.
    Parameters:
        df: Database of (a) journal(s).
    Returns:
        target_df: Database including entities, i.e. "Year" and "Years of references"in references.
    """
    year_list = []
    for references in cleaner.clean_references(df):
        years = [[re.sub('\D', '', year) for year in utils.find_year(reference)] for reference in references]
        year_list.append(years)
    df['Years of references'] = year_list
    target_df = df[['Year', 'Years of references']]
    return target_df


def get_author_reference(df: pd.DataFrame, delimiter: Literal[',', None]) -> pd.DataFrame:
    """
    Return Database including entities, i.e. "Year" and "Authors of references" in references.
    Parameters:
        df: Database of a journal.
        delimiter: Delimiter to distinguish authors.
    Returns:
        target_df: Database including entities, i.e. "Year" and "Authors of references" in references.
    Notes:
        * Notations for surname of authors can be various accroding to referencing styles or journals.
        * For exemple, "Gwanghun P., Stilometrische Analyse ..., (2024)", if delimiter is ",".
        * For exemple, "Gwanghun P. (2024) Stilometrische Analyse ...", if delimiter is None.
    """
    name_list = []
    char = ':' # a restriction for searching names
    pattern = r'[a-zA-Z]\s[A-Z]\.'
    stopwords = [' ed', ' ed.', '(ed)', '(ed.)', ' Ed', ' Ed.', '(Ed)', '(Ed.)',
                 ' eds', ' eds.', '(eds)', '(eds.)', ' Eds', 'Eds.', '(Eds)', '(Eds.)',
                 'et al', 'et al.', 'et. al.', 'Et al', 'Et al.', 'Et. al.', 'al.']
    for references in cleaner.clean_references(df):
        names = []
        for reference in references:
            candidate_list = []
            if delimiter == ',':
                splitted_reference = reference.split(',')
                if len(splitted_reference) > 1 and char not in splitted_reference[0]:
                    candidates = []
                    num = 0
                    for cnt, component in enumerate(splitted_reference):
                        if len(component.split()) > 4:             
                            num = cnt
                            break
                    candidates.extend(splitted_reference[:num])
                    if num == 0:
                        candidates.append(splitted_reference[0].strip())
                    for candidate in candidates:
                        if utils.search_pattern(pattern, candidate):
                            for stopword in stopwords:
                                candidate = candidate.replace(stopword, '')
                            names.append(candidate.strip())
            elif delimiter == None:
                year = utils.find_year(reference)
                splitted_reference = reference.split(''.join(year))
                if ''.join(year) != '0' and char not in splitted_reference[0]:
                    if ', and ' in splitted_reference[0]:
                        splitted_reference[0] = splitted_reference[0].replace(', and ', ', ')
                    if ' and 'or ' et ' in splitted_reference[0]:
                        splitted_reference[0] = splitted_reference[0].replace(' and ', ', ').replace(' et ', ', ')
                    if '.,' in splitted_reference[0]:
                        candidates = splitted_reference[0].split(',')
                        for candidate in candidates:
                            if utils.search_pattern(pattern, candidate):
                                for stopword in stopwords:
                                    candidate = candidate.replace(stopword, '')
                                names.append(candidate.strip())
                    else:
                        candidate = utils.search_pattern(pattern, splitted_reference[0])
                        if candidate:
                            for stopword in stopwords:
                                candidate = candidate.replace(stopword, '')
                            names.append(candidate.strip())
        name_list.append(names)
    df['Authors of references'] = name_list
    target_df = df[['Year', 'Authors of references']]
    target_df = target_df.explode('Authors of references').explode('Authors of references')
    target_df.dropna(subset=['Authors of references'], inplace=True)
    return target_df


def calculate_age_of_citations(target_age_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate count of citations and median, mean, and standard error of age of citations by year.
    Parameters:
        target_age_df: Database for age of citations by year.
    Returns:
        target_df: Database including count of citations and median, mean, and standard error of age of citations by year.
    Notes:
        * It is connected to the function tools.plot_age_of_citataions.
    """
    target_df = pd.DataFrame()
    target_df.index = ['Count', 'Median', 'Mean', 'SE']
    for year in sorted(target_age_df['Year'].unique()):
        df = target_age_df[target_age_df['Year'] == year]
        target_df[year] = [
            len(df['Age of Citations']),
            np.median(df['Age of Citations']),
            np.mean(df['Age of Citations']),
            df['Age of Citations'].sem()
        ]
    return target_df


def convert_text_to_jsonl(zip_name: str, target_list: list[tuple[str, str]]) -> list[dict[str:str]]:
    """
    Convert texts of PDF files to a JSONL file.
    Parameters:
        zip_name: A name of ZIP file to save a JSONL file.
        target_list: A list of titles and texts of PDF files.
    Returns:
        jsonl_content: A list of dictionaries {file, id, title, text}.
    Notes:
        * It is connected to the function tools.extract_texts_from_zip.
    """
    jsonl_content = []
    for idx, pair in enumerate(target_list):
        jsonl_content.append({
            'id': idx,
            'title': pair[0],
            'text': pair[-1]
        })
    with open(join(SAVE_PATH, f'{zip_name}_texts.jsonl'), 'w') as f:
        json.dump(jsonl_content, f, ensure_ascii=False)
    print(f'{zip_name}_texts.jsonl is created.')
    return jsonl_content


def get_top_keywords(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return top keywords with a minimum value of 2 or more by year.
    Parameters:
        merged_df: Database for top keywords by year.
    Returns:
        df: Database for top keywords with a minimum value of 2 or more by year.
    Notes:
        * It is connected to the function tools.plot_keywords.
    """
    merged_df = merged_df.loc[merged_df['Keywords'] != '']
    years = []
    keywords = []
    cnt = []
    for k,v in merged_df.groupby(['Year', 'Keywords']).size().items():
        years.append(k[0])
        keywords.append(k[-1])
        cnt.append(v)
    data = {'Year':years, 'Keyword':keywords, 'Count':cnt}
    df = pd.DataFrame(data)
    return df