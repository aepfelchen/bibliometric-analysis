import re
from ast import literal_eval

import pandas as pd
import pycountry


def merge_keyword(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a combined database of author keywords and index keywords from an integrated database of multiple journals.
    Parameters:
        df: Database of journals.
    Returns:
        merged_df: A combined database of author keywords and index keywords.
    """
    merged_df = pd.DataFrame()
    for journal in df['Source title'].unique():
        temp = df[df['Source title'] == journal]
        if len(temp['Author Keywords'].dropna()) < len(temp['Index Keywords'].dropna()):
            temp['Keywords'] = temp['Index Keywords']
        else:
            temp['Keywords'] = temp['Author Keywords']
        merged_df = pd.concat([merged_df, temp], ignore_index=True)
        merged_df.dropna(subset=['Keywords'], inplace=True)
    return merged_df


def get_co_occurrence_dict(co_occurrence_list: list[list[str]]) -> dict[str:int]:
    """
    Counts frequencies of two elements among co-occurring elements in a list.
    Parameters:
        co_occurence_list: A list of lists of co-occurring elements.
    Returns:
        count_dict: A dictionary of frequencies of two elements among co-occurring elements in a list.
    """
    count_dict= {}
    for co_occurrences in co_occurrence_list:
        for cnt, i in enumerate(co_occurrences[:-1], start=1):
            for j in co_occurrences[cnt:]:
                if i == j:
                    pass
                else:
                    count_dict[i.strip(), j.strip()] = count_dict.get((i.strip(), j.strip()), 0)+1
    return count_dict


def get_country_name() -> list[str]:
    """
    Return country names in english.
    Returns:
        names: A list of all countries.
    """
    countries = list(pycountry.countries)
    names = [country.name for country in countries]
    return names


def find_year(reference: list[str]) -> list[str]:
    """
    Search years of publications by the pattern "(yyyy)".
    For exemple, "Gwanghun P., Stilometrische Analyse ..., (2024)".
    Parameters:
        reference: A target reference.
    Returns:
        year_list: A list of years of publications in references.
    Notes:
        * If year of publication is not found, the element "0" is added to the list.
        * Years of publications can have various pattern und position according to referencing styles or journals.
    """
    pattern = r'\(\d\d\d\d\w?\)'
    p = re.compile(pattern)
    year_list = []
    match = p.search(reference)
    if match:
        year_list.append(match.group())
    else:
        year_list.append('0')
    return year_list


def search_pattern(pattern: str, candidate: str) -> str:
    """
    Search a word according to a pattern.
    Parameters:
        pattern: A pattern to find.
        candidate: A target word to find a pattern.
    Returns:
        candidate: A word accroding to a pattern.
    Notes:
        * Here it is used to find surnames of authors in citations.
    """
    p = re.compile(pattern)
    if p.search(candidate):
        return candidate
    

def str_to_list(elem):
    """
    Convert an argument data type in a series from "str" to "list".
    Parameters:
        elem: An argument in a series.
    Raises:
        ValueError: If a data type other than "str" or "list" comes, the error occurs.
    Returns:
        elem: An argument converted to list.
    """
    try:
        if type(elem) == str:
            return literal_eval(elem)
        elif type(elem) == list:
            return elem
    except ValueError:
        print('The series olny includes the data types "str" or "list".')