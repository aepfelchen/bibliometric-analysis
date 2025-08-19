import re
from typing import Literal, Optional

import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import utils

lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer('\w+')
STOP_WORDS = set(stopwords.words('english'))
COUNTRIES = [name.lower() for name in utils.get_country_name()]


def clean_authors(df: pd.DataFrame) -> list[str]:
    """
    Clean full names of authors of a document.
    Parameters:
        df: Database of a journal
    Returns:
        clenaed_author_list: A list of cleaned names of authors.
    """
    clenaed_author_list = []
    for author in df['Author full names']:
        if ';' in str(author):
            authors = author.split(';')
            author_list = []
            for author in authors:
                if ' (' in str(author):
                    name, id = author.split(' (')
                    name = name.strip()
                    author_list.append(name.strip())
                else:
                    author_list.append(author)
            author_list = '; '.join(author_list)
            clenaed_author_list.append(author_list)
        else:
            if ' (' in str(author):
                name, id = author.split(' (')
                name = name.strip()
                clenaed_author_list.append(name)
            else:
                clenaed_author_list.append(str(author).strip())
    return clenaed_author_list


def clean_references(df: pd.DataFrame) -> list[str]:
    """
    Clean and split references of a document.
    Parameters:
        df: Database of a journal
    Returns:
        res_list: A list of cleaned references.
    """
    res_list = []
    for reference in df['References']:
        if ';' in str(reference):
            references = reference.split(';')
            references = [reference.strip() for reference in references]
            res_list.append(references)
        else:
            res_list.append(str(reference).strip())
    return res_list


def clean_reference_without_year(reference: str) -> str:
    """
    Remove a year of publication from a reference.
    Parameters:
        reference: A target reference.
    Returns:
        reference_without_year: A reference without a year.
    """
    reference_without_year = []
    res = utils.find_year(reference)
    if ''.join(res) in reference:
        reference_without_year.append(reference.replace(*res, '').strip())
    else:
        reference_without_year.append(reference)
    return reference_without_year


def clean_omissions_references(df: pd.DataFrame) -> list[list[str]]:
    """
    Correct an error (=an omission) due to repetition of previous reference.
    Parameters:
        df: Database of a journal.
    Returns:
        new_reference_list: A list of lists of corrected references
    """
    references_list = clean_references(df)
    references_list = [references for references in references_list if references]
    year_criteria = []
    for references in references_list:
        years = []
        for reference in references:
            res = utils.find_year(reference)
            years.extend(res)
        year_criteria.append(years)
    new_reference_list = []
    for cnt, references in enumerate(references_list):
        new_references = []
        for num, reference in enumerate(references):
            year = year_criteria[cnt][num]
            if len(reference.split()) == 1 and year != '0':
                prev = 1
                while True:
                    reference = references[num-prev]
                    if len(reference.split()) > 1 and '0' not in utils.find_year(reference):
                        break
                    else:
                        prev += 1
                reference = clean_reference_without_year(reference)
                reference.append(year)
                reference = ' '.join(reference)
                new_references.append(reference)
            else:
                new_references.append(reference)
        new_reference_list.append(new_references)
    return new_reference_list


def clean_(df: pd.DataFrame,
           category: Literal['author', 'institution', 'country', 'keyword', 'author keyword', 'index keyword', 'reference']) -> list[str]:
    """
    Clean objects according to a category.
    Parameters:
        df: Database of a journal.
        category: A type of objects (author, institution, country, keyword, author keyword, index keyword and reference).
    Returns:
        res_list: A list of cleaned objects.
    Notes:
        * It is used for the function "tools.plot_top_".
    """
    res_list = []
    if category == 'author':
        author_list = []
        for author in df['Author full names'].dropna():
            if ';' in str(author):
                authors = author.split(';')
                authors = [author.strip() for author in authors]
                author_list.extend(authors)
            else:
                author_list.append(author)
        for author in author_list:
            if ' (' in str(author):
                name, id = author.split(' (')
                name = name.strip()
                res_list.append(name)
            else:
                res_list.append(author.strip())
    elif category == 'institution':
        for entities in df['Affiliations'].dropna():
            if ';' in str(entities):
                affiliations = entities.split(';')
                institutions = []
                for affiliation in affiliations:
                    if len(affiliation.split(',')) > 2:
                        institutions.append(affiliation.split(',')[0].strip())
                    else:
                        if affiliation.lower() not in COUNTRIES:
                            institutions.append(affiliation.strip())
                res_list.extend(institutions)
            else:
                if entities.split(',')[0].lower().strip() not in COUNTRIES:
                    res_list.append(entities.split(',')[0].strip())
    elif category == 'country':
        for entities in df['Affiliations'].dropna():
            if ';' in str(entities):
                affiliations = entities.split(';')
                countries = []
                for affiliation in affiliations:
                    if len(affiliation.split(',')) > 2:
                        candidate = affiliation.split(',')[-1].strip()
                        if candidate.lower() in COUNTRIES:
                            countries.append(candidate)
                    else:
                        if affiliation.lower() in COUNTRIES:
                            countries.append(affiliation.strip())
                res_list.extend(countries)
            else:
                if entities.split(',')[-1].lower().strip() in COUNTRIES:
                    res_list.append(entities.split(',')[-1].strip())
    elif category == 'keyword':
        df = utils.merge_keyword(df)
        for keyword in df['Keywords']:
            if ';' in str(keyword):
                keywords = keyword.split(';')
                keywords = [lemmatizer.lemmatize(keyword.strip().lower()) for keyword in keywords]
                res_list.extend(keywords)
            else:
                res_list.append(lemmatizer.lemmatize(str(keyword).strip().lower()))
    elif category == 'author keyword':
        for author_keyword in df['Author Keywords'].dropna():
            if ';' in str(author_keyword):
                author_keywords = author_keyword.split(';')
                author_keywords = [lemmatizer.lemmatize(author_keyword.strip().lower()) for author_keyword in author_keywords]
                res_list.extend(author_keywords)
            else:
                res_list.append(lemmatizer.lemmatize(str(author_keyword).strip().lower()))
    elif category == 'index keyword':
        for index_keyword in df['Index Keywords'].dropna():
            if ';' in str(index_keyword):
                index_keywords = index_keyword.split(';')
                index_keywords = [lemmatizer.lemmatize(index_keyword.strip().lower()) for index_keyword in index_keywords]
                res_list.extend(index_keywords)
            else:
                res_list.append(lemmatizer.lemmatize(str(index_keyword).strip().lower()))
    elif category == 'reference':
        for references in clean_omissions_references(df):
            for reference in references:
                if len(reference) != 1:
                    res_list.append(reference)
    return res_list


def clean_co_(df: pd.DataFrame,
              category: Literal['author', 'institution', 'country', 'keyword', 'author keyword', 'index keyword', 'reference']) -> list[str]:
    """
    Clean co-occurring elements within a category.
    Parameters:
        df: Database of a journal.
        category: A type of objects (author, institution, country, keyword, author keyword, index keyword and reference).
    Returns:
        co_res_list: A list of cleaned co-occurring elements within a category.
    """
    co_res_list = []
    if category == 'author':
        co_author_list = []
        for author in df['Author full names']:
            if ';' in str(author):
                authors = author.split(';')
                authors = [author.strip() for author in authors]
                co_author_list.append(authors)
        for authors in co_author_list:
            co_author = []
            for author in authors:
                if ' (' in str(author):
                    name, id = author.split(' (')
                    name = name.strip()
                    co_author.append(name)
                else:
                    co_author.append(str(author).strip())
            co_res_list.append(co_author)
    elif category == 'institution':
        for entities in df['Affiliations']:
            if ';' in str(entities):
                affiliations = entities.split(';')
                institutions = []
                for affiliation in affiliations:
                    if len(affiliation.split(',')) > 2:
                        institutions.append(affiliation.split(',')[0].strip())
                    else:
                        if affiliation.lower() not in COUNTRIES:
                            institutions.append(affiliation.strip())
                co_res_list.append(institutions)
    elif category == 'country':
        for entities in df['Affiliations']:
            if ';' in str(entities):
                affiliations = entities.split(';')
                countries = []
                for affiliation in affiliations:
                    if len(affiliation.split(',')) > 2:
                        candidate = affiliation.split(',')[-1].strip()
                        if candidate.lower() in COUNTRIES:
                            countries.append(candidate)
                    else:
                        if affiliation.lower() in COUNTRIES:
                            countries.append(affiliation.strip())
                co_res_list.append(countries)
    elif category == 'keyword':
        df = utils.merge_keyword(df)
        for keyword in df['Keywords']:
            if ';' in str(keyword):
                keywords = keyword.split(';')
                keywords = [lemmatizer.lemmatize(keyword.strip().lower()) for keyword in keywords]
                co_res_list.append(keywords)
    elif category == 'author keyword':
        for keyword in df['Author Keywords'].dropna():
            if ';' in str(keyword):
                keywords = keyword.split(';')
                keywords = [lemmatizer.lemmatize(keyword.strip().lower()) for keyword in keywords]
                co_res_list.append(keywords)
    elif category == 'index keyword':
        for keyword in df['Index Keywords'].dropna():
            if ';' in str(keyword):
                keywords = keyword.split(';')
                keywords = [lemmatizer.lemmatize(keyword.strip().lower()) for keyword in keywords]
                co_res_list.append(keywords)
    elif category == 'reference':
        for references in clean_omissions_references(df):
            if len(references) > 1:
                co_res_list.append(references)
    return co_res_list


def clean_keywords(df: pd.DataFrame, category: Literal['keyword', 'author keyword', 'index keyword']) -> list[str]:
    """
    Clean and split keywords of a document.
    Parameters:
        df: Database of a journal.
        use_index_keyword: If author keywords do not exist in database, index keywords are used.
    Returns:
        res_list: A list of cleaned keywords.
    """
    res_list = []
    if category == 'keyword':
        df = utils.merge_keyword(df)
        for keyword in df['Keywords']:
            if ';' in str(keyword):
                keywords = keyword.split(';')
                keywords = [lemmatizer.lemmatize(keyword.strip().lower()) for keyword in keywords]
                res_list.append(keywords)
            else:
                res_list.append(lemmatizer.lemmatize(str(keyword).strip().lower()))
        df['Keywords'] = res_list
        return df
    elif category == 'author keyword':
        for keyword in df['Author Keywords']:
            if ';' in str(keyword):
                keywords = keyword.split(';')
                keywords = [lemmatizer.lemmatize(keyword.strip().lower()) for keyword in keywords]
                res_list.append(keywords)
            else:
                res_list.append(lemmatizer.lemmatize(str(keyword).strip().lower()))
    else:
        for keyword in df['Index Keywords']:
            if ';' in str(keyword):
                keywords = keyword.split(';')
                keywords = [lemmatizer.lemmatize(keyword.strip().lower()) for keyword in keywords]
                res_list.append(keywords)
            else:
                res_list.append(lemmatizer.lemmatize(str(keyword).strip().lower()))
    return res_list


def clean_texts(texts: list[str], stopwords: Optional[list[str]] = None) -> list[str]:
    """
    Convert to lowercase letters, remove special characters and stopwords, Tokenize texts, and lemmatize tokens.
    Parameters:
        texts: A list of texts to be preprocessed.
        stopwords: A list of personal stopwords.
    Returns:
        preprocessed_texts/target_texts: A list of refined texts.
    """
    lowercase_texts = [text.lower() for text in texts]
    cleaned_texts = [re.sub('\W', ' ', text) for text in lowercase_texts]
    tokenized_texts = [tokenizer.tokenize(cleaned_text) for cleaned_text in cleaned_texts]
    cleaned_texts = [[token for token in tokenized_text if len(token) != 1] for tokenized_text in tokenized_texts]
    cleaned_texts = [[token for token in cleaned_text if token not in STOP_WORDS] for cleaned_text in cleaned_texts]
    lemmatized_texts = [[lemmatizer.lemmatize(token) for token in cleaned_text] for cleaned_text in cleaned_texts]
    preprocessed_texts = [' '.join(lemmatized_text) for lemmatized_text in lemmatized_texts]
    if not stopwords:
        print('Texts are cleaned!')
        return preprocessed_texts
    else:
        target_texts = []
        for preprocessed_text in preprocessed_texts:
            for stopword in stopwords:
                preprocessed_text = re.sub(stopword, '', preprocessed_text)
            target_texts.append(preprocessed_text)
        print('Texts are cleaned!')
        return target_texts
