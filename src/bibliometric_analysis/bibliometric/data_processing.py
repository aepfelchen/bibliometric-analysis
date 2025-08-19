from os import makedirs, listdir
from os.path import abspath, dirname, join, isfile, exists, split, splitext
from io import BytesIO
from zipfile import ZipFile
import json

from pybtex.database.input import bibtex
import pandas as pd
import pdfplumber
from pdfminer.high_level import extract_text

import helper

BASE_PATH = abspath(__file__)
ROOT_PATH = dirname(dirname(dirname(dirname(BASE_PATH))))
SAVE_PATH = join(ROOT_PATH, 'resources')
if not exists(SAVE_PATH):
    makedirs(SAVE_PATH)


def parse_bib(file_path: str):
    """
    Convert and save a BIB file to a CSV file.
    Parameters:
        file_path: A path of BIB file.
    Notes:
        * Even if a key error occurs, ignore it.
        * The delimiter is "|".
    """
    parser = bibtex.Parser()
    bibdata = parser.parse_file(file_path)
    rows_list = []
    for bib_id in bibdata.entries:
        row_list = []
        try:
            b = bibdata.entries[bib_id].fields
            title = b['title']
            journal = b['journal']
            year = b['year']
            volume = b['volume']
            number = b['number']
            pages= b['pages']
            doi = b['doi']
            name_list = []
            for author in bibdata.entries[bib_id].persons['author']:
                first_name = ''.join(author.first())
                if len(author.middle()) == 1:
                    middle_name = ''.join(author.middle())
                else:
                    middle_name = ' '.join(author.middle())
        
                if len(author.last()) == 1:
                    last_name = ''.join(author.last()).replace('.', '')
                else:
                    last_name = ' '.join(author.last())
                name = last_name+', '+first_name+' '+middle_name
                name_list.append(' '.join(name.split()))
            chain_name = ';'.join(name_list)
            row_list.append(chain_name)
            row_list.append(title)
            row_list.append(journal)
            row_list.append(year)
            row_list.append(volume)
            row_list.append(number)
            row_list.append(pages)
            row_list.append(doi)
            rows_list.append(row_list)
        except:
            continue
    columns=['Author full names', 'Title', 'Source title', 'Year', 'Volume', 'Art. No.', 'Page count', 'DOI']
    df = pd.DataFrame(rows_list, columns=columns)
    df = df.sort_values(by='Year')
    journal = str(*df['Source title'].unique()).lower().replace(' ', '_')
    if isfile(join(SAVE_PATH, f'{journal}.csv')):
        print(f'{journal}.csv is already created.')
    else:
        df.to_csv(join(SAVE_PATH, f'{journal}.csv'), sep='|', index=False, encoding='utf-8')
        print(f'The csv file \'{journal}.csv\' is created.')
        print('The data frame is organized into', df.shape[0], 'rows and', df.shape[1], 'columns.')


def merge_csv(file_path: str) -> None:
    """
    Merge and save CSV files.
    Parameters:
        file_path: A path of CSV files.
    Notes:
        * The delimiter is ",".
    """
    merged_df = pd.DataFrame()
    for file in listdir(file_path):
        temp = pd.read_csv(join(file_path, file))
        merged_df = pd.concat([merged_df, temp], ignore_index=True)
    merged_df.to_csv(join(SAVE_PATH, f'merged_journals.csv'), sep=',', index=False, encoding='utf-8')
    print(f'The csv file \'merged_journals.csv\' is created.')
    print('The data frame is organized into', merged_df.shape[0], 'rows and', merged_df.shape[1], 'columns.')


def extract_texts_from_zip(zip_path: str) -> list[tuple[str, str]]:
    """
    Open a ZIP file and extract texts of PDF files inside it.
    Parameters:
        zip_path: A path of ZIP files.
    Returns:
        helfer.convert_text_to_jsonl(zip_name, target_list): A list of dictionaries {file, id, title, text}.
    Notes:
        * The issue of Non-Ascii85: https://github.com/jsvine/pdfplumber/issues/1293
    """
    dir_name, file_name = split(zip_path)
    zip_name, ext = splitext(file_name)
    if isfile(join(SAVE_PATH, f'{zip_name}_texts.jsonl')):
        print(f'{zip_name}_texts.jsonl is already created.')
        with open(join(SAVE_PATH, f'{zip_name}_texts.jsonl'), 'r') as f:
            jsonl_content = json.load(f)
            return jsonl_content
    else:
        with ZipFile(zip_path, 'r') as zf:
            file_list = zf.namelist()
            metadata_list = []
            text_list = []
            for file in file_list:
                if file.endswith('.pdf'): 
                    read_file = zf.read(file)
                    with pdfplumber.open(BytesIO(read_file)) as pdf:
                        metadata = pdf.metadata
                        try:
                            text = extract_text(BytesIO(read_file), codec='utf-8')
                            lines = [line.strip() for line in text.splitlines() if line.strip()]
                            cleaned_text = ' '.join(lines)
                            text_list.append(cleaned_text)
                            metadata_list.append(metadata.get('Title'))
                        except:
                            print(f'{file} has a problem with Non-Ascii85.')
            target_list = list(zip(metadata_list, text_list))
            num_pdfs = len(file_list)
            num_texts = len(text_list)
            print(f'{num_texts} texts are successfully extracted from {num_pdfs} PDFs!')
            return helper.convert_text_to_jsonl(zip_name, target_list)