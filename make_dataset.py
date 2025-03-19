import json
import glob
from bs4 import BeautifulSoup
import argparse
from typing import Literal, List, Dict, Union
from datetime import datetime
import re
from tqdm import tqdm

class NonmatchException(Exception):
    def __init__(self, message:str):
        self.message = message

    def __str__(self):
        return str(self.message)


def pre_processing_page_data(page_data:str, url:str)->Dict[str, Union[int, str]]:

    soup = BeautifulSoup(page_data, 'html.parser')

    publish_datetime =  soup.find_all("span", attrs= {"class": "pdate", "data-role": "publishdate"})[0].text
    publish_data = publish_datetime.split(' - ')[0]
    publish_data_datetime_object = datetime.strptime(publish_data, "%d-%m-%Y")


    title = soup.find_all("h1", attrs= {"class": "title", "data-role": "title"})[0].text
    title = title.strip()
    
    lines = soup.text.strip().split('\n')
    lines = [line.strip() for line in lines if line != '']

    title_idx = [ith for ith, line in enumerate(lines) if line == title][-1]

    end_corpus_idx = [ith for ith, line in enumerate(lines) if line == 'Lấy link!'][-1]

    main_corpus = "\n".join(lines[title_idx+1:end_corpus_idx-4])

    # find news for ACB and other banks only
    regex_result = re.findall(r"\bACB\b|\bNgân hàng\b|\bngân hàng\b", main_corpus)

    if len(set(regex_result)) > 0:
        return {
            'url': url,
            'day': publish_data_datetime_object.day,
            'month': publish_data_datetime_object.month,
            'year': publish_data_datetime_object.year,
            'corpus': main_corpus
        }
    else:
        raise NonmatchException('main corpus does not have target kewwords')


def main()->None:
    for json_file in glob.glob('stage_3_data/*.json'):
        file_name = json_file.split('/')[-1].replace('.json','')

        total_data = []
        with open(json_file,'r') as fp:
            stage3_data = json.load(fp)

            for page in tqdm(stage3_data, total = len(stage3_data)):
                try:
                    total_data.append(pre_processing_page_data(
                        page_data = page['page_data'],
                        url = page['url']
                    ))
                except (IndexError, NonmatchException) as err:
                    continue
        
        with open(f'stage_4_data/{file_name}.json','w') as fp:
                json.dump(total_data, fp, indent= 4)

if __name__ == '__main__':
    main()
