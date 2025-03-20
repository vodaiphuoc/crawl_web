import json
import glob
from bs4 import BeautifulSoup
import argparse
from typing import Literal, List, Dict, Union
from datetime import datetime
import re
from tqdm import tqdm
from collections import defaultdict
from vnstock3 import Vnstock
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

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
    regex_result = re.findall(r"\bACB\b|\bNgân hàng\b|\bngân hàng\b|\bgiá vàng\b|\bvàng\b", main_corpus)

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


class PostProcessing(object):
    def __init__(self):
        total_data = []
        for json_file in glob.glob('stage_4_data/page_data_*.json'):
            with open(json_file,'r') as fp:
                stage3_data = json.load(fp)
                total_data.extend(stage3_data)


        self.grouped_data = defaultdict(list)
        for item in total_data:
            date_tuple = (item['year'], item['month'], item['day'])
            self.grouped_data[date_tuple].append({
                'url': item['url'],
                'corpus':item['corpus']
            })

        self.stock_values = self._post_processing_vnstock()

        # convert merge corpus to embedding vetors
        self.sentence_model = SentenceTransformer('dangvantuan/vietnamese-document-embedding', 
                                    trust_remote_code=True)
        self.sentence_model.compile(fullgraph = True)

    def _post_processing_vnstock(self)->pd.DataFrame:
        acb_stocks = Vnstock().stock(symbol="ACB", source= "TCBS")
        stock_values =  acb_stocks.quote.history(
            start = "2022-01-01", 
            end = str(datetime.now().date()),
            interval = '1D'
        )
        stock_values = stock_values.sort_values(by = 'time')

        stock_values['year'] = stock_values['time'].dt.year
        stock_values['month'] = stock_values['time'].dt.month
        stock_values['day'] = stock_values['time'].dt.day

        print('length before: ', len(stock_values))
        stock_values.drop_duplicates(inplace=True)
        stock_values.reset_index(drop = True, inplace = True)
        print('length after: ', len(stock_values))

        stock_values['time'] = pd.to_datetime(stock_values['time'], format="%Y-%m-%d")
        stock_values['time_diff'] = stock_values['time'].diff()
        
        stock_values.to_csv('temp.csv')

        return stock_values


    def _get_corpus(self, row):
        r"""
        Merge corpus and get embedding
        """    
        query_key = (row.year, row.month, row.day)
        
        try:
            merge_corpus = "\n".join([
                element['corpus'] 
                for element in self.grouped_data[query_key]
            ])

            return {
                'merge_corpus': merge_corpus
            }
        except IndexError as err:
            return {
                'corpus_list': []
            }

    def align(self):
        r"""
        Align corpus with time in price dataframe
        """
        self.stock_values['merge_corpus'] = self.stock_values.apply(
            lambda x: self._get_corpus(x), 
            axis = 1, 
            result_type = "expand"
        )
        
        
        corpus_list = self.stock_values['merge_corpus'].tolist()
        batch_size = len(corpus_list)//3
        
        total_embeddings = []
        for _ith in range(0, len(corpus_list), batch_size):
            if _ith + batch_size > len(corpus_list):
                start_ids = _ith
                end_ids = len(corpus_list)
            else:
                start_ids = _ith
                end_ids = _ith+batch_size
            
            batch_data = corpus_list[start_ids: end_ids]
            embeddings = self.sentence_model.encode(batch_data)
            total_embeddings.append(embeddings)
        

        total_embeddings = np.concatenate(total_embeddings, axis= 0)

        _final_embeddings_length = total_embeddings.shape[0]

        assert _final_embeddings_length == len(corpus_list), f"Found {_final_embeddings_length} vs {len(corpus_list)}"

        self.stock_values['embeddings'] = self.stock_values.apply(lambda x: total_embeddings[x.index,:].tolist())

        return self.stock_values


def main()->None:
    for json_file in glob.glob('stage_3_data/*.json'):
        file_name = json_file.split(os.sep)[-1].replace('.json','')

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


    # post processing
    engine = PostProcessing()
    result_data = engine.align()

    print('result data length: ', len(result_data))

    result_data.to_csv('total.csv')


if __name__ == '__main__':
    main()
