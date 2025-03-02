import glob
from typing import Dict, Any, List
import pickle
from bs4 import BeautifulSoup
import json

def process_each_file(data:Dict[str,str])->List[Dict[str,str]]:
    total_link = []

    for ith, soup_as_str in enumerate(data['list_tags']):
        soup = BeautifulSoup(soup_as_str, 'html.parser')
        
        link = None
        for tag in soup.find_all('a',href=True):
            if tag.get("class")==None:
                link = tag['href']

        if link is None:
            raise Exception(f"cannot find link in ith: {ith}, key: {data['key']}")
        else:
            total_link.append({
                'key': data['key'],
                'link': link
            })
    
    return total_link

if __name__ == '__main__':
    stage_data = []
    for _path in glob.glob('stage_1_data/*.pkl'):
        with open(_path, 'rb') as fp:
            data = pickle.load(fp)
            
            try:
                stage_data.extend(process_each_file(data))
            except Exception as e:
                print(f'has eception: {e}')
                continue
    
    with open('stage_2_data/links.json','w') as fp:
        json.dump(stage_data, fp, indent= 4)