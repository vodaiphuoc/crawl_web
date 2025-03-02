import urllib3
from bs4 import BeautifulSoup
import pickle
import time
import json

def url_extract(
        url:str,
        key:int,
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:11.0) Gecko/20100101',
        host = 'cafef.vn',
        referer = 'https://cafef.vn/thi-truong-chung-khoan.chn',
        connection = 'keep-alive'
        ):

    reponse = urllib3.request(
        method= "GET", 
        url= 'https://cafef.vn' +url, 
        headers={
            'User-Agent':user_agent,
            'Host': host,
            'Referer': referer,
            'Connection': connection
            }
    )

    if reponse.status == 200:
        soup = BeautifulSoup(reponse.data, 'html.parser')
        return {
            'key': key,
            'url': 'https://cafef.vn' +url,
            'page_data': str(soup)
        }
    else:
        return None

if __name__ == '__main__':
    with open('stage_2_data/links.json','r') as fp:
        stage2_data = json.load(fp)
    
    stage3_data = []
    for data_dict in stage2_data:
        out_dict = url_extract(
            url = data_dict['link'], 
            key = data_dict['key']
        )
        if out_dict is not None:
            stage3_data.append(out_dict)
        
        time.sleep(3)

    with open('stage_3_data/page_data.json','w') as fp:
        json.dump(stage3_data, fp, indent= 4)