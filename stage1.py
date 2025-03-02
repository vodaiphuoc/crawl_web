import urllib3
from bs4 import BeautifulSoup
import pickle
import time

def url_extract(
        url = 'https://cafef.vn/timelinelist/18831/{key}.chn', 
        key: int = 100,
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:11.0) Gecko/20100101',
        host = 'cafef.vn',
        referer = 'https://cafef.vn/thi-truong-chung-khoan.chn',
        connection = 'keep-alive'
        ):

    reponse = urllib3.request(
        method= "GET", 
        url= url.format(key = key), 
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
            'list_tags': [
                str(tag) 
                for tag in 
                soup.find_all(
                name= 'div',
                attrs= {'class': 'tlitem box-category-item'}
                )
            ] 
        }
    else:
        return None

if __name__ == '__main__':
    for i in range(500, 1000):
        output = url_extract(key= i)
        if output is not None:
            with open(f'stage_1_data/{i}.pkl','wb') as fp:
                pickle.dump(output, fp)
        time.sleep(3)