import json
import glob
from model.inference import Inference
from bs4 import BeautifulSoup

def pre_processing_page_data(page_data:str)->str:
    soup = BeautifulSoup(page_data, 'html.parser')

    title = soup.find_all("h1", attrs= {"class": "title", "data-role": "title"})[0].text
    title = title.strip()
    
    lines = soup.text.strip().split('\n')
    lines = [line.strip() for line in lines if line != '']

    title_idx = [ith for ith, line in enumerate(lines) if line == title][-1]

    end_corpus_idx = [ith for ith, line in enumerate(lines) if line == 'Lấy link!'][-1]

    main_corpus = "\n".join(lines[title_idx+1:end_corpus_idx-4])

    return main_corpus


model = Inference(model_id_list = "google/gemma-3-1b-it")
BATCH_SIZE = 3


total_data = []
_ith = 0

for json_file in glob.glob('stage_3_data/*.json'):
    with open(json_file,'r') as fp:
        stage3_data = json.load(fp)

    for ith in range(0,len(stage3_data), BATCH_SIZE):
        print('batch start id: ', ith)

        batch_pages_data = stage3_data[ith: ith + BATCH_SIZE] \
            if ith + BATCH_SIZE < len(stage3_data)  \
            else stage3_data[ith: ith : ]


        batch_page = []
        for page in batch_pages_data:
            try:
                batch_page.append(pre_processing_page_data(page['page_data']))
            except IndexError as err:
                continue
        
        batch_reponses = model.forward(batch_page)

        output_batch = [{
                "key": origin_data["key"],
                "url": origin_data["url"],
                "page_data": origin_data["page_data"],
                "NER": reponse_data
            }
            for origin_data, reponse_data \
            in zip(batch_pages_data, batch_reponses)
        ]

        total_data.extend(output_batch)

        if len(total_data) > 100:
            _ith += 1
            with open(f'stage_4_data/page_data_{_ith}.json','w') as fp:
                json.dump(total_data, fp, indent= 4)
            
            total_data = []
