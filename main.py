import json
import glob
from model.inference import Inference
from bs4 import BeautifulSoup
import argparse

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


def main(
    batch_size:int, 
    max_length:int, 
    quantization: Literal["8_bits", "4bits", "None"]
    )->None:
    
    model = Inference(
        model_id_list = "google/gemma-3-1b-it", 
        tokenizer_max_length = max_length,
        quantization = quantization
    )


    total_data = []
    _ith = 0

    for json_file in glob.glob('stage_3_data/*.json'):
        with open(json_file,'r') as fp:
            stage3_data = json.load(fp)

        for ith in range(0,len(stage3_data), batch_size):
            print('batch start id: ', ith)

            batch_pages_data = stage3_data[ith: ith + batch_size] \
                if ith + batch_size < len(stage3_data)  \
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type = int)
    parser.add_argument("--max_length", type = int)
    parser.add_argument("--quantization", type = str)
    
    args = vars(parser.parse_args())

    main(**args)
    