import json
import glob
from model.inference import Inference

model = Inference(model_id_list = "google/gemma-3-1b-it")
BATCH_SIZE = 3


total_data = []
_ith = 0

for json_file in glob.glob('stage_3_data/*.json'):
    with open(json_file,'r') as fp:
        stage3_data = json.load(fp)

    for ith in range(0,len(stage3_data), BATCH_SIZE):
        batch_pages_data = stage3_data[ith: ith + BATCH_SIZE] \
            if ith + BATCH_SIZE < len(stage3_data)  \
            else stage3_data[ith: ith : ]

        batch_page = [page['page_data'] for page in batch_pages_data]

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
