from collections import defaultdict
import glob
import json

def group_and_sort_dictionaries(list_of_dictionaries):
    grouped_data = defaultdict(list)
    
    for item in list_of_dictionaries:
        date_tuple = (item['year'], item['month'], item['day'])
        grouped_data[date_tuple].append({
            'url': item['url'],
            'corpus':item['corpus']
        })

    sorted_dates = sorted(grouped_data.keys())  # Sort the unique dates
    
    sorted_grouped_list = []
    
    for year, month, day in sorted_dates:
        sorted_grouped_list.append({
            'year': year,
            'month': month,
            'day': day,
            'data': grouped_data[(year, month, day)]
        })
        
    return sorted_grouped_list

if __name__ == '__main__':
    
    total_data = []
    for json_file in glob.glob('stage_4_data/page_data_*.json'):
        with open(json_file,'r') as fp:
            stage3_data = json.load(fp)
            total_data.extend(stage3_data)

    sorted_data = group_and_sort_dictionaries(total_data)

    with open(f'stage_4_data/total.json','w') as fp:
        json.dump(sorted_data, fp, indent= 4)