from transformers import (
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoModelForCausalLM, 
    Gemma3ForCausalLM,
    DynamicCache,
    GenerationConfig
)
import torch
from typing import Literal, List
from functools import partial

from .example import get_total_example
from .data_model import get_schema


class Inference(object):

    system_instruction = "You are an professional financial expert."

    _prompt_template = f"""
Dựa vào đoạn văn bản dưới đây, hãy bóc tách ra các sự kiện liên quan đến tài chính được đề cập để xây
dựng knowledge graph
- Mỗi sự kiện là một cặp S - R - O, trong đó S là một entity vai trò là chủ ngữ, O là một entity với vai
trò là vị ngữ được S đề cập đến, R là relations, mối liên hệ giữa S và O
- Ứng với mỗi sự kiện, hãy xác định thời gian của sự kiện đó theo định dạng ngày, tháng, năm hoặc ngày/tháng/năm
- S là entity, là danh từ, có thể là tên một tổ chức như công ty, tổ phức chính phủ, hoặc tên của một cá nhân
nào đó như chủ tịch công ty, tập đoàn, tên các cổ phiếu, tên loại vàng, vv..
- O là entity có thể là giá cổ phiếu, giá vàng (theo đồng, lượng, đô-la, VND), có thể là một string rỗng nếu không tìm được 
- R là mối quan hệ giữa S và O, có thể là động từ (verb) hoặc tính từ (adjective) trong hoạt động tài chính như mua vào, bán ra, tăng hoặc giảm
đối với giá cả, niêm yết, bốc hơi, vv ..., R không thể chứa các danh từ như trong S và O.

- Đầu ra của bạn cần phải tuân theo định dạng JSON dưới đây:
{{response_schema}}

- Dưới đây là một vài ví dụ bao gồm đầu vào và đầu ra tương ứng
{{example}}

Bây giờ, hãy đưa ra output cho đoạn văn bản dưới đây
{{statement}}
"""

    def __init__(self, 
            model_id_list: Literal["google/gemma-2-2b-it", "google/gemma-3-1b-it"],
            quantization: Literal["8_bits", "4_bits", "None"],
            tokenizer_max_length:int = 4000,
        )->None:
        model_id = model_id_list
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if quantization == "8_bits":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "4_bits":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit = True, 
                bnb_4bit_compute_dtype = torch.bfloat16
            )
        else:
            quantization_config = None
        
        self.model = Gemma3ForCausalLM.from_pretrained(
            model_id,
            device_map = "auto",
            torch_dtype = "auto" if use_8_bits else torch.float16,
            attn_implementation = "sdpa",
            quantization_config = quantization_config
        ).eval()

        self.generation_config =  GenerationConfig(
            max_new_tokens = 8000,
            do_sample= True,
            temperature = 0.5,
            num_beams = 3,
            use_cache = True,
            cache_implementation = "hybrid"
        )

        total_examples = get_total_example()
        self.reponse_schema = get_schema()

        self.pre_built_template = partial(Inference._prompt_template.format,
            example = total_examples,
            response_schema = self.reponse_schema
        )

        self.tokenizer_max_length = tokenizer_max_length

    def forward(self, corpus:List[str])->List[str]:
        batch_msgs = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_instruction},]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "text", 
                        "text": self.pre_built_template(statement = statement)
                        },]
                },
            ]
            for statement in corpus
        ]


        inputs = self.tokenizer.apply_chat_template(
            batch_msgs,
            add_generation_prompt=True,
            tokenize=True,
            padding = True,
            padding_side = "left",
            truncation = True,
            max_length = self.tokenizer_max_length,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, 
                generation_config = self.generation_config
            )

        return  self.tokenizer.batch_decode(
            outputs[:, inputs['input_ids'].shape[-1]:], 
            skip_special_tokens = True
        )