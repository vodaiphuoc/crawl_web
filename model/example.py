from dataclasses import asdict
from pydantic.dataclasses import dataclass
from .data_model import Output, Event

@dataclass
class Example:
    corpus: str
    output: Output


exp1 = asdict(Example(
    corpus = """Trong nước, giá vàng miếng SJC ngày 15/3 được các công ty SJC, DOJI, PNJ niêm yết 94,3 – 95,8 triệu đồng/ lượng 
    (mua vào – bán ra), giá vàng nhẫn và vàng trang sức các loại cũng tăng vọt, lập đỉnh mới.""",
    output = Output(event_list = [
        Event(
            time = "ngày 15/3",
            S = "giá vàng miếng SJC",
            R = "được các công ty SJC, DOJI, PNJ niêm yết",
            O = "94,3 – 95,8 triệu đồng/ lượng"
        ),
        Event(
            time = "ngày 15/3",
            S = "giá vàng nhẫn và vàng trang sức các loại",
            R = "tăng vọt, lập đỉnh mới.",
            O = ""
        )
    ])
    )
)

exp2 = asdict(Example(
    corpus = """Vốn hóa thị trường theo đó “bốc hơi” gần 7.000 tỷ, còn chưa đến 30.000 tỷ đồng.""",
    output = Output(event_list = [
        Event(
            time = "",
            S = "Vốn hóa thị trường",
            R = "bốc hơi",
            O = "gần 7.000 tỷ, còn chưa đến 30.000 tỷ đồng."
        )
    ])
    )
)

def get_total_example()->str:
    total_exps = "\n".join([
        f"example {{ith}}:\n{{content}}".format(ith = ith+1, content = content) 
        for ith, content in enumerate([exp1, exp2])
    ])
    return total_exps