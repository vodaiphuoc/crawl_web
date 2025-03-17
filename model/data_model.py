from pydantic.dataclasses import dataclass
from pydantic import Field, TypeAdapter
import json

@dataclass
class Event:
    r"""
    Dataclass for modeling an Event
    """
    time: str = Field(description = "Time (in datetime) related to event")
    S: str = Field(description = "Entity as main subject in event")
    R: str = Field(description = "Relation between subject and object")
    O: str = Field(description = "Enity as object which is mentioned in this event")

@dataclass
class Output:
    event_list: list[Event] = Field(description = "list of events found in the corpus")

def get_schema()->str:
    dict_schema = TypeAdapter(Output).json_schema()
    del dict_schema['title']
    del dict_schema['type']

    return json.dumps(dict_schema)