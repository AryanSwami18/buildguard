from pydantic import BaseModel


class BashSource(BaseModel):
    name: str
    content: str
