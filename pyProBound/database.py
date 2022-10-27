import json
import pandas as pd
import os

current = os.path.split(os.path.realpath(__file__))[0]
jardir = f"{current}/jardir"


class MotifCentral:
    def __init__(self):
        # loads database
        ...