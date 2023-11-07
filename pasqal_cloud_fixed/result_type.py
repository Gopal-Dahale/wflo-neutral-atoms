from pasqal_cloud.utils.strenum import StrEnum


class MyResultType(StrEnum):
    COUNTER: str = "counter"
    RUN: str = "run"
    SAMPLE: str = "sample"
    EXPECTATION: str = "expectation"
