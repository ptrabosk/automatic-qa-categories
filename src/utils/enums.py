from enum import StrEnum


class Role(StrEnum):
    AGENT = "agent"
    CUSTOMER = "customer"
    SYSTEM = "system"
    OTHER = "other"


class Method(StrEnum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"
    HYBRID = "hybrid"


class PassFail(StrEnum):
    PASS = "pass"
    FAIL = "fail"
