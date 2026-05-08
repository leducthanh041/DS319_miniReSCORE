# Compatibility shim for outlines versions that import pyairports.airports.
# ReSCORE does not use guided airport-code grammars, but vLLM imports outlines
# at request time, so AIRPORT_LIST only needs to be importable.
AIRPORT_LIST = [
    ("", "", "", "AAA"),
]
