"""Phase 0: schemas/ files are valid JSON Schema 2020-12 and the example
instance embedded in each schema validates against it."""
import json
import pathlib

import pytest

jsonschema = pytest.importorskip("jsonschema")

SCHEMA_DIR = pathlib.Path(__file__).resolve().parent.parent / "schemas"
SCHEMA_FILES = sorted(SCHEMA_DIR.glob("*.schema.json"))


def test_schemas_exist():
    assert SCHEMA_FILES, "schemas/ must contain at least the v0 snapshot schema"


@pytest.mark.parametrize("path", SCHEMA_FILES, ids=lambda p: p.name)
def test_schema_is_valid_and_example_validates(path):
    schema = json.loads(path.read_text())
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    assert schema.get("examples"), f"{path.name} must embed >=1 example"
    for example in schema["examples"]:
        jsonschema.validate(example, schema)
    assert schema["properties"]["schema_name"]["const"]
    assert schema["properties"]["schema_version"]["const"] == 0
