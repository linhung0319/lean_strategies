import importlib
from datetime import datetime

from variants import ALGORITHM_CLASSES, VARIANTS, slug


def test_all_sixteen_variants_from_main_py_are_present():
    # Counted by module prefix rather than by len(VARIANTS): the registry also
    # carries variants that run against converted data, and adding one of
    # those must not look like losing one of these.
    spy_variants = [name for name, (module, _) in VARIANTS.items() if module.startswith("spy_")]
    assert len(spy_variants) == 16


def test_every_variant_points_at_a_known_algorithm_module():
    for name, (module_name, _) in VARIANTS.items():
        assert module_name in ALGORITHM_CLASSES, name


def test_every_algorithm_module_imports_and_defines_its_class():
    for module_name, class_name in ALGORITHM_CLASSES.items():
        module = importlib.import_module(module_name)
        assert hasattr(module, class_name), f"{module_name}.{class_name}"


# How each algorithm coerces the raw (stringified) parameter value, keyed by
# parameter name — mirrors the `int(...)`, `float(...)`, `str(...).upper()`
# and date-splitting calls in algorithms/*.py's initialize().
INT_PARAMETERS = {"ma_period", "lookback", "lookback_months"}
UPPERCASE_STRING_PARAMETERS = {"frequency", "ticker"}
DATE_PARAMETERS = {"start_date", "end_date"}


def _coerce(key, value):
    if key in UPPERCASE_STRING_PARAMETERS:
        return str(value).upper()
    if key in INT_PARAMETERS:
        return int(value)
    if key in DATE_PARAMETERS:
        return datetime.strptime(str(value), "%Y-%m-%d")
    return float(value)


def test_every_variant_parameter_is_accepted_by_its_algorithm():
    for name, (module_name, parameters) in VARIANTS.items():
        module = importlib.import_module(module_name)
        algorithm = getattr(module, ALGORITHM_CLASSES[module_name])()
        algorithm._parameters = {k: str(v) for k, v in parameters.items()}
        algorithm.initialize()
        for key, value in parameters.items():
            assert hasattr(algorithm, key), f"{name}: {key}"
            assert getattr(algorithm, key) == _coerce(key, value), f"{name}: {key}"


def test_slugs_are_unique_and_filesystem_safe():
    slugs = [slug(name) for name in VARIANTS]
    assert len(set(slugs)) == len(slugs)
    for value in slugs:
        assert value.replace("_", "").isalnum(), value
