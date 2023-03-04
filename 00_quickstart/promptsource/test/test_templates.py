import time
from jinja2 import meta, TemplateError
import pytest
import promptsource.templates
from promptsource.utils import get_dataset_builder
from uuid import UUID

# Sets up Jinja environment
env = promptsource.templates.env

# Loads templates and iterates over each data (sub)set
template_collection = promptsource.templates.TemplateCollection()


def test_uuids():
    """
    Checks that all UUIDs across promptsource are unique. (Although collisions
    are unlikely, copying and pasting YAML files could lead to duplicates.
    """
    all_uuids = {}

    # Iterates over all datasets
    for dataset_name, subset_name in template_collection.keys:

        # Iterates over each template for current data (sub)set
        dataset_templates = template_collection.get_dataset(dataset_name, subset_name)
        for template_name in dataset_templates.all_template_names:
            template = dataset_templates[template_name]

            uuid = template.get_id()

            if uuid in all_uuids:
                raise ValueError(f"Template {template_name} for dataset {dataset_name}/{subset_name} "
                                 f"has duplicate uuid {template.get_id()} as "
                                 f"{all_uuids[uuid][0]}/{all_uuids[uuid][1]}.")

            all_uuids[uuid] = (dataset_name, subset_name)


@pytest.mark.parametrize("dataset", template_collection.keys)
def test_dataset(dataset):
    """
    Validates all the templates in the repository with simple syntactic checks:
    0. Are all templates parsable YAML?
    1. Do all templates parse in Jinja and are all referenced variables in the dataset schema?
    2. Does the template contain a prompt/output separator "|||" ?
    3. Are all names and templates within a data (sub)set unique?
    4. Is the YAML dictionary properly formatted?
    5. Is the UUID valid?

    :param dataset: (dataset_name, subset_name) pair to test

    """
    dataset_name, subset_name = dataset

    # Loads dataset information
    tries = 0
    max_tries = 3
    while True:
        try:
            builder_instance = get_dataset_builder(dataset_name, subset_name)
            break
        except ConnectionError as e:
            if tries < max_tries:
                time.sleep(2)
                tries += 1
            else:
                raise e

    has_features = builder_instance.info.features is not None
    if has_features:
        features = builder_instance.info.features.keys()
        features = set([feature.replace("-", "_") for feature in features])

    # Initializes sets for checking uniqueness among templates
    template_name_set = set()
    template_jinja_set = set()

    # Iterates over each template for current data (sub)set
    dataset_templates = template_collection.get_dataset(dataset_name, subset_name)
    any_original = False
    for template_name in dataset_templates.all_template_names:
        template = dataset_templates[template_name]
        any_original = any_original or template.metadata.original_task
        # Check 1: Jinja and all features valid?
        try:
            parse = env.parse(template.jinja)
        except TemplateError as e:
            raise ValueError(f"Template for dataset {dataset_name}/{subset_name} "
                             f"with uuid {template.get_id()} failed to parse.") from e

        variables = meta.find_undeclared_variables(parse)
        for variable in variables:
            if has_features and variable not in features and variable != "answer_choices":
                raise ValueError(f"Template for dataset {dataset_name}/{subset_name} "
                                 f"with uuid {template.get_id()} has unrecognized variable {variable}.")

        # Check 2: Prompt/output separator present?
        if "|||" not in template.jinja:
            raise ValueError(f"Template for dataset {dataset_name}/{subset_name} "
                             f"with uuid {template.get_id()} has no prompt/output separator.")

        # Check 3: Unique names and templates?
        if template.get_name() in template_name_set:
            raise ValueError(f"Template for dataset {dataset_name}/{subset_name} "
                             f"with uuid {template.get_id()} has duplicate name.")

        if template.jinja in template_jinja_set:
            raise ValueError(f"Template for dataset {dataset_name}/{subset_name} "
                             f"with uuid {template.get_id()} has duplicate definition.")

        template_name_set.add(template.get_name())
        template_jinja_set.add(template.jinja)

        # Check 4: Is the YAML dictionary properly formatted?
        try:
            if dataset_templates.templates[template.get_id()] != template:
                raise ValueError(f"Template for dataset {dataset_name}/{subset_name} "
                                 f"with uuid {template.get_id()} has wrong YAML key.")
        except KeyError as e:
            raise ValueError(f"Template for dataset {dataset_name}/{subset_name} "
                             f"with uuid {template.get_id()} has wrong YAML key.") from e

        # Check 5: Is the UUID valid?
        UUID(template.get_id())

    # Turned off for now until we fix.
    #assert any_original, "There must be at least one original task template for each dataset"
