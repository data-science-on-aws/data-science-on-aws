# Manipulating prompts
PromptSource implements 4 classes to store, manipulate and use prompts and their metadata: `Template`, `Metadata`, `DatasetTemplates` and `TemplateCollection`. All of them are implemented in [`templates.py`](promptsource/templates.py)

## Class `Template` and `Metadata`
`Template` is a class that wraps a prompt, its associated metadata, and implements the helper functions to use the prompt.

Instances of `Template` have the following main methods that will come handy:
* `apply(example, truncate=True, highlight_variables=False)`: Create a prompted example by applying the template to the given example
  - `example` (Dict): the dataset example to create a prompt for
  - `truncate` (Bool, default to `True`): if True, example fields will be truncated to `TEXT_VAR_LENGTH` chars
  - `highlight_variables`(Bool, default to `False`): highlight the added variables (internal use for the app rendering)
* `get_id()`: Get the uuid of the prompt
* `get_name()`: Get the name of the prompt
* `get_reference()`: Get any additional information about the prompt (such as bibliographic reference)
* `get_answer_choices_list(example)`: If applicable, returns a list of answer choices for a given example.

Each `Template` also has a `metadata` attribute, an instance of the class `Metadata` that encapsulates the following 3 attributes:
* `original_task`: If True, this prompt asks a model to perform the original task designed for this dataset.
* `choices_in_prompt`: If True, the answer choices are included in the templates such that models see those choices in the input. Only applicable to classification tasks.
* `metrics`: List of strings denoting metrics to use for evaluation

## Class `DatasetTemplates`
`DatasetTemplates` is a class that wraps all the prompts (each of them are instances of `Template`) for a specific dataset/subset and implements all the helper functions necessary to read/write to the YAML file in which the prompts are saved.

You will likely mainly be interested in getting the existing prompts and their names for a given dataset. You can do that with the following instantiation:
```python
>>> template_key = f"{dataset_name}/{subset_name}" if subset_name is not None else dataset_name
>>> prompts = DatasetTemplates(template_key)
>>> len(prompts) # Returns the number of prompts for the given dataset
>>> prompts.all_template_names # Returns a sorted list of all templates names for this dataset
```

## Class `TemplateCollection`
`TemplateCollection` is a class that encapsulates all the prompts available under PromptSource by wrapping the `DatasetTemplates` class. It initializes the `DatasetTemplates` for all existing template folders, gives access to each `DatasetTemplates`, and provides aggregated counts overall `DatasetTemplates`.

The main methods are:
* `get_dataset(dataset_name, subset_name)`: Return the DatasetTemplates object corresponding to the dataset name
  - `dataset_name` (Str): name of the dataset to get
  - `subset_name` (Str, default to None): name of the subset
* `get_templates_count()`: Return the overall number count over all datasets. NB: we don't breakdown datasets into subsets for the count, i.e subsets count are included into the dataset count
