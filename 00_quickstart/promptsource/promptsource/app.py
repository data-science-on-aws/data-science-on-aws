import argparse
import functools
import multiprocessing
import os
import textwrap
from hashlib import sha256
from multiprocessing import Manager, Pool

import pandas as pd
import plotly.express as px
import streamlit as st
from datasets import get_dataset_infos
from datasets.info import DatasetInfosDict
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import DjangoLexer

from promptsource import DEFAULT_PROMPTSOURCE_CACHE_HOME
from promptsource.session import _get_state
from promptsource.templates import INCLUDED_USERS, LANGUAGES, METRICS, DatasetTemplates, Template, TemplateCollection
from promptsource.utils import (
    get_dataset,
    get_dataset_confs,
    list_datasets,
    removeHyphen,
    renameDatasetColumn,
    render_features,
)


DATASET_INFOS_CACHE_DIR = os.path.join(DEFAULT_PROMPTSOURCE_CACHE_HOME, "DATASET_INFOS")
os.makedirs(DATASET_INFOS_CACHE_DIR, exist_ok=True)

# Python 3.8 switched the default start method from fork to spawn. OS X also has
# some issues related to fork, eee, e.g., https://github.com/bigscience-workshop/promptsource/issues/572
# so we make sure we always use spawn for consistency
multiprocessing.set_start_method("spawn", force=True)


def get_infos(all_infos, d_name):
    """
    Wrapper for mutliprocess-loading of dataset infos

    :param all_infos: multiprocess-safe dictionary
    :param d_name: dataset name
    """
    d_name_bytes = d_name.encode("utf-8")
    d_name_hash = sha256(d_name_bytes)
    foldername = os.path.join(DATASET_INFOS_CACHE_DIR, d_name_hash.hexdigest())
    if os.path.isdir(foldername):
        infos_dict = DatasetInfosDict.from_directory(foldername)
    else:
        infos = get_dataset_infos(d_name)
        infos_dict = DatasetInfosDict(infos)
        os.makedirs(foldername)
        infos_dict.write_to_directory(foldername)
    all_infos[d_name] = infos_dict


def format_language(tag):
    """
    Formats a language tag for display in the UI.

    For example, if the tag is "en", then the function returns "en (English)"
    :param tag: language tag
    :return: formatted language name
    """
    return tag + " (" + LANGUAGES[tag] + ")"


# add an argument for read-only
# At the moment, streamlit does not handle python script arguments gracefully.
# Thus, for read-only mode, you have to type one of the below two:
# streamlit run promptsource/app.py -- -r
# streamlit run promptsource/app.py -- --read-only
# Check https://github.com/streamlit/streamlit/issues/337 for more information.
parser = argparse.ArgumentParser(description="run app.py with args")
parser.add_argument("-r", "--read-only", action="store_true", help="whether to run it as read-only mode")

args = parser.parse_args()
if args.read_only:
    select_options = ["Helicopter view", "Prompted dataset viewer"]
    side_bar_title_prefix = "Promptsource (Read only)"
else:
    select_options = ["Helicopter view", "Prompted dataset viewer", "Sourcing"]
    side_bar_title_prefix = "Promptsource"

#
# Cache functions
#
get_dataset = st.cache(allow_output_mutation=True)(get_dataset)
get_dataset_confs = st.cache(get_dataset_confs)
list_datasets = st.cache(list_datasets)


def run_app():
    #
    # Loads session state
    #
    state = _get_state()

    def reset_template_state():
        state.template_name = None
        state.jinja = None
        state.reference = None

    #
    # Initial page setup
    #
    st.set_page_config(page_title="Promptsource", layout="wide")
    st.sidebar.markdown(
        "<center><a href='https://github.com/bigscience-workshop/promptsource'>💻Github - Promptsource\n\n</a></center>",
        unsafe_allow_html=True,
    )
    mode = st.sidebar.selectbox(
        label="Choose a mode",
        options=select_options,
        index=0,
        key="mode_select",
    )
    st.sidebar.title(f"{side_bar_title_prefix} 🌸 - {mode}")

    #
    # Adds pygments styles to the page.
    #
    st.markdown(
        "<style>" + HtmlFormatter(style="friendly").get_style_defs(".highlight") + "</style>", unsafe_allow_html=True
    )

    WIDTH = 140

    def show_jinja(t, width=WIDTH):
        def replace_linebreaks(t):
            """
            st.write does not handle double breaklines very well. When it encounters `\n\n`, it exit the curent <div> block.
            Explicitely replacing all `\n` with their html equivalent to bypass this issue.
            Also stripping the trailing `\n` first.
            """
            return t.strip("\n").replace("\n", "<br/>")

        wrap = textwrap.fill(t, width=width, replace_whitespace=False)
        out = highlight(wrap, DjangoLexer(), HtmlFormatter())
        out = replace_linebreaks(out)
        st.write(out, unsafe_allow_html=True)

    def show_text(t, width=WIDTH, with_markdown=False):
        wrap = [textwrap.fill(subt, width=width, replace_whitespace=False) for subt in t.split("\n")]
        wrap = "\n".join(wrap)
        if with_markdown:
            st.write(wrap, unsafe_allow_html=True)
        else:
            st.text(wrap)

    if mode == "Helicopter view":
        st.title("High level metrics")
        st.write("This will take a minute to collect.")
        st.write(
            "If you want to contribute, please refer to the instructions in "
            + "[Contributing](https://github.com/bigscience-workshop/promptsource/blob/main/CONTRIBUTING.md)."
        )

        #
        # Loads template data
        #
        try:
            template_collection = TemplateCollection()
        except FileNotFoundError:
            st.error(
                "Unable to find the prompt folder!\n\n"
                "We expect the folder to be in the working directory. "
                "You might need to restart the app in the root directory of the repo."
            )
            st.stop()

        #
        # Global metrics
        #
        counts = template_collection.get_templates_count()
        nb_prompted_datasets = len(counts)
        st.write(f"## Number of *prompted datasets*: `{nb_prompted_datasets}`")
        nb_prompts = sum(counts.values())
        st.write(f"## Number of *prompts*: `{nb_prompts}`")

        #
        # Metrics per dataset/subset
        #
        # Download dataset infos (multiprocessing download)
        manager = Manager()
        all_infos = manager.dict()
        all_datasets = list(set([t[0] for t in template_collection.keys]))

        pool = Pool(processes=multiprocessing.cpu_count())
        pool.map(functools.partial(get_infos, all_infos), all_datasets)
        pool.close()
        pool.join()

        results = []
        for (dataset_name, subset_name) in template_collection.keys:
            # Collect split sizes (train, validation and test)
            if dataset_name not in all_infos:
                infos = get_dataset_infos(dataset_name)
                all_infos[dataset_name] = infos
            else:
                infos = all_infos[dataset_name]
            if infos:
                if subset_name is None:
                    subset_infos = infos[list(infos.keys())[0]]
                else:
                    subset_infos = infos[subset_name]

                try:
                    split_sizes = {k: v.num_examples for k, v in subset_infos.splits.items()}
                except Exception:
                    # Fixing bug in some community datasets.
                    # For simplicity, just filling `split_sizes` with nothing, so the displayed split sizes will be 0.
                    split_sizes = {}
            else:
                split_sizes = {}

            # Collect template counts, original task counts and names
            dataset_templates = template_collection.get_dataset(dataset_name, subset_name)
            results.append(
                {
                    "Dataset name": dataset_name,
                    "Subset name": "∅" if subset_name is None else subset_name,
                    "Train size": split_sizes["train"] if "train" in split_sizes else 0,
                    "Validation size": split_sizes["validation"] if "validation" in split_sizes else 0,
                    "Test size": split_sizes["test"] if "test" in split_sizes else 0,
                    "Number of prompts": len(dataset_templates),
                    "Number of original task prompts": sum(
                        [bool(t.metadata.original_task) for t in dataset_templates.templates.values()]
                    ),
                    "Prompt names": [t.name for t in dataset_templates.templates.values()],
                }
            )
        results_df = pd.DataFrame(results)
        results_df.sort_values(["Number of prompts"], inplace=True, ascending=False)
        results_df.reset_index(drop=True, inplace=True)

        nb_training_instances = results_df["Train size"].sum()
        st.write(f"## Number of *training instances*: `{nb_training_instances}`")

        plot_df = results_df[["Dataset name", "Subset name", "Train size", "Number of prompts"]].copy()
        plot_df["Name"] = plot_df["Dataset name"] + " - " + plot_df["Subset name"]
        plot_df.sort_values(["Train size"], inplace=True, ascending=False)
        fig = px.bar(
            plot_df,
            x="Name",
            y="Train size",
            hover_data=["Dataset name", "Subset name", "Number of prompts"],
            log_y=True,
            title="Number of training instances per data(sub)set - y-axis is in logscale",
        )
        fig.update_xaxes(visible=False, showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            f"- Top 3 training subsets account for `{100 * plot_df[:3]['Train size'].sum() / nb_training_instances:.2f}%` of the training instances."
        )
        biggest_training_subset = plot_df.iloc[0]
        st.write(
            f"- Biggest training subset is *{biggest_training_subset['Name']}* with `{biggest_training_subset['Train size']}` instances"
        )
        smallest_training_subset = plot_df[plot_df["Train size"] > 0].iloc[-1]
        st.write(
            f"- Smallest training subset is *{smallest_training_subset['Name']}* with `{smallest_training_subset['Train size']}` instances"
        )

        st.markdown("***")
        st.write("Details per dataset")
        st.table(results_df)

    else:
        # Combining mode `Prompted dataset viewer` and `Sourcing` since the
        # backbone of the interfaces is the same
        assert mode in ["Prompted dataset viewer", "Sourcing"], ValueError(
            f"`mode` ({mode}) should be in `[Helicopter view, Prompted dataset viewer, Sourcing]`"
        )

        #
        # Loads dataset information
        #

        dataset_list = list_datasets()
        ag_news_index = dataset_list.index("ag_news")

        #
        # Select a dataset - starts with ag_news
        #
        dataset_key = st.sidebar.selectbox(
            "Dataset",
            dataset_list,
            key="dataset_select",
            index=ag_news_index,
            help="Select the dataset to work on.",
        )

        #
        # If a particular dataset is selected, loads dataset and template information
        #
        if dataset_key is not None:

            #
            # Check for subconfigurations (i.e. subsets)
            #
            configs = get_dataset_confs(dataset_key)
            conf_option = None
            if len(configs) > 0:
                conf_option = st.sidebar.selectbox("Subset", configs, index=0, format_func=lambda a: a.name)

            subset_name = str(conf_option.name) if conf_option else None
            try:
                dataset = get_dataset(dataset_key, subset_name)
            except OSError as e:
                st.error(
                    f"Some datasets are not handled automatically by `datasets` and require users to download the "
                    f"dataset manually. It is possibly the case for {dataset_key}{f'/{subset_name}' if subset_name is not None else ''}. "
                    f"\n\nIf so, please download the raw dataset to `~/.cache/promptsource/{dataset_key}{f'/{subset_name}' if subset_name is not None else ''}`. "
                    f"\n\nYou can choose another cache directory by overriding `PROMPTSOURCE_MANUAL_DATASET_DIR` environment "
                    f"variable and downloading raw dataset to `$PROMPTSOURCE_MANUAL_DATASET_DIR/{dataset_key}{f'/{subset_name}' if subset_name is not None else ''}`"
                    f"\n\nOriginal error:\n{str(e)}"
                )
                st.stop()
            except Exception as e:
                st.error(
                    f"An error occured while loading the dataset {dataset_key}{f'/{subset_name}' if subset_name is not None else ''}. "
                    f"\\n\nOriginal error:\n{str(e)}"
                )

            splits = list(dataset.keys())
            index = 0
            if "train" in splits:
                index = splits.index("train")
            split = st.sidebar.selectbox("Split", splits, key="split_select", index=index)
            dataset = dataset[split]
            dataset = renameDatasetColumn(dataset)

            #
            # Loads template data
            #
            try:
                dataset_templates = DatasetTemplates(dataset_key, conf_option.name if conf_option else None)
            except FileNotFoundError:
                st.error(
                    "Unable to find the prompt folder!\n\n"
                    "We expect the folder to be in the working directory. "
                    "You might need to restart the app in the root directory of the repo."
                )
                st.stop()

            template_list = dataset_templates.all_template_names
            num_templates = len(template_list)
            st.sidebar.write(
                "No of prompts created for "
                + f"`{dataset_key + (('/' + conf_option.name) if conf_option else '')}`"
                + f": **{str(num_templates)}**"
            )

            if mode == "Prompted dataset viewer":
                if num_templates > 0:
                    template_name = st.sidebar.selectbox(
                        "Prompt name",
                        template_list,
                        key="template_select",
                        index=0,
                        help="Select the prompt to visualize.",
                    )

                step = 50
                example_index = st.sidebar.number_input(
                    f"Select the example index (Size = {len(dataset)})",
                    min_value=0,
                    max_value=len(dataset) - step,
                    value=0,
                    step=step,
                    key="example_index_number_input",
                    help="Offset = 50.",
                )
            else:  # mode = Sourcing
                st.sidebar.subheader("Select Example")
                example_index = st.sidebar.slider("Select the example index", 0, len(dataset) - 1)

                example = dataset[example_index]
                example = removeHyphen(example)

                st.sidebar.write(example)

            st.sidebar.subheader("Dataset Schema")
            rendered_features = render_features(dataset.features)
            st.sidebar.write(rendered_features)

            #
            # Display dataset information
            #
            st.header("Dataset: " + dataset_key + " " + (("/ " + conf_option.name) if conf_option else ""))

            # If we have a custom dataset change the source link to the hub
            split_dataset_key = dataset_key.split("/")
            possible_user = split_dataset_key[0]
            if len(split_dataset_key) > 1 and possible_user in INCLUDED_USERS:
                source_link = "https://huggingface.co/datasets/%s/blob/main/%s.py" % (
                    dataset_key,
                    split_dataset_key[-1],
                )
            else:
                source_link = "https://github.com/huggingface/datasets/blob/master/datasets/%s/%s.py" % (
                    dataset_key,
                    dataset_key,
                )

            st.markdown("*Homepage*: " + dataset.info.homepage + "\n\n*Dataset*: " + source_link)

            md = """
            %s
            """ % (
                dataset.info.description.replace("\\", "") if dataset_key else ""
            )
            st.markdown(md)

            #
            # Body of the app: display prompted examples in mode `Prompted dataset viewer`
            # or text boxes to create new prompts in mode `Sourcing`
            #
            if mode == "Prompted dataset viewer":
                #
                # Display template information
                #
                if num_templates > 0:
                    template = dataset_templates[template_name]
                    st.subheader("Prompt")
                    st.markdown("##### Name")
                    st.text(template.name)
                    st.markdown("##### Reference")
                    st.text(template.reference)
                    st.markdown("##### Original Task? ")
                    st.text(template.metadata.original_task)
                    st.markdown("##### Choices in template? ")
                    st.text(template.metadata.choices_in_prompt)
                    st.markdown("##### Metrics")
                    st.text(", ".join(template.metadata.metrics) if template.metadata.metrics else None)
                    st.markdown("##### Prompt Languages")
                    if template.metadata.languages:
                        st.text(", ".join([format_language(tag) for tag in template.metadata.languages]))
                    else:
                        st.text(None)
                    st.markdown("##### Answer Choices")
                    if template.get_answer_choices_expr() is not None:
                        show_jinja(template.get_answer_choices_expr())
                    else:
                        st.text(None)
                    st.markdown("##### Jinja template")
                    splitted_template = template.jinja.split("|||")
                    st.markdown("###### Input template")
                    show_jinja(splitted_template[0].strip())
                    if len(splitted_template) > 1:
                        st.markdown("###### Target template")
                        show_jinja(splitted_template[1].strip())
                    st.markdown("***")

                #
                # Display a couple (steps) examples
                #
                for ex_idx in range(example_index, example_index + step):
                    if ex_idx >= len(dataset):
                        continue
                    example = dataset[ex_idx]
                    example = removeHyphen(example)
                    col1, _, col2 = st.beta_columns([12, 1, 12])
                    with col1:
                        st.write(example)
                    if num_templates > 0:
                        with col2:
                            prompt = template.apply(example, highlight_variables=False)
                            if prompt == [""]:
                                st.write("∅∅∅ *Blank result*")
                            else:
                                st.write("Input")
                                show_text(prompt[0])
                                if len(prompt) > 1:
                                    st.write("Target")
                                    show_text(prompt[1])
                    st.markdown("***")
            else:  # mode = Sourcing
                st.markdown("## Prompt Creator")

                #
                # Create a new template or select an existing one
                #
                col1a, col1b, _, col2 = st.beta_columns([9, 9, 1, 6])

                # current_templates_key and state.templates_key are keys for the templates object
                current_templates_key = (dataset_key, conf_option.name if conf_option else None)

                # Resets state if there has been a change in templates_key
                if state.templates_key != current_templates_key:
                    state.templates_key = current_templates_key
                    reset_template_state()

                with col1a, st.form("new_template_form"):
                    new_template_name = st.text_input(
                        "Create a New Prompt",
                        key="new_template",
                        value="",
                        help="Enter name and hit enter to create a new prompt.",
                    )
                    new_template_submitted = st.form_submit_button("Create")
                    if new_template_submitted:
                        if new_template_name in dataset_templates.all_template_names:
                            st.error(
                                f"A prompt with the name {new_template_name} already exists "
                                f"for dataset {state.templates_key}."
                            )
                        elif new_template_name == "":
                            st.error("Need to provide a prompt name.")
                        else:
                            template = Template(new_template_name, "", "")
                            dataset_templates.add_template(template)
                            reset_template_state()
                            state.template_name = new_template_name
                    else:
                        state.new_template_name = None

                with col1b, st.beta_expander("or Select Prompt", expanded=True):
                    template_list = dataset_templates.all_template_names
                    if state.template_name:
                        index = template_list.index(state.template_name)
                    else:
                        index = 0
                    state.template_name = st.selectbox(
                        "", template_list, key="template_select", index=index, help="Select the prompt to work on."
                    )

                    if st.button("Delete Prompt", key="delete_prompt"):
                        dataset_templates.remove_template(state.template_name)
                        reset_template_state()

                variety_guideline = """
                :heavy_exclamation_mark::question:Creating a diverse set of prompts whose differences go beyond surface wordings (i.e. marginally changing 2 or 3 words) is highly encouraged.
                Ultimately, the hope is that exposing the model to such a diversity will have a non-trivial impact on the model's robustness to the prompt formulation.
                \r**To get various prompts, you can try moving the cursor along theses axes**:
                \n- **Interrogative vs affirmative form**: Ask a question about an attribute of the inputs or tell the model to decide something about the input.
                \n- **Task description localization**: where is the task description blended with the inputs? In the beginning, in the middle, at the end?
                \n- **Implicit situation or contextualization**: how explicit is the query? For instance, *Given this review, would you buy this product?* is an indirect way to ask whether the review is positive.
                """

                col1, _, _ = st.beta_columns([18, 1, 6])
                with col1:
                    if state.template_name is not None:
                        show_text(variety_guideline, with_markdown=True)

                #
                # Edit the created or selected template
                #
                col1, _, col2 = st.beta_columns([18, 1, 6])
                with col1:
                    if state.template_name is not None:
                        template = dataset_templates[state.template_name]
                        #
                        # If template is selected, displays template editor
                        #
                        with st.form("edit_template_form"):
                            updated_template_name = st.text_input("Name", value=template.name)
                            state.reference = st.text_input(
                                "Prompt Reference",
                                help="Short description of the prompt and/or paper reference for the prompt.",
                                value=template.reference,
                            )

                            # Metadata
                            state.metadata = template.metadata
                            state.metadata.original_task = st.checkbox(
                                "Original Task?",
                                value=template.metadata.original_task,
                                help="Prompt asks model to perform the original task designed for this dataset.",
                            )
                            state.metadata.choices_in_prompt = st.checkbox(
                                "Choices in Template?",
                                value=template.metadata.choices_in_prompt,
                                help="Prompt explicitly lists choices in the template for the output.",
                            )

                            state.metadata.metrics = st.multiselect(
                                "Metrics",
                                sorted(METRICS),
                                default=template.metadata.metrics,
                                help="Select all metrics that are commonly used (or should "
                                "be used if a new task) to evaluate this prompt.",
                            )

                            state.metadata.languages = st.multiselect(
                                "Prompt Languages",
                                sorted(LANGUAGES.keys()),
                                default=template.metadata.languages,
                                format_func=format_language,
                                help="Select all languages used in this prompt. "
                                "This annotation is independent from the language(s) "
                                "of the dataset.",
                            )

                            # Answer choices
                            if template.get_answer_choices_expr() is not None:
                                answer_choices = template.get_answer_choices_expr()
                            else:
                                answer_choices = ""
                            state.answer_choices = st.text_input(
                                "Answer Choices",
                                value=answer_choices,
                                help="A Jinja expression for computing answer choices. "
                                "Separate choices with a triple bar (|||).",
                            )

                            # Jinja
                            state.jinja = st.text_area("Template", height=40, value=template.jinja)

                            # Submit form
                            if st.form_submit_button("Save"):
                                if (
                                    updated_template_name in dataset_templates.all_template_names
                                    and updated_template_name != state.template_name
                                ):
                                    st.error(
                                        f"A prompt with the name {updated_template_name} already exists "
                                        f"for dataset {state.templates_key}."
                                    )
                                elif updated_template_name == "":
                                    st.error("Need to provide a prompt name.")
                                else:
                                    # Parses state.answer_choices
                                    if state.answer_choices == "":
                                        updated_answer_choices = None
                                    else:
                                        updated_answer_choices = state.answer_choices

                                    dataset_templates.update_template(
                                        state.template_name,
                                        updated_template_name,
                                        state.jinja,
                                        state.reference,
                                        state.metadata,
                                        updated_answer_choices,
                                    )
                                    # Update the state as well
                                    state.template_name = updated_template_name
                #
                # Displays template output on current example if a template is selected
                # (in second column)
                #
                with col2:
                    if state.template_name is not None:
                        st.empty()
                        template = dataset_templates[state.template_name]
                        prompt = template.apply(example)
                        if prompt == [""]:
                            st.write("∅∅∅ *Blank result*")
                        else:
                            st.write("Input")
                            show_text(prompt[0], width=40)
                            if len(prompt) > 1:
                                st.write("Target")
                                show_text(prompt[1], width=40)

    #
    # Must sync state at end
    #
    state.sync()


if __name__ == "__main__":
    run_app()
