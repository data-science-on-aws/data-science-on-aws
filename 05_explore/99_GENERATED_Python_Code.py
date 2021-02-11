from pyspark.sql.session import SparkSession
from pyspark.sql.dataframe import DataFrame

#  You may want to configure the Spark Context with the right credentials provider.
spark = SparkSession.builder.master("local").getOrCreate()

mode = None


def capture_stdout(func, *args, **kwargs):
    """Capture standard output to a string buffer"""

    from contextlib import redirect_stdout
    import io

    stdout_string = io.StringIO()
    with redirect_stdout(stdout_string):
        func(*args, **kwargs)
    return stdout_string.getvalue()


def convert_or_coerce(pandas_df, spark):
    """Convert pandas df to pyspark df and coerces the mixed cols to string"""
    import re

    try:
        return spark.createDataFrame(pandas_df)
    except TypeError as e:
        match = re.search(r".*field (\w+).*Can not merge type.*", str(e))
        if match is None:
            raise e
        mixed_col_name = match.group(1)
        # Coercing the col to string
        pandas_df[mixed_col_name] = pandas_df[mixed_col_name].astype("str")
        return pandas_df


def default_spark(value):
    return {"default": value}


def default_spark_with_stdout(df, stdout):
    return {
        "default": df,
        "stdout": stdout,
    }


def default_spark_with_trained_parameters(value, trained_parameters):
    return {"default": value, "trained_parameters": trained_parameters}


def default_spark_with_trained_parameters_and_state(df, trained_parameters, state):
    return {"default": df, "trained_parameters": trained_parameters, "state": state}


def dispatch(key_name, args, kwargs, funcs):
    """
    Dispatches to another operator based on a key in the passed parameters.
    This also slices out any parameters using the parameter_name passed in,
    and will reassemble the trained_parameters correctly after invocation.

    Args:
        key_name: name of the key in kwargs used to identify the function to use.
        args: dataframe that will be passed as the first set of parameters to the function.
        kwargs: keyword arguments that key_name will be found in; also where args will be passed to parameters.
                These are also expected to include trained_parameters if there are any.
        funcs: dictionary mapping from value of key_name to (function, parameter_name)

    """
    if key_name not in kwargs:
        raise OperatorCustomerError(f"Missing required parameter {key_name}")

    operator = kwargs[key_name]

    if operator not in funcs:
        raise OperatorCustomerError(f"Invalid choice selected for {key_name}. {operator} is not supported.")

    func, parameter_name = funcs[operator]

    # Extract out the parameters that should be available.
    func_params = kwargs.get(parameter_name, {})
    if func_params is None:
        func_params = {}

    # Extract out any trained parameters.
    specific_trained_parameters = None
    if "trained_parameters" in kwargs:
        trained_parameters = kwargs["trained_parameters"]
        if trained_parameters is not None and parameter_name in trained_parameters:
            specific_trained_parameters = trained_parameters[parameter_name]
    func_params["trained_parameters"] = specific_trained_parameters

    # Execute the function (should return a dict).
    result = func(*args, **func_params)

    # Check if the result contains any trained parameters and remap them to the proper structure.
    if result is not None and "trained_parameters" in result:
        existing_trained_parameters = kwargs.get("trained_parameters")
        updated_trained_parameters = result["trained_parameters"]

        if existing_trained_parameters is not None or updated_trained_parameters is not None:
            existing_trained_parameters = (
                existing_trained_parameters if existing_trained_parameters is not None else {}
            )
            existing_trained_parameters[parameter_name] = result["trained_parameters"]

            # Update the result trained_parameters so they are part of the original structure.
            result["trained_parameters"] = existing_trained_parameters
        else:
            # If the given trained parameters were None and the returned trained parameters were None, don't return anything.
            del result["trained_parameters"]

    return result


def get_dataframe_with_sequence_ids(df: DataFrame):
    df_cols = df.columns
    rdd_with_seq = df.rdd.zipWithIndex()
    df_with_seq = rdd_with_seq.toDF()
    df_with_seq = df_with_seq.withColumnRenamed("_2", "_seq_id_")
    for col_name in df_cols:
        df_with_seq = df_with_seq.withColumn(col_name, df_with_seq["_1"].getItem(col_name))
    df_with_seq = df_with_seq.drop("_1")
    return df_with_seq


def get_execution_state(status: str, message=None):
    return {"status": status, "message": message}


class OperatorCustomerError(Exception):
    """Error type for Customer Errors in Spark Operators"""


from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler,
    StandardScalerModel,
    RobustScaler,
    RobustScalerModel,
    MinMaxScaler,
    MinMaxScalerModel,
    MaxAbsScaler,
    MaxAbsScalerModel,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as sf
from pyspark.sql.types import NumericType


def process_numeric_standard_scaler(
    df, input_column=None, center=None, scale=None, output_column=None, trained_parameters=None
):
    expects_column(df, input_column, "Input column")
    expects_valid_column_name(output_column, "Output column", nullable=True)
    process_numeric_expects_numeric_column(df, input_column)

    temp_vector_col = temp_col_name(df)
    assembled = VectorAssembler(inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="keep").transform(
        df
    )
    assembled_wo_nans = VectorAssembler(
        inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="skip"
    ).transform(df)
    temp_normalized_vector_col = temp_col_name(assembled)

    trained_parameters = load_trained_parameters(
        trained_parameters, {"input_column": input_column, "center": center, "scale": scale}
    )

    scaler_model, scaler_model_loaded = load_pyspark_model_from_trained_parameters(
        trained_parameters, StandardScalerModel, "scaler_model"
    )

    if scaler_model is None:
        scaler = StandardScaler(
            inputCol=temp_vector_col,
            outputCol=temp_normalized_vector_col,
            withStd=parse_parameter(bool, scale, "scale", True),
            withMean=parse_parameter(bool, center, "center", False),
        )
        scaler_model = fit_and_save_model(trained_parameters, "scaler_model", scaler, assembled_wo_nans)

    output_df = transform_using_trained_model(scaler_model, assembled, scaler_model_loaded)

    # convert the resulting vector back to numeric
    temp_flattened_vector_col = temp_col_name(output_df)
    output_df = output_df.withColumn(temp_flattened_vector_col, vector_to_array(temp_normalized_vector_col))

    # keep only the final scaled column.
    output_column = input_column if output_column is None or not output_column else output_column
    output_column_value = sf.col(temp_flattened_vector_col)[0].alias(output_column)
    output_df = output_df.withColumn(output_column, output_column_value)
    final_columns = list(dict.fromkeys((list(df.columns) + [output_column])))
    output_df = output_df.select(final_columns)

    return default_spark_with_trained_parameters(output_df, trained_parameters)


def process_numeric_robust_scaler(
    df,
    input_column=None,
    lower_quantile=None,
    upper_quantile=None,
    center=None,
    scale=None,
    output_column=None,
    trained_parameters=None,
):
    expects_column(df, input_column, "Input column")
    expects_valid_column_name(output_column, "Output column", nullable=True)
    process_numeric_expects_numeric_column(df, input_column)

    temp_vector_col = temp_col_name(df)
    assembled = VectorAssembler(inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="keep").transform(
        df
    )
    assembled_wo_nans = VectorAssembler(
        inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="skip"
    ).transform(df)
    temp_normalized_vector_col = temp_col_name(assembled)

    trained_parameters = load_trained_parameters(
        trained_parameters,
        {
            "input_column": input_column,
            "center": center,
            "scale": scale,
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
        },
    )

    scaler_model, scaler_model_loaded = load_pyspark_model_from_trained_parameters(
        trained_parameters, RobustScalerModel, "scaler_model"
    )

    if scaler_model is None:
        scaler = RobustScaler(
            inputCol=temp_vector_col,
            outputCol=temp_normalized_vector_col,
            lower=parse_parameter(float, lower_quantile, "lower_quantile", 0.25),
            upper=parse_parameter(float, upper_quantile, "upper_quantile", 0.75),
            withCentering=parse_parameter(bool, center, "with_centering", False),
            withScaling=parse_parameter(bool, scale, "with_scaling", True),
        )
        scaler_model = fit_and_save_model(trained_parameters, "scaler_model", scaler, assembled_wo_nans)

    output_df = transform_using_trained_model(scaler_model, assembled, scaler_model_loaded)

    # convert the resulting vector back to numeric
    temp_flattened_vector_col = temp_col_name(output_df)
    output_df = output_df.withColumn(temp_flattened_vector_col, vector_to_array(temp_normalized_vector_col))

    # keep only the final scaled column.
    output_column = input_column if output_column is None or not output_column else output_column
    output_column_value = sf.col(temp_flattened_vector_col)[0].alias(output_column)
    output_df = output_df.withColumn(output_column, output_column_value)
    final_columns = list(dict.fromkeys((list(df.columns) + [output_column])))
    output_df = output_df.select(final_columns)

    return default_spark_with_trained_parameters(output_df, trained_parameters)


def process_numeric_min_max_scaler(
    df, input_column=None, min=None, max=None, output_column=None, trained_parameters=None
):
    expects_column(df, input_column, "Input column")
    expects_valid_column_name(output_column, "Output column", nullable=True)
    process_numeric_expects_numeric_column(df, input_column)

    temp_vector_col = temp_col_name(df)
    assembled = VectorAssembler(inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="keep").transform(
        df
    )
    assembled_wo_nans = VectorAssembler(
        inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="skip"
    ).transform(df)
    temp_normalized_vector_col = temp_col_name(assembled)

    trained_parameters = load_trained_parameters(
        trained_parameters,
        {
            "input_column": input_column,
            "min": min,
            "max": max,
        },
    )

    scaler_model, scaler_model_loaded = load_pyspark_model_from_trained_parameters(
        trained_parameters, MinMaxScalerModel, "scaler_model"
    )

    if scaler_model is None:
        scaler = MinMaxScaler(
            inputCol=temp_vector_col,
            outputCol=temp_normalized_vector_col,
            min=parse_parameter(float, min, "min", 0.0),
            max=parse_parameter(float, max, "max", 1.0),
        )
        scaler_model = fit_and_save_model(trained_parameters, "scaler_model", scaler, assembled_wo_nans)

    output_df = transform_using_trained_model(scaler_model, assembled, scaler_model_loaded)

    # convert the resulting vector back to numeric
    temp_flattened_vector_col = temp_col_name(output_df)
    output_df = output_df.withColumn(temp_flattened_vector_col, vector_to_array(temp_normalized_vector_col))

    # keep only the final scaled column.
    output_column = input_column if output_column is None or not output_column else output_column
    output_column_value = sf.col(temp_flattened_vector_col)[0].alias(output_column)
    output_df = output_df.withColumn(output_column, output_column_value)
    final_columns = list(dict.fromkeys((list(df.columns) + [output_column])))
    output_df = output_df.select(final_columns)

    return default_spark_with_trained_parameters(output_df, trained_parameters)


def process_numeric_max_absolute_scaler(df, input_column=None, output_column=None, trained_parameters=None):
    expects_column(df, input_column, "Input column")
    expects_valid_column_name(output_column, "Output column", nullable=True)
    process_numeric_expects_numeric_column(df, input_column)

    temp_vector_col = temp_col_name(df)
    assembled = VectorAssembler(inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="keep").transform(
        df
    )
    assembled_wo_nans = VectorAssembler(
        inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="skip"
    ).transform(df)
    temp_normalized_vector_col = temp_col_name(assembled)

    trained_parameters = load_trained_parameters(
        trained_parameters,
        {
            "input_column": input_column,
        },
    )

    scaler_model, scaler_model_loaded = load_pyspark_model_from_trained_parameters(
        trained_parameters, MinMaxScalerModel, "scaler_model"
    )

    if scaler_model is None:
        scaler = MinMaxScaler(inputCol=temp_vector_col, outputCol=temp_normalized_vector_col)
        scaler_model = fit_and_save_model(trained_parameters, "scaler_model", scaler, assembled_wo_nans)

    output_df = transform_using_trained_model(scaler_model, assembled, scaler_model_loaded)

    scaler = MaxAbsScaler(inputCol=temp_vector_col, outputCol=temp_normalized_vector_col)

    output_df = scaler.fit(assembled_wo_nans).transform(assembled)

    # convert the resulting vector back to numeric
    temp_flattened_vector_col = temp_col_name(output_df)
    output_df = output_df.withColumn(temp_flattened_vector_col, vector_to_array(temp_normalized_vector_col))

    # keep only the final scaled column.
    output_column = input_column if output_column is None or not output_column else output_column
    output_column_value = sf.col(temp_flattened_vector_col)[0].alias(output_column)
    output_df = output_df.withColumn(output_column, output_column_value)
    final_columns = list(dict.fromkeys((list(df.columns) + [output_column])))
    output_df = output_df.select(final_columns)

    return default_spark_with_trained_parameters(output_df, trained_parameters)


def process_numeric_expects_numeric_column(df, input_column):
    column_type = df.schema[input_column].dataType
    if not isinstance(column_type, NumericType):
        raise OperatorSparkOperatorCustomerError(
            f'Numeric column required. Please cast column to a numeric type first. Column "{input_column}" has type {column_type.simpleString()}.'
        )


def process_numeric_scale_values(df, **kwargs):
    return dispatch(
        "scaler",
        [df],
        kwargs,
        {
            "Standard scaler": (process_numeric_standard_scaler, "standard_scaler_parameters"),
            "Robust scaler": (process_numeric_robust_scaler, "robust_scaler_parameters"),
            "Min-max scaler": (process_numeric_min_max_scaler, "min_max_scaler_parameters"),
            "Max absolute scaler": (process_numeric_max_absolute_scaler, "max_absolute_scaler_parameters"),
        },
    )


#########################
# Athena helper methods #
#########################
import logging
import re
import uuid

from botocore.exceptions import ClientError


def glue_create_database_core(client, request):
    """Helper method for invoking the Glue service API call."""
    database_name = request["DatabaseInput"]["Name"]
    logging.debug("Attempting to create %s database in the Glue data catalog. Request is: %s", database_name, request)
    try:
        client.create_database(**request)
        logging.debug("%s database created successfully.", database_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "AlreadyExistsException":
            logging.debug("%s database already exists.", request["DatabaseInput"]["Name"])
        elif e.response["Error"]["Code"] == "AccessDeniedException":
            raise SparkOperatorCustomerError(
                "An error occurred when trying to create {} database in the Glue data catalog: {}".format(
                    database_name, e
                )
            )
        elif e.response["Error"]["Code"] == "ResourceNumberLimitExceededException":
            raise SparkOperatorCustomerError(
                "An error occurred when trying to create {} database in the Glue data catalog: {}".format(
                    database_name, e
                )
            )
        else:
            raise RuntimeError(
                "An error occurred when trying to create {} database in the Glue data catalog: {}".format(
                    database_name, e
                )
            )


def athena_start_query_execution_core(client, request):
    """Helper method for invoking the Athena service API call."""
    try:
        result = client.start_query_execution(**request)
    except Exception as e:
        raise RuntimeError(
            f"An error ({type(e).__name__}) occurred when trying to invoke `start_query_execution`: {e}"
        )
    return result


def athena_get_query_execution_core(client, request):
    """Helper method for invoking the Athena service API call."""
    try:
        result = client.get_query_execution(**request)
    except Exception as e:
        raise RuntimeError(f"An error ({type(e).__name__}) occurred when trying to invoke `get_query_execution`: {e}")
    return result


def athena_parse_query_execution_state(get_query_execution_response):
    """Helper method for parsing query execution state."""
    try:
        return get_query_execution_response["QueryExecution"]["Status"]["State"]
    except Exception as e:
        raise KeyError(f"Unable to parse state from Athena `get_query_execution` response: {e}")


def athena_delete_tmp_table(glue_client, ctas_table_name):
    # We will always create the temp table (output of CTAS execution) in the sagemaker_data_wrangler database with the
    # sagemaker_tmp_ prefix. This method attempts to delete the temp table from the sagemaker_data_wrangler database.

    ctas_database_name = "sagemaker_data_wrangler"

    if re.match("sagemaker_tmp_.*", ctas_table_name):
        try:
            logging.debug("Attempting to delete table %s, in database %s", ctas_table_name, ctas_database_name)
            glue_client.delete_table(DatabaseName=ctas_database_name, Name=ctas_table_name)
            logging.debug("%s table has been deleted successfully.", ctas_table_name)
        except ClientError as e:
            if e.response["Error"]["Code"] == "EntityNotFoundException":
                logging.debug(
                    "%s table does not exist in the database %s. It must have been deleted already.",
                    ctas_table_name,
                    ctas_database_name,
                )
            elif e.response["Error"]["Code"] == "AccessDeniedException":
                raise SparkOperatorCustomerError(
                    "An error occurred when trying to delete {} table from {} database: {}".format(
                        ctas_table_name, ctas_database_name, e
                    )
                )
            else:
                raise RuntimeError(
                    "An error occurred when trying to delete {} table from {} database: {}".format(
                        ctas_table_name, ctas_database_name, e
                    )
                )


def athena_create_database_if_not_exists(glue_client):
    # This method creates the sagemaker_data_wranger database in the Glue data catalog if it doesn't exist
    # already. This database is used to create and delete temporary tables that are created by the Athena CTAS
    # execution. Method is a no-op if the database exists already.

    database_name = "sagemaker_data_wrangler"
    create_database_request = {"DatabaseInput": {"Name": database_name}}
    glue_create_database_core(client=glue_client, request=create_database_request)


def athena_start_query_execution(dataset_definition, client):
    table_name_prefix = "sagemaker_tmp_"
    s3_base_location = dataset_definition["s3OutputLocation"]
    sql_query_string = dataset_definition["queryString"]
    output_format = dataset_definition["outputFormat"]
    database_name = dataset_definition["databaseName"]
    catalog_name = dataset_definition["catalogName"]

    # We will always execute the CTAS statement in the following catalog and database
    ctas_catalog_name = "awsdatacatalog"
    ctas_database_name = "sagemaker_data_wrangler"

    dataset_uuid = str(uuid.uuid4()).replace("-", "")
    ctas_table_name = table_name_prefix + dataset_uuid
    ctas_s3_output_location = f"{s3_base_location}{dataset_uuid}/data/"
    metadata_s3_output_location = f"{s3_base_location}{dataset_uuid}/metadata/"

    ctas_query = (
        f"CREATE TABLE {ctas_catalog_name}.{ctas_database_name}.{ctas_table_name} "
        f"WITH (external_location = '{ctas_s3_output_location}', format = '{output_format.upper()}') "
        f"AS {sql_query_string}"
    )

    query_request = {
        "QueryString": ctas_query,
        "QueryExecutionContext": {
            "Database": database_name,
            "Catalog": catalog_name,
        },
        "ResultConfiguration": {"OutputLocation": metadata_s3_output_location},
    }
    logging.debug("Query request is: %s", query_request)
    result = athena_start_query_execution_core(client=client, request=query_request)
    query_execution_id = result.get("QueryExecutionId")
    if query_execution_id is None:
        raise KeyError(f"QueryExecutionId not found in result: {result}")

    return {
        "query_execution_id": query_execution_id,
        "ctas_table_name": ctas_table_name,
        "ctas_s3_output_location": ctas_s3_output_location,
    }


from enum import Enum

from pyspark.sql.types import BooleanType, DateType, DoubleType, LongType, StringType
from pyspark.sql import functions as f


class NonCastableDataHandlingMethod(Enum):
    REPLACE_WITH_NULL = "replace_null"
    REPLACE_WITH_NULL_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN = "replace_null_with_new_col"
    REPLACE_WITH_FIXED_VALUE = "replace_value"
    REPLACE_WITH_FIXED_VALUE_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN = "replace_value_with_new_col"
    DROP_NON_CASTABLE_ROW = "drop"

    @staticmethod
    def get_names():
        return [item.name for item in NonCastableDataHandlingMethod]

    @staticmethod
    def get_values():
        return [item.value for item in NonCastableDataHandlingMethod]


class MohaveDataType(Enum):
    BOOL = "bool"
    DATE = "date"
    FLOAT = "float"
    LONG = "long"
    STRING = "string"
    OBJECT = "object"

    @staticmethod
    def get_names():
        return [item.name for item in MohaveDataType]

    @staticmethod
    def get_values():
        return [item.value for item in MohaveDataType]


PYTHON_TYPE_MAPPING = {
    MohaveDataType.BOOL: bool,
    MohaveDataType.DATE: str,
    MohaveDataType.FLOAT: float,
    MohaveDataType.LONG: int,
    MohaveDataType.STRING: str,
}

MOHAVE_TO_SPARK_TYPE_MAPPING = {
    MohaveDataType.BOOL: BooleanType,
    MohaveDataType.DATE: DateType,
    MohaveDataType.FLOAT: DoubleType,
    MohaveDataType.LONG: LongType,
    MohaveDataType.STRING: StringType,
}

SPARK_TYPE_MAPPING_TO_SQL_TYPE = {
    BooleanType: "BOOLEAN",
    LongType: "BIGINT",
    DoubleType: "DOUBLE",
    StringType: "STRING",
    DateType: "DATE",
}

SPARK_TO_MOHAVE_TYPE_MAPPING = {value: key for (key, value) in MOHAVE_TO_SPARK_TYPE_MAPPING.items()}


def cast_single_column_type_helper(df, column_name_to_cast, column_name_to_add, mohave_data_type, date_formatting):
    if mohave_data_type == MohaveDataType.DATE:
        df = df.withColumn(column_name_to_add, f.to_date(df[column_name_to_cast], date_formatting))
    else:
        df = df.withColumn(
            column_name_to_add, df[column_name_to_cast].cast(MOHAVE_TO_SPARK_TYPE_MAPPING[mohave_data_type]())
        )
    return df


def cast_single_column_type(
    df, column, mohave_data_type, invalid_data_handling_method, replace_value=None, date_formatting="dd-MM-yyyy"
):
    """Cast single column to a new type

    Args:
        df (DataFrame): spark dataframe
        column (Column): target column for type casting
        mohave_data_type (Enum): Enum MohaveDataType
        invalid_data_handling_method (Enum): Enum NonCastableDataHandlingMethod
        replace_value (str): value to replace for invalid data when "replace_value" is specified
        date_formatting (str): format for date. Default format is "dd-MM-yyyy"

    Returns:
        df (DataFrame): casted spark dataframe
    """
    cast_to_date = f.to_date(df[column], date_formatting)
    cast_to_non_date = df[column].cast(MOHAVE_TO_SPARK_TYPE_MAPPING[mohave_data_type]())
    non_castable_column = f"{column}_typecast_error"
    temp_column = "temp_column"

    if invalid_data_handling_method == NonCastableDataHandlingMethod.REPLACE_WITH_NULL:
        # Replace non-castable data to None in the same column. pyspark's default behaviour
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+---+--+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long
        # +---+------+
        # | id | txt |
        # +---+------+
        # | 1 | None |
        # | 2 | None |
        # | 3 | 1    |
        # +---+------+
        return df.withColumn(column, cast_to_date if (mohave_data_type == MohaveDataType.DATE) else cast_to_non_date)
    if invalid_data_handling_method == NonCastableDataHandlingMethod.DROP_NON_CASTABLE_ROW:
        # Drop non-castable row
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+---+--+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long, _ non-castable row
        # +---+----+
        # | id|txt |
        # +---+----+
        # |  3|  1 |
        # +---+----+
        df = df.withColumn(column, cast_to_date if (mohave_data_type == MohaveDataType.DATE) else cast_to_non_date)
        return df.where(df[column].isNotNull())

    if (
        invalid_data_handling_method
        == NonCastableDataHandlingMethod.REPLACE_WITH_NULL_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN
    ):
        # Replace non-castable data to None in the same column and put non-castable data to a new column
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+------+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long
        # +---+----+------------------+
        # | id|txt |txt_typecast_error|
        # +---+----+------------------+
        # |  1|None|      foo         |
        # |  2|None|      bar         |
        # |  3|  1 |                  |
        # +---+----+------------------+
        df = df.withColumn(
            temp_column, cast_to_date if (mohave_data_type == MohaveDataType.DATE) else cast_to_non_date
        )
        df = df.withColumn(
            non_castable_column,
            f.when(df[temp_column].isNotNull(), "").otherwise(df[column]),
        )
    elif invalid_data_handling_method == NonCastableDataHandlingMethod.REPLACE_WITH_FIXED_VALUE:
        # Replace non-castable data to a value in the same column
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+------+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long, replace non-castable value to 0
        # +---+-----+
        # | id| txt |
        # +---+-----+
        # |  1|  0  |
        # |  2|  0  |
        # |  3|  1  |
        # +---+----+
        value = _validate_and_cast_value(value=replace_value, mohave_data_type=mohave_data_type)

        df = df.withColumn(
            temp_column, cast_to_date if (mohave_data_type == MohaveDataType.DATE) else cast_to_non_date
        )

        replace_date_value = f.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(
            f.to_date(f.lit(value), date_formatting)
        )
        replace_non_date_value = f.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(value)

        df = df.withColumn(
            temp_column, replace_date_value if (mohave_data_type == MohaveDataType.DATE) else replace_non_date_value
        )
    elif (
        invalid_data_handling_method
        == NonCastableDataHandlingMethod.REPLACE_WITH_FIXED_VALUE_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN
    ):
        # Replace non-castable data to a value in the same column and put non-castable data to a new column
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+---+--+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long, replace non-castable value to 0
        # +---+----+------------------+
        # | id|txt |txt_typecast_error|
        # +---+----+------------------+
        # |  1|  0  |   foo           |
        # |  2|  0  |   bar           |
        # |  3|  1  |                 |
        # +---+----+------------------+
        value = _validate_and_cast_value(value=replace_value, mohave_data_type=mohave_data_type)

        df = df.withColumn(
            temp_column, cast_to_date if (mohave_data_type == MohaveDataType.DATE) else cast_to_non_date
        )
        df = df.withColumn(
            non_castable_column,
            f.when(df[temp_column].isNotNull(), "").otherwise(df[column]),
        )

        replace_date_value = f.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(
            f.to_date(f.lit(value), date_formatting)
        )
        replace_non_date_value = f.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(value)

        df = df.withColumn(
            temp_column, replace_date_value if (mohave_data_type == MohaveDataType.DATE) else replace_non_date_value
        )
    # drop temporary column
    df = df.withColumn(column, df[temp_column]).drop(temp_column)

    df_cols = df.columns
    if non_castable_column in df_cols:
        # Arrange columns so that non_castable_column col is next to casted column
        df_cols.remove(non_castable_column)
        column_index = df_cols.index(column)
        arranged_cols = df_cols[: column_index + 1] + [non_castable_column] + df_cols[column_index + 1 :]
        df = df.select(*arranged_cols)
    return df


def _validate_and_cast_value(value, mohave_data_type):
    if value is None:
        return value
    try:
        return PYTHON_TYPE_MAPPING[mohave_data_type](value)
    except ValueError as e:
        raise ValueError(
            f"Invalid value to replace non-castable data. "
            f"{mohave_data_type} is not in mohave supported date type: {MohaveDataType.get_values()}. "
            f"Please use a supported type",
            e,
        )


import os
import collections
import tempfile
import zipfile
import base64
import logging
from io import BytesIO
import numpy as np


class OperatorSparkOperatorCustomerError(Exception):
    """Error type for Customer Errors in Spark Operators"""


def temp_col_name(df, *illegal_names):
    """Generates a temporary column name that is unused."""
    name = "temp_col"
    idx = 0
    name_set = set(list(df.columns) + list(illegal_names))
    while name in name_set:
        name = f"_temp_col_{idx}"
        idx += 1

    return name


def get_temp_col_if_not_set(df, col_name):
    """Extracts the column name from the parameters if it exists, otherwise generates a temporary column name."""
    if col_name:
        return col_name, False
    else:
        return temp_col_name(df), True


def replace_input_if_output_is_temp(df, input_column, output_column, output_is_temp):
    """Replaces the input column in the dataframe if the output was not set

    This is used with get_temp_col_if_not_set to enable the behavior where a
    transformer will replace its input column if an output is not specified.
    """
    if output_is_temp:
        df = df.withColumn(input_column, df[output_column])
        df = df.drop(output_column)
        return df
    else:
        return df


def parse_parameter(typ, value, key, default=None, nullable=False):
    if value is None:
        if default is not None or nullable:
            return default
        else:
            raise OperatorSparkOperatorCustomerError(f"Missing required input: '{key}'")
    else:
        try:
            value = typ(value)
            if isinstance(value, (int, float, complex)) and not isinstance(value, bool):
                if np.isnan(value) or np.isinf(value):
                    raise OperatorSparkOperatorCustomerError(
                        f"Invalid value provided for '{key}'. Expected {typ.__name__} but received: {value}"
                    )
                else:
                    return value
            else:
                return value
        except (ValueError, TypeError):
            raise OperatorSparkOperatorCustomerError(
                f"Invalid value provided for '{key}'. Expected {typ.__name__} but received: {value}"
            )


def expects_valid_column_name(value, key, nullable=False):
    if nullable and value is None:
        return

    if value is None or len(str(value).strip()) == 0:
        raise OperatorSparkOperatorCustomerError(
            f"Column name cannot be null, empty, or whitespace for parameter '{key}': {value}"
        )


def expects_parameter(value, key, condition=None):
    if value is None:
        raise OperatorSparkOperatorCustomerError(f"Missing required input: '{key}'")
    elif condition is not None and not condition:
        raise OperatorSparkOperatorCustomerError(f"Invalid value provided for '{key}': {value}")


def expects_column(df, value, key):
    if not value or value not in df.columns:
        raise OperatorSparkOperatorCustomerError(
            f"Expected column in dataframe for '{key}' however received '{value}'"
        )


def expects_parameter_value_in_list(key, value, items):
    if value not in items:
        raise OperatorSparkOperatorCustomerError(
            f"Illegal parameter value. {key} expected to be in {items}, but given {value}"
        )


def encode_pyspark_model(model):
    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = os.path.join(dirpath, "model")
        # Save the model
        model.save(dirpath)

        # Create the temporary zip-file.
        mem_zip = BytesIO()
        with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            # Zip the directory.
            for root, dirs, files in os.walk(dirpath):
                for file in files:
                    rel_dir = os.path.relpath(root, dirpath)
                    zf.write(os.path.join(root, file), os.path.join(rel_dir, file))

        zipped = mem_zip.getvalue()
        encoded = base64.b85encode(zipped)
        return str(encoded, "utf-8")


def decode_pyspark_model(model_factory, encoded):
    with tempfile.TemporaryDirectory() as dirpath:
        zip_bytes = base64.b85decode(encoded)
        mem_zip = BytesIO(zip_bytes)
        mem_zip.seek(0)

        with zipfile.ZipFile(mem_zip, "r") as zf:
            zf.extractall(dirpath)

        model = model_factory.load(dirpath)
        return model


def hash_parameters(value):
    # pylint: disable=W0702
    try:
        if isinstance(value, collections.Hashable):
            return hash(value)
        if isinstance(value, dict):
            return hash(frozenset([hash((hash_parameters(k), hash_parameters(v))) for k, v in value.items()]))
        if isinstance(value, list):
            return hash(frozenset([hash_parameters(v) for v in value]))
        raise RuntimeError("Object not supported for serialization")
    except:  # noqa: E722
        raise RuntimeError("Object not supported for serialization")


def load_trained_parameters(trained_parameters, operator_parameters):
    trained_parameters = trained_parameters if trained_parameters else {}
    parameters_hash = hash_parameters(operator_parameters)
    stored_hash = trained_parameters.get("_hash")
    if stored_hash != parameters_hash:
        trained_parameters = {"_hash": parameters_hash}
    return trained_parameters


def load_pyspark_model_from_trained_parameters(trained_parameters, model_factory, name):
    if trained_parameters is None or name not in trained_parameters:
        return None, False

    try:
        model = decode_pyspark_model(model_factory, trained_parameters[name])
        return model, True
    except Exception as e:
        logging.error(f"Could not decode PySpark model {name} from trained_parameters: {e}")
        del trained_parameters[name]
        return None, False


def fit_and_save_model(trained_parameters, name, algorithm, df):
    model = algorithm.fit(df)
    trained_parameters[name] = encode_pyspark_model(model)
    return model


def transform_using_trained_model(model, df, loaded):
    try:
        return model.transform(df)
    except Exception as e:
        if loaded:
            raise OperatorSparkOperatorCustomerError(
                f"Encountered error while using stored model. Please delete the operator and try again. {e}"
            )
        else:
            raise e


import re
from datetime import date

import numpy as np
import pandas as pd
from pyspark.sql.types import (
    BooleanType,
    IntegralType,
    FractionalType,
    StringType,
)


def type_inference(df):  # noqa: C901 # pylint: disable=R0912
    """Core type inference logic

    Args:
        df: spark dataframe

    Returns: dict a schema that maps from column name to mohave datatype

    """
    columns_to_infer = [col for (col, col_type) in df.dtypes if col_type == "string"]

    pandas_df = df.toPandas()
    report = {}
    for (columnName, _) in pandas_df.iteritems():
        if columnName in columns_to_infer:
            column = pandas_df[columnName].values
            report[columnName] = {
                "sum_string": len(column),
                "sum_numeric": sum_is_numeric(column),
                "sum_integer": sum_is_integer(column),
                "sum_boolean": sum_is_boolean(column),
                "sum_date": sum_is_date(column),
                "sum_null_like": sum_is_null_like(column),
                "sum_null": sum_is_null(column),
            }

    # Analyze
    numeric_threshold = 0.8
    integer_threshold = 0.8
    date_threshold = 0.8
    bool_threshold = 0.8

    column_types = {}

    for col, insights in report.items():
        # Convert all columns to floats to make thresholds easy to calculate.
        proposed = MohaveDataType.STRING.value
        if (insights["sum_numeric"] / insights["sum_string"]) > numeric_threshold:
            proposed = MohaveDataType.FLOAT.value
            if (insights["sum_integer"] / insights["sum_numeric"]) > integer_threshold:
                proposed = MohaveDataType.LONG.value
        elif (insights["sum_boolean"] / insights["sum_string"]) > bool_threshold:
            proposed = MohaveDataType.BOOL.value
        elif (insights["sum_date"] / insights["sum_string"]) > date_threshold:
            proposed = MohaveDataType.DATE.value
        column_types[col] = proposed

    for f in df.schema.fields:
        if f.name not in columns_to_infer:
            if isinstance(f.dataType, IntegralType):
                column_types[f.name] = MohaveDataType.LONG.value
            elif isinstance(f.dataType, FractionalType):
                column_types[f.name] = MohaveDataType.FLOAT.value
            elif isinstance(f.dataType, StringType):
                column_types[f.name] = MohaveDataType.STRING.value
            elif isinstance(f.dataType, BooleanType):
                column_types[f.name] = MohaveDataType.BOOL.value
            else:
                # unsupported types in mohave
                column_types[f.name] = MohaveDataType.OBJECT.value

    return column_types


def _is_numeric_single(x):
    try:
        x_float = float(x)
        return np.isfinite(x_float)
    except ValueError:
        return False
    except TypeError:  # if x = None
        return False


def sum_is_numeric(x):
    """count number of numeric element

    Args:
        x: numpy array

    Returns: int

    """
    castables = np.vectorize(_is_numeric_single)(x)
    return np.count_nonzero(castables)


def _is_integer_single(x):
    try:
        if not _is_numeric_single(x):
            return False
        return float(x) == int(x)
    except ValueError:
        return False
    except TypeError:  # if x = None
        return False


def sum_is_integer(x):
    castables = np.vectorize(_is_integer_single)(x)
    return np.count_nonzero(castables)


def _is_boolean_single(x):
    boolean_list = ["true", "false"]
    try:
        is_boolean = x.lower() in boolean_list
        return is_boolean
    except ValueError:
        return False
    except TypeError:  # if x = None
        return False
    except AttributeError:
        return False


def sum_is_boolean(x):
    castables = np.vectorize(_is_boolean_single)(x)
    return np.count_nonzero(castables)


def sum_is_null_like(x):  # noqa: C901
    def _is_empty_single(x):
        try:
            return bool(len(x) == 0)
        except TypeError:
            return False

    def _is_null_like_single(x):
        try:
            return bool(null_like_regex.match(x))
        except TypeError:
            return False

    def _is_whitespace_like_single(x):
        try:
            return bool(whitespace_regex.match(x))
        except TypeError:
            return False

    null_like_regex = re.compile(r"(?i)(null|none|nil|na|nan)")  # (?i) = case insensitive
    whitespace_regex = re.compile(r"^\s+$")  # only whitespace

    empty_checker = np.vectorize(_is_empty_single)(x)
    num_is_null_like = np.count_nonzero(empty_checker)

    null_like_checker = np.vectorize(_is_null_like_single)(x)
    num_is_null_like += np.count_nonzero(null_like_checker)

    whitespace_checker = np.vectorize(_is_whitespace_like_single)(x)
    num_is_null_like += np.count_nonzero(whitespace_checker)
    return num_is_null_like


def sum_is_null(x):
    return np.count_nonzero(pd.isnull(x))


def _is_date_single(x):
    try:
        return bool(date.fromisoformat(x))  # YYYY-MM-DD
    except ValueError:
        return False
    except TypeError:
        return False


def sum_is_date(x):
    return np.count_nonzero(np.vectorize(_is_date_single)(x))


def cast_df(df, schema):
    """Cast datafram from given schema

    Args:
        df: spark dataframe
        schema: schema to cast to. It map from df's col_name to mohave datatype

    Returns: casted dataframe

    """
    # col name to spark data type mapping
    col_to_spark_data_type_map = {}

    # get spark dataframe's actual datatype
    fields = df.schema.fields
    for f in fields:
        col_to_spark_data_type_map[f.name] = f.dataType
    cast_expr = []
    # iterate given schema and cast spark dataframe datatype
    for col_name in schema:
        mohave_data_type_from_schema = MohaveDataType(schema.get(col_name, MohaveDataType.OBJECT.value))
        if mohave_data_type_from_schema != MohaveDataType.OBJECT:
            spark_data_type_from_schema = MOHAVE_TO_SPARK_TYPE_MAPPING[mohave_data_type_from_schema]
            # Only cast column when the data type in schema doesn't match the actual data type
            if not isinstance(col_to_spark_data_type_map[col_name], spark_data_type_from_schema):
                # use spark-sql expression instead of spark.withColumn to improve performance
                expr = f"CAST (`{col_name}` as {SPARK_TYPE_MAPPING_TO_SQL_TYPE[spark_data_type_from_schema]})"
            else:
                # include column that has same dataType as it is
                expr = f"`{col_name}`"
        else:
            # include column that has same mohave object dataType as it is
            expr = f"`{col_name}`"
        cast_expr.append(expr)
    if len(cast_expr) != 0:
        df = df.selectExpr(*cast_expr)
    return df, schema


def validate_schema(df, schema):
    """Validate if every column is covered in the schema

    Args:
        schema ():
    """
    columns_in_df = df.columns
    columns_in_schema = schema.keys()

    if len(columns_in_df) != len(columns_in_schema):
        raise ValueError(
            f"Invalid schema column size. "
            f"Number of columns in schema should be equal as number of columns in dataframe. "
            f"schema columns size: {len(columns_in_schema)}, dataframe column size: {len(columns_in_df)}"
        )

    for col in columns_in_schema:
        if col not in columns_in_df:
            raise ValueError(
                f"Invalid column name in schema. "
                f"Column in schema does not exist in dataframe. "
                f"Non-existed columns: {col}"
            )


def athena_source(spark, mode, dataset_definition, trained_parameters=None):  # noqa: C901
    """Represents a source that handles Athena.

    Note: Input dataset_definition is a Mohave specific dataset definition and is different from Processing Job
    dataset definition.
    """

    import time
    import boto3

    athena_client = boto3.client("athena")
    glue_client = boto3.client("glue")

    athena_create_database_if_not_exists(glue_client=glue_client)
    trained_parameters = athena_start_query_execution(dataset_definition=dataset_definition, client=athena_client)

    query_execution_id = trained_parameters["query_execution_id"]
    ctas_table_name = trained_parameters["ctas_table_name"]
    request = {"QueryExecutionId": query_execution_id}
    response = athena_get_query_execution_core(client=athena_client, request=request)
    state = athena_parse_query_execution_state(get_query_execution_response=response).upper()

    while state in ("RUNNING", "QUEUED"):
        time.sleep(5)
        response = athena_get_query_execution_core(client=athena_client, request=request)
        state = athena_parse_query_execution_state(get_query_execution_response=response).upper()

    if state == "SUCCEEDED":
        ctas_s3_output_location = trained_parameters["ctas_s3_output_location"]
        path = ctas_s3_output_location.replace("s3://", "s3a://")
        athena_delete_tmp_table(glue_client=glue_client, ctas_table_name=ctas_table_name)
        # clear ctas_table_name once we delete temp table
        trained_parameters["ctas_table_name"] = ""
        try:
            return default_spark_with_trained_parameters_and_state(
                df=spark.read.parquet(path),
                trained_parameters=trained_parameters,
                state=get_execution_state(state),
            )
        except Exception as e:
            raise RuntimeError(
                f"Error while reading Athena ctas output from S3 location: {path}, with the exception: {e}"
            )
    if state in ("FAILED", "CANCELLED"):
        try:
            message = response["QueryExecution"]["Status"]["StateChangeReason"]
        except Exception as e:  # pylint: disable=W0703
            message = str(e)
            logging.error(message)
        raise RuntimeError(f"Athena query execution did not complete with the following message: {message}")

    raise RuntimeError("Unknown Athena query state: {}".format(state))


def infer_and_cast_type(df, spark, inference_data_sample_size=1000, trained_parameters=None):
    """Infer column types for spark dataframe and cast to inferred data type.

    Args:
        df: spark dataframe
        spark: spark session
        inference_data_sample_size: number of row data used for type inference
        trained_parameters: trained_parameters to determine if we need infer data types

    Returns: a dict of pyspark df with column data type casted and trained parameters

    """
    from pyspark.sql.utils import AnalysisException

    # if trained_parameters is none or doesn't contain schema key, then type inference is needed
    if trained_parameters is None or not trained_parameters.get("schema", None):
        # limit first 1000 rows to do type inference

        limit_df = df.limit(inference_data_sample_size)
        schema = type_inference(limit_df)
    else:
        schema = trained_parameters["schema"]
        try:
            validate_schema(df, schema)
        except ValueError as e:
            raise OperatorCustomerError(e)
    try:
        df, schema = cast_df(df, schema)
    except (AnalysisException, ValueError) as e:
        raise OperatorCustomerError(e)
    trained_parameters = {"schema": schema}
    return default_spark_with_trained_parameters(df, trained_parameters)


def process_numeric(df, spark, **kwargs):

    return dispatch(
        "operator",
        [df],
        kwargs,
        {
            "Scale values": (process_numeric_scale_values, "scale_values_parameters"),
        },
    )


def custom_pyspark(df, spark, code):
    """Apply custom pyspark operation on the input dataframe

    Example:
        The custom code expects the user to provide an output df.
        code = \"""
        from pyspark.sql.functions import UserDefinedFunction
        from pyspark.sql.types import StringType
        target_column = 'txt'
        udf = UserDefinedFunction(lambda x: f'{x}_new', StringType())
        df = df.withColumn(target_column, udf(df.txt))
        \"""

    Notes:
        This operation expects the user code to store the output in df variable.

    Args:
        spark: spark session
        params (dict): dictionary that has various params. Required param for this operation is "code"
        df: input pyspark dataframe

    Returns:
        df: pyspark dataframe with custom pyspark code executed on the input df.
    """
    import ast

    exec_block = ast.parse(code, mode="exec")

    _globals, _locals = {}, {"df": df}
    stdout = capture_stdout(exec, compile(exec_block, "<string>", mode="exec"), _locals)  # pylint: disable=W0122
    output_df = eval("df", _globals, _locals)  # pylint: disable=W0123
    return default_spark_with_stdout(output_df, stdout)


op_1_output = athena_source(
    spark=spark,
    mode=mode,
    **{
        "dataset_definition": {
            "datasetSourceType": "Athena",
            "name": "amazon_reviews",
            "catalogName": "AwsDataCatalog",
            "databaseName": "dsoaws",
            "queryString": "select * from amazon_reviews_parquet",
            "s3OutputLocation": "s3://sagemaker-us-east-1-835319576252/athena/",
            "outputFormat": "parquet",
        }
    },
)
op_2_output = infer_and_cast_type(op_1_output["default"], spark=spark, **{})
op_5_output = process_numeric(
    op_2_output["default"],
    spark=spark,
    **{
        "operator": "Scale values",
        "scale_values_parameters": {
            "scaler": "Min-max scaler",
            "min_max_scaler_parameters": {
                "min": -1,
                "max": 1,
                "input_column": "star_rating",
                "output_column": "star_rating_min_max_scaled_builtin",
            },
            "standard_scaler_parameters": {},
        },
    },
)
op_6_output = custom_pyspark(
    op_5_output["default"],
    spark=spark,
    **{
        "code": '# Table is available as variable `df`\nfrom pyspark.sql.functions import stddev, mean, col, floor\ndf = df.withColumn("sentiment", (floor(col("star_rating_min_max_scaled_builtin"))))'
    },
)

#  Glossary: variable name to node_id
#
#  op_1_output: 14039109-2da9-49b4-8eee-df39306c9c47
#  op_2_output: 98b4c198-d379-42ab-af96-165dcd1a01d8
#  op_5_output: 93919dab-601b-4aa2-93d3-d223b6f46e25
#  op_6_output: 019d2a9b-9601-4cca-8395-9a976db0ada5
