# This code is auto-generated.
import http.client as http_client
import io
import json
import logging
import os

import numpy as np
from joblib import load
from scipy import sparse

from sagemaker_containers.beta.framework import encoders
from sagemaker_containers.beta.framework import worker
from sagemaker_sklearn_extension.externals import read_csv_data


def _is_inverse_label_transform():
    """Returns True if if it's running in inverse label transform."""
    return os.getenv('AUTOML_TRANSFORM_MODE') == 'inverse-label-transform'


def _is_feature_transform():
    """Returns True if it's running in feature transform mode."""
    return os.getenv('AUTOML_TRANSFORM_MODE') == 'feature-transform'


def _get_selected_input_keys():
    """Returns a list of ordered content keys for container's input."""
    return [key.strip().lower() for key in os.environ['SAGEMAKER_INFERENCE_INPUT'].split(',')]


def _get_selected_output_keys():
    """Returns a list of ordered content keys for container's output."""
    return [key.strip().lower() for key in os.environ['SAGEMAKER_INFERENCE_OUTPUT'].split(',')]


def _sparsify_if_needed(x):
    """Returns a sparse matrix if the needed for encoding to sparse recordio protobuf."""
    if os.getenv('AUTOML_SPARSE_ENCODE_RECORDIO_PROTOBUF') == '1' \
            and not sparse.issparse(x):
        return sparse.csr_matrix(x)
    return x


def _split_features_target(x):
    """Returns the features and target by splitting the input array."""
    if os.getenv('AUTOML_TRANSFORM_MODE') == 'feature-transform':
        return _sparsify_if_needed(x), None

    if sparse.issparse(x):
        return x[:, 1:], x[:, 0].toarray()
    return _sparsify_if_needed(x[:, 1:]), np.ravel(x[:, 0])


def model_fn(model_dir):
    """Loads the model.

    The SageMaker Scikit-learn model server loads model by invoking this method.

    Parameters
    ----------
    model_dir: str
        the directory where the model files reside

    Returns
    -------
    : AutoMLTransformer
        deserialized model object that can be used for model serving

    """
    return load(filename=os.path.join(model_dir, 'model.joblib'))


def predict_fn(input_object, model):
    """Generates prediction for the input_object based on the model.

    The SageMaker Scikit-learn model server invokes this method with the return value of the input_fn.
    This method invokes the loaded model's transform method, if it's expected to transform the input data
    and invokes the loaded model's inverse_label_transform, if the task is to perform
    inverse label transformation.

    Parameters
    ----------
    input_object : array-like
        the object returned from input_fn

    model : AutoMLTransformer
        the model returned from model_fn

    Returns
    -------
    : ndarray
        transformed input data or inverse transformed label data

    """
    if isinstance(input_object, worker.Response):
        return input_object

    if _is_inverse_label_transform():
        return _generate_post_processed_response(input_object, model)
    try:
        return model.transform(input_object)
    except ValueError as e:
        return worker.Response(
            response='{}'.format(str(e) or 'Unknown error.'),
            status=http_client.BAD_REQUEST
        )


def _generate_post_processed_response(array, model):
    """Build the predictions response.

    If selectable inference mode is on, only parts of the input array will be inverse label transformed.

    Parameters
    ----------
    array : array-like
        array representation of predictions

    model : AutoMLTransformer
        the model returned from model_fn

    Returns
    -------
    : ndarray
        transformed prediction data

    """
    try:
        input_keys = _get_selected_input_keys()
        output_keys = _get_selected_output_keys()
    except KeyError:
        # Selectable inference is not turned on
        return model.inverse_label_transform(array.ravel().astype(np.float).astype(np.int))

    output_array = np.empty((len(array), len(output_keys)), dtype=object)
    for output_key_idx, output_key in enumerate(output_keys):
        if output_key == "predicted_label" and output_key in input_keys:
            input_key_idx = input_keys.index(output_key)
            output_array[:, output_key_idx] = model.inverse_label_transform(array[:, input_key_idx]
                                                                            .ravel().astype(np.float).astype(np.int))
        elif output_key == "labels":
            output_array[:, output_key_idx][:] = str(list(model.target_transformer.get_classes()))
        elif output_key in input_keys:
            input_key_idx = input_keys.index(output_key)
            output_array[:, output_key_idx] = array[:, input_key_idx].tolist()
        else:
            output_array[:, output_key_idx][:] = np.nan
    return output_array


def input_fn(request_body, request_content_type):
    """Decodes request body to 2D numpy array.

    The SageMaker Scikit-learn model server invokes this method to deserialize the request data into an object
    for prediction.

    Parameters
    ----------
    request_body : str or bytes
        the request body
    request_content_type : str
        the media type for the request body

    Returns
    -------
    : array-like
        decoded data as 2D numpy array

    """
    content_type = request_content_type.lower(
    ) if request_content_type else "text/csv"
    content_type = content_type.split(";")[0].strip()

    if content_type == 'text/csv':
        if isinstance(request_body, str):
            byte_buffer = request_body.encode()
        else:
            byte_buffer = request_body
        val = read_csv_data(source=byte_buffer)
        logging.info(f"Shape of the requested data: '{val.shape}'")
        return val

    return worker.Response(
        response=f"'{request_content_type}' is an unsupported content type.",
        status=http_client.UNSUPPORTED_MEDIA_TYPE
    )


def output_fn(prediction, accept_type):
    """Encodes prediction to accept type.

    The SageMaker Scikit-learn model server invokes this method with the result of prediction and
    serializes this according to the response MIME type. It expects the input to be numpy array and encodes
    to requested response MIME type.

    Parameters
    ----------
    prediction : array-like
        the object returned from predict_fn

    accept_type : str
        the expected MIME type of the response

    Returns
    -------
    : Response obj
        serialized predictions in accept type

    """
    if isinstance(prediction, worker.Response):
        return prediction

    if _is_inverse_label_transform():
        try:
            output_keys = _get_selected_output_keys()
            return worker.Response(
                response=encoder_factory[accept_type](prediction, output_keys),
                status=http_client.OK,
                mimetype=accept_type
            )
        except KeyError:
            # Selectable inference is not turned on
            if accept_type == 'text/csv':
                return worker.Response(
                    response=encoders.encode(prediction, accept_type),
                    status=http_client.OK,
                    mimetype=accept_type
                )
            return worker.Response(
                response=f"Accept type '{accept_type}' is not supported "
                         f"during inverse label transformation.",
                status=http_client.NOT_ACCEPTABLE
            )

    if isinstance(prediction, tuple):
        X, y = prediction
    else:
        X, y = _split_features_target(prediction)

    if accept_type == 'application/x-recordio-protobuf':
        return worker.Response(
            response=encoders.array_to_recordio_protobuf(
                _sparsify_if_needed(X).astype('float32'),
                y.astype('float32') if y is not None else y
            ),
            status=http_client.OK,
            mimetype=accept_type
        )

    if accept_type == 'text/csv':
        if y is not None:
            X = np.column_stack(
                (np.ravel(y), X.todense() if sparse.issparse(X) else X)
            )

        return worker.Response(
            response=encoders.encode(X, accept_type),
            status=http_client.OK,
            mimetype=accept_type
        )
    return worker.Response(
        response=f"Accept type '{accept_type}' is not supported.",
        status=http_client.NOT_ACCEPTABLE
    )


def execution_parameters_fn():
    """Return the response for execution-parameters request used by SageMaker Batch transform.

    The SageMaker Scikit-learn model server invokes when execution-parameters endpoint is called.
    For the models generated by AutoML Jobs, returns the MaxPayloadInMB to be 1MB for feature transform
    used during inference and defaults to 6MB otherwise.
    """
    if _is_feature_transform():
        return worker.Response(
            response='{"MaxPayloadInMB":1}',
            status=http_client.OK,
            mimetype="application/json"
        )
    return worker.Response(
        response='{"MaxPayloadInMB":6}',
        status=http_client.OK,
        mimetype="application/json"
    )


def numpy_array_to_csv(array, output_keys):
    """Encode numpy array in csv format.

    Parameters
    ----------
    array : numpy array
        2D array of predictions

    output_keys : list of strings (not used)
    |   keys for selected output predictions

    Returns
    -------
    : string
        predictions in csv response format

    """
    return encoders.array_to_csv(array)


def numpy_array_to_json(array, output_keys):
    """Convert a 2D numpy array to json format.

    Parameters
    ----------
    array : numpy array
        2D array of predictions

    output_keys : list of strings
    |   keys for selected output predictions, matches order of the columns in input array

    Returns
    -------
    : string
        predictions in json response format

    """
    predictions = []
    for single_prediction in array:
        single_prediction_response = {}
        for (key, item) in zip(output_keys, single_prediction):
            single_prediction_response[key] = item
        predictions.append(single_prediction_response)
    return json.dumps({"predictions": predictions})


def numpy_array_to_jsonlines(array, output_keys):
    """Converts a 2D numpy array to jsonlines format.

    Parameters
    ----------
    array : numpy array
        2D array of predictions

    output_keys : list of strings
    |   keys for selected output predictions, matches order of the columns in input array

    Returns
    -------
    : bytes
        predictions in jsonlines response format

    """
    bio = io.BytesIO()
    for single_prediction in array:
        single_prediction_response = {}
        for (key, item) in zip(output_keys, single_prediction):
            single_prediction_response[key] = item
        bio.write(bytes(json.dumps(single_prediction_response) + "\n", "UTF-8"))
    return bio.getvalue()


encoder_factory = {
    'text/csv': numpy_array_to_csv,
    'application/json': numpy_array_to_json,
    'application/jsonlines': numpy_array_to_jsonlines
}
