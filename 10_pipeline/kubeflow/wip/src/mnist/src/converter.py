import json


def convert_mnist_experiment_result(experiment_result) -> str:
    """
    convert results into args string
    """
    import json
    r = json.loads(experiment_result)
    args = []
    for hp in r:
        print(hp)
        args.append("%s=%s" % (hp["name"], hp["value"]))

    return " ".join(args)