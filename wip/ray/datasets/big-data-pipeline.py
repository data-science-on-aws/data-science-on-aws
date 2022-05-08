import argparse
import tempfile
import time
from typing import List

import pandas
import pyarrow

import ray
from ray.data.dataset_pipeline import DatasetPipeline
from ray.data.datasource.datasource import RandomIntRowDatasource

def create_shuffle_pipeline(
    training_data_dir: str, num_epochs: int, num_shards: int
) -> List[DatasetPipeline]:

    return (
#        ray.data.read_parquet('s3://dsoaws/parquet/') #training_data_dir)
        ray.data.read_parquet(training_data_dir)
        .repeat(num_epochs)
        .random_shuffle_each_window()
        .split(num_shards, equal=True)
    )


@ray.remote
class TrainingWorker:
    def __init__(self, rank: int, shard: DatasetPipeline):
        self.rank = rank
        self.shard = shard

    def train(self):
        for epoch, training_dataset in enumerate(self.shard.iter_epochs()):
            # Following code emulates epoch based SGD training.
            print(f"Training... worker: {self.rank}, epoch: {epoch}")
            for i, batch in enumerate(training_dataset.iter_batches()):
                # TODO: replace the code for real training.
                pass


parser = argparse.ArgumentParser()
parser.add_argument(
    "--large-scale-test",
    action="store_true",
    help="Run large scale test (100GiB of data).",
)

args, _ = parser.parse_known_args()


if not args.large_scale_test:

    NUM_TRAINING_WORKERS = 4
    NUM_EPOCHS = 5
    NUM_COLUMNS = 10
    SIZE_100MiB = 100 * 1024 * 1024

    # create a local ray cluster.
    ray.init(address="auto")

    def generate_example_files(size_bytes: int) -> str:
        tmpdir = tempfile.mkdtemp()
        ray.data.read_datasource(
            RandomIntRowDatasource(),
            n=size_bytes // 8 // NUM_COLUMNS,
            num_columns=NUM_COLUMNS,
        ).write_parquet(tmpdir)
        return tmpdir

    example_files_dir = generate_example_files(SIZE_100MiB)

    splits = create_shuffle_pipeline(
        example_files_dir, NUM_EPOCHS, NUM_TRAINING_WORKERS
    )

    training_workers = [
        TrainingWorker.remote(rank, shard) for rank, shard in enumerate(splits)
    ]

    # Let's run the e2e pipeline
    start = time.time()
    ray.get([worker.train.remote() for worker in training_workers])
    print(f"total ingestion time: {int(time.time() - start)}s")

    # -> Write Progress: 100%|████████████████████| 201/201 [00:00<00:00, 228.67it/s]
    # -> Stage 0:   0%|          | 0/5 [00:00<?, ?it/s]
    # -> Stage 0:  40%|████      | 2/5 [00:11<00:17,  5.75s/it]
    # -> Stage 0:  60%|██████    | 3/5 [00:23<00:16,  8.15s/it]
    # -> ...
    # -> (TrainingWorker pid=1651600) Training... worker: 2, epoch: 0
    # -> Stage 0:  80%|████████  | 4/5 [00:35<00:09,  9.59s/it]
    # -> ...
    # -> (TrainingWorker pid=1651599) Training... worker: 0, epoch: 1
    # -> Stage 0: 100%|██████████| 5/5 [00:46<00:00, 10.34s/it]
    # -> ...
    # -> (TrainingWorker pid=1651387) Training... worker: 3, epoch: 4
    # -> total ingestion time: 61s


def create_large_shuffle_pipeline(
    data_size_bytes: int, num_epochs: int, num_columns: int, num_shards: int
) -> List[DatasetPipeline]:
    return (
        ray.data.read_datasource(
            RandomIntRowDatasource(),
            n=data_size_bytes // 8 // num_columns,
            num_columns=num_columns,
        )
        .repeat(num_epochs)
        .random_shuffle_each_window()
        .split(num_shards, equal=True)
    )


if args.large_scale_test:
    NUM_TRAINING_WORKERS = 10 # 16
    NUM_EPOCHS = 5
    NUM_COLUMNS = 10
    GiB = 1024 * 1024 * 1024
    SIZE_500GiB = 500 * GiB
    TOTAL_NUM_NODES = 10 # 70 + 16 + 1

    # use the AWS cluster we just set up.
    ray.init(address="auto")

    # waiting for cluster nodes to come up.
    while len(ray.nodes()) < TOTAL_NUM_NODES:
        print(f"waiting for nodes to start up: {len(ray.nodes())}/{TOTAL_NUM_NODES}")
        time.sleep(5)

    splits = create_large_shuffle_pipeline(
        SIZE_500GiB, NUM_EPOCHS, NUM_COLUMNS, NUM_TRAINING_WORKERS
    )

    # Note we set num_gpus=1 for workers so that
    # the workers will only run on GPU nodes.
    training_workers = [
        TrainingWorker.options(num_gpus=0).remote(rank, shard)
        for rank, shard in enumerate(splits)
    ]

    start = time.time()

    # Let's run the large scale test.
    ray.get([worker.train.remote() for worker in training_workers])
    print(f"total ingestion time: {int(time.time() - start)}s")
    throughput = SIZE_500GiB * NUM_EPOCHS / (time.time() - start) / GiB
    print("throughput: {0:0.2f}GiB/s".format(throughput))


