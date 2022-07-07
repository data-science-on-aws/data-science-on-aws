import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl

import ray
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch

from ray_lightning.tune import TuneReportCallback, get_tune_resources
from ray_lightning import RayPlugin
from ray_lightning.tests.utils import LightningMNISTClassifier
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything

from torch.optim.lr_scheduler import OneCycleLR

from torchmetrics.functional import accuracy

seed_everything(42)



from sagemaker_ray_helper import RayHelper



def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class LitResnet(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        return {f"{stage}_loss": loss, f"{stage}_accuracy": acc}

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc) 

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config.get("lr", 0.01),
            momentum=self.config.get("momentum", 0.9),
            weight_decay=5e-4,
        )
        steps_per_epoch = (len(self.trainer.datamodule.train_dataloader()) // self.trainer.accumulate_grad_batches)
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                self.config.get("lr", 0.01) * 1.5,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}




def train_cifar(config,
                data_dir=None,
                num_epochs=10,
                num_workers=1,
                use_gpu=False,
                callbacks=None):

    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    cifar10_dm = CIFAR10DataModule(
        data_dir=data_dir,
        batch_size=config.get("batch_size", 256),
        num_workers=4,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    model = LitResnet(config)
    model.datamodule = cifar10_dm

    callbacks = callbacks or []

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        progress_bar_refresh_rate=0,
        plugins=[
            RayPlugin(
                num_workers=num_workers,
                use_gpu=use_gpu
            )
        ])
    
    trainer.fit(model, cifar10_dm)


def tune_cifar(data_dir,
               num_samples=10,
               num_epochs=10,
               num_workers=1,
               use_gpu=False):
    
    cluster_resources = ray.cluster_resources()
    concurrent_jobs = cluster_resources["GPU"] // num_workers
    available_cpus = cluster_resources["CPU"] - concurrent_jobs
    cpus_per_worker = cpus_per_worker = (available_cpus // num_workers) // concurrent_jobs
    
    config = {
            "lr": tune.uniform(0.001, 0.5),
            "momentum": tune.uniform(0.1, 0.9),
            "batch_size": tune.choice([256, 512, 1024, 2048])
            }

    # Add Tune callback.
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
    
    trainable = tune.with_parameters(
        train_cifar,
        data_dir=data_dir,
        num_epochs=num_epochs,
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=callbacks)
    
    optuna_search = OptunaSearch(
        metric="loss",
        mode="min"
    )
    
    analysis = tune.run(
        trainable,
        metric="loss",
        mode="min",
        search_alg=optuna_search,
        config=config,
        num_samples=num_samples,
        resources_per_trial=get_tune_resources(
            num_workers=num_workers, 
            use_gpu=use_gpu, 
            cpus_per_worker=cpus_per_worker),
        name="tune_cifar")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    import argparse
    
    ray_helper = RayHelper()
    ray_helper.start_ray()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of training workers to use.",
        default=1)
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU for training.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to tune.")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs to train for.")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")

    args, _ = parser.parse_known_args()

    num_epochs = 1 if args.smoke_test else args.num_epochs
    num_workers = 1 if args.smoke_test else args.num_workers
    use_gpu = False if args.smoke_test else args.use_gpu
    num_samples = 1 if args.smoke_test else args.num_samples

#     if args.smoke_test:
#         ray.init(num_cpus=2)
#     else:
#         ray.init(address=args.address)

    data_dir = os.environ.get("SM_CHANNEL_TRAIN")
    
    tune_cifar(data_dir, num_samples, num_epochs, num_workers, use_gpu)