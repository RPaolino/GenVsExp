import argparse
from collections import Counter
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import Constant, Compose
import tqdm
from datetime import datetime
import wandb

from src.utils import seed_all
from src.models import (
    GNN,
    GNNnClassifier
)
from src.synthetic_dataset import SyntheticDataset
from src.cosine_scheduler import get_cosine_schedule_with_warmup
from src.transforms import (
    AddSubgraphCycleCounts,
    AddHomomorphismCycleCounts,
    AddCycleBasisCounts,
    AddTaskCounts,
    AddLabel,
    AddSubgraph
)
from src.tmd import pairwise_TMD 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_loss(args, data, model, criterion):
    data, target = data.to(device), data.y.to(device)
    output = model(data)
    if len(output.shape) != len(target.shape):
        output = torch.squeeze(output, dim=1)
    loss = criterion(output, target)
    return loss, output

def compute_metric_batch(args, output, target, storage):
    if ("sum_sub_C" in args.task 
        or "sum_basis_C" in args.task 
        or "hom_C" in args.task 
        or "hom_D" in args.task 
        or args.task == "tmd"):
        # Classification - Accuracy
        if storage is None:
            storage = 0
        pred = output.argmax(dim=1)
        storage += pred.eq(target).sum().item()
    elif args.task == "node_count":
        # Regression - MAE
        if storage is None:
            storage = 0
        l1_sum_reduction = torch.nn.L1Loss(reduction="sum")
        storage += float(l1_sum_reduction(output, target))
    else:
        raise Exception("Unknown Task")
    return storage

def train(args, model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss = 0
    storage = None
    for data in loader:
        optimizer.zero_grad()
        loss, output = compute_loss(args, data, model, criterion)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        storage = compute_metric_batch(args, output, data.y, storage)
    if scheduler is not None:
        scheduler.step()
    return total_loss / len(loader), storage / len(loader.dataset)

def evaluate(args, model, loader, criterion):
    model.eval()
    total_loss = 0
    storage = None
    with torch.no_grad():
        for data in loader:
            loss, output = compute_loss(args, data, model, criterion)
            total_loss += loss.item()
            storage = compute_metric_batch(args, output, data.y, storage)
    return total_loss / len(loader), storage / len(loader.dataset)

def get_mlp(num_layers, in_dim, out_dim, hidden_dim, activation, dropout_rate):
    layers = []
    for i in range(num_layers):
        in_size = hidden_dim if i > 0 else in_dim
        out_size = hidden_dim if i < num_layers - 1 else out_dim
        layers.append(
            torch.nn.Linear(in_size, out_size)
        )
        layers.append(
            torch.nn.BatchNorm1d(out_size)
        )          
        if num_layers > 0 and i < num_layers - 1:
            layers.append(
                torch.nn.Dropout(p=dropout_rate)
            )
            layers.append(activation)       
    return torch.nn.Sequential(*layers)  

def main(config = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        default="MP"
    )
    parser.add_argument(
        "--max_distance", 
        type=int, 
        default=5, 
        help="Distance encoding."
    )
    parser.add_argument(
        "--dataset", 
        type=str,
        default="er", 
        choices=["er", "ba", "sbm"],
        help=(
            "Which dataset to construct."
            "-er: Erods Renyi graph;"
            "-ba: Barabasi Albert graph;"
            "-sbm: Stochastic Block Model graph."
        )
    )
    parser.add_argument(
        "--num_graphs", 
        type=int,
        default=3000, 
        help="Number of graphs to generate."
    )
    parser.add_argument(
        "--device", 
        type=int, 
        default=0, 
        help="CUDA device."
    )
    parser.add_argument(
        "--num_layers", 
        type=int, 
        default=5, 
        help="Number of layers."
    )
    parser.add_argument(
        "--hidden_channels", 
        type=int, 
        default=64, 
        help="Number of hidden channels."
    )
    parser.add_argument(
        "--pool", 
        type=str, 
        default="sum", 
        choices=["sum", "mean"],
        help="Pooling method."
    )
    parser.add_argument(
        "--bs", 
        type=int, 
        default=128, 
        help="Batch size."
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-3, 
        help="Learning rate."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100, 
        help="Number of training epochs."
    )
    parser.add_argument(
        "--pe", 
        type=str, 
        default="sub_C4", 
        help=(
            "Using cycle counts as node features."
            "- hom_CN: add homomorphism counts of cycle whose length is at most N;"
            "- basis_CN: add counts of cycle whose length is at most N in the cycle_basis."
            "- sub_CN: add subgraph counts of cycle whose length is at most N."
            "Note that sub_C2 does not add any positional encoding."
        )
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="sum_basis_C4", 
        help=(
            "Task to use during learning."
            "-sum_sub_CN: the label will be the number of cycles of length at most N;"
            "-sum_basis_CN: the label will be the number of cycles of length at most N in the cycle_basis;"
            "-hom_D: the label will be the number of homomorphism from the dragon graph;"
            "-hom_CN: the label will be the number of homomorphism from the cycle of length N;"
            "-num_nodes: the label will be the number of nodes;"
            "-tmd: labels are correlated to the Tree Mover's Distance."
        )
    )
    parser.add_argument(
        "--optim", 
        type=str, 
        default="Adam", 
        help="Optimizer to use."
    )
    parser.add_argument(
        "--scheduler", 
        type=str, 
        default="cosine",
        choices=["none", "cosine"],
        help="Scheduler to use."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help=""
    )
    if config is None:
        args = parser.parse_args()
    else:
        print(transform_dict_to_args_list(config))
        args = parser.parse_known_args(transform_dict_to_args_list(config))[0]
    if "tmd" in args.task:
        print("Overwriting hyperparams for TMD task")
        if args.dataset != "er":
            print(f"Dataset from {args.dataset} to ER")
            args.dataset = "er"
        args.num_nodes_lower = 15
        args.num_nodes_upper = 35
        if args.num_graphs > 500:
            print(f"Num. graphs from {args.num_graphs} to 500")
            args.num_graphs = 500
    else:
        args.num_nodes_lower = 35
        args.num_nodes_upper = 55
    # Seeding
    seed_all(args.seed)
    # Creating folder for the results
    results_dir = f'results/{args.dataset}/{datetime.now().strftime("%Y%m%d%H%M%S")}'
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/args.json", "w") as file:
        json.dump(vars(args), file, indent=4)
    # Get transforms
    transforms = [
        Constant(cat=False),
    ]
    if "hom_C" in args.pe:
        transforms.append(
            AddHomomorphismCycleCounts(
                length_bound=int(args.pe.replace("hom_C", ""))
            )
        )
    elif "sub_C" in args.pe:
        transforms.append(
            AddSubgraphCycleCounts(
                length_bound=int(args.pe.replace("sub_C", ""))
            )
        )
    elif "basis_C" in args.pe:
        transforms.append(
            AddCycleBasisCounts(
                length_bound=int(args.pe.replace("basis_C", ""))
            )
        )
    else:
        raise NotImplementedError("Positional encoding not implemented!")
    if ("tmd" not in args.task):
        transforms.append(
            AddTaskCounts(
                task=args.task
            )
        )
    transforms = Compose(
       transforms
    )
    # Get Dataset
    dataset = SyntheticDataset(
        num_graphs=args.num_graphs,
        name=args.dataset,
        pe=args.pe,
        transform=transforms,
        num_nodes_lower=args.num_nodes_lower,
        num_nodes_upper=args.num_nodes_upper,
        task=args.task,
        depth=args.num_layers+1
    )
    print("X shape", dataset._data.x.shape)
    print("Y", Counter(dataset._data.y.cpu().tolist()))
    skf_dataset, test_dataset, _, _ = train_test_split(
        dataset, 
        dataset._data.y, 
        stratify=dataset._data.y,
        test_size=0.1, 
        random_state=42
    ) 
    # print(len(skf_dataset), len(test_dataset))
    # print(skf_dataset[0])
    # filename = os.path.join(
    #         results_dir,
    #         f"{args.task}_train_test_{args.num_layers+1}_{args.pe}.pt"
    # )
    # tmd_matrix = pairwise_TMD(
    #         skf_dataset,
    #         test_dataset,
    #         depth=args.num_layers+1
    # )
    # torch.save(
    #     tmd_matrix,
    #     filename
    # )
    # print("Max: ", tmd_matrix.min(0).values.max())
    # print("Test TMDs: ", tmd_matrix.min(0).values)
    # exit()
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.bs, 
        shuffle=False
    )
    # Dataset split
    skf = StratifiedKFold(n_splits=10)
    for fold, (train_idx, val_idx) in enumerate(skf.split(skf_dataset, [d.y for d in skf_dataset])):
        train_loader = DataLoader(
            [skf_dataset[idx] for idx in train_idx], 
            batch_size=args.bs, 
            shuffle=True
        )
        val_loader = DataLoader(
            [skf_dataset[idx] for idx in val_idx], 
            batch_size=args.bs, 
            shuffle=False
        )
        dataset_name = args.dataset.lower()
        # Build model
        if dataset_name in ["er", "ba", "sbm", "tmd"]:
            x_encoder = torch.nn.Linear(dataset._data.x.shape[1], args.hidden_channels)
            edge_attr_encoder = None#Linear(1, args.hidden_channels)
    
        message_passing = GNN(
            model=args.model, 
            hidden_channels=args.hidden_channels, 
            num_layers=args.num_layers, 
            max_distance=args.max_distance,
            task="g",
            x_encoder=x_encoder,
            edge_attr_encoder=edge_attr_encoder,
            output_channels=1,
            pool=args.pool
        ).to(device)

        if ("sum_sub_C" in args.task 
            or "sum_basis_C" in args.task 
            or "hom_" in args.task
            or args.task == "tmd"):
            criterion = torch.nn.CrossEntropyLoss()
            classifier = get_mlp(
                2, 
                args.hidden_channels, 
                2, 
                args.hidden_channels, 
                torch.nn.ReLU(), 
                0
            )
            metric_name, task_type = "accuracy", "maximize"
        elif args.task == "node_count":
            criterion = torch.nn.L1Loss()
            classifier = get_mlp(
                2, 
                args.hidden_channels, 
                1, 
                args.hidden_channels, 
                torch.nn.ReLU(), 
                0
            )
            metric_name, task_type = "mae", "minimize"
        else:
            raise Exception("Unknown task")
        if args.dataset == "tmd":
            assert args.task == "tmd"
        classifier = classifier.to(device)
        model = GNNnClassifier(message_passing, classifier)

        optimizer = getattr(torch.optim, args.optim)(
            model.parameters(), 
            lr=args.lr
        )
        scheduler = None
        if args.scheduler=="cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer, 
                num_warmup_steps=5, 
                num_training_steps=args.epochs
            )
        # Initialize metrics
        losses, metrics = [[], [], []], [[], [], []]
        best_epoch = 0
        if task_type == "maximize":
            best_metric = -1e9
        elif task_type == "minimize":
            best_metric = 1e9
        else:
            raise Exception("Unknown task type")
        # Starting the learning
        wandb.init(project = "GenVsExp")
        progress = tqdm.trange(args.epochs)
        for epoch in progress:
            train_loss, train_metric = train(
                args, 
                model, 
                train_loader, 
                criterion, 
                optimizer, 
                scheduler
            )
            losses[0].append(train_loss)
            metrics[0].append(train_metric)
            val_loss, val_metric = evaluate(
                args, 
                model, 
                val_loader, 
                criterion
            )
            losses[1].append(val_loss)
            metrics[1].append(val_metric)  
            test_loss, test_metric = evaluate(
                args, 
                model, 
                test_loader, 
                criterion
            )
            losses[2].append(test_loss)
            metrics[2].append(test_metric)  
            # Save metrics     
            with open(f"{results_dir}/metrics_{fold}.csv", "w", newline="\n") as metric:
                writer = csv.writer(metric)
                # Write the header
                writer.writerow([
                    "loss_train", "loss_val", "loss_test", f"{metric_name}_train", f"{metric_name}_val", f"{metric_name}_test"
                ])
                # Write the data rows
                writer.writerows(
                    list(
                        map(
                            lambda l: [f"{element:.4f}" for element in l], 
                            [l.tolist() + m.tolist() for l, m in zip(np.transpose(losses), np.transpose(metrics))]
                        )
                    )
                )
            fig, ax = plt.subplots(ncols=2, figsize=(4.8*2, 6.4))
            for y, label in zip(losses, ["train", "val", "test"]):
                ax[0].plot(
                    np.arange(epoch+1),
                    y,
                    label=label
                )
            ax[0].set_title("Loss")
            ax[0].legend()
            for y, label in zip(metrics, ["train", "val", "test"]):
                ax[1].plot(
                    np.arange(epoch+1),
                    y,
                    label=label
                )
            ax[1].set_title(f"{metric_name}")
            ax[1].legend()
            fig.savefig(f"{results_dir}/metrics_{fold}.pdf", bbox_inches="tight")
            plt.close(fig)
            if ((task_type == "maximize" and val_metric > best_metric) 
                or (task_type == "minimize" and val_metric < best_metric)):
                best_metric = val_metric
                best_epoch = epoch
            progress.set_description(
                f"Train loss: {train_loss:.4f}, {metric_name}: {train_metric:.4f}"
                + f" - Best {metric_name}: {best_metric:.4f}"
                + f" - Best epoch {best_epoch}"
            )
            wandb.log({
                "epoch": epoch,
                "loss_train": train_loss,
                "loss_val": val_loss,
                "loss_test": test_loss,
                f"{metric_name}_train": train_metric,
                f"{metric_name}_val": val_metric,
                f"{metric_name}_test": test_metric,
                f"loss_generalization_gap": train_loss-test_loss,
                f"{metric_name}_generalization_gap": train_metric-test_metric,
                "learning_rate": optimizer.param_groups[-1]["lr"]
            })
        wandb.log({
            "final/epoch": best_epoch,
            "final/loss_train": losses[0][best_epoch],
            "final/loss_val": losses[1][best_epoch],
            "final/loss_test": losses[2][best_epoch],
            f"final/{metric_name}_train": metrics[0][best_epoch],
            f"final/{metric_name}_val": metrics[1][best_epoch],
            f"final/{metric_name}_test": metrics[2][best_epoch],
            f"final/loss_generalization_gap": losses[0][best_epoch]-losses[2][best_epoch],
            f"final/{metric_name}_generalization_gap": metrics[0][best_epoch]-metrics[2][best_epoch]
        })
        wandb.finish()

    return {
        "best_epoch": best_epoch,
        "losses": losses,
        "metrics": metrics
    }
    
if __name__ == "__main__":
    main()