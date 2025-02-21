import argparse
import copy
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.models import GIN
from torch_geometric.transforms import Compose, Constant
import tqdm

from src.cosine_scheduler import get_cosine_schedule_with_warmup
from src.models import MLP
from src.tmd import pairwise_TMD
from src.transforms import (
    AddSubgraphCycleCounts,
    AddHomomorphismCycleCounts,
    AddCycleBasisCounts
)
from src.utils import seed_all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, classifier, loader, criterion, optimizer, scheduler):
    model.train()
    classifier.train()
    total_loss = 0.0
    correct = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        embedding = model(data.x, data.edge_index)
        output = global_add_pool(embedding, data.batch)
        output = classifier(output)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pred = output.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
    if scheduler is not None:
        scheduler.step()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def evaluate(model, classifier, loader, criterion):
    model.eval()
    classifier.eval()
    total_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device=device)
            embedding = model(data.x, data.edge_index)
            output = global_add_pool(embedding, data.batch)
            output = classifier(output)
            loss = criterion(output, data.y)

            total_loss += loss.item() * data.num_graphs
            pred = output.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()

            all_preds.append(pred.cpu())
            all_labels.append(data.y.cpu())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return avg_loss, accuracy#, all_preds, all_labels


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        default="MP"
    )
    parser.add_argument(
        "--dataset", 
        type=str,
        default="MUTAG", 
        help="Dataset to train on."
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
        default="sub_C2", 
        help=(
            "Using cycle counts as node features."
            "- hom_CN: add homomorphism counts of cycle whose length is at most N;"
            "- basis_CN: add counts of cycle whose length is at most N in the cycle_basis."
            "- sub_CN: add subgraph counts of cycle whose length is at most N."
            "Note that sub_C2 does not add any positional encoding."
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
    args = parser.parse_args()
    # Seeding
    seed_all(args.seed)
    # Creating folder for the results
    results_dir = f'results/{args.dataset}'
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/args.json", "w") as file:
        json.dump(vars(args), file, indent=4)
    
    transforms =[Constant(cat=False)]
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
    transforms = Compose(
       transforms
    )
    dataset = TUDataset(
        root="data",
        name=args.dataset,
        transform=transforms
    )
    skf_dataset, test_dataset, _, _ = train_test_split(
        dataset, 
        dataset._data.y, 
        stratify=dataset._data.y,
        test_size=0.1, 
        random_state=42
    ) 
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.bs, 
        shuffle=False
    )
    filename = os.path.join(
        results_dir,
        f"tmd_train_test_{args.num_layers+1}.pt"
    )
    if os.path.exists(filename):
        tmd_train_test = torch.load(
            filename
        )
    else:
        tmd_train_test = pairwise_TMD(
            skf_dataset,
            test_dataset,
            depth=args.num_layers+1
        )
        torch.save(
            tmd_train_test,
            filename
        )
    distance_values = np.unique(tmd_train_test.min(0).values)
    # Dataset split
    skf = StratifiedKFold(n_splits=10)
    tmd_metrics = np.zeros(
        (len(distance_values), 10)
    )
    tmd_len = np.zeros(
        (len(distance_values), 10)
    )
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(skf_dataset)), [d.y.item() for d in skf_dataset])):
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
        model = GIN(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers
        ).to(device)   
        classifier = MLP(
            input_channels=args.hidden_channels,
            output_channels=dataset.num_classes,
            norm=False
        ).to(device)
        print(dataset.num_classes)
        optimizer = getattr(torch.optim, args.optim)(
            list(model.parameters()) + list(classifier.parameters()), 
            lr=args.lr
        )
        criterion = torch.nn.CrossEntropyLoss()
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
        best_metric = 0.
        progress = tqdm.trange(args.epochs)
        for epoch in progress:
            train_loss, train_metric = train(
                model, 
                classifier,
                train_loader, 
                criterion, 
                optimizer, 
                scheduler
            )
            losses[0].append(train_loss)
            metrics[0].append(train_metric)
            val_loss, val_metric = evaluate(
                model, 
                classifier,
                val_loader, 
                criterion
            )
            losses[1].append(val_loss)
            metrics[1].append(val_metric)  
            test_loss, test_metric = evaluate(
                model,
                classifier, 
                test_loader, 
                criterion
            )
            losses[2].append(test_loss)
            metrics[2].append(test_metric)  
            if (val_metric > best_metric) :
                best_metric = val_metric
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                best_classifier = copy.deepcopy(classifier)
            progress.set_description(
                f"Train loss: {train_loss:.4f}, acc: {train_metric:.4f}"
                + f" - Best acc: {best_metric:.4f}"
                + f" - Best epoch {best_epoch}"
            )
        min_distance = tmd_train_test[train_idx].min(0).values
        sorted_idx = np.argsort(min_distance)
        
        for n, upper in enumerate(distance_values):
            tmd_dataset = [
                test_dataset[i] for i, cond in enumerate(min_distance <= upper) if cond
            ]
            tmd_loader = DataLoader(
                tmd_dataset, 
                batch_size=args.bs, 
                shuffle=False
            )
            tmd_loss, tmd_metric = evaluate(
                best_model,
                best_classifier, 
                tmd_loader, 
                criterion
            )
            tmd_metrics[n, fold] = tmd_metric
            tmd_len[n, fold] = len(tmd_dataset)
        fig, ax = plt.subplots()
        mean = tmd_metrics[:, :fold+1].mean(1)
        std = tmd_metrics[:, :fold+1].std(1)
        ax.plot(
            distance_values,
            mean
        )
        ax.fill_between(
            distance_values,
            mean - std,
            mean + std,
            alpha=.5
        )
        ax.set_xlabel("TMD from training dataset")
        ax.set_ylabel("Accuracy")
        ax.set_xscale("symlog")
        fig.savefig(
            os.path.join(
                results_dir,
                f"metrics_{args.num_layers+1}.pdf"
            ),
            bbox_inches="tight"
        )
        plt.close(fig)
        filename = os.path.join(
            results_dir,
            f"metrics_{args.num_layers+1}.np"
        )
        np.savetxt(
            filename,
            np.concatenate(
                [distance_values.reshape(-1, 1),
                mean.reshape(-1, 1),
                std.reshape(-1, 1)],
                axis=-1
            ),
            fmt="%5.4f"
        )
