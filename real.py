import argparse
import copy
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from ogb.graphproppred import PygGraphPropPredDataset
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
from src.models import MLP, LinearClassifier, GINModel
from src.tmd import pairwise_TMD
from src.transforms import (
    AddSubgraphCycleCounts,
    AddHomomorphismCycleCounts,
    AddCycleBasisCounts
)
from src.utils import seed_all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_max_degree(dataset):
    """Compute maximum node degree over all graphs in the dataset."""
    max_degree = 0
    for d in dataset:
        edge_index = d.edge_index
        if edge_index is not None:
            row = edge_index[0]
            degs = torch.bincount(row)
            max_degree = max(max_degree, degs.max().item())
    return max_degree

def sum_of_squares_params(*modules):
    """
    Sum of L2-norm^2 of all parameters in the given modules.
    i.e. sum_{t} ||W_t||_2^2 + sum_{l} ||tilde{W}_l||_2^2
    """
    total = 0.0
    for m in modules:
        for p in m.parameters():
            total += p.norm(2).item()**2
    return total

import math

def fro_norm_sq(module_list):
    """Sum of Frobenius norms squared over the given modules: 
       This acts as |w|^2_2 = sum_i ||W_i||^2_F."""
    total = 0.0
    for m in module_list:
        for p in m.parameters():
            # Frobenius norm^2 = sum of squares of all entries
            total += p.pow(2).sum().item()
    return total

def spectral_norms(module_list):
    """
    Compute the spectral (2-)norm of each layer W_i (as ||W_i||_2).
    In a real project, you might do a proper spectral norm estimate 
    or use a direct call if the layer is a simple linear. Here, we do 
    an L2-operator *approximation* by taking ||W_i||_F for demonstration.
    """
    norms_2 = []
    for m in module_list:
        # naive approach: approximate spectral norm by the Frobenius norm
        spectral_norm = 0.0
        for p in m.parameters():
            if p.ndim == 1:
                # Skip biases
                continue
            S = torch.linalg.svdvals(p)
            spectral_norm += S.max().item()
        norms_2.append(spectral_norm)
    return norms_2

def pac_bayes_bound(
    module_list,
    m,
    B=1.0,
    delta=0.1,
    gamma=0.1,
    l=5,              # number of GNN layers
    h=64,             # hidden dimension
    max_degree=2,     # d: maximum node degree in train/test
    cphi=1.0,
    crho=1.0,
    cg=1.0
):
    """
    Computes a PAC-Bayes style bound (Theorem 3.4-like) but revised so that:
      - C = Cφ·Cρ·Cg · maxᵢ||Wᵢ||₂,
      - d is the maximum degree in train/test set.
    
    train_error : L_{S,γ}(f_w) or approximate margin-based error on training
    module_list : e.g. [best_model, best_classifier]
    m           : size of training set
    B, delta, gamma : Theorem constants/hyperparams
    l           : number of GNN layers
    h           : e.g. hidden_channels
    max_degree  : the maximum node degree (for d)
    cphi, crho, cg : your constants Cφ, Cρ, Cg
    
    Returns: float, the bound: L_{D,0}(f_w) ≤ train_error + <penalty>
    """
    
    # 1) Compute norms
    w_fro_sq = fro_norm_sq(module_list)    # sum of Frobenius norms^2 = |w|₂²
    norms_2_list = spectral_norms(module_list)
    
    # For the theorem, we typically identify W1, Wl, etc. 
    # We'll just do a naive check in case not enough layers:
    if len(norms_2_list) < 2:
        return float('nan')  # or raise an exception
    
    # largest spectral norm:
    maxW_2 = max(norms_2_list)
    # smallest spectral norm (for ζ):
    zeta = min(norms_2_list)  
    
    # let W1 = norms_2_list[0], Wl = norms_2_list[-1]
    W1_2 = norms_2_list[0]
    Wl_2 = norms_2_list[-1]
    
    # λ = ||W1||₂ * ||Wl||₂
    lam = W1_2 * Wl_2
    
    # 2) Build the constant C = Cφ·Cρ·Cg·maxᵢ||Wᵢ||₂
    bigC = cphi * crho * cg * maxW_2
    
    # 3) Construct ξ (xi). For instance, you might do:
    #     xi = C · (d^(l-2)) / (d - 1)
    # but *check your exact theorem statement* for the exponent. 
    # Here is a sample version:
    xi = cphi*((bigC*max_degree)**(l-1)-1) / (bigC*max_degree - 1)
    
    # 4) The factor: max( ζ^{-(l+1)}, (λ·ξ)^{(l+1)/l} )
    partA = zeta ** (-(l+1))
    partB = (lam * xi) ** ((l+1)/l)
    big_max = max(partA, partB)
    
    # 5) The sqrt(...) portion: 
    #    B² (big_max²) · l² h log(lh) · |w|₂² + log( m^(l+1)/δ )   all / (γ²·m)
    log_part = math.log( (m**(l+1)) / delta + 1e-9 )
    
    numerator = (B**2) * (big_max**2) * (l**2) * h * math.log(l*h + 1e-9) * w_fro_sq
    numerator += log_part
    denominator = (gamma**2) * m
    
    penalty = math.sqrt(numerator / denominator)
    return penalty

def train(model, classifier, loader, criterion, optimizer, scheduler):
    model.train()
    classifier.train()
    total_loss = 0.0
    correct = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        embedding = model(data.x, data.edge_index, data.batch)
        output = classifier(embedding)
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
            embedding = model(data.x, data.edge_index, data.batch)
            output = classifier(embedding)
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
    # Extras for theoretical bound:
    parser.add_argument("--alpha", type=float, default=0.25, help="Alpha in the theoretical bound.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma in the theoretical bound.")
    parser.add_argument("--delta", type=float, default=0.1, help="Delta in the theoretical bound.")
    parser.add_argument("--K", type=float, default=1.0, help="K in the theoretical bound.")
    parser.add_argument("--C_const", type=float, default=1.0, help="C in the theoretical bound.")
    
    args = parser.parse_args()
    # Seeding
    seed_all(args.seed)
    # Creating folder for the results
    results_dir = f'results/{args.dataset}'
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/args.json", "w") as file:
        json.dump(vars(args), file, indent=4)
    
    transforms =[
        Constant()
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
    transforms = Compose(
       transforms
    )
    dataset = TUDataset(
        root="data",
        name=args.dataset,
        transform=transforms
    )
    max_deg = compute_max_degree(dataset)
    train_dataset, test_dataset, _, _ = train_test_split(
        dataset, 
        [d.y.item() for d in dataset], 
        stratify=[d.y.item() for d in dataset],
        test_size=0.1, 
        random_state=42
    ) 
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.bs, 
        shuffle=False
    )
    if "_C2" in args.pe:
        filename = os.path.join(
            results_dir,
            f"tmd_train_test_{args.num_layers+1}.pt"
        )
    else:
        filename = os.path.join(
            results_dir,
            f"tmd_train_test_{args.num_layers+1}_{args.pe}.pt"
        )
    if os.path.exists(filename):
        tmd_train_test = torch.load(
            filename
        )
    else:
        tmd_train_test = pairwise_TMD(
            train_dataset,
            test_dataset,
            depth=args.num_layers+1
        )
        torch.save(
            tmd_train_test,
            filename
        )
    quantiles_list = [.15, .3, .45, .6, .75, .9, 1]
    all_distance_values = tmd_train_test.min(0).values #np.unique()

    distance_values = np.quantile(
        all_distance_values,
        quantiles_list
    )

    # We'll track final train/test accuracies (per fold) and also final error bounds for entire test
    fold_train_accuracies = []
    fold_test_accuracies = []
    fold_test_err_bounds = []  # The final bound on test error (for entire test set, or overall distribution)
    fold_pac_bounds = []
    # Dataset split
    skf = StratifiedKFold(n_splits=10)
    tmd_metrics = np.zeros(
        (len(distance_values), 10)
    )
    generalization_tmd_metrics = np.zeros(
        (len(distance_values), 10)
    )
    tmd_len = np.zeros(
        (len(distance_values), 10)
    )
    subset_err_bound = np.zeros(
        (len(quantiles_list), 10)
    )
    # For quantile-based partition: empirical accuracy & error bound
    subset_sizes = np.zeros(
        (len(quantiles_list), 10)
    )
    # We'll also store the *max TMD* in that subset (per fold) to show in the final table
    subset_max_tmd = np.zeros(
        (len(quantiles_list), 10)
    )
    


    splits = skf.split(train_dataset, [d.y for d in train_dataset])
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_loader = DataLoader(
            [train_dataset[idx] for idx in train_idx], 
            batch_size=args.bs, 
            shuffle=True
        )
        val_loader = DataLoader(
            [train_dataset[idx] for idx in val_idx], 
            batch_size=args.bs, 
            shuffle=False
        ) 
        model = GINModel(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers
        ).to(device)   
        classifier = LinearClassifier(
            in_channels=args.hidden_channels,
            out_channels=dataset.num_classes
        ).to(device)
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
        #min_distance = tmd_train_test[train_idx].min(0).values
        #sorted_idx = np.argsort(all_distance_values)


        # Evaluate the best model on the full train set & test set
        train_loss_best, train_acc_best = evaluate(best_model, best_classifier, train_loader, criterion)
        test_loss_best,  test_acc_best  = evaluate(best_model, best_classifier, test_loader, criterion)

        fold_train_accuracies.append(train_acc_best)
        fold_test_accuracies.append(test_acc_best)

        # Compute a theoretical error bound for the entire distribution
        import math
        N_tr  = len(train_dataset)
        alpha = args.alpha
        gamma = args.gamma
        delta = args.delta
        K_val = args.K
        C_val = args.C_const
        B_val = torch.norm(dataset.x, p=2, dim=1).max().item()

        b_val = args.hidden_channels
        D_val = 2*(args.num_layers + 1) # total # of learnable matrices (excluding biases)
        sum_sq_params = sum_of_squares_params(best_model, best_classifier)

        # test_min_d = tmd_train_test[train_idx].min(0).values

        # maximum TMD from training to test for *this fold*
        xi_zeta_alltest = float(all_distance_values.max()) if len(all_distance_values) > 0 else 0.0

        def bound_on_test_error(xi_zeta):
            # You can refine these formula details as needed
            term1 = (b_val * sum_sq_params) / (
                (N_tr ** (2*alpha)) * ((gamma/8) ** (2.0/D_val))
            ) * (xi_zeta ** (2.0/D_val)) if N_tr>0 else 0.0

            ln_part = math.log(
                2*b_val*D_val*C_val*((2*max_deg*B_val)**(1.0/D_val)) + 1e-10
            )
            term2 = (b_val**2 * ln_part) / (
                (N_tr**(2*alpha)) * (gamma**(1.0/D_val)) * (delta+1e-10)
            ) if N_tr>0 else 0.0

            term3 = 1.0 / (N_tr**(1.0 - 2.0*alpha)) if N_tr>0 else 0.0
            term4 = (C_val * K_val * xi_zeta)
            return term1 + term2 + term3 + term4

        fold_err_bound_alltest = bound_on_test_error(xi_zeta_alltest)
        fold_test_err_bounds.append(fold_err_bound_alltest)

        # 3) PAC-Bayes bound from Theorem 3.4
        #    Put your actual values for B, delta, gamma, l, h, etc.  
        #    E.g. l = args.num_layers, h = args.hidden_channels, ...
        fold_pac_bound = pac_bayes_bound(
            module_list=[best_model, best_classifier],
            m=len(train_dataset),
            B=1.0,
            delta=0.1,
            gamma=0.1,
            l=D_val,        # number of GNN layers
            h=args.hidden_channels,             # hidden dimension
            max_degree=max_deg,     # d: maximum node degree in train/test
            cphi=1.0,
            crho=1.0,
            cg=1.0
        )
        fold_pac_bounds.append(fold_pac_bound)

        for n, upper in enumerate(distance_values):
            tmd_dataset = [
                test_dataset[i] for i, cond in enumerate(all_distance_values <= upper) if cond
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
            generalization_tmd_metrics[n, fold] = metrics[0][best_epoch] - tmd_metric
            tmd_len[n, fold] = len(tmd_dataset)
            # # average TMD in this subset (for that fold)
            # print(test_min_d.shape)
            # sub_indices = np.where(test_min_d <= upper)[0]
            # print(sub_indices.shape)
            # if len(sub_indices) > 0:
            #     print(test_min_d[sub_indices].shape)
            #     exit()
            #     subset_max_tmd_val = test_min_d[sub_indices].max()  # min(0).values.
            # else:
            #     subset_max_tmd_val = 0.0
            # subset_max_tmd[n, fold] = subset_max_tmd_val

            # Specialized error bound for that subset by plugging in xi_zeta = threshold_q
            err_bound_sub = bound_on_test_error(upper)
            subset_err_bound[n, fold] = err_bound_sub
        print(tmd_len)
        print(generalization_tmd_metrics)
        fig, ax = plt.subplots()
        mean = tmd_metrics[:, :fold+1].mean(1)
        std = tmd_metrics[:, :fold+1].std(1)
        ax.plot(
            distance_values,
            mean,
            marker="o"
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
        if "_C2" in args.pe:
            figname = os.path.join(
                results_dir,
                f"metrics_{args.num_layers+1}.pdf"
            )
        else:
            figname = os.path.join(
                results_dir,
                f"metrics_{args.num_layers+1}_{args.pe}.pdf"
            )
        fig.savefig(
            figname,
            bbox_inches="tight"
        )
        plt.close(fig)
        filename = figname.replace(".pdf", ".np")
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
            
    # =======================
    # AFTER all folds
    # =======================
    mean_train_acc = np.mean(fold_train_accuracies)
    std_train_acc  = np.std(fold_train_accuracies)
    mean_test_acc  = np.mean(fold_test_accuracies)
    std_test_acc   = np.std(fold_test_accuracies)

    # Theoretical test ERROR bound across folds (entire test distribution)
    mean_bound_error = np.mean(fold_test_err_bounds)
    std_bound_error  = np.std(fold_test_err_bounds)

    print("Cross-validation summary:")
    print(f"Train Accuracy: {mean_train_acc:.4f} +/- {std_train_acc:.4f}")
    print(f"Test Accuracy:  {mean_test_acc:.4f} +/- {std_test_acc:.4f}")
    print("Theoretical Error Bound (entire test distribution):")
    print(f"   Mean: {mean_bound_error:.4f}  +/- {std_bound_error:.4f}")
    print(f"   PAC-Bayes bound: {np.mean(fold_pac_bounds):.4f} +/- {np.std(fold_pac_bounds):.4f}")

    # 2) Quantile-based slicing
    mean_subset_acc      = generalization_tmd_metrics.mean(axis=1)
    std_subset_acc       = generalization_tmd_metrics.std(axis=1)
    mean_subset_err      = subset_err_bound.mean(axis=1)
    std_subset_err       = subset_err_bound.std(axis=1)
    avg_subset_size      = tmd_len.mean(axis=1)
    mean_subset_max_tmd = subset_max_tmd.mean(axis=1)
    std_subset_max_tmd  = subset_max_tmd.std(axis=1)

    # Finally, print the expanded table
    print("\nQuantile-based results across folds (mean +/- std):")
    print("   q    SubsetSize    MeanTMD      StdTMD      EmpAcc      EmpAccStd     ErrBound     ErrBoundStd     TrainAcc  TrainAccStd")
    for i, q in enumerate(quantiles_list):
        print(f"{q:5.2f}   {avg_subset_size[i]:9.1f}   {mean_subset_max_tmd[i]:9.4f}"
              f"   {std_subset_max_tmd[i]:9.4f}   {mean_subset_acc[i]:9.4f}   {std_subset_acc[i]:9.4f}"
              f"   {mean_subset_err[i]:9.4f}   {std_subset_err[i]:9.4f}"
              f"   {mean_train_acc:9.4f}   {std_train_acc:9.4f}")
    fig, ax = plt.subplots()
    axtwinx = ax.twinx()
    ax.plot(
        distance_values,
        mean_subset_acc,
        marker="o",
        color="tab:orange",
        label="Empirical Generalization Error"
    )
    ax.fill_between(
        distance_values,
        mean_subset_acc - std_subset_acc,
        mean_subset_acc + std_subset_acc,
        color="tab:orange",
        alpha=.5
    )
    ax.tick_params(axis='y', labelcolor="tab:orange")
    ax.set_ylabel('Empirical generalization error', color="tab:orange")
    axtwinx.plot(
        distance_values,
        mean_subset_err,
        marker="o",
        color="black",
        label="Error Bound"
    )
    axtwinx.fill_between(
        distance_values,
        (mean_subset_err - std_subset_err),
        (mean_subset_err + std_subset_err),
        color="black",
        alpha=.5
    )
    axtwinx.tick_params(axis='y', labelcolor="black")
    axtwinx.set_ylabel('Error bound', color="black")
    ax.set_xlabel("TMD to training dataset")
    ax.set_xscale("symlog")
    ax.set_title(args.dataset)
    if "_C2" in args.pe:
        figname = os.path.join(
            results_dir,
            f"bound_{args.num_layers+1}.svg"
        )
    else:
        figname = os.path.join(
            results_dir,
            f"bound_{args.num_layers+1}_{args.pe}.svg"
        )
    fig.savefig(
        figname,
        bbox_inches="tight"
    )
    plt.close(fig)