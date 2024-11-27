"""
Trains and evaluates a model a single time for given hyperparameters.
"""

import random
import time 
import os
import sys
import json

import wandb
import torch
import numpy as np

from Exp.parser import parse_args
from Misc.config import config
from Misc.utils import list_of_dictionary_to_dictionary_of_lists
from Exp.preparation import load_dataset, get_model, get_optimizer_scheduler, get_loss
from Exp.training_loop_functions import train, eval, step_scheduler
from Exp.utils import get_classes_tasks_feats


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def track_epoch(epoch, metric_name, train_result, val_result, test_result, lr):
    log_dict = {
        "Epoch": epoch,
        "Train/Loss": train_result["total_loss"],
        "Val/Loss": val_result["total_loss"],
        f"Val/{metric_name}": val_result[metric_name],
        "Test/Loss": test_result["total_loss"],
        f"Test/{metric_name}": test_result[metric_name],
        "LearningRate": lr,
        "Train/loss_pred": train_result["loss_pred"],
        "Train/loss_graph_emb": train_result["loss_graph_emb"],
        "Train/loss_layer_emb": train_result["loss_layer_emb"]
        }
    
    # Track layer embedding losses
    layer_emb_keys = list(filter(lambda key: "weighted" in key and "loss_layer" in key,train_result.keys()))
    for key in layer_emb_keys:
        log_dict[f"Train/{key}"] = train_result[key]
    
    wandb.log(log_dict)
    
def print_progress(train_loss, val_loss, test_loss, metric_name, val_metric, test_metric):
    print(f"\tTRAIN\t loss: {train_loss:6.4f}")
    print(f"\tVAL\t loss: {val_loss:6.4f}\t  {metric_name}: {val_metric:10.4f}")
    print(f"\tTEST\t loss: {test_loss:6.4f}\t  {metric_name}: {test_metric:10.4f}")
    
def main(args):
    print(args)
    device = args.device
    use_tracking = args.use_tracking
    
    set_seed(args.seed)
    train_loader, val_loader, test_loader = load_dataset(args, config)
    num_classes, num_tasks, num_vertex_features = get_classes_tasks_feats(train_loader.dataset, args.dataset)
        
    print(f"#Features: {num_vertex_features}")
    print(f"#Classes: {num_classes}")
    print(f"#Tasks: {num_tasks}")
    print(f"#Training samples: {len(train_loader.dataset)}")
    print(f"Using graph embedding: {args.use_graph_emb}")
    print(f"Using node embedding: {args.use_node_emb}")

    model = get_model(args, num_classes, num_vertex_features, num_tasks)
    nr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device)
    optimizer, scheduler = get_optimizer_scheduler(model, args)
    loss_dict = get_loss(args)
    loss_fct = loss_dict["loss"]
    eval_name = loss_dict["metric"]
    metric_method = loss_dict["metric_method"]

    if not os.path.isdir(config.MODELS_PATH):
        os.mkdir(config.MODELS_PATH)

    model_name = f'{args.model}_{args.dataset}_{time.strftime("%H:%M:%S", time.localtime())}.pt'
    model_path = os.path.join(config.MODELS_PATH, model_name)
    print(model_path)

    if eval_name in ["mae", "rmse (ogb)"]:
        mode = "min"
    else:
        mode = "max"

    if use_tracking:
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(
            config = args,
            project = config.project)

    do_predict_logits = args.teacher != ""

    print("Begin training.\n")
    time_start = time.time()
    train_results, val_results, test_results = [], [], []
    best_val_result = sys.float_info.max if mode == "min" else sys.float_info.min
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}")
        train_result = train(model, device, train_loader, optimizer, loss_fct, eval_name, use_tracking, metric_method=metric_method, use_graph_emb = args.use_graph_emb, use_node_emb = args.use_node_emb, graph_emb_fac = args.graph_emb_fac, node_emb_fac=args.node_emb_fac, use_pred_loss=args.use_pred_loss, final_mp_layers=args.final_mp_layers, do_predict_logits=do_predict_logits)
        val_result = eval(model, device, val_loader, loss_fct, eval_name, metric_method=metric_method)
        test_result = eval(model, device, test_loader, loss_fct, eval_name, metric_method=metric_method)

        train_results.append(train_result)
        val_results.append(val_result)
        test_results.append(test_result)

        print_progress(train_result['total_loss'], val_result['total_loss'], test_result['total_loss'], eval_name, val_result[eval_name], test_result[eval_name])

        if use_tracking:
            track_epoch(epoch, eval_name, train_result, val_result, test_result, optimizer.param_groups[0]['lr'])

        step_scheduler(scheduler, args, val_result["total_loss"])

        # Store the model 
        if args.do_store_model and ((mode == "min" and val_result[eval_name] < best_val_result) or (mode == "max" and val_result[eval_name] > best_val_result)):
            torch.save(model.state_dict(), model_path)
            best_val_result = val_result[eval_name]

        # EXIT CONDITIONS
        if optimizer.param_groups[0]['lr'] < args.min_lr:
                print("\nLR reached minimum: exiting.")
                break

        runtime = (time.time()-time_start)/3600
        if args.max_time > 0 and runtime > args.max_time:
            print("\nMaximum training time reached: exiting.")
            break

    # Final result
    train_results = list_of_dictionary_to_dictionary_of_lists(train_results)
    val_results = list_of_dictionary_to_dictionary_of_lists(val_results)
    test_result = list_of_dictionary_to_dictionary_of_lists(test_results)

    if args.selection == "best_val":
        if mode == "min":
            best_val_epoch = np.argmin(val_results[eval_name])
        else:
            best_val_epoch = np.argmax(val_results[eval_name])
    else:
        print(f"Selector: last. Best_val_epoch is set to the last epoch")
        best_val_epoch = len(train_results['total_loss']) - 1

    loss_train, loss_val, loss_test = train_results['total_loss'][best_val_epoch], val_results['total_loss'][best_val_epoch], test_result['total_loss'][best_val_epoch]
    result_val, result_test = val_results[eval_name][best_val_epoch], test_result[eval_name][best_val_epoch]

    print("\n\nFinal Result:")
    print(f"\tRuntime: {runtime:.2f}h")
    print(f"\tBest epoch {best_val_epoch} / {epoch}")
    print_progress(loss_train, loss_val, loss_test, eval_name, result_val, result_test)

    if use_tracking:
        wandb.log({
            "Final/Train/Loss": loss_train,
            "Final/Val/Loss": loss_val,
            f"Final/Val/{eval_name}": result_val,
            "Final/Test/Loss": loss_test,
            f"Final/Test/{eval_name}": result_test})

        wandb.finish()

    if args.do_store_model:
        print("Storing model.")
        
        # Load the best model
        model.load_state_dict(torch.load(model_path))

        # Delete temporary model
        os.unlink(model_path)      
        
        # Store final model
        model_name = f'{args.model}_{args.dataset}__{args.dataset}_val_{result_val:.4f}_test_{result_test:.4f}_{time.strftime("%H:%M:%S", time.localtime())}'
        model_path = os.path.join(config.MODELS_PATH, model_name)
        torch.save(model.state_dict(), model_path + ".pt")
        
        # Store model information
        with open(os.path.join(config.MODELS_PATH, model_name + ".json"), "w") as file:
            json.dump(args.__dict__, file, indent=4)

    return {        
        "mode": mode,
        "loss_train": loss_train, 
        "loss_val":loss_val,
        "loss_test": loss_test,
        "val": result_val,
        "test": result_test,
        "runtime_hours":  runtime,
        "epochs": epoch,
        "best_val_epoch": int(best_val_epoch),
        "parameters": nr_parameters,
        "details_train": train_results,
        "details_val": val_results,
        "details_test": test_results,
        }        

def run(passed_args = None):
    args = parse_args(passed_args)
    return main(args)

if __name__ == "__main__":
    run()