from collections import defaultdict

import tqdm
import torch
import wandb
from ogb.graphproppred import Evaluator
import torch.nn.functional as F
import torch.nn as nn

from Exp.preparation import get_evaluator
from Exp.emb_extraction_utils import EmbStorageObject

l2_loss = nn.MSELoss()
cos_loss = torch.nn.CosineEmbeddingLoss()
l1 = torch.nn.L1Loss()

def get_tracking_dict():
    return {"correct_classifications": 0, "y_preds":[], "y_true":[],  "total_loss":0, "batch_losses":[], "loss_pred": 0, "loss_graph_emb": 0, "loss_layer_emb" : 0, "loss_layer_unweighted": defaultdict(lambda: 0.0), "loss_layer_weighted": defaultdict(lambda: 0.0)}

def get_graph_embs(model, device, loader, num_layer, classfication = True):
    """
    Get the embeddings of the graph the GNN generates
    """
    model.eval()

    emb_storage = EmbStorageObject(classfication, loader.dataset.num_tasks)

    for _, batch in enumerate(tqdm.tqdm(loader)):
        with torch.no_grad():
            prediction, graph_emb, node_embs = model(prepare_batch(batch, model, device))               
            emb_storage.update(batch, prediction, graph_emb, node_embs)

    return emb_storage.finalize()

def prepare_batch(batch, model, device):
    if batch.edge_index is not None:
        batch.edge_index = batch.edge_index.long()

    if batch.edge_attr is not None and len(batch.edge_attr.shape) == 1:
        batch.edge_attr = batch.edge_attr.view(-1, 1)
    batch = batch.to(device)
    return batch

def compute_loss_predictions(batch, model, metric, device, loss_fn, tracking_dict, use_graph_emb = False, use_node_emb = False, graph_emb_fac = 1, node_emb_fac = 1, use_pred_loss = True, final_mp_layers=3, do_predict_logits = False):
    batch_size = batch.y.shape[0]
    
    predictions, pred_graph_emb, pred_node_emb = model(prepare_batch(batch, model, device))

    y = batch.y

    if model.num_tasks == 1:
        y = y.view(batch_size, -1)
    else:
        y = y.view(batch_size, model.num_tasks)

    if y.shape[1] == 1 and metric == "accuracy":
        y = F.one_hot(torch.squeeze(y, 1), 10)
        
    is_labeled = y == y

    if y.dtype == torch.int64:
        y = y.float()

    losses = []

    if use_pred_loss:
        if do_predict_logits:
            logits_target = batch.logits
            if len(logits_target.shape) != len(predictions.shape):
                logits_target = torch.unsqueeze(logits_target, 1)
            pred_loss = loss_fn(predictions, logits_target)
        else:
            if metric in ['accuracy']:
                pred_loss = loss_fn(predictions, y)  
            else:
                pred_loss = loss_fn(predictions[is_labeled], y.float()[is_labeled])            
            
        tracking_dict["loss_pred"] += pred_loss.item()*batch_size
        losses.append(pred_loss)

    # print("\tOriginal loss: ", loss)

    if use_graph_emb:
        target_graph_emb = batch.graph_emb.view(batch_size, -1)
        graph_emb_loss = l2_loss(pred_graph_emb, target_graph_emb)
        # print("\tGraph emb loss: ", graph_emb_loss)
        tracking_dict["loss_graph_emb"] += graph_emb_loss.item()*batch_size
        losses.append(graph_emb_loss*graph_emb_fac)

    if use_node_emb:
        model_layers = len(pred_node_emb)
        teacher_model_layers = len(batch.node_emb)
        assert model_layers >= teacher_model_layers

        node_emb_losses = []

        for i in range(1, teacher_model_layers + 1):
            relevant_model_layer = i - 1

            if model_layers > teacher_model_layers:
                relevant_model_layer = ((model_layers-final_mp_layers)//teacher_model_layers)*(i) - 1

            target_node_emb = batch.node_emb[i-1]
            unweighted_node_loss = l2_loss(pred_node_emb[relevant_model_layer], target_node_emb)
            weighted_node_loss = torch.exp(torch.tensor([i - teacher_model_layers])).to(device)*unweighted_node_loss
            node_emb_losses.append(unweighted_node_loss)
            tracking_dict["loss_layer_unweighted"][i] += unweighted_node_loss
            tracking_dict["loss_layer_weighted"][i] += weighted_node_loss          
        
        node_emb_loss = sum(node_emb_losses) / (teacher_model_layers - 1)
        losses.append(node_emb_loss*node_emb_fac)
        tracking_dict["loss_layer_emb"] += node_emb_loss.item()*batch_size 
                
    loss = sum(losses)

    if metric == 'accuracy':
        tracking_dict["correct_classifications"] += torch.sum(predictions.argmax(dim=1)== y.argmax(dim=1)).item()

    tracking_dict["y_preds"] += predictions.cpu()
    tracking_dict["y_true"] += y.cpu()
    
    tracking_dict["batch_losses"].append(loss.item())
    tracking_dict["total_loss"] += loss.item()*batch_size
    
    del losses, batch
    return loss

def compute_final_tracking_dict(tracking_dict, output_dict, loader, metric, metric_method=None,train=False):
    output_dict["total_loss"] = float(tracking_dict["total_loss"] / len(loader.dataset))
    output_dict["loss_pred"] = float(tracking_dict["loss_pred"] / len(loader.dataset))
    output_dict["loss_graph_emb"] = float(tracking_dict["loss_graph_emb"] / len(loader.dataset))
    output_dict["loss_layer_emb"] = float(tracking_dict["loss_layer_emb"] / len(loader.dataset))

    if train:
        for (key, value) in tracking_dict["loss_layer_weighted"].items():
            output_dict[f"loss_layer_{key}_weighted"] = float(value / len(loader.dataset))
                        
        for (key, value) in tracking_dict["loss_layer_unweighted"].items():
            output_dict[f"loss_layer_{key}_unweighted"] = float(value / len(loader.dataset))
        
    if metric == 'accuracy':
        output_dict["accuracy"] = float(tracking_dict["correct_classifications"]) / len(loader.dataset)
    elif "(ogb)" in metric:
        y_preds = torch.stack(tracking_dict["y_preds"])
        y_true = torch.stack(tracking_dict["y_true"])

        if len(y_preds.shape) == 1:
            y_preds = torch.unsqueeze(y_preds, dim = 1)
            y_true = torch.unsqueeze(y_true, dim = 1)

        output_dict[metric] = float(metric_method(y_true, y_preds)[metric.replace(" (ogb)", "")])
        
    elif metric == 'mae':
        y_preds = torch.stack(tracking_dict["y_preds"])
        y_true = torch.stack(tracking_dict["y_true"])
        if "std" in tracking_dict:
            std = tracking_dict["std"]
            num_samples = y_true.shape[0]
            # The std should not actually do anything here?
            output_dict["mae"] = (((y_true * std - y_preds * std).abs() / std).sum(keepdim=True, dim = 0)/num_samples).mean().item()
        else:
            output_dict["mae"] = float(l1(y_preds, y_true))
        del y_preds, y_true
       
    del tracking_dict 
    return output_dict

def train(model, device, train_loader, optimizer, loss_fct, eval_name, use_tracking, metric_method=None, use_graph_emb = False, use_node_emb = False, graph_emb_fac = 1, node_emb_fac = 1, use_pred_loss = True, final_mp_layers=3, do_predict_logits=False):
    """
        Performs one training epoch, i.e. one optimization pass over the batches of a data loader.
    """
    model.train()

    tracking_dict = get_tracking_dict()
    if train_loader.dataset.has_mean:
        tracking_dict["mean"], tracking_dict["std"] = train_loader.dataset.mean, train_loader.dataset.std
        
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        loss = compute_loss_predictions(batch, model, eval_name, device, loss_fct, tracking_dict, use_graph_emb, use_node_emb, graph_emb_fac, node_emb_fac, use_pred_loss, final_mp_layers, do_predict_logits,)

        loss.backward()
        optimizer.step()

        # if use_tracking:
        #     wandb.log({"Train/BatchLoss": loss.item()})
        del loss
            
    return compute_final_tracking_dict(tracking_dict, {}, train_loader, eval_name, metric_method=metric_method, train=True)

def eval(model, device, loader, loss_fn, eval_name, metric_method=None):
    """
        Evaluates a model over all the batches of a data loader.
    """
    model.eval()

    tracking_dict = get_tracking_dict()
    if loader.dataset.has_mean:
        tracking_dict["mean"], tracking_dict["std"] = loader.dataset.mean, loader.dataset.std
    for step, batch in enumerate(loader):
        with torch.no_grad():
            compute_loss_predictions(batch, model, eval_name, device, loss_fn, tracking_dict)

    eval_dict = compute_final_tracking_dict(tracking_dict, {}, loader, eval_name, metric_method=metric_method)
    
    return eval_dict

def step_scheduler(scheduler, args, val_loss):
    """
        Steps the learning rate scheduler forward by one
    """
    if args.lr_scheduler in ["StepLR", "Cosine"]:
        scheduler.step()
    elif args.lr_scheduler == 'None':
        pass
    elif args.lr_scheduler == "ReduceLROnPlateau":
         scheduler.step(val_loss)
    else:
        raise NotImplementedError(f'Scheduler {args.lr_scheduler} is not currently supported.')
