

def get_classes_tasks_feats(dataset, dataset_name):
    num_classes, num_vertex_features = dataset.num_classes, dataset.num_node_features    
    if dataset_name.lower() == "zinc" or "ogb" in dataset_name.lower():
        num_classes = 1
    try:
        num_tasks = dataset.num_tasks
    except:
        num_tasks = 1
    return num_classes, num_tasks, num_vertex_features