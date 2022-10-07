import torch
import torch_geometric.utils as util
import torch_geometric.datasets as datasets
import tools
import torch.nn.functional as F

def preprocess(dataset_name:str='crocodile',normalization='sym',url:str='.\\',rm_selfloop=True,
                    train_rat:float=0.85,test_rat:float=0.1,neg_rat:int=1):
    #
    '''
    A function used to preprocess the **UNDIRECTED** graph data. Return feat, posidx, adj
    dataset_name:str,'Cora'|'CiteSeer'|'PubMed'|'Photo'|'chameleon'|'crocodile'|'squirrel'|'Cornell'|'Texas'|'Wisconsin'|'EN'|'DE'|'johnshopkins55'|'amherst41'
        all datasets come from torch_geometric.datasets,depend on the paper: '2018-Pitfalls of Graph Neural Network Evaluation' 
        '2016-Revisiting Semi-Supervised Learning with Graph Embeddings', '2020-Multi-scale Attributed Node Embedding' and
        '2020-Geom-GCN: Geometric Graph Convolutional Networks'
    1.Parameters:
    normlization:str,'sym'|'rw'|type(None),choose a method to process the Graph Laplacian;
    url:str,default is '.\',head url of the datasets;
    rm_selfloop:bool,whether removes the self-loop or not,default True because of link-prediction;
    train_rat:float,ratio of training edges,default 0.85;
    test_rat:float,ratio of testing edges,default 0.1;
    neg_rat:int,ratio of (negative sampling : positive edges),default 1;
    2.Returns:
    N_dataset:int,number of nodes;
    feat_dataset:tensor,nodes'feature;
    posidx_dataset:tensor,edges of undirected original graph;
    adj_dataset:tensor,adjacency of undirected original graph;
    train_adj_dataset:tensor,adjacency of undirected train graph;
    train_posidx_dataset:tensor,training edges of undirected graph;
    test_posidx_dataset:tensor,tesing edges of undirected graph;
    valid_posidx_dataset:tensor,validating edges of undirected graph,it would be type(None) if train+test=1;
    train_negidx_dataset:tensor,training edges of undirected graph;
    test_negidx_dataset:tensor,tesing edges of undirected graph;
    valid_negidx_dataset:tensor,validating edges of undirected graph,it would be type(None) if train+test=1;
    train_nlap:tensor,normalized laplacian of training set; 
    '''
    ## Download the dataset into Class(Data) and extract the data.
    if (dataset_name=='Cora')|(dataset_name=='CiteSeer')|(dataset_name=='PubMed'):
        dataset = datasets.Planetoid(root=url,name=dataset_name)
    elif (dataset_name=='Photo'):
        dataset = datasets.Amazon(root=url,name=dataset_name)
    elif dataset_name=='crocodile':
        dataset = datasets.WikipediaNetwork(root=url,name=dataset_name,geom_gcn_preprocess=False)
    elif (dataset_name=='chameleon')|(dataset_name=='squirrel'):
        dataset = datasets.WikipediaNetwork(root=url,name=dataset_name)
    elif (dataset_name=='Cornell')|(dataset_name=='Texas')|(dataset_name=='Wisconsin'):
        dataset = datasets.WebKB(root=url,name=dataset_name)
    elif (dataset_name=='amherst41')|(dataset_name=='johnshopkins55'):
        dataset = datasets.LINKXDataset(root=url,name=dataset_name)
    elif (dataset_name=='DE')|(dataset_name=='EN'):
        dataset = datasets.Twitch(root=url,name=dataset_name)
    
    ## Extract the data.
    feat_dataset = dataset.data['x']   ## Features and Nodes number.
    feat_dataset = F.normalize(feat_dataset, p=2, dim=1)    ## Normalize the feature.

    label_dataset = dataset.data['y']

    N_dataset = feat_dataset.shape[0]

    if dataset.data.is_undirected():
        ## When the graph is undirected:
        posidx_dataset = dataset.data['edge_index']
    else:
        ## When the graph is directed:
        posidx_dataset = dataset.data['edge_index']
        posidx_dataset = util.to_undirected(edge_index=posidx_dataset)

    if rm_selfloop:
        posidx_dataset = util.remove_self_loops(posidx_dataset)[0]  ## To remove the self-loop

    ## split it into train/test(or add valid) set.All of them are undirected.
    train_posidx_dataset,train_negidx_dataset,test_posidx_dataset,test_negidx_dataset,valid_posidx_dataset,valid_negidx_dataset,negidx_dataset = tools.train_test_valid_set(num_nodes=N_dataset,edge_index=posidx_dataset,train_rat=train_rat,test_rat=test_rat,neg_rat=neg_rat)

    ## Adjacency. Training set/Original set/Testing set/Validation set
    adj_dataset = torch.sparse_coo_tensor(indices=posidx_dataset,values=torch.ones_like(posidx_dataset[0,:]),size=(N_dataset,N_dataset))
    adj_dataset = adj_dataset.to_dense()
    train_adj_dataset = torch.sparse_coo_tensor(indices=train_posidx_dataset,values=torch.ones_like(train_posidx_dataset[0,:]),size=(N_dataset,N_dataset))
    train_adj_dataset = train_adj_dataset.to_dense()

    ##Graph Laplacian
    ##Normailized graph laplacian, eigenvalue is constrainted in range(0,2)
    train_nlap = util.get_laplacian(edge_index=train_posidx_dataset,normalization=normalization,num_nodes=N_dataset)
    train_nlap = torch.sparse_coo_tensor(indices=train_nlap[0],values=train_nlap[1],size=(N_dataset,N_dataset))  ##Generate a sparse matrix with index.
    train_nlap = train_nlap.to_dense()  ##pad the matrix with 0.
    
    ## Note that all the returns are type(tensor.long()). If you want to use torch.matmul 
    ##      or others functions in torch, you can trans the returns into type(tensor.float()) with [return].float()
    return N_dataset,feat_dataset,posidx_dataset,negidx_dataset,adj_dataset,train_adj_dataset,train_posidx_dataset,train_negidx_dataset,test_posidx_dataset,test_negidx_dataset,valid_posidx_dataset,valid_negidx_dataset,train_nlap,label_dataset
