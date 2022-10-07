import torch
import torch_geometric.utils as util
from torch_geometric.nn import APPNP
from sklearn.metrics import average_precision_score, roc_auc_score

class GCGP(torch.nn.Module):
    def __init__(self, x_train, hops):
        super(GCGP, self).__init__()
        self.graphconv1 = APPNP(hops,0.5)
        self.graphconv2 = APPNP(hops,0.5)
        self.x_train = x_train
        self.kernel_scale = torch.nn.Linear(x_train.shape[1],1,False)
        self.kernel = self.RBF_kernel

    def RBF_kernel(self,input):
        var = torch.zeros((input.shape[0],input.shape[0]))

        for i in range(0,input.shape[0]-1):
            for j in range(i+1,input.shape[0]):
                var[i,j] = self.kernel_scale((input[i,:]-input[j,:]) * (input[i,:]-input[j,:]))
        
        var = var + var.T + torch.eye(input.shape[0])

        return var

    def forward(self,edge_index):
        kff = self.kernel(self.x_train)

        kgg = self.graphconv1(kff,edge_index)
        kgg = self.graphconv1(kgg.T,edge_index)

        adj = torch.sparse_coo_tensor(indices=edge_index,values=torch.ones_like(edge_index[0,:]),\
            size=(self.x_train.shape[0],self.x_train.shape[0]))
        adj = adj.to_dense()
        diag = (torch.diag(adj.sum(dim=1)) + 1)**(-0.5)

        kgg = diag * kgg
        kgg = diag * (kgg.T)

        return kgg

class EdgeGCGP(torch.nn.Module):
    def __init__(self, x_train, hops):
        super(EdgeGCGP, self).__init__()
        self.GCGP = GCGP(x_train, hops)
    
    def forward(self,edge_index):
        kgg = self.GCGP(edge_index)

        diag = torch.diag(kgg).reshape((1,kgg.shape[0]))
        
        krr = torch.matmul(diag.T,diag)
        krr += kgg**2

        radj = (krr - 0.5).sigmoid()

        return radj

def recon_loss(radj,train_index):
    neg_train_index = util.negative_sampling(edge_index=train_index, num_nodes=radj.size(0),\
        num_neg_samples=2*train_index.shape[1],force_undirected=True)
    
    pos_loss = -torch.log(1e-15 + radj[train_index[0],train_index[1]]).mean()

    neg_loss = -torch.log(1-radj[neg_train_index[0],neg_train_index[1]] + 1e-15).mean()

    return pos_loss + neg_loss

def test(radj,test_index,edge_index):
    neg_test_index = util.negative_sampling(edge_index=edge_index, num_nodes=radj.size(0),\
        num_neg_samples=2*test_index.shape[1],force_undirected=True)
    
    pos_y = torch.ones(test_index.shape[1])
    neg_y = torch.zeros(neg_test_index.shape[1])
    y = torch.cat([pos_y,neg_y],dim=0)

    pos_pred = radj[test_index[0],test_index[1]]
    neg_pred = radj[neg_test_index[0],neg_test_index[1]]
    pred = torch.cat([pos_pred,neg_pred],dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return roc_auc_score(y, pred), average_precision_score(y, pred) 
