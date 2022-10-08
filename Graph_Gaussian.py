import torch
import data_preprocessing as data_preprocess
import Graph_Gaussian_Models as Model

#torch.manual_seed(12346)
## Cuda selection
use_gpu = True

## Use Cuda or not
device = torch.device('cuda:0' if (use_gpu & torch.cuda.is_available()) else 'cpu')

num_node,feat,posidx,negidx,adj,train_adj,train_posidx,train_negidx,test_posidx,test_negidx,valid_posidx,\
    valid_negidx,train_nlap,label = data_preprocess.preprocess(dataset_name='Texas',\
    neg_rat=1,train_rat=0.89,test_rat=0.1)
print("The edge rates of the dataset used now is: ")
print(posidx.shape[1]/(num_node**2-num_node))

## Move to gpu
feat,posidx,adj,train_adj,train_posidx,test_posidx,valid_posidx,train_nlap = feat.to(device),\
    posidx.to(device),adj.to(device),train_adj.to(device),train_posidx.to(device),test_posidx.to(device),\
        valid_posidx.to(device),train_nlap.to(device)

model = Model.EdgeGCGP(feat,3)
model.to(device)

print('Now start training')

## Original Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    if epoch%1==0:
        with torch.no_grad():
            model.eval()
            radj = model(train_posidx)
            print("Train result: " , 100 * Model.test(radj,train_posidx,posidx)[0])
            print("Test result: " , 100 * Model.test(radj,test_posidx,posidx)[0])
            print("Epoch: " , epoch)
        
    model.train()
    
    radj = model(train_posidx)
    radj = radj.to(device)

    ## Recontruction loss
    loss = Model.recon_loss(radj,train_posidx)

    print("recon_loss:")
    print(loss)

    loss.backward()

    optimizer.step()


## The model.eval() is necessary,because of the torch.nn.module
print('Now start testing')
    
## Note that only in testing do we need to fixed the edges.
with torch.no_grad():
    model.eval()
    radj = model(train_posidx)
    print("Train result: " , 100 * Model.test(radj,train_posidx,posidx)[0])
    print("Test result: " , 100 * Model.test(radj,test_posidx,posidx)[0])