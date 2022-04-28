import pickle
import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
import torch.utils.data as data
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import os
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from sklearn.model_selection import train_test_split
import torchmetrics

GNN_LAYER_BY_NAME = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv,
    "GGCN": geom_nn.GatedGraphConv,
}

BATCH_SIZE = 50

class GNNModel(nn.Module):
    
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.2, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = GNN_LAYER_BY_NAME[layer_name]
        
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            if "GGCN" not in layer_name:
                layers += [
                    gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs), nn.ReLU(inplace=True), 
                    nn.Dropout(dp_rate)
                ]
            else:
                layers += [
                    gnn_layer(out_channels=out_channels, num_layers=5, **kwargs), nn.ReLU(inplace=True), 
                    nn.Dropout(dp_rate)
                ]
            in_channels = c_hidden
        if "GGCN" not in layer_name:
            layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]
        else:
            layers += [gnn_layer(out_channels=c_out, num_layers=5, **kwargs)]
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x

class GraphGNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of output features (usually number of classes)
            dp_rate_linear - Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs - Additional arguments for the GNNModel object
        """
        super().__init__()
        self.GNN = GNNModel(c_in=c_in,
                            c_hidden=c_hidden,
                            c_out=c_hidden, # Not our prediction output yet!
                            **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out)
        )

    def forward(self, x, edge_index, batch_idx):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx - Index of batch element for each node
        """
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx) # Average pooling
        x = self.head(x)
        return x

class GraphLevelGNN(pl.LightningModule):

    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = GraphGNNModel(**model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)

        if self.hparams.c_out == 1:
            preds = (x > 0).float()
            data.y = data.y.float()
        else:
            preds = x.argmax(dim=-1)
        loss = self.loss_module(x[0], data.y[0])
        acc = torchmetrics.functional.accuracy(preds.int(), data.y.int())
        f1 = torchmetrics.functional.f1_score(preds.int(), data.y.int())
        auroc = torchmetrics.functional.auroc(preds.int(), data.y.int(), num_classes=1)
        precision = torchmetrics.functional.precision(preds.int(), data.y.int())
        return loss, acc, f1, auroc, precision

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1.3e-6)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc, f1, auroc, precision = self.forward(batch, mode="train")
        self.log('train_loss', loss, batch_size=BATCH_SIZE)
        self.log('train_acc', acc, batch_size=BATCH_SIZE)
        self.log('train_f1', f1, batch_size=BATCH_SIZE)
        self.log('train_auroc', auroc, batch_size=BATCH_SIZE)
        self.log('train_precision', precision, batch_size=BATCH_SIZE)
        #self.log(name='batch_size', value=torch.Tensor(100).to(torch.float32).to('cuda'), batch_size=torch.Tensor(100).to(torch.float32).to('cuda'))
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc, f1, auroc, precision = self.forward(batch, mode="val")
        self.log('val_acc', acc, batch_size=BATCH_SIZE)
        self.log('val_f1', f1, batch_size=BATCH_SIZE)
        self.log('val_auroc', auroc, batch_size=BATCH_SIZE)
        self.log('val_precision', precision, batch_size=BATCH_SIZE)
        #self.log(name='batch_size', value=torch.Tensor(100).to(torch.float32).to('cuda'), batch_size=torch.Tensor(100).to(torch.float32).to('cuda'))

    def test_step(self, batch, batch_idx):
        _, acc, f1, auroc, precision = self.forward(batch, mode="test")
        self.log('test_acc', acc, batch_size=BATCH_SIZE)
        self.log('test_f1', f1, batch_size=BATCH_SIZE)
        self.log('test_auroc', auroc, batch_size=BATCH_SIZE)
        self.log('val_precision', precision, batch_size=BATCH_SIZE)
        #self.log(name='batch_size', value=torch.Tensor(100).to(torch.float32).to('cuda'), batch_size=torch.Tensor(100).to(torch.float32).to('cuda'))

class GraphClassifier():

    def __init__(self, model_name, **model_kwargs):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        # Create a PyTorch Lightning trainer with the generation callback
        root_dir = os.path.join("models", "lightning_models", "GraphLevel" + model_name)
        os.makedirs(root_dir, exist_ok=True)
        self.trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(self.device).startswith("cuda") else 0,
                         max_epochs=5)
        self.trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    def run_model(self, data_list, batch_size=BATCH_SIZE):
        '''
        '''
        train, split1 = train_test_split(data_list, test_size=.25)
        test, validate = train_test_split(split1, test_size=.2)
        train_batches = DataLoader(train, batch_size=batch_size, num_workers=4)
        test_batches = DataLoader(test, batch_size=batch_size, num_workers=4)
        validate_batches = DataLoader(validate, batch_size=batch_size, num_workers=4)
        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = os.path.join("lightning_models", f"GraphLevel{self.model_name}.ckpt")
        if os.path.isfile(pretrained_filename):
            print("Found pretrained model, loading...")
            model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
        else:
            pl.seed_everything(42)
        model = GraphLevelGNN(c_in=100,
                              c_out=1,
                              **self.model_kwargs)
        self.trainer.fit(model, train_batches, validate_batches)
        model = GraphLevelGNN.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path)
        # Test best model on validation and test set
        train_result = self.trainer.test(model, dataloaders=validate_batches, verbose=True)
        test_result = self.trainer.test(model, dataloaders=test_batches, verbose=True)
        #result = {"test": test_result[0]['test_acc'], "train": train_result[0]['test_acc']}
        return model, train_result, test_result