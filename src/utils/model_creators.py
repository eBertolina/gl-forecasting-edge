import json
from pathlib import Path
import keras
from keras.layers import Dropout
from laplace import Laplace
import pandas as pd
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import numpy as np


from torchsummary import summary
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from gossiplearning.config import Config
from utils.metrics import compute_metrics, Metrics
from laplace.curvature import CurvlinopsGGN, AsdlGGN


@keras.saving.register_keras_serializable()
def create_LSTM(config: Config) -> Model:
    optz = Adam(learning_rate=0.001, epsilon=1e-6)

    input_timesteps = 4

    inputs = Input(shape=(input_timesteps, config.training.n_input_features))

    lstm_layers = Sequential(
        [
            LSTM(
                50,
                activation="tanh",
                return_sequences=True,
            ),
            LSTM(
                50,
                activation="tanh",
                return_sequences=False,
            ),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
        ]
    )(inputs)

    outputs = [
        Dense(1, activation="relu", name=f"fn_{i}")(lstm_layers)
        for i in range(config.training.n_output_vars)
    ]

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optz,
        loss={f"fn_{i}": "mse" for i in range(config.training.n_output_vars)},
        metrics=["mae", "msle", "mse", "mape", RootMeanSquaredError()],
    )

    return model

def create_MLP(config):
    class MLPModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            #input_timesteps = 4
            #input_dim = (input_timesteps * config.training.n_input_features) 
            input_dim = config.training.n_input_features  #17 Ton_Iot dim #5 o 9 SE-CIC-IDS dim

            self.fc1 = nn.Linear(input_dim, 128)
            #self.dropout1 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(128, 64)
            #self.dropout2 = nn.Dropout(0.3)

            self.output_layer = nn.Linear(64, config.training.n_output_vars)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            #x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            #x = self.dropout2(x)
            logit = self.output_layer(x)  
            return logit
        
        def train_centr(self, train_data, validation_data, batch_size, shuffle, epochs, path):
            X_train = torch.tensor(train_data[0].astype("float32"), dtype=torch.float32)
            Y_train = torch.tensor(train_data[1], dtype=torch.long)
            X_val = torch.tensor(validation_data[0].astype("float32"), dtype=torch.float32)
            Y_val = torch.tensor(validation_data[1], dtype=torch.long)

            train_loader = torch.utils.data.DataLoader(
                NetworkDataset(X_train, Y_train),
                batch_size=batch_size,
                shuffle=shuffle,
            )
            val_loader = torch.utils.data.DataLoader(
                NetworkDataset(X_val, Y_val),
                batch_size=batch_size,
                shuffle=False,
            )

            model = self
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            optimizer = optim.Adam(model.parameters(), lr=0.00001)
            train_loss_list = []
            val_loss_list = []
            all_preds = []
            all_labels = []
            y_pred_list = []
            best_val_loss = float("inf")
            patience_counter = 0
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                for X_batch, Y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, Y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * X_batch.size(0)

                train_loss /= len(train_loader.dataset)
                train_loss_list.append(train_loss)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, Y_batch in val_loader:
                        outputs = model(X_batch)
                        preds = torch.argmax(outputs, dim=1)
                        #Y_batch = Y_batch.squeeze(1).long()
                        loss = criterion(outputs, Y_batch)
                        val_loss += loss.item() * X_batch.size(0)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(Y_batch.cpu().numpy())
                        y_pred_list.append(preds)

                #y_pred1 = torch.cat(y_pred_list).numpy()
                val_loss /= len(val_loader.dataset)
                val_loss_list.append(val_loss)
                print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

                # checkpoint
                if val_loss < best_val_loss - config.training.min_delta:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), path / "centralized.pt")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.training.patience:
                        print("Early stopping")
                        break
        
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            target_names = [str(c) for c in np.unique(y_true)]
            pred_names = [str(c) for c in np.unique(y_pred)]
            report = classification_report(y_true, 
                                        y_pred, 
                                        target_names=target_names, zero_division=0,
                                        output_dict=True
                                        )

            cm = confusion_matrix(y_true, y_pred,)

            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=target_names, yticklabels=target_names)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            # plt.show()

            plt.savefig(path/ f"confusion_matrix_{epochs}_epochs.png")

            metrics = compute_metrics(y_true, y_pred)

            history = {
                "loss": train_loss_list,
                "val_loss": val_loss_list, 
                "accuracy": metrics.acc,
                "precision_macro": metrics.prec,
                "recall_macro": metrics.rec,
                "f1_macro": metrics.f1,
                "f1_weighted": metrics.f1_weighted
            }

            with open(path/"history.json", "w") as outfile:
                json.dump(history, outfile, indent=3)
                json.dump(report, outfile, indent=3)
            outfile.close

            return history
        
        def train_one_epoch(self, data, batch_size, 
                            shuffle, epoch, id, path, 
                            current_update, fisher):
            X_train = torch.tensor(data["X_train"].astype('float32'), dtype=torch.float32)
            Y_train = torch.tensor(data["Y_train"], dtype=torch.long)
            X_val = torch.tensor(data["X_val"].astype('float32'), dtype=torch.float32)
            Y_val = torch.tensor(data["Y_val"], dtype=torch.long)

            train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
            val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)


            train_loader = torch.utils.data.DataLoader(
                NetworkDataset(X_train, Y_train),
                batch_size=batch_size,
                shuffle=shuffle,
            )
            val_loader = torch.utils.data.DataLoader(
                NetworkDataset(X_val, Y_val),
                batch_size=batch_size,
                shuffle=False,
            )

            model = self
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            #criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))
            optimizer = optim.Adam(model.parameters(), lr=0.001)  #0.001 #0.0000001  #0.00001
            fisher_diag = torch.tensor(fisher)
            theta_star = torch.cat([p.detach().flatten() for p in model.parameters()])
            λ = 0.01

            model.train()
            train_loss = 0.0
            for X_batch, Y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                if len(fisher) != 0:
                    #theta = torch.cat([p.flatten() for p in model.parameters()])
                    #reg = (torch.sum(fisher_diag * (theta - theta_star)**2))/batch_size
                    loss = loss_fn(outputs, Y_batch) #+ λ * reg
                    #print(f"Node {id} Loss: ", loss)
                    #print(f"Node {id} Reg: ", reg)
                else:
                    loss = loss_fn(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation
            all_preds = []
            all_labels = []
            y_pred_list = []

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    outputs = model(X_batch)
                    preds = torch.argmax(outputs, dim=1)
                    #Y_batch = Y_batch.squeeze(1).long()
                    loss = loss_fn(outputs, Y_batch)
                    val_loss += loss.item() * X_batch.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(Y_batch.cpu().numpy())
                    y_pred_list.append(preds)

            #y_pred1 = torch.cat(y_pred_list).numpy()
            val_loss /= len(val_loader.dataset)

            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            target_names = [str(c) for c in np.unique(y_true)]
            pred_names = [str(c) for c in np.unique(y_pred)]
            report = classification_report(y_true, 
                                        y_pred, 
                                        labels=range(7), #8
                                        target_names=target_names, zero_division=0,
                                        output_dict=True
                                        )
            #print(report)
            #print("\nClassification Report:\n")

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred,
                                  labels=range(7)
                                )

            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=range(7), yticklabels=range(7))
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            # plt.show()
            
            epoch_plots_folder = Path(path) / "plots" / f"node_{id}" / f"update_{current_update}"
            epoch_plots_folder.mkdir(parents=True, exist_ok=True)
            plt.savefig(epoch_plots_folder/ f"plot_epoch_{epoch}.png")
            plt.close()

            #df = pd.DataFrame(report)
            #df.to_json(epoch_plots_folder/ f"report_epoch_{epoch}.json", indent=3)
            with open(epoch_plots_folder/ f"report_epoch_{epoch}.json", "w") as outfile:
                json.dump(report, outfile, indent=3)
            outfile.close

            metrics = compute_metrics(y_true, y_pred)
            #print("Event metrics from training one epoch \n", metrics)
            #self.eval_metrics.append(metrics)

            history = {
                "loss": [train_loss],
                "val_loss": [val_loss],
                "accuracy": metrics.acc,
                "precision_macro": metrics.prec,
                "recall_macro": metrics.rec,
                "f1_macro": metrics.f1,
                "f1_weighted": metrics.f1_weighted
            }
            return metrics, history
        

        def train_single_node(self, train_data, validation_data, batch_size, shuffle, epochs, path, id):
            X_train = torch.tensor(train_data[0].astype("float32"), dtype=torch.float32)
            Y_train = torch.tensor(train_data[1], dtype=torch.long)
            X_val = torch.tensor(validation_data[0].astype("float32"), dtype=torch.float32)
            Y_val = torch.tensor(validation_data[1], dtype=torch.long)

            train_loader = torch.utils.data.DataLoader(
                NetworkDataset(X_train, Y_train),
                batch_size=batch_size,
                shuffle=shuffle,
            )
            val_loader = torch.utils.data.DataLoader(
                NetworkDataset(X_val, Y_val),
                batch_size=batch_size,
                shuffle=False,
            )

            model = self
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            optimizer = optim.Adam(model.parameters(), lr=0.00001)
            train_loss_list = []
            val_loss_list = []
            all_preds = []
            all_labels = []
            y_pred_list = []
            best_val_loss = float("inf")
            patience_counter = 0
            extended_path = Path(path / f"node_{id}")
            extended_path.mkdir(parents=True, exist_ok=True)

            print("Starting training node ", id)
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                for X_batch, Y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, Y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * X_batch.size(0)

                train_loss /= len(train_loader.dataset)
                train_loss_list.append(train_loss)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, Y_batch in val_loader:
                        outputs = model(X_batch)
                        preds = torch.argmax(outputs, dim=1)
                        loss = criterion(outputs, Y_batch)
                        val_loss += loss.item() * X_batch.size(0)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(Y_batch.cpu().numpy())
                        y_pred_list.append(preds)

                val_loss /= len(val_loader.dataset)
                val_loss_list.append(val_loss)
                print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

                # checkpoint
                if val_loss < best_val_loss - config.training.min_delta:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), extended_path / f"single_node_{id}.pt")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.training.patience:
                        print("Early stopping")
                        break
        
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            target_names = [str(c) for c in np.unique(y_true)]
            pred_names = [str(c) for c in np.unique(y_pred)]
            report = classification_report(y_true, 
                                        y_pred, 
                                        target_names=target_names, zero_division=0,
                                        output_dict=True
                                        )

            cm = confusion_matrix(y_true, y_pred,)

            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=target_names, yticklabels=target_names)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            # plt.show()

            plt.savefig(extended_path / f"confusion_matrix_{epochs}_epochs.png")
            plt.close()

            metrics = compute_metrics(y_true, y_pred)

            history = {
                "loss": train_loss_list,
                "val_loss": val_loss_list, 
                "accuracy": metrics.acc,
                "precision_macro": metrics.prec,
                "recall_macro": metrics.rec,
                "f1_macro": metrics.f1,
                "f1_weighted": metrics.f1_weighted
            }

            with open(extended_path / "history.json", "w") as outfile:
                json.dump(history, outfile, indent=3)
                json.dump(report, outfile, indent=3)
            outfile.close

            return history
        
        def compute_fisher_diag(self, data):

            """X = data["X_train"]
            Y = data["Y_train"]
            idx = np.random.choice(len(X), size=256, replace=False)
            subset_X, subset_Y = [], []

            for i in idx:
                subset_X.append(X[i])
                subset_Y.append(Y[i])

            subset_X = np.stack(subset_X)
            subset_Y = np.array(subset_Y)

            X_train = torch.tensor(subset_X.astype("float32"), dtype=torch.float32)
            Y_train = torch.tensor(subset_Y, dtype=torch.long)"""

            X_train = torch.tensor(data["X_train"].astype("float32"), dtype=torch.float32)
            Y_train = torch.tensor(data["Y_train"], dtype=torch.long)
            
            train_loader = torch.utils.data.DataLoader(
                NetworkDataset(X_train, Y_train),
                batch_size=512,
                shuffle=False,
            )

            self.eval()
            la = Laplace(
                self,
                likelihood='classification',
                subset_of_weights='all',
                hessian_structure='diag',
                backend=AsdlGGN,
                prior_precision=0.1
            )
            la.fit(train_loader, 
                   progress_bar=True
                   )
            #print("posterior: ", la.posterior_precision)
            #return la.H.diag().detach().cpu().numpy()
            return la.posterior_precision

    model = MLPModel(config)

    #optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-6)

    return model
    #return model, optimizer

class NetworkDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
