"""
@Author: Siri Leane Rüegg
@Contact: sirrueeg@ethz.ch
@File: ConvNets.py
"""

import pytorch_lightning as pl

import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet
import torchvision.models as models

import numpy as np
import plotly.graph_objs as go
from pathlib import Path
import pandas as pd
import json
import yaml
import os


class ConvNets(pl.LightningModule):
    def __init__(self, outputNr: int = 30, run_name: str = "default", loss_func: torch.nn.Module = nn.MSELoss(),
                 pretrained: bool = False, ckpt_path: str = '', pkl_path: str = "originalDataset.pkl", network: str = "inception", only_last_layer=False):
        """
        Args:
          outputNr: how many dimensions are we predicting (could infer from DataModule)
          run_name: name that summarizes the settings of the run
          loss_func: define desired loss function
          pretrained: bool for using pretrained model
          ckpt_path: path of .ckpt or .pth file for using pretrained weights
          pkl_path: path to pickle file that defines directories of back scans
          network: choose network {resnext, densenet, inception}
        """
        super().__init__()
        self.save_hyperparameters(ignore=['pkl_path', 'only_last_layer'])
        self.output_nr = outputNr
        self.loss_func = loss_func
        self.run_name = run_name
        self.pretrained = pretrained
        self.ckpt_path = ckpt_path
        self.pkl_path = pkl_path
        self.only_last_layer = only_last_layer
        self.network = network
        self.model = self.get_model(network=self.network)

        assert os.path.exists(pkl_path), f"File '{pkl_path}' does not exist."

        # metric functions
        self.pairwise_distance = nn.PairwiseDistance()
        self.mae = nn.L1Loss()

        # metric values
        self.test_meanL2norm = []
        self.test_x_mae = []
        self.test_z_mae = []
        self.test_mae = []
        self.test_rmse = []

        # for pytorch lightning's summary
        self.example_input_array = torch.Tensor(1, 1, 640, 480)

    def on_train_start(self):
        self.hparams["optimizer"] = self.trainer.optimizers[0].__class__.__name__
        self.hparams["learning_rate"] = self.trainer.optimizers[0].param_groups[0]['lr']
        self.hparams["batch_size"] = self.trainer.datamodule.__dict__['batch_size']
        self.hparams["train_per"] = self.trainer.datamodule.__dict__['train_per']
        self.hparams["test_per"] = self.trainer.datamodule.__dict__['test_per']
        self.logger.log_hyperparams(self.hparams,
                                    {"hp/L2": 0, "hp/L2_trim": 0, "hp/MAE": 0, "hp/RMSE": 0, "hp/MAE_x": 0,
                                     "hp/MAE_z": 0})

    def get_model(self, network: str = "resnext"):
        if network == "resnext":
            model = self.resnext50()

        elif network == "densenet":
            model = self.densenet()

        elif network == "inception":
            model = self.inception()

        else:
            print('No valid network chosen. Choose between "resnext", "densenet" and "inception"')

        return model

    def configure_optimizers(self):
        if self.only_last_layer:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
            return optimizer

    def resnext50(self):
        # ResNext50 Model with 1-channel as input and self.output_nr outputs
        kwargs = {"groups": 32, "width_per_group": 4}
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

        if self.pretrained and self.ckpt_path.endswith('.pth'):
            print("Using pre-trained model")
            # Load pre-trained weights from the saved file
            pretrained_weights = torch.load(self.ckpt_path)
            model.load_state_dict(pretrained_weights)

        # changing from 3-channel input layer to 1-channel input layer
        new = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        new.weight = nn.Parameter(model.conv1.weight[:, 0:1, :, :])
        model.conv1 = new

        # changing the number of outputs in output layer
        model.fc = nn.Linear(model.fc.in_features, self.output_nr)

        if self.pretrained and self.ckpt_path.endswith('.ckpt'):
            print("Using pre-trained model")

            # Load pre-trained weights from the .ckpt file
            pretrained_state_dict = torch.load(self.ckpt_path)['state_dict']

            # Remove the "model." prefix from the keys in the state_dict
            pretrained_state_dict = {k.replace("model.", ""): v for k, v in pretrained_state_dict.items()}

            # Load the state_dict into the model
            model.load_state_dict(pretrained_state_dict)

        if self.only_last_layer:
            # Freeze pretrained layers
            for param in model.parameters():
                param.requires_grad = False

            # Enable gradient computation for the last layer
            for param in model.fc.parameters():
                param.requires_grad = True

        return model

    def densenet(self):
        # DenseNet121 Model with 1-channel as input and self.output_nr outputs
        model = models.densenet121()

        if self.pretrained and self.ckpt_path.endswith('.pth'):
            print("Using pre-trained model")
            # Load pre-trained weights from the saved file
            pretrained_weights = torch.load(self.ckpt_path)
            model.load_state_dict(pretrained_weights)

        # Changing from 3-channel input layer to 1-channel input layer
        new_conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        new_conv.weight = nn.Parameter(model.features.conv0.weight[:, 0:1, :, :])
        model.features.conv0 = new_conv

        # Changing the number of outputs in the final linear layer
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, self.output_nr)

        if self.pretrained and self.ckpt_path.endswith('.ckpt'):
            print("Using pre-trained model")

            # Load pre-trained weights from the .ckpt file
            pretrained_state_dict = torch.load(self.ckpt_path)['state_dict']

            # Remove the "model." prefix from the keys in the state_dict
            pretrained_state_dict = {k.replace("model.", ""): v for k, v in pretrained_state_dict.items()}

            # Load the state_dict into the model
            model.load_state_dict(pretrained_state_dict)


        if self.only_last_layer:
            # Freeze pretrained layers
            for param in model.parameters():
                param.requires_grad = False

            # Enable gradient computation for the last layer
            for param in model.classifier.parameters():
                param.requires_grad = True

        return model

    def inception(self):
        # InceptionV3 Model with 1-channel as input and self.output_nr outputs
        model = models.inception_v3()

        if self.pretrained and self.ckpt_path.endswith('.pth'):
            print("Using pre-trained model")
            # Load pre-trained weights from the saved file
            pretrained_weights = torch.load(self.ckpt_path)
            model.load_state_dict(pretrained_weights)

            model.transform_input = False  # can maybe be deleted

        # Modify the input stem to accept 1-channel input
        model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Adjusting the number of outputs in auxiliary classifier
        model.AuxLogits.fc = nn.Linear(768, self.output_nr)

        # Changing the number of outputs in the final fully connected layer
        model.fc = nn.Linear(2048, self.output_nr)

        if self.pretrained and self.ckpt_path.endswith('.ckpt'):
            print("Using pre-trained model")

            # Load pre-trained weights from the .ckpt file
            pretrained_state_dict = torch.load(self.ckpt_path)['state_dict']

            # Remove the "model." prefix from the keys in the state_dict
            pretrained_state_dict = {k.replace("model.", ""): v for k, v in pretrained_state_dict.items()}

            # Load the state_dict into the model
            model.load_state_dict(pretrained_state_dict)

        if self.only_last_layer:
            # Freeze pretrained layers
            for param in model.parameters():
                param.requires_grad = False

            # Enable gradient computation for the last layer
            for param in model.fc.parameters():
                param.requires_grad = True

        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets, path_id = batch
        batch_size = images.size(0)

        # Forward pass
        if self.network == "inception":
            outputs, _ = self.model(images)  # Access the primary output using self.model.fc
        else:
            outputs = self.model(images)
        loss = self.loss_func(outputs, targets)

        reshaped_outputs = outputs.view(2, int(batch_size * self.output_nr / 2)).t()
        reshaped_targets = targets.view(2, int(batch_size * self.output_nr / 2)).t()
        meanl2norm = torch.mean(self.pairwise_distance(reshaped_outputs, reshaped_targets))  # Mean Euclidian Distance

        # Log loss to Tensorboard
        self.log('loss/train', loss, batch_size=batch_size)
        self.log('acc/train', meanl2norm, batch_size=batch_size)
        self.log("batch_idx", float(batch_idx), batch_size=batch_size)
        # self.logger.experiment.add_scalars('loss', {'train_loss': loss}, self.current_epoch)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, targets, path_id = batch
        batch_size = images.size(0)

        # Forward pass
        outputs = self.model(images)
        loss = self.loss_func(outputs, targets)

        reshaped_outputs = outputs.view(2, int(batch_size * self.output_nr / 2)).t()
        reshaped_targets = targets.view(2, int(batch_size * self.output_nr / 2)).t()
        meanl2norm = torch.mean(self.pairwise_distance(reshaped_outputs, reshaped_targets))  # Mean Euclidian Distance

        # Log loss to Tensorboard
        self.log('loss/val', loss, batch_size=batch_size)
        self.log('acc/val', meanl2norm, batch_size=batch_size)
        self.log("batch_idx", float(batch_idx), batch_size=batch_size)
        # self.logger.experiment.add_scalars('loss', {'val_loss': loss}, self.current_epoch)

    def test_step(self, batch, batch_idx):
        images, targets, path_id = batch
        batch_size = images.size(0)
        path_id = path_id[0]
        num_points = int(batch_size * self.output_nr / 2)

        # Forward pass
        outputs = self.model(images)
        loss = self.loss_func(outputs, targets)

        df_cleaned = pd.read_pickle(self.pkl_path).reset_index(drop=True)
        df_cleaned['Subject_ID'] = df_cleaned['Subject_ID'].astype(str)  # maybe delete with new pickle file
        scaling_factor = df_cleaned.loc[df_cleaned['Subject_ID'] == path_id, 'scaling_factor'].iloc[0]

        outputs = outputs / scaling_factor
        targets = targets / scaling_factor

        reshaped_outputs = outputs.view(2, int(batch_size * self.output_nr / 2)).t()
        reshaped_targets = targets.view(2, int(batch_size * self.output_nr / 2)).t()

        meanl2norm = torch.mean(self.pairwise_distance(reshaped_outputs, reshaped_targets))  # Mean Euclidian Distance
        mae = self.mae(reshaped_outputs, reshaped_targets)
        x_mae = torch.mean(abs(outputs[:, :num_points] - targets[:, :num_points]))  # MAE in x-direction
        z_mae = torch.mean(abs(outputs[:, num_points:] - targets[:, num_points:]))  # MAE in x-direction
        rmse = torch.sqrt(nn.functional.mse_loss(reshaped_outputs, reshaped_targets))
        self.test_meanL2norm.append(meanl2norm.item())
        self.test_mae.append(mae.item())
        self.test_x_mae.append(x_mae.item())
        self.test_z_mae.append(z_mae.item())
        self.test_rmse.append(rmse.item())

        # Log loss to Tensorboard
        self.log('loss/test', loss, batch_size=batch_size)
        self.log('acc/test', meanl2norm, batch_size=batch_size)
        self.log("batch_idx", float(batch_idx), batch_size=batch_size)

        targets_np = targets.cpu().numpy()
        outputs_np = outputs.cpu().numpy()

        path_back = df_cleaned.loc[df_cleaned['Subject_ID'] == path_id, 'back_pp'].iloc[0]
        path_fixPoints = df_cleaned.loc[df_cleaned['Subject_ID'] == path_id, 'fixPoints_pp'].iloc[0]

        if "original1024" in self.pkl_path:
            path_back = path_back.replace("./preprocessed", "./data/preprocessed_original1024")
            path_fixPoints = path_fixPoints.replace("./preprocessed", "./data/preprocessed_original1024")

        elif "original2048" in self.pkl_path:
            path_back = path_back.replace("./preprocessed", "./data/preprocessed_original2048")
            path_fixPoints = path_fixPoints.replace("./preprocessed", "./data/preprocessed_original2048")

        elif "original4096" in self.pkl_path:
            path_back = path_back.replace("./preprocessed", "./data/preprocessed_original4096")
            path_fixPoints = path_fixPoints.replace("./preprocessed", "./data/preprocessed_original4096")

        elif "original8192" in self.pkl_path:
            path_back = path_back.replace("./preprocessed", "./data/preprocessed_original8192")
            path_fixPoints = path_fixPoints.replace("./preprocessed", "./data/preprocessed_original8192")

        elif "augmented" in self.pkl_path:
            path_back = path_back.replace("./preprocessed", "./preprocessed_original")
            path_fixPoints = path_fixPoints.replace("./preprocessed", "./preprocessed_original")

        elif "balanced" in self.pkl_path:
            path_back = path_back.replace("./preprocessed", "./preprocessed_balanced")
            path_fixPoints = path_fixPoints.replace("./preprocessed", "./preprocessed_balanced")

        elif "italian" in self.pkl_path:
            path_back = path_back.replace("./preprocessed", "./data/preprocessed_italian8192")
            path_fixPoints = path_fixPoints.replace("./preprocessed", "./data/preprocessed_italian8192")

        elif "noArms" in self.pkl_path:
            path_back = path_back.replace("./preprocessed", "./preprocessed_noArms")
            path_fixPoints = path_fixPoints.replace("./preprocessed", "./preprocessed_noArms")

        else:
            ValueError("Invalid .pkl file")

        back_np = np.load(path_back)
        with open(path_fixPoints, "r") as f:
            fixPoints_dict = json.load(f)

        fixPoints_np = np.array(list(fixPoints_dict.values()))

        # scale back to original size: back, fixPoints
        corrected_back = back_np / scaling_factor
        corrected_fixPoints_np = fixPoints_np / scaling_factor
        corrected_fixPoints = {}

        for key, value in fixPoints_dict.items():
                corrected_fixPoints[key] = np.asarray(value) / scaling_factor

        # create two arrays for the prediction and ground-truth spine
        array1 = np.zeros((num_points, 3))
        array2 = np.zeros((num_points, 3))

        # fill the new array with data
        array1[:, 0] = targets_np[0, :num_points]
        array1[:, 1] = np.linspace(np.asarray(corrected_fixPoints['fix_C7'][1]), np.asarray(corrected_fixPoints['fix_DM'][1]), num_points)
        array1[:, 2] = targets_np[0, num_points:]
        array2[:, 0] = outputs_np[0, :num_points]
        array2[:, 1] = np.linspace(np.asarray(corrected_fixPoints['fix_C7'][1]), np.asarray(corrected_fixPoints['fix_DM'][1]), num_points)
        array2[:, 2] = outputs_np[0, num_points:]

        distances = np.sqrt(np.sum((array2 - array1) ** 2, axis=1))

        # create the traces
        lines = []
        for i in range(num_points):
            lines.append(
                go.Scatter3d(
                    x=[array1[i, 0], array2[i, 0]],
                    y=[array1[i, 1], array2[i, 1]],
                    z=[array1[i, 2], array2[i, 2]],
                    mode='lines',
                    line=dict(color='blue'),
                    showlegend=False,
                    text=['Distance: {:.2f}'.format(distances[i]),
                          'Distance: {:.2f}'.format(distances[i])],
                )
            )

        red_line = go.Scatter3d(
            x=array1[:, 0],
            y=array1[:, 1],
            z=array1[:, 2],
            mode='lines',
            line=dict(color='red'),
            showlegend=False,
        )

        green_line = go.Scatter3d(
            x=array2[:, 0],
            y=array2[:, 1],
            z=array2[:, 2],
            mode='lines',
            line=dict(color='green'),
            showlegend=False,
        )

        red_dots = go.Scatter3d(
            x=array1[:, 0],
            y=array1[:, 1],
            z=array1[:, 2],
            name='ground-truth',
            mode='markers',
            marker=dict(color='red', size=4),
        )

        green_dots = go.Scatter3d(
            x=array2[:, 0],
            y=array2[:, 1],
            z=array2[:, 2],
            name='prediction',
            mode='markers',
            marker=dict(color='green', size=4),
        )

        back = go.Scatter3d(
            x=corrected_back[:, 0],
            y=corrected_back[:, 1],
            z=corrected_back[:, 2],
            mode='markers',
            name='back',
            marker=dict(
                size=1,
                color='blue',
                opacity=0.5,
            )
        )

        # create the layout
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X [mm]'),
                yaxis=dict(title='Y [mm]'),
                zaxis=dict(title='Z [mm]'),
                aspectmode='data',
            ),

            updatemenus=[
                dict(
                    type='buttons',
                    showactive=True,
                    buttons=[
                        dict(
                            label='View X-Y Plane',
                            method='relayout',
                            args=[{
                                'scene.camera.up': {'x': 0, 'y': 1, 'z': 0},
                                'scene.camera.eye': {'x': 0, 'y': 0, 'z': 3},
                            }]
                        ),
                        dict(
                            label='View Y-Z Plane',
                            method='relayout',
                            args=[{
                                'scene.camera.up': {'x': 0, 'y': 1, 'z': 0},
                                'scene.camera.eye': {'x': 3, 'y': 0, 'z': 0},
                            }]
                        ),
                    ]
                )
            ],
            showlegend=True,
            title=f"<span style='font-size: 16px;'>ISL Comparison - Subject ID: {path_id} </span><br>"
                  f"<span style='font-size: 14px;'>Mean Pairwise Distance/Mean L2-Norm: {meanl2norm:.2f} [mm], MAE: {mae:.2f} [mm], RMSE: {rmse:.2f} [mm] </span><br>"
                  f"<span style='font-size: 14px;'>MAE in x-direction: {x_mae:.2f} [mm], MAE in z-direction: {z_mae:.2f} [mm]</span>",
        )

        # plot the figure
        fig = go.Figure(data=lines + [red_line, green_line, red_dots, green_dots, back], layout=layout)

        # create the folder if it does not exist
        folder_path = Path(f'./logs/{self.logger.name}/version_{str(self.logger.version)}/plotlys')
        if not folder_path.exists():
            folder_path.mkdir(parents=True)

        # if meanl2norm <= min(self.test_meanL2norm) or meanl2norm >= max(self.test_meanL2norm):
        fig.write_html(f"./logs/{self.logger.name}/version_{str(self.logger.version)}/plotlys/subject_{path_id}_acc_{meanl2norm:.2f}.html")

    def trimmed_mean(self, data, trim_percentage=5):
        # Sort the data in ascending order
        sorted_data = sorted(data)

        # Calculate the number of elements to remove from each end
        num_elements_to_remove = int(len(sorted_data) * trim_percentage / 100)

        if num_elements_to_remove == 0:
            num_elements_to_remove = 1

        # Remove the specified number of elements from both ends
        trimmed_data = sorted_data[num_elements_to_remove:-num_elements_to_remove]

        # Calculate the mean of the remaining values
        mean = sum(trimmed_data) / len(trimmed_data)
        return mean

    def on_test_epoch_end(self):
        # Calculate mean values
        mean_dist = round(sum(self.test_meanL2norm) / len(self.test_meanL2norm), 2)
        trimmed_dist = round(self.trimmed_mean(self.test_meanL2norm), 2)
        mean_mae = round(sum(self.test_mae) / len(self.test_mae), 2)
        mean_rmse = round(sum(self.test_rmse) / len(self.test_rmse), 2)
        mean_x_mae = round(sum(self.test_x_mae) / len(self.test_x_mae), 2)
        mean_z_mae = round(sum(self.test_z_mae) / len(self.test_z_mae), 2)

        # Log to tensorboard hparams
        self.log("hp/L2", mean_dist)
        self.log("hp/L2_trim", trimmed_dist)
        self.log("hp/MAE", mean_mae)
        self.log("hp/RMSE", mean_rmse)
        self.log("hp/MAE_x", mean_x_mae)
        self.log("hp/MAE_z", mean_z_mae)

        # Create the boxplot traces
        boxplot_trace = go.Box(y=self.test_meanL2norm, name='Mean Pairwise Distance')
        mae_trace = go.Box(y=self.test_mae, name='MAE')
        rmse_trace = go.Box(y=self.test_rmse, name='RMSE')

        # Create the layout
        layout = go.Layout(
            title=f"<span style='font-size: 16px;'>Boxplot of Accuracies [n={len(self.test_meanL2norm)}]</span><br>"
                  f"<span style='font-size: 14px;'>Mean Pairwise Distance/Mean L2-Norm: {mean_dist:.2f} [mm], Mean MAE: {mean_mae:.2f} [mm], Mean RMSE: {mean_rmse:.2f} [mm] </span><br>",
            showlegend=True,
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True, title='Error [mm]'),
        )

        # Create the figure
        fig = go.Figure(data=[boxplot_trace, mae_trace, rmse_trace], layout=layout)

        fig.write_html(
            f"./logs/{self.logger.name}/version_{str(self.logger.version)}/plotlys/acc_{self.run_name}_{mean_dist:.2f}.html")

        # Load the YAML file
        with open(f'./logs/{self.logger.name}/version_{str(self.logger.version)}/config.yaml', "r") as f:
            config = yaml.safe_load(f)

        # Add the new line
        new_config = {"metric": {
            "meanL2norm": mean_dist,
            "trimmedL2norm": trimmed_dist,
            "mae": mean_mae,
            "x_mae": mean_x_mae,
            "z_mae": mean_z_mae,
            "rmse": mean_rmse,
        },
            "Version": self.logger.version
        }
        config.update(new_config)

        # Write the updated config back to the file
        with open(f'./logs/{self.logger.name}/version_{str(self.logger.version)}/config.yaml', "w") as f:
            yaml.dump(config, f)
