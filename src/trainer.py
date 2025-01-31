import torch
import mlflow
import numpy as np
from tqdm import tqdm
from src.utils import plot_y_comp, plot_x_comp


class FISTANetTrainer():
    def __init__(self, model, Psi, args):
        self.device = args['device']

        self.model = model.to(self.device)
        
        self.Psi_np = Psi
        self.Psi = torch.from_numpy(Psi).to(self.device)
        
        self.load_model_run = args['load_model_run']
        self.load_model_epoch = args['load_model_epoch']

        if self.load_model_epoch is not None:
            self.model = mlflow.pytorch.load_model(
                f'runs:/{self.load_model_run}/models/FISTA-Net_ep{self.load_model_epoch}').to(self.device)

        self.batch_size = args['batch_size']
        self.lr = args['lr']
        
        # define optimizer with learnable parameters
        self.optimizer = torch.optim.Adam([
                {'params': self.model.fcs.parameters()},
                {'params': self.model.w_theta, 'lr': 0.001},
                {'params': self.model.b_theta, 'lr': 0.001},
                {'params': self.model.w_mu, 'lr': 0.001},
                {'params': self.model.b_mu, 'lr': 0.001},
                {'params': self.model.w_rho, 'lr': 0.001},
                {'params': self.model.b_rho, 'lr': 0.001}
            ],
            lr=self.lr, weight_decay=0.001)

        # loss weights
        self.lambda_Lspa = args['lambda_Lspa']
        self.lambda_LFsym = args['lambda_LFsym']
        self.lambda_LFspa = args['lambda_LFspa']
        
        # discrepancy loss Lmse
        self.mse_loss = torch.nn.MSELoss()

        self.metric_list = ['L', 'Lmse', 'Lspa', 'LFsym', 'LFspa', 'l0-norm']

    def preprocess_batch(self, Yn, Y):
        # repeat the dictionary for each batch
        Psi = self.Psi.repeat((Yn.shape[0], 1, 1))

        # zero initialization of X
        X0 = torch.from_numpy(np.zeros((Yn.shape[0], self.Psi.shape[1])))
        
        # adapt dimensions and device to the model
        Yn = torch.unsqueeze(Yn, 2).to(self.device)
        Y = torch.unsqueeze(Y, 2).to(self.device)
        X0 = torch.unsqueeze(X0, 2).to(self.device)

        return Yn, Y, X0, Psi

    def loss_L(self, Yhat, Y, Xhat, layers_Rdiff, layers_Rst, losses_dict):
        # discrepancy loss Lmse
        loss_Lmse = self.mse_loss(Yhat, Y)
        
        # sparsity loss Lspa with abs-max normalization
        mins = Xhat.squeeze().min(dim=1).values.repeat((Xhat.squeeze().shape[1], 1)).T
        maxs = Xhat.squeeze().max(dim=1).values.repeat((Xhat.squeeze().shape[1], 1)).T
        absmax = torch.stack([mins, maxs]).abs().max(dim=0).values
        loss_Lspa = torch.mean(torch.abs(Xhat.squeeze() / absmax))
        
        # symmetry loss of F and Ftilde transformations LFsym
        loss_LFsym = sum([torch.mean(torch.pow(Rdiff, 2)) for Rdiff in layers_Rdiff])

        # sparsity loss in F feature space LFspa
        loss_LFspa = sum([torch.mean(torch.abs(Rst)) for Rst in layers_Rst])

        # compound loss L with lambda weights
        loss = loss_Lmse + \
                    self.lambda_Lspa * loss_Lspa + \
                    self.lambda_LFsym * loss_LFsym + \
                    self.lambda_LFspa * loss_LFspa

        # append losses to global dit for logging
        losses_dict['L'].append(loss.item())
        losses_dict['Lmse'].append(loss_Lmse.item())
        losses_dict['Lspa'].append(self.lambda_Lspa * loss_Lspa.item())
        losses_dict['LFsym'].append(self.lambda_LFsym * loss_LFsym.item())
        losses_dict['LFspa'].append(self.lambda_LFspa * loss_LFspa.item())
        losses_dict['l0-norm'].append((torch.sum(Xhat.abs() > 1e-3) / \
                                      (Xhat.shape[0] * Xhat.shape[1])).item())

        return loss, losses_dict

    def train(self, train_loader, valid_loader, epochs, start_epoch=0,
              log_model_every=10, log_comp_fig_every=10, comp_fig_samples=[0, 500, 950]):
        for epoch in tqdm(range(1 + start_epoch, epochs + start_epoch + 1)):
            losses_dict = {k: [] for k in self.metric_list}

            self.model.train(True)
            for batch_idx, (Yn, Y) in enumerate(train_loader):
                Yn, Y, X0, Psi = self.preprocess_batch(Yn, Y)
                
                self.model.zero_grad(set_to_none=True)
                self.optimizer.zero_grad()

                Xhat, layers_Rdiff, layers_Rst = self.model(X0, Yn, Psi)
                Yhat = Yn - torch.bmm(Psi, Xhat)

                loss, losses_dict = self.loss_L(Yhat, Y, Xhat,
                                                layers_Rdiff, layers_Rst, losses_dict)

                loss.backward()
                self.optimizer.step()
                
            # log average epoch loss values to MLflow
            for k, v in losses_dict.items():
                mlflow.log_metric(k + '_train', np.mean(v), step=epoch)

            losses_dict = {k: [] for k in self.metric_list}
            self.model.eval()
            with torch.no_grad():
                for batch_idy, (Yn, Y) in enumerate(valid_loader):
                    Yn, Y, X0, Psi = self.preprocess_batch(Yn, Y)
                    
                    Xhat, layers_Rdiff, layers_Rst = self.model(X0, Yn, Psi)
                    Yhat = Yn - torch.bmm(Psi, Xhat)

                    loss, losses_dict = self.loss_L(Yhat, Y, Xhat,
                                                    layers_Rdiff, layers_Rst, losses_dict)
                    
                    # plot validation batch as MLflow artifact
                    if not epoch % log_comp_fig_every and not batch_idy:
                        plot_y_comp(Yn, Yhat, Y, self.Psi_np, 'valid', epoch,
                                    batch_idy, comp_fig_samples)
                        plot_x_comp(X0, Xhat, 'valid', epoch,
                                    batch_idy, comp_fig_samples)
            
                # log average epoch loss values and model parameters to MLflow
                for k, v in losses_dict.items():
                    mlflow.log_metric(k + '_valid', np.mean(v), step=epoch)
                for k, v in {'param_w_theta': self.model.w_theta.item(),
                             'param_b_theta': self.model.b_theta.item(),
                             'param_w_mu': self.model.w_mu.item(),
                             'param_b_mu': self.model.b_mu.item(),
                             'param_w_rho': self.model.w_rho.item(),
                             'param_b_rho': self.model.b_rho.item()}.items():
                    mlflow.log_metric(k, v, step=epoch)
        
            # log model to MLflow
            if not (epoch % log_model_every) and epoch > 0:
                mlflow.pytorch.log_model(self.model, artifact_path=f'models/FISTA-Net_ep{epoch}')

    def evaluate(self, test_loader, criterion=None, crit_text=None, mlflow_log=True):
        assert not ((type(criterion) != type(None)) ^ (crit_text is not None))
        losses_dict = {k: [] for k in self.metric_list}
        self.model.eval()
        with torch.no_grad():
            for batch_idy, (Yn, Y) in enumerate(test_loader):
                Yn, Y, X0, Psi = self.preprocess_batch(Yn, Y)
                
                Xhat, layers_Rdiff, layers_Rst = self.model(X0, Yn, Psi)
                Yhat = Yn - torch.bmm(Psi, Xhat)

                if type(criterion) == type(None):
                    loss, losses_dict = self.loss_L(Yhat, Y, Xhat,
                                                    layers_Rdiff, layers_Rst, losses_dict)
                    for k, v in losses_dict.items():
                        if mlflow_log:
                            mlflow.log_metric(k + '_test', np.mean(v))
                        else:
                            print(k + '_test: ', np.mean(v))
                else:
                    loss = criterion(Yhat, Y)
                    if mlflow_log:
                        mlflow.log_metric(f'{crit_text}_test', loss.item())
                    else:
                        print(f'{crit_text}_test: ', loss.item())

    def infer(self, Yn):
        self.model.eval()
        with torch.no_grad():
            Yn, _, X0, Psi = self.preprocess_batch(Yn, Yn)
            
            Xhat, layers_Rdiff, layers_Rst = self.model(X0, Yn, Psi)
            Yhat = Yn - torch.bmm(Psi, Xhat)

            return [Yhat, Xhat]
