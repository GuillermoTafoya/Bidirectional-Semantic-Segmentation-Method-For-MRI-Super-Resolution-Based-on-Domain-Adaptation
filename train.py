import torch
from torch.nn import DataParallel
import torch.optim as optim

from model import model
from model.utils import *
from model.utils.losses import *

class Trainer:
    def __init__(self, device, batch):
        self.model = model()
        self.device = device
        self.batch = batch
        
        train_dl, val_dl = loader(source)
        self.loader = {"tr": train_dl, "ts": val_dl}

        self.optimizer_r = optim.Adam(self.model.decoder_r.parameters(), lr=1e-4, weight_decay=1e-5)
        self.optimizer_pdd = optim.Adam(self.model.pdd.parameters(), lr=1e-4, weight_decay=1e-5)
        self.optimizer_s = optim.Adam(self.model.decoder_s.parameters(), lr=1e-4, weight_decay=1e-5)
        self.optimizer_odd = optim.Adam(self.model.odd.parameters(), lr=1e-4, weight_decay=1e-5)

    def train(self, epochs):
        current_loader = self.loader["tr"]

        self.writer = open(self.tensor_path, 'w')
        self.writer.write('Epoch, train_r, train_pdd, train_s, train_pdd, test_r, test_pdd, test_s, test_pdd'+'\n')

        self.best_loss = 10000

        c_As_list = dict
        At_list = dict

        module_E = DataParallel(self.model.module_e).to(self.device).eval()
        decoder_r = DataParallel(self.model.decoder_r).to(self.device).train()
        pdd = DataParallel(self.model.pdd).to(self.device).train()
        module_S = DataParallel(self.model.decoder_s).to(self.device).train()
        odd = DataParallel(self.model.odd).to(self.device).train()

        # ------ PRE-TRAIN -------
        for id, data in enumerate(current_loader):
            # Inputs
            Is = data['Is'].to(self.device)
            As = data['As'].to(self.device)
            It = data['It'].to(self.device)

            d_It = downsample(It)

            # Module E
            Is_Unet = module_E(Is)
            d_It_Unet = module_E(d_It)

            # Module R
            r_Is, layers_R_Is = decoder_r(Is_Unet)
            r_It, layers_R_It = decoder_r(d_It_Unet)

            # Module S
            s_Is, layers_S_Is = module_S(layers_R_Is, Is_Unet)
            s_It, layers_S_It = module_S(layers_R_It, d_It_Unet)

            # Labels
            At = max_threshold(r_It)
            c_As = LCS(r_Is, As)
            c_As_list[id] = c_As
            At_list[id] = At

            # Composite
            d_composite = downsample(r_Is)
            composite_Unet = module_E(d_composite)
            r_composite, layers_R_composite = decoder_r(composite_Unet)
            s_composite, layers_S_composite = module_S(layers_R_composite, composite_Unet)

            # -------Update Module R-------
            for param in self.model.module_r.parameters():
                param.requires_grad = True
            for param in self.model.pdd.parameters():
                param.requires_grad = True
            for param in self.model.module_s.parameters():
                param.requires_grad = False
            for param in self.model.odd.parameters():
                param.requires_grad = False

            u_Is = upsample(Is)
            u_As = upsample(As)

            pdd_r_Is = pdd(r_Is)
            pdd_It = pdd(It)

            R_loss = r_loss(r_It, It, r_Is, u_Is)
            Pdd_loss = pdd_loss(pdd_r_Is, pdd_It)

            self.optimizer_r.zero_grad()
            R_loss.backward()
            self.optimizer_r.step()

            self.optimizer_pdd.zero_grad()
            Pdd_loss.backward()
            self.optimizer_pdd.step()
            
            # -------Update Module S-------

            for param in self.model.module_r.parameters():
                param.requires_grad = False
            for param in self.model.pdd.parameters():
                param.requires_grad = False
            for param in self.model.module_s.parameters():
                param.requires_grad = True
            for param in self.model.odd.parameters():
                param.requires_grad = True

            odd_d_It = odd(d_It)
            odd_Is = odd(Is)

            S_loss = s_loss(s_Is, u_As, s_composite, layers_S_Is, layers_R_Is['step1'])
            Odd_loss = odd_loss(odd_d_It, odd_Is)

            self.optimizer_s.zero_grad()
            S_loss.backward()
            self.optimizer_s.step()

            self.optimizer_odd.zero_grad()
            Odd_loss.backward()
            self.optimizer_odd.step()

        # Trains for all epochs
        for epoch in range(epochs):
            print('-'*15)
            print(f'epoch {epoch+1}/{epochs}')

            decoder_r = DataParallel(self.model.decoder_r).to(self.device).train()
            pdd = DataParallel(self.model.pdd).to(self.device).train()
            module_S = DataParallel(self.model.decoder_s).to(self.device).train()
            odd = DataParallel(self.model.odd).to(self.device).train()

            epoch_r_loss, epoch_pdd_loss, epoch_S_loss, epoch_odd_loss = 0.0, 0.0, 0.0, 0.0
            for id, data in enumerate(current_loader):
                # Inputs
                Is = data['Is'].to(self.device)
                As = data['As'].to(self.device)
                It = data['It'].to(self.device)
                old_c_As = c_As_list[id]
                old_At = At_list[id]

                d_It = downsample(It)

                # Module E
                Is_Unet = module_E(Is)
                d_It_Unet = module_E(d_It)

                # Module R
                r_Is, layers_R_Is = decoder_r(Is_Unet)
                r_It, layers_R_It = decoder_r(d_It_Unet)

                # Module S
                s_Is, layers_S_Is = module_S(layers_R_Is, Is_Unet)
                s_It, layers_S_It = module_S(layers_R_It, d_It_Unet)

                # Labels
                At = max_threshold(r_It)
                c_As = LCS(r_Is, As)

                # Composite
                d_composite = downsample(r_Is)
                composite_Unet = module_E(d_composite)
                r_composite, layers_R_composite = decoder_r(composite_Unet)
                s_composite, layers_S_composite = module_S(layers_R_composite, composite_Unet)

                # -------Update Module R-------
                for param in self.model.module_r.parameters():
                    param.requires_grad = True
                for param in self.model.pdd.parameters():
                    param.requires_grad = True
                for param in self.model.module_s.parameters():
                    param.requires_grad = False
                for param in self.model.odd.parameters():
                    param.requires_grad = False

                u_Is = upsample(Is)
                u_As = upsample(As)

                pdd_r_Is = pdd(r_Is)
                pdd_It = pdd(It)

                R_loss = r_loss(r_It, It, r_Is, u_Is, s_composite, u_As)
                Pdd_loss = pdd_loss(pdd_r_Is, pdd_It)

                self.optimizer_r.zero_grad()
                R_loss.backward()
                self.optimizer_r.step()

                self.optimizer_pdd.zero_grad()
                Pdd_loss.backward()
                self.optimizer_pdd.step()
                
                # -------Update Module S-------

                for param in self.model.module_r.parameters():
                    param.requires_grad = False
                for param in self.model.pdd.parameters():
                    param.requires_grad = False
                for param in self.model.module_s.parameters():
                    param.requires_grad = True
                for param in self.model.odd.parameters():
                    param.requires_grad = True

                odd_d_It = odd(d_It)
                odd_Is = odd(Is)

                S_loss = s_loss(s_Is, u_As, s_composite, layers_S_Is, layers_R_Is['step1'], old_At, At, c_As)
                Odd_loss = odd_loss(odd_d_It, odd_Is)

                self.optimizer_s.zero_grad()
                S_loss.backward()
                self.optimizer_s.step()

                self.optimizer_odd.zero_grad()
                Odd_loss.backward()
                self.optimizer_odd.step()
                
                ## ------- Losses -------

                epoch_r_loss += R_loss
                epoch_pdd_loss += Pdd_loss
                epoch_S_loss += S_loss
                epoch_odd_loss += Odd_loss
            
            epoch_r_loss /= len(self.loader["tr"])
            epoch_pdd_loss /= len(self.loader["tr"])
            epoch_S_loss /= len(self.loader["tr"])
            epoch_odd_loss /= len(self.loader["tr"])

            tr_loss = {'r': epoch_r_loss, 'pdd' : epoch_pdd_loss, 
                        's' : epoch_S_loss, 'odd' : epoch_S_loss}
            val_loss = self.test()

            self.log(epoch, epochs, val_loss['s'], tr_loss, val_loss)

    def test(self,):
        self.model.eval()
        current_loader = self.loader['ts']

        epoch_r_loss, epoch_pdd_loss, epoch_S_loss, epoch_odd_loss = 0.0, 0.0, 0.0, 0.0

        for id, data in enumerate(current_loader):
            # Inputs
            Is = data['Is'].to(self.device)
            As = data['As'].to(self.device)
            It = data['It'].to(self.device)

            d_It = downsample(It)

            # Module E
            Is_Unet = self.module_E(Is)
            d_It_Unet = self.module_E(d_It)

            # Module R
            r_Is, layers_R_Is = self.decoder_r(Is_Unet)
            r_It, layers_R_It = self.decoder_r(d_It_Unet)

            # Module S
            s_Is, layers_S_Is = self.module_S(layers_R_Is, Is_Unet)
            s_It, layers_S_It = self.module_S(layers_R_It, d_It_Unet)

            # Labels
            At = max_threshold(r_It)
            c_As = LCS(r_Is, As)

            # Composite
            d_composite = downsample(r_Is)
            composite_Unet = self.module_E(d_composite)
            r_composite, layers_R_composite = self.decoder_r(composite_Unet)
            s_composite, layers_S_composite = self.module_S(layers_R_composite, composite_Unet)

            # -------Losses-------

            u_Is = upsample(Is)
            u_As = upsample(As)

            pdd_r_Is = self.pdd(r_Is)
            pdd_It = self.pdd(It)

            R_loss = r_loss(r_It, It, r_Is, u_Is)
            Pdd_loss = pdd_loss(pdd_r_Is, pdd_It)

            odd_d_It = self.odd(d_It)
            odd_Is = self.odd(Is)

            S_loss = s_loss(s_Is, u_As, s_composite, layers_S_Is, layers_R_Is['step1'])
            Odd_loss = odd_loss(odd_d_It, odd_Is)

            epoch_r_loss += R_loss
            epoch_pdd_loss += Pdd_loss
            epoch_S_loss += S_loss
            epoch_odd_loss += Odd_loss
        
        epoch_r_loss /= len(self.loader["ts"])
        epoch_pdd_loss /= len(self.loader["ts"])
        epoch_S_loss /= len(self.loader["ts"])
        epoch_odd_loss /= len(self.loader["ts"])

        return {'r': epoch_r_loss, 'pdd' : epoch_pdd_loss, 
                's' : epoch_S_loss, 'odd' : epoch_S_loss}
    
    def log(self, epoch, epochs, val_loss, tr, ts):
        model_path = self.model_path
        self.writer.write(str(epoch+1) + ', ' +
                          str(tr[0]) + ', ' +
                          str(tr[0]) + ', ' +
                          str(tr[0]) + ', ' +
                          str(tr[0]) + ', ' +
                          str(ts[0]) + ', ' +
                          str(ts[0]) + ', ' +
                          str(ts[0]) + ', ' +
                          str(ts[0]) + ', ' + '\n')

        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch + 1,
                'decoder_r': self.model.decoder_r.state_dict(),
            }, model_path + f'/decoder_r_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'pdd': self.model.pdd.state_dict(),
            }, model_path + f'/pdd_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'decoder_s': self.model.decoder_s.state_dict(),
            }, model_path + f'/decoder_s_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'odd': self.model.odd.state_dict(),
            }, model_path + f'/odd_{epoch + 1}.pth')

        if val_loss[0] < self.best_loss:
            self.best_loss = val_loss[0]
            torch.save({
                'epoch': epoch + 1,
                'decoder_r': self.model.decoder_r.state_dict(),
            }, model_path + f'/decoder_r_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'pdd': self.model.pdd.state_dict(),
            }, model_path + f'/pdd_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'decoder_s': self.model.decoder_s.state_dict(),
            }, model_path + f'/decoder_s_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'odd': self.model.odd.state_dict(),
            }, model_path + f'/odd_{epoch + 1}.pth')