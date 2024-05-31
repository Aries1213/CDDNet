import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from thop import profile
from ptflops import get_model_complexity_info
import time

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.losses.LDL_loss import get_refined_artifact_map

from torch.autograd import gradcheck

@MODEL_REGISTRY.register()
class SRGANDynamicModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRGANDynamicModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        self.net_p = build_network(opt['network_p'])
        self.net_p = self.model_to_device(self.net_p)

        if self.is_train:
            self.net_d = build_network(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        load_key = self.opt['path'].get('param_key_g', None)
        if load_path is not None:
            if 'pretrained_models' in load_path and self.is_train:
                self.load_network_init_alldynamic(self.net_g, load_path, self.opt['num_networks'], self.opt['path'].get('strict_load_g', True), load_key)
            else:
                self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), load_key)

        load_path_p = self.opt['path'].get('pretrain_network_p', None)
        if load_path_p is not None:
            self.load_network(self.net_p, load_path_p, self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            load_path_d = self.opt['path'].get('pretrain_network_d', None)
            load_key = self.opt['path'].get('param_key_g', None)
            if load_path_d is not None:
                self.load_network(self.net_d, load_path_d, self.opt['path'].get('strict_load_d', True), load_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        self.net_p.train()
        self.net_d.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            for p in self.net_g_ema.parameters():
                p.requires_grad = False
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                if 'pretrained_models' in load_path:
                    self.load_network_init_alldynamic(self.net_g_ema, load_path, self.opt['num_networks'], self.opt['path'].get('strict_load_g', True), 'params_ema')
                else:
                    self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('regress_opt'):
            self.cri_regress = build_loss(train_opt['regress_opt']).to(self.device)
        else:
            self.cri_regress = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
            
        if train_opt.get('artifacts_opt'):
            self.cri_artifacts = build_loss(train_opt['artifacts_opt']).to(self.device)
        else:
            self.cri_artifacts = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)


        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_type = train_opt['optim_g'].pop('type')
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        for k, v in self.net_p.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.lq_path = data['lq_path']
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        # 一次迭代步骤的优化。优化一次生成器，接着优化一次判别器。
        # optimize net_g
        # 1. 首先优化 生成网络net_g, net_d 判别网络不更新weight
        for p in self.net_d.parameters():
            p.requires_grad = False
        # 2. 梯度归0
        self.optimizer_g.zero_grad()
        # 3. 前向生成网络，输入的是一个低质低分辨率图像
        # predicted_params, weights分别是33dim的退化类型参数，net_g的动态卷积参数
        # 图像先经过退化网络预测退化，并融入超分生成网络，生成超分图像output
        predicted_params, weights = self.net_p(self.lq)
        self.output = self.net_g(self.lq.contiguous(), weights)
        self.output_ema = self.net_g_ema(self.lq.contiguous(), weights)
        # 4. 计算训练生成网络的损失
        # 主要包括 pixel loss 重建损失 self.cri_pix(self.output, self.gt)
        # 主要包括 退化预测回归损失 self.cri_regress(predicted_params, self.degradation_params)
        # 图像内容和风格感知损失    self.cri_perceptual(self.output, self.gt)
        # gan损失，使预测迷惑判别器 self.cri_gan(fake_g_pred, True, is_disc=False)
        l_g_total = 0
        loss_dict = OrderedDict()
        if current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters:
            # pixel loss
            if self.cri_pix:
                l_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_pix
                loss_dict['l_pix'] = l_pix
            if self.cri_regress:
                l_regression = self.cri_regress(predicted_params, self.degradation_params)
                l_g_total += l_regression
                loss_dict['l_regression'] = l_regression
            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                if l_percep is not None:
                    l_g_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_g_total += l_style
                    loss_dict['l_style'] = l_style

            if self.cri_artifacts:
                pixel_weight = get_refined_artifact_map(self.gt, self.output, self.output_ema, 7)
                l_g_artifacts = self.cri_artifacts(torch.mul(pixel_weight, self.output),torch.mul(pixel_weight, self.gt))
                l_g_total += l_g_artifacts
                loss_dict['l_g_artifacts'] = l_g_artifacts

            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            # 5. 计算梯度和优化
            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        # 6. 优化判别器网络，首先requires_grad设为ture,可训练
        for p in self.net_d.parameters():
            p.requires_grad = True
        # 7. 梯度归0
        self.optimizer_d.zero_grad()
        # real
        # 8. 计算gt进入判别器的损失，使gt 尽量为 1
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        # 9. 计算fake进入判别器的损失，使predict output 尽量为 0
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        # 10. 梯度计算和优化
        l_d_fake.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        self.net_p.eval()
        with torch.no_grad():
            predicted_params, weights = self.net_p(self.lq)
        self.net_p.train()

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq, weights)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq.contiguous(), weights)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')

                imwrite(sr_img, save_img_path)


            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_p, 'net_p', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

