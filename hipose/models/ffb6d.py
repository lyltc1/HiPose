""" adapt ffb6d to training mask and binary code output, with rgbd branch only """

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn.pspnet import PSPNet
import models.pytorch_utils as pt_utils
from models.RandLA.RandLANet import Network as RandLANet


psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'convnext_base':lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='convnext_base'),
    'convnext_large':lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=768, deep_features_size=384, backend='convnext_large'),
}


class FFB6D(nn.Module):
    def __init__(
        self, n_classes, n_pts, rndla_cfg, number_of_outputs=17, fusion=False, convnext=False,
    ):
        super().__init__()

        # ######################## prepare stages#########################
        self.n_cls = n_classes
        self.n_pts = n_pts
        self.number_of_outputs = number_of_outputs
        self.fusion = fusion
        if self.fusion:
            self.fusion_layer = DenseFusion()
        if convnext == False or convnext is None:
            cnn = psp_models['resnet34'.lower()]()
        else:
            cnn = psp_models[convnext.lower()]()

        rndla = RandLANet(rndla_cfg)
        if convnext == False:
            self.cnn_pre_stages = nn.Sequential(
                cnn.feats.conv1,  # stride = 2, [bs, c, 240, 320]
                cnn.feats.bn1, cnn.feats.relu,
                cnn.feats.maxpool  # stride = 2, [bs, 64, 120, 160]
            )
        else:
            self.cnn_pre_stages = cnn.feats.stem
        self.rndla_pre_stages = rndla.fc0

        # ####################### downsample stages#######################
        if convnext == False:
            self.ds_rgb_oc = [64, 128, 512, 1024]
            self.up_rgb_oc = [256, 64, 64]
            self.cnn_ds_stages = nn.ModuleList([
                cnn.feats.layer1,    # stride = 1, [bs, 64, 120, 160]
                cnn.feats.layer2,    # stride = 2, [bs, 128, 60, 80]
                nn.Sequential(cnn.feats.layer3, cnn.feats.layer4),  # stride = 1, [bs, 512, 60, 80]
                nn.Sequential(cnn.psp, cnn.drop_1)   # [bs, 1024, 60, 80]
            ])
            self.cnn_up_stages = nn.ModuleList([
                nn.Sequential(cnn.up_1, cnn.drop_2),  # [bs, 256, 120, 160]
                nn.Sequential(cnn.up_2, cnn.drop_2),  # [bs, 64, 240, 320]
                nn.Sequential(cnn.final),            # [bs, 64, 240, 320]
                nn.Sequential(cnn.up_3, cnn.final)  # [bs, 64, 480, 640]
            ])
            self.up_4 = nn.Identity()
        elif convnext.startswith('convnext_base'):
            self.ds_rgb_oc = [128, 256, 512, 1024]
            self.up_rgb_oc = [256, 64, 64]
            self.cnn_ds_stages = nn.ModuleList([
                cnn.feats.stages[0],    # stride = 1, [bs, 128, H//4, W//4]
                cnn.feats.stages[1],    # stride = 2, [bs, 256, H//8, W//8]
                cnn.feats.stages[2],    # stride = 2, [bs, 512, H//16, H//16]
                nn.Sequential(cnn.psp, cnn.drop_1)   # stride = 1, [bs, 1024, H//16, W//16]
            ])
            self.cnn_up_stages = nn.ModuleList([
                nn.Sequential(cnn.up_1, cnn.drop_2),  # [bs, 256, 120, 160]
                nn.Sequential(cnn.up_2, cnn.drop_2),  # [bs, 64, 240, 320]
                nn.Sequential(cnn.final),            # [bs, 64, 240, 320]
                nn.Sequential(cnn.up_3, cnn.final)  # [bs, 64, 480, 640]
            ])
            self.up_4 = nn.Sequential(cnn.up_4, nn.Conv2d(64, 64, kernel_size=1), nn.LogSoftmax())
        elif convnext.startswith('convnext_large'):
            self.ds_rgb_oc = [192, 384, 768, 1024]
            self.up_rgb_oc = [256, 64, 64]
            self.cnn_ds_stages = nn.ModuleList([
                cnn.feats.stages[0],    # stride = 1, [bs, 192, H//4, W//4]
                cnn.feats.stages[1],    # stride = 2, [bs, 384, H//8, W//8]
                cnn.feats.stages[2],    # stride = 2, [bs, 768, H//16, W//16]
                nn.Sequential(cnn.psp, cnn.drop_1)   # stride = 1, [bs, 1024, H//16, W//16]
            ])
            self.cnn_up_stages = nn.ModuleList([
                nn.Sequential(cnn.up_1, cnn.drop_2),  # [bs, 256, 120, 160]
                nn.Sequential(cnn.up_2, cnn.drop_2),  # [bs, 64, 240, 320]
                nn.Sequential(cnn.final),            # [bs, 64, 240, 320]
                nn.Sequential(cnn.up_3, cnn.final)  # [bs, 64, 480, 640]
            ])
            self.up_4 = nn.Sequential(cnn.up_4, nn.Conv2d(64, 64, kernel_size=1), nn.LogSoftmax())

        self.rndla_ds_stages = rndla.dilated_res_blocks

        self.ds_rndla_oc = [item * 2 for item in rndla_cfg.d_out]
        self.ds_fuse_r2p_pre_layers = nn.ModuleList()
        self.ds_fuse_r2p_fuse_layers = nn.ModuleList()
        self.ds_fuse_p2r_pre_layers = nn.ModuleList()
        self.ds_fuse_p2r_fuse_layers = nn.ModuleList()
        for i in range(4):
            self.ds_fuse_r2p_pre_layers.append(
                pt_utils.Conv2d(
                    self.ds_rgb_oc[i], self.ds_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.ds_fuse_r2p_fuse_layers.append(
                pt_utils.Conv2d(
                    self.ds_rndla_oc[i]*2, self.ds_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

            self.ds_fuse_p2r_pre_layers.append(
                pt_utils.Conv2d(
                    self.ds_rndla_oc[i], self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.ds_fuse_p2r_fuse_layers.append(
                pt_utils.Conv2d(
                    self.ds_rgb_oc[i]*2, self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

        # ###################### upsample stages #############################

        
        self.up_rndla_oc = []
        for j in range(rndla_cfg.num_layers):
            if j < 3:
                self.up_rndla_oc.append(self.ds_rndla_oc[-j-2])
            else:
                self.up_rndla_oc.append(self.ds_rndla_oc[0])

        self.rndla_up_stages = rndla.decoder_blocks

        n_fuse_layer = 3
        self.up_fuse_r2p_pre_layers = nn.ModuleList()
        self.up_fuse_r2p_fuse_layers = nn.ModuleList()
        self.up_fuse_p2r_pre_layers = nn.ModuleList()
        self.up_fuse_p2r_fuse_layers = nn.ModuleList()
        for i in range(n_fuse_layer):
            self.up_fuse_r2p_pre_layers.append(
                pt_utils.Conv2d(
                    self.up_rgb_oc[i], self.up_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.up_fuse_r2p_fuse_layers.append(
                pt_utils.Conv2d(
                    self.up_rndla_oc[i]*2, self.up_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

            self.up_fuse_p2r_pre_layers.append(
                pt_utils.Conv2d(
                    self.up_rndla_oc[i], self.up_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.up_fuse_p2r_fuse_layers.append(
                pt_utils.Conv2d(
                    self.up_rgb_oc[i]*2, self.up_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

        # ####################### prediction headers #############################
        # We use 3D keypoint prediction header for pose estimation following PVN3D
        # You can use different prediction headers for different downstream tasks.
        self.prediction_head = (
            pt_utils.Seq(self.up_rndla_oc[-1]+self.up_rgb_oc[-1])
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(self.number_of_outputs, activation=None)
        )

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        if len(feature.size()) > 3:
            feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(
            feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        ).contiguous()
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(
            feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        ).contiguous()
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features

    def _break_up_pc(self, pc):
        xyz = pc[:, :3, :].transpose(1, 2).contiguous()
        features = (
            pc[:, 3:, :].contiguous() if pc.size(1) > 3 else None
        )
        return xyz, features

    def forward(
        self, inputs, end_points=None, scale=1,
    ):
        """
        Params:
        inputs: dict of :
            rgb         : FloatTensor [bs, 3, h, w]
            dpt_nrm     : FloatTensor [bs, 6, h, w], 3c xyz in meter + 3c normal map
            cld_rgb_nrm : FloatTensor [bs, 9, npts]
            choose      : LongTensor [bs, 1, npts]
            xmap, ymap: [bs, h, w]
            K:          [bs, 3, 3]
        Returns:
            end_points:
        """
        # ###################### prepare stages #############################
        if not end_points:
            end_points = {}
        # ResNet pre + layer1 + layer2
        rgb_emb = self.cnn_pre_stages(inputs['rgb'])  # [bs, 3, h, w] -> [bs, 128, h // 4, w // 4]
        # rndla pre
        xyz, p_emb = self._break_up_pc(inputs['cld_rgb_nrm'])  # [bs, 9, npts] -> [bs, 6, npts]
        p_emb = inputs['cld_rgb_nrm']
        p_emb = self.rndla_pre_stages(p_emb)  # [bs, 9, npts] -> [bs, 8, N]
        p_emb = p_emb.unsqueeze(dim=3)  # [bs, 8, N, 1]

        # ###################### encoding stages #############################
        ds_emb = []
        for i_ds in range(4):
            # encode rgb downsampled feature
            rgb_emb0 = self.cnn_ds_stages[i_ds](rgb_emb)  # [bs, 128, h // 4, w // 4], [bs, 256, h // 8, w // 8]
            bs, c, hr, wr = rgb_emb0.size()

            # encode point cloud downsampled feature
            f_encoder_i = self.rndla_ds_stages[i_ds](  # [bs, 64, N, 1], [bs, 128, N//4, 1]
                p_emb, inputs['cld_xyz%d' % i_ds], inputs['cld_nei_idx%d' % i_ds]
            )
            f_sampled_i = self.random_sample(f_encoder_i, inputs['cld_sub_idx%d' % i_ds])  # [bs, N // 4, 16], [bs, 128, N // 16, 1]
            p_emb0 = f_sampled_i
            if i_ds == 0:
                ds_emb.append(f_encoder_i)

            # fuse point feauture to rgb feature
            p2r_emb = self.ds_fuse_p2r_pre_layers[i_ds](p_emb0)  # [bs, 128, N//4, 1], [bs, 255, N//16, 1]
            p2r_emb = self.nearest_interpolation(
                p2r_emb, inputs['p2r_ds_nei_idx%d' % i_ds]
            )
            p2r_emb = p2r_emb.view(bs, -1, hr, wr)  # [bs, 128, h // 4, w // 4]
            rgb_emb = self.ds_fuse_p2r_fuse_layers[i_ds](  # [bs, 128, h // 4, w // 4]
                torch.cat((rgb_emb0, p2r_emb), dim=1)
            )

            # fuse rgb feature to point feature
            r2p_emb = self.random_sample(  # [bs, 128, N //4, 1]
                rgb_emb0.reshape(bs, c, hr*wr, 1), inputs['r2p_ds_nei_idx%d' % i_ds]
            ).view(bs, c, -1, 1)
            r2p_emb = self.ds_fuse_r2p_pre_layers[i_ds](r2p_emb)  # [bs, 64, N//4, 1]
            p_emb = self.ds_fuse_r2p_fuse_layers[i_ds](  # [bs, 64, N//4, 1]
                torch.cat((p_emb0, r2p_emb), dim=1)
            )
            ds_emb.append(p_emb)

        # ###################### decoding stages #############################
        n_up_layers = len(self.rndla_up_stages)
        for i_up in range(n_up_layers-1):
            # decode rgb upsampled feature
            rgb_emb0 = self.cnn_up_stages[i_up](rgb_emb)  #[bs, 1024, H //16, W//16], [bs, 256, H//8,W//8]
            bs, c, hr, wr = rgb_emb0.size()

            # decode point cloud upsampled feature
            f_interp_i = self.nearest_interpolation(
                p_emb, inputs['cld_interp_idx%d' % (n_up_layers-i_up-1)]
            )
            f_decoder_i = self.rndla_up_stages[i_up](
                torch.cat([ds_emb[-i_up - 2], f_interp_i], dim=1)
            )
            p_emb0 = f_decoder_i  # [bs, 256, 42, 1]

            # fuse point feauture to rgb feature
            p2r_emb = self.up_fuse_p2r_pre_layers[i_up](p_emb0)
            p2r_emb = self.nearest_interpolation(
                p2r_emb, inputs['p2r_up_nei_idx%d' % i_up]
            )
            p2r_emb = p2r_emb.view(bs, -1, hr, wr)  
            rgb_emb = self.up_fuse_p2r_fuse_layers[i_up](
                torch.cat((rgb_emb0, p2r_emb), dim=1)
            )

            # fuse rgb feature to point feature
            r2p_emb = self.random_sample(
                rgb_emb0.reshape(bs, c, hr*wr), inputs['r2p_up_nei_idx%d' % i_up]
            ).view(bs, c, -1, 1)
            r2p_emb = self.up_fuse_r2p_pre_layers[i_up](r2p_emb)
            p_emb = self.up_fuse_r2p_fuse_layers[i_up](
                torch.cat((p_emb0, r2p_emb), dim=1)
            )

        # final upsample layers:
        rgb_emb = self.cnn_up_stages[n_up_layers-1](rgb_emb)

        f_interp_i = self.nearest_interpolation(
            p_emb, inputs['cld_interp_idx%d' % (0)]
        )
        p_emb = self.rndla_up_stages[n_up_layers-1](
            torch.cat([ds_emb[0], f_interp_i], dim=1)
        ).squeeze(-1)
        rgb_emb = self.up_4(rgb_emb)
        bs, di, _, _ = rgb_emb.size()
        rgb_emb_c = rgb_emb.view(bs, di, -1)
        choose_emb = inputs['choose'].repeat(1, di, 1)
        rgb_emb_c = torch.gather(rgb_emb_c, 2, choose_emb).contiguous()

        if self.fusion:
            rgbd_emb = self.fusion_layer(rgb_emb_c, p_emb)
        else:
            rgbd_emb = torch.cat([rgb_emb_c, p_emb], dim=1)

        # ###################### prediction stages #############################
        predictions = self.prediction_head(rgbd_emb)

        mask,binary_code = torch.split(predictions,[1,self.number_of_outputs-1],1)
        return mask, binary_code


# Copy from PVN3D: https://github.com/ethnhe/PVN3D
class DenseFusion(nn.Module):
    def __init__(self):
        super(DenseFusion, self).__init__()
        self.conv2_rgb = torch.nn.Conv1d(64, 256, 1)
        self.conv2_cld = torch.nn.Conv1d(64, 256, 1)

        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AdaptiveAvgPool1d(1)
        self.conv5 = torch.nn.Conv1d(1664, 128, 1)  # 128 + 512 + 1024

    def forward(self, rgb_emb, cld_emb):
        bs, _, n_pts = cld_emb.size()
        feat_1 = torch.cat((rgb_emb, cld_emb), dim=1)
        rgb = F.relu(self.conv2_rgb(rgb_emb))
        cld = F.relu(self.conv2_cld(cld_emb))

        feat_2 = torch.cat((rgb, cld), dim=1)

        rgbd = F.relu(self.conv3(feat_1))
        rgbd = F.relu(self.conv4(rgbd))

        ap_x = self.ap1(rgbd).repeat(1, 1, n_pts)

        return self.conv5(torch.cat([feat_1, feat_2, ap_x], 1))  # 96+ 512 + 1024 = 1644


def main():
    from common import ConfigRandLA
    rndla_cfg = ConfigRandLA

    n_cls = 22
    model = FFB6D(n_cls, rndla_cfg.num_points, rndla_cfg)
    print(model)

    print(
        "model parameters:", sum(param.numel() for param in model.parameters())
    )


if __name__ == "__main__":
    main()
