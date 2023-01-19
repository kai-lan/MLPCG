import argparse
import glob
import importlib.util
import os
import pdb
import random
import sys
from shutil import copyfile
from timeit import default_timer

import numpy as np
import numpy.ma as ma
import pyevtk.hl as vtk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

sys.path.insert(1, "lib")
from lib.GLOBAL_VARS import *
from lib.dataset_load import FluidNetDataset
from lib.velocity_divergence import velocityDivergence
from model import FluidNet
##########################
# Only support 2D for now
##########################
b_size = 20 # batch size
lr = 1e-4
cuda = torch.device('cuda:0')


# By default we use all available data for training and testing
train_set = FluidNetDataset(NUM_IMAGES_TR_2D__, NUM_FRAMES_TR_2D__)
test_set = FluidNetDataset(NUM_IMAGES_TE_2D__, NUM_FRAMES_TE_2D__)


train_loader = torch.utils.data.DataLoader(train_set, batch_size=b_size, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=b_size, shuffle=False, pin_memory=True)

net = FluidNet()
# When we have multiple GPUs ...
if 0: net = nn.DataParallel(net)
net = net.to(cuda)
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)

        net = net.cuda()

# Initialize network weights with Kaiming normal method (a.k.a MSRA)
net.apply(init_weights)
# Optimizer: Adam
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# Gradually reduce learning rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.6, patience=10, verbose=True, threshold=3e-4, threshold_mode='rel')

# ************************ Training and Validation*******************
# Test set Scenes idx to plot during validation
list_to_plot = [0, 64, 2560, 5120, 11392]

exit()
def run_epoch(epoch, loader, training=True):
    if training:
        # set model to train
        net.train()
    else:
        # otherwise, set it to eval.
        net.eval()

    # initialise loss scores
    total_loss = 0
    p_l2_total_loss = 0
    div_l2_total_loss = 0
    p_l1_total_loss = 0
    div_l1_total_loss = 0
    div_lt_total_loss = 0

    n_batches = 0  # Number of processed batches

    # Loss types
    _pL2Loss = nn.MSELoss() # MSE of pressure
    _divL2Loss = nn.MSELoss() # MSE of divergence (Ground truth is zero divergence)
    _divLTLoss = nn.MSELoss() # Absolute difference of pressure
    _pL1Loss = nn.L1Loss() # Absolute difference of divergence
    _divL1Loss = nn.L1Loss() # MSE of long term divergence. If > 0, implements the Long Term divergence concept from FluidNet

    # Loss lambdas (multiply the corresponding loss)
    pL2Lambda = 0
    divL2Lambda = 1
    pL1Lambda = 0
    divL1Lambda = 0
    divLTLambda = 0

    # loop through data, sorted into batches
    for batch_idx, data in enumerate(loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        if training:
            # Set gradients to zero, clearing previous batches.
            optimizer.zero_grad()

        is3D = data.size(1) == 6 #TODO
        assert (is3D and data.size(1) == 6) or (not is3D and data.size(1) == 5), "Data must have \
                5 input chan for 2D, 6 input chan for 3D"  # TODO

        # Run the model forward
        # flags = data[:, 3].unsqueeze(1).contiguous()

        # New inputs for modle analysis
        out_p, out_U = net(data, epoch, path)

        # Calculate targets
        target_p = target[:, 0].unsqueeze(1)
        out_div = velocityDivergence(out_U.contiguous(), flags)
        target_div = torch.zeros_like(out_div)

        # For plotting purposes
        pressure_from_net = out_p
        pressure_target = target_p
        input_diver = velocityDivergence(data[:, 1:4].contiguous(), flags)
        output_diver = out_div
        input_vel = data[:, 1:4]
        output_vel = out_U.contiguous()
        target_vel = target[:, 1:4]

        # Measure loss and save it
        pL2Loss = pL2Lambda * _pL2Loss(out_p, target_p)
        divL2Loss = divL2Lambda * _divL2Loss(out_div, target_div)
        pL1Loss = pL1Lambda * _pL1Loss(out_p, target_p)
        divL1Loss = divL1Lambda * _divL1Loss(out_div, target_div)

        loss_size = pL2Loss + divL2Loss + pL1Loss + divL1Loss

        # We calculate the divergence of a future frame.
        if (divLTLambda > 0):
            # Check if additional buoyancy or gravity is added to future frames.
            # Adding Buoyancy means adding a source term in the momentum equation, of
            # the type f = delta_rho*g and rho = rho_0 + delta_rho (constant term + fluctuation)
            # with rho' << rho_0
            # Adding gravity: source of the type f = rho_0*g
            # Source term is a vector (direction and magnitude).

            oldBuoyancyScale = 0
            trainBuoyancyScale = 2.0
            buoyancyScale = 0
            trainBuoyancyProb = 0.3
            oldGravityScale = 0
            trainGravityProb = 0
            trainGravityScale = 0
            gravityVec = [0, 0, 0]
            oldGravity = gravityVec
            # rand(1) is an uniform dist on the interval [0,1)
            if torch.rand(1)[0] < trainBuoyancyProb:
                # Add buoyancy to this batch (only in the long term frames)
                var = torch.tensor([1.], device=cuda)
                buoyancyScale = torch.normal(trainBuoyancyScale, var)
            # rand(1) is an uniform dist on the interval [0,1)
            if torch.rand(1)[0] < trainGravityProb:
                # Add gravity to this batch (only in the long term frames)
                var = torch.tensor([1.], device=cuda)
                gravityScale = torch.normal(trainGravityScale, var)


            if buoyancyScale > 0 or gravityScale > 0:
                # Set to 0 gravity vector (direction of gravity)
                gravityVec = [0, 0, 0]
                # Chose randomly one of three cardinal directions and set
                # random + or - dir
                card_dir = 0
                if is3D:
                    card_dir = random.randint(0, 2)
                else:
                    card_dir = random.randint(0, 1)

                updown = random.randint(0, 1) * 2 - 1
                if card_dir == 0:
                    gravityVec[0] = updown
                elif card_dir == 1:
                    gravityVec = updown
                elif card_dir == 2:
                    gravityVec = updown

            dt = 0.1
            base_dt = dt # Time step: default simulation timestep.
            timeScaleSigma = 1
            if timeScaleSigma > 0:
                # FluidNet: randn() returns normal distribution with mean 0 and var 1.
                # The mean of abs(randn) ~= 0.7972, hence the 0.2028 value
                # below.
                scale_dt = 0.2028 + torch.abs(torch.randn(1))[0] * timeScaleSigma
                dt = base_dt * scale_dt
            longTermDivNumSteps = [4, 16]
            num_future_steps = longTermDivNumSteps[0]
            # rand(1) is an uniform dist on the interval [0,1)
            # longTermDivProbability is the prob that longTermDivNumSteps[0] is taken.
            # otherwise, longTermDivNumSteps[1] is taken with prob 1 -
            # longTermDivProbability
            longTermDivProbability = 0.9
            if torch.rand(1)[0] > longTermDivProbability:
                num_future_steps = longTermDivNumSteps[1]

            batch_dict = {}
            batch_dict['p'] = out_p
            batch_dict['U'] = out_U
            batch_dict['flags'] = flags

            resY = out_U.size(3)
            resX = out_U.size(4)
            resZ = out_U.size(2)

            # Some extra features for the batch_dict
            batch_dict['Ustar'] = torch.zeros_like(out_U)
            batch_dict['Div_in'] = torch.zeros_like(out_p)
            batch_dict['Test_case'] = 'Train'

            #it = 0

            dt = base_dt
            Outside_Ja = False
            Threshold_Div = 0.0

            max_iter = max_epochs
            method = 'convnet'
            it = 0
            folder = path

            # Time Vec Declaration
            Time_vec = np.zeros(max_iter)
            Time_Pres = np.zeros(max_iter)
            Jacobi_switch = np.zeros(max_iter)
            Max_Div = np.zeros(max_iter)
            Mean_Div = np.zeros(max_iter)
            Max_Div_All = np.zeros(max_iter)
            time_big = np.zeros(max_iter)

            # Set the simulation forward n steps (using model, no grad calculation),
            # but on the last do not perform a pressure projection.
            # We then input last state to model with grad calculation and
            # add to global loss.
            with torch.no_grad():
                for i in range(0, num_future_steps):
                    output_div = (i == num_future_steps)
                    lib.simulate(mconf, batch_dict, net, method, Time_vec, Time_Pres, Jacobi_switch,
                                    Max_Div, Mean_Div, Max_Div_All, folder, it, Threshold_Div, dt, Outside_Ja)
                    # lib.simulate(mconf, batch_dict, net, \
                    #        'convnet', epoch, output_div=output_div)

            # data indexes     |           |
            #       (dim 1)    |    2D     |    3D
            # ----------------------------------------
            #   DATA:
            #       pDiv       |    0      |    0
            #       UDiv       |    1:3    |    1:4
            #       flags      |    3      |    4
            #       densityDiv |    4      |    5
            #   TARGET:
            #       p          |    0      |    0
            #       U          |    1:3    |    1:4
            #       density    |    3      |    4

            data_lt = torch.zeros_like(data)
            if is3D:
                data_lt[:, 0] = batch_dict['p'].squeeze(1)
                data_lt[:, 1:4] = batch_dict['U']
                data_lt[:, 4] = batch_dict['flags'].squeeze(1)
                data_lt = data_lt.contiguous()
            else:
                data_lt[:, 0] = batch_dict['p'].squeeze(1)
                data_lt[:, 1:3] = batch_dict['U']
                data_lt[:, 3] = batch_dict['flags'].squeeze(1)
                data_lt = data_lt.contiguous()

            dt = base_dt

            out_p_LT, out_U_LT, time = net(data_lt, epoch, path)
            out_div_LT = fluid.velocityDivergence(
                out_U_LT.contiguous(), flags)
            target_div_LT = torch.zeros_like(out_div)
            divLTLoss = divLTLambda * _divLTLoss(out_div_LT, target_div_LT)

            loss_size += divLTLoss

            # After lt, save for plotting purpouses
            save_vtk = False
            outIter = 1
            if save_vtk and it % outIter == 0 and batch_idx == 0:
                px, py, pz = 100, 100, 100
                dpi = 25
                figx = px / dpi
                figy = py / dpi
                figz = pz / dpi

                nx = resX
                ny = resY
                nz = resZ
                ncells = nx * ny * nz

                ratio_x = nx / nz
                ratio_y = ny / nz
                lx, ly, lz = ratio_x, ratio_y, 1.0
                dx, dy, dz = lx / nx, ly / ny, lz / nz

                # Coordinates
                x = np.arange(0, lx + 0.1 * dx, dx, dtype='float32')
                y = np.arange(0, ly + 0.1 * dy, dy, dtype='float32')
                z = np.arange(0, lz + 0.1 * dz, dz, dtype='float32')

                minY = 0
                maxY = resY
                maxY_win = resY
                minX = 0
                maxX = resX
                maxX_win = resX
                minZ = 0
                maxZ = resZ
                maxZ_win = resZ
                X, Y, Z = np.linspace(0, resX - 1, num=resX),\
                    np.linspace(0, resY - 1, num=resY),\
                    np.linspace(0, resZ - 1, num=resZ)

                # Variables
                div_input = batch_dict['Div_in'][0, 0].clone()
                div = fluid.velocityDivergence(
                    batch_dict['U'].clone(),
                    batch_dict['flags'].clone())[0, 0]
                velstar_div = fluid.velocityDivergence(
                    batch_dict['Ustar'].clone(),
                    batch_dict['flags'].clone())[0, 0]
                vel = fluid.getCentered(batch_dict['U'].clone())[
                    0].unsqueeze(0)
                velstar = fluid.getCentered(batch_dict['Ustar'].clone())[
                    0].unsqueeze(0)
                density = batch_dict['density'].clone()[0].unsqueeze(0)
                pressure = batch_dict['p'].clone()[0].unsqueeze(0)
                b = 1
                w = pressure.size(4)
                h = pressure.size(3)
                d = pressure.size(2)

                rho = density.narrow(
                    4, 1, w - 2).narrow(3, 1, h - 2).narrow(2, 1, d - 2)
                rho = rho.clone().expand(b, 3, d - 2, h - 2, w - 2)
                rho_m = rho.clone().expand(b, 3, d - 2, h - 2, w - 2)
                rho_m[:, 0] = density.narrow(
                    4, 0, w - 2).narrow(3, 1, h - 2).narrow(2, 1, d - 2).squeeze(1)
                rho_m[:, 1] = density.narrow(
                    4, 1, w - 2).narrow(3, 0, h - 2).narrow(2, 1, d - 2).squeeze(1)
                rho_m[:, 2] = density.narrow(
                    4, 1, w - 2).narrow(3, 1, h - 2).narrow(2, 0, d - 2).squeeze(1)
                gradRho_center = torch.zeros_like(vel)[:, 0:3].contiguous()
                gradRho_faces = rho - rho_m
                gradRho_center[:, 0:3, 1:(
                    d - 1), 1:(h - 1), 1:(w - 1)] = fluid.getCentered(gradRho_faces)[0, 0:3]

                Pijk = pressure.narrow(
                    4, 1, w - 2).narrow(3, 1, h - 2).narrow(2, 1, d - 2)
                Pijk = Pijk.clone().expand(b, 3, d - 2, h - 2, w - 2)
                Pijk_m = Pijk.clone().expand(b, 3, d - 2, h - 2, w - 2)
                Pijk_m[:, 0] = pressure.narrow(
                    4, 0, w - 2).narrow(3, 1, h - 2).narrow(2, 1, d - 2).squeeze(1)
                Pijk_m[:, 1] = pressure.narrow(
                    4, 1, w - 2).narrow(3, 0, h - 2).narrow(2, 1, d - 2).squeeze(1)
                Pijk_m[:, 2] = pressure.narrow(
                    4, 1, w - 2).narrow(3, 1, h - 2).narrow(2, 0, d - 2).squeeze(1)
                gradP_center = torch.zeros_like(vel)[:, 0:3].contiguous()
                gradP_faces = Pijk - Pijk_m
                gradP_center[:, 0:3, 1:(
                    d - 1), 1:(h - 1), 1:(w - 1)] = fluid.getCentered(gradP_faces)[:, 0:3]

                # Debug Variables
                p_net = pressure_from_net[0, 0].clone()
                p_target = pressure_target[0, 0].clone()
                input_diver = input_diver[0, 0].clone()
                output_diver = output_diver[0, 0].clone()
                input_velocity_x = input_vel[0, 0].clone()
                input_velocity_y = input_vel[0, 1].clone()
                input_velocity_z = input_vel[0, 2].clone()
                output_velocity_x = output_vel[0, 0].clone()
                output_velocity_y = output_vel[0, 1].clone()
                output_velocity_z = output_vel[0, 2].clone()
                target_velocity_x = target_vel[0, 0].clone()
                target_velocity_y = target_vel[0, 1].clone()
                target_velocity_z = target_vel[0, 2].clone()

                pressure = pressure[0, 0]
                density = density[0, 0]

                velX = vel[0, 0].clone()
                velY = vel[0, 1].clone()
                velZ = vel[0, 2].clone()
                velstarX = velstar[0, 0].clone()
                velstarY = velstar[0, 1].clone()
                velstarZ = velstar[0, 2].clone()
                gradRhoX = gradRho_center[0, 0].clone()
                gradRhoY = gradRho_center[0, 1].clone()
                gradRhoZ = gradRho_center[0, 2].clone()
                gradPX = gradP_center[0, 0].clone()
                gradPY = gradP_center[0, 1].clone()
                gradPZ = gradP_center[0, 2].clone()
                flags = batch_dict['flags'][0, 0].clone()

                # Change shape form (D,H,W) to (W,H,D)
                div.transpose_(0, 2).contiguous()
                div_input.transpose_(0, 2).contiguous()
                velstar_div.transpose_(0, 2).contiguous()
                density.transpose_(0, 2).contiguous()
                pressure.transpose_(0, 2).contiguous()
                velX.transpose_(0, 2).contiguous()
                velY.transpose_(0, 2).contiguous()
                velZ.transpose_(0, 2).contiguous()
                velstarX.transpose_(0, 2).contiguous()
                velstarY.transpose_(0, 2).contiguous()
                velstarZ.transpose_(0, 2).contiguous()
                gradRhoX.transpose_(0, 2).contiguous()
                gradRhoY.transpose_(0, 2).contiguous()
                gradRhoZ.transpose_(0, 2).contiguous()
                gradPX.transpose_(0, 2).contiguous()
                gradPY.transpose_(0, 2).contiguous()
                gradPZ.transpose_(0, 2).contiguous()
                flags.transpose_(0, 2).contiguous()

                # Debug Variables
                p_net.transpose_(0, 2).contiguous()
                p_target.transpose_(0, 2).contiguous()
                input_diver.transpose_(0, 2).contiguous()
                output_diver.transpose_(0, 2).contiguous()
                input_velocity_x.transpose_(0, 2).contiguous()
                input_velocity_y.transpose_(0, 2).contiguous()
                input_velocity_z.transpose_(0, 2).contiguous()
                output_velocity_x.transpose_(0, 2).contiguous()
                output_velocity_y.transpose_(0, 2).contiguous()
                output_velocity_z.transpose_(0, 2).contiguous()
                target_velocity_x.transpose_(0, 2).contiguous()
                target_velocity_y.transpose_(0, 2).contiguous()
                target_velocity_z.transpose_(0, 2).contiguous()

                div_np = div.cpu().data.numpy()
                div_input_np = div_input.cpu().data.numpy()
                velstardiv_np = velstar_div.cpu().numpy()
                density_np = density.cpu().data.numpy()
                pressure_np = pressure.cpu().data.numpy()
                velX_np = velX.cpu().data.numpy()
                velY_np = velY.cpu().data.numpy()
                velZ_np = velZ.cpu().data.numpy()
                velstarX_np = velstarX.cpu().data.numpy()
                velstarY_np = velstarY.cpu().data.numpy()
                velstarZ_np = velstarY.cpu().data.numpy()
                np_mask = flags.eq(2).cpu().data.numpy().astype(float)
                pressure_masked = ma.array(pressure_np, mask=np_mask)
                velx_masked = ma.array(velX_np, mask=np_mask)
                vely_masked = ma.array(velY_np, mask=np_mask)
                velz_masked = ma.array(velZ_np, mask=np_mask)
                velstarx_masked = ma.array(velstarX_np, mask=np_mask)
                velstary_masked = ma.array(velstarY_np, mask=np_mask)
                velstarz_masked = ma.array(velstarZ_np, mask=np_mask)
                ma.set_fill_value(pressure_masked, np.nan)
                ma.set_fill_value(velx_masked, np.nan)
                ma.set_fill_value(vely_masked, np.nan)
                ma.set_fill_value(velz_masked, np.nan)
                pressure_masked = pressure_masked.filled()
                velx_masked = velx_masked.filled()
                vely_masked = vely_masked.filled()
                velz_masked = velz_masked.filled()
                velstarx_masked = velstarx_masked.filled()
                velstary_masked = velstary_masked.filled()
                velstarz_masked = velstarz_masked.filled()

                # Debug Variables
                p_net_np = p_net.cpu().data.numpy()
                p_target_np = p_target.cpu().data.numpy()
                in_vx_np = input_velocity_x.cpu().data.numpy()
                in_vy_np = input_velocity_y.cpu().data.numpy()
                in_vz_np = input_velocity_z.cpu().data.numpy()
                out_vx_np = output_velocity_x.cpu().data.numpy()
                out_vy_np = output_velocity_y.cpu().data.numpy()
                out_vz_np = output_velocity_z.cpu().data.numpy()
                tar_vx_np = target_velocity_x.cpu().data.numpy()
                tar_vy_np = target_velocity_y.cpu().data.numpy()
                tar_vz_np = target_velocity_z.cpu().data.numpy()

                in_vx_masked = ma.array(in_vx_np, mask=np_mask)
                in_vy_masked = ma.array(in_vy_np, mask=np_mask)
                in_vz_masked = ma.array(in_vz_np, mask=np_mask)
                out_vx_masked = ma.array(out_vx_np, mask=np_mask)
                out_vy_masked = ma.array(out_vy_np, mask=np_mask)
                out_vz_masked = ma.array(out_vz_np, mask=np_mask)
                tar_vx_masked = ma.array(tar_vx_np, mask=np_mask)
                tar_vy_masked = ma.array(tar_vy_np, mask=np_mask)
                tar_vz_masked = ma.array(tar_vz_np, mask=np_mask)
                ma.set_fill_value(in_vx_masked, np.nan)
                ma.set_fill_value(in_vy_masked, np.nan)
                ma.set_fill_value(in_vz_masked, np.nan)
                ma.set_fill_value(out_vx_masked, np.nan)
                ma.set_fill_value(out_vy_masked, np.nan)
                ma.set_fill_value(out_vz_masked, np.nan)
                ma.set_fill_value(tar_vx_masked, np.nan)
                ma.set_fill_value(tar_vy_masked, np.nan)
                ma.set_fill_value(tar_vz_masked, np.nan)
                in_vx_masked = in_vx_masked.filled()
                in_vy_masked = in_vy_masked.filled()
                in_vz_masked = in_vz_masked.filled()
                out_vx_masked = out_vx_masked.filled()
                out_vy_masked = out_vy_masked.filled()
                out_vz_masked = out_vz_masked.filled()
                tar_vx_masked = tar_vx_masked.filled()
                tar_vy_masked = tar_vy_masked.filled()
                tar_vz_masked = tar_vz_masked.filled()

                p_net_masked = ma.array(p_net_np, mask=np_mask)
                p_target_masked = ma.array(p_target_np, mask=np_mask)

                input_div_np = input_diver.cpu().data.numpy()
                output_div_np = output_diver.cpu().data.numpy()

                divergence = np.ascontiguousarray(
                    div_np[minX:maxX, minY:maxY, minZ:maxZ])
                divergence_input = np.ascontiguousarray(
                    div_input_np[minX:maxX, minY:maxY, minZ:maxZ])
                rho = np.ascontiguousarray(
                    density_np[minX:maxX, minY:maxY, minZ:maxZ])
                p = np.ascontiguousarray(
                    pressure_masked[minX:maxX, minY:maxY, minZ:maxZ])
                velx = np.ascontiguousarray(
                    velx_masked[minX:maxX, minY:maxY, minZ:maxZ])
                vely = np.ascontiguousarray(
                    vely_masked[minX:maxX, minY:maxY, minZ:maxZ])
                velz = np.ascontiguousarray(
                    velz_masked[minX:maxX, minY:maxY, minZ:maxZ])
                velstarx = np.ascontiguousarray(
                    velstarx_masked[minX:maxX, minY:maxY, minZ:maxZ])
                velstary = np.ascontiguousarray(
                    velstary_masked[minX:maxX, minY:maxY, minZ:maxZ])
                velstarz = np.ascontiguousarray(
                    velstarz_masked[minX:maxX, minY:maxY, minZ:maxZ])
                velstardiv = np.ascontiguousarray(
                    velstardiv_np[minX:maxX, minY:maxY, minZ:maxZ])
                gradRhox = np.ascontiguousarray(gradRhoX.cpu().data.numpy()[
                                                minX:maxX, minY:maxY, minZ:maxZ])
                gradRhoy = np.ascontiguousarray(gradRhoY.cpu().data.numpy()[
                                                minX:maxX, minY:maxY, minZ:maxZ])
                gradRhoz = np.ascontiguousarray(gradRhoZ.cpu().data.numpy()[
                                                minX:maxX, minY:maxY, minZ:maxZ])
                gradPx = np.ascontiguousarray(gradPX.cpu().data.numpy()[
                                                minX:maxX, minY:maxY, minZ:maxZ])
                gradPy = np.ascontiguousarray(gradPY.cpu().data.numpy()[
                                                minX:maxX, minY:maxY, minZ:maxZ])
                gradPz = np.ascontiguousarray(gradPZ.cpu().data.numpy()[
                                                minX:maxX, minY:maxY, minZ:maxZ])

                # Debug Variables
                p_net_out = np.ascontiguousarray(
                    p_net_masked[minX:maxX, minY:maxY, minZ:maxZ])
                p_target_out = np.ascontiguousarray(
                    p_target_masked[minX:maxX, minY:maxY, minZ:maxZ])
                div_in_net = np.ascontiguousarray(
                    input_div_np[minX:maxX, minY:maxY, minZ:maxZ])
                div_out_net = np.ascontiguousarray(
                    output_div_np[minX:maxX, minY:maxY, minZ:maxZ])

                in_vx = np.ascontiguousarray(
                    in_vx_masked[minX:maxX, minY:maxY, minZ:maxZ])
                in_vy = np.ascontiguousarray(
                    in_vy_masked[minX:maxX, minY:maxY, minZ:maxZ])
                in_vz = np.ascontiguousarray(
                    in_vz_masked[minX:maxX, minY:maxY, minZ:maxZ])
                out_vx = np.ascontiguousarray(
                    out_vx_masked[minX:maxX, minY:maxY, minZ:maxZ])
                out_vy = np.ascontiguousarray(
                    out_vy_masked[minX:maxX, minY:maxY, minZ:maxZ])
                out_vz = np.ascontiguousarray(
                    out_vz_masked[minX:maxX, minY:maxY, minZ:maxZ])
                tar_vx = np.ascontiguousarray(
                    tar_vx_masked[minX:maxX, minY:maxY, minZ:maxZ])
                tar_vy = np.ascontiguousarray(
                    tar_vy_masked[minX:maxX, minY:maxY, minZ:maxZ])
                tar_vz = np.ascontiguousarray(
                    tar_vz_masked[minX:maxX, minY:maxY, minZ:maxZ])

                folder_vtk = folder + '/Images'
                filename = folder_vtk + '/output_{0:05}'.format(it)
                vtk.gridToVTK(filename, x, y, z, cellData={
                    'density': rho,
                    'divergence': divergence,
                    'pressure': p,
                    'ux': velx,
                    'uy': vely,
                    'uz': velz,
                    'u_star_x': velstarx,
                    'u_star_y': velstary,
                    'u_star_z': velstarz,
                    'u_star_div': velstardiv,
                    'divergence_input': divergence_input,
                    'gradPx': gradPx,
                    'gradPy': gradPy,
                    'gradPz': gradPz,
                    'gradRhox': gradRhox,
                    'gradRhoy': gradRhoy,
                    'gradRhoz': gradRhoz,
                    'p_net': p_net_out,
                    'p_target': p_target_out,
                    'div_in_net': div_in_net,
                    'div_out_net': div_out_net,
                    'in_ux': in_vx,
                    'in_uy': in_vy,
                    'in_uz': in_vz,
                    'out_ux': out_vx,
                    'out_uy': out_vy,
                    'out_uz': out_vz,
                    'tar_ux': tar_vx,
                    'tar_uy': tar_vy,
                    'tar_uz': tar_vz,
                })

        # Print statistics
        p_l2_total_loss += pL2Loss.data.item()
        div_l2_total_loss += divL2Loss.data.item()
        p_l1_total_loss += pL1Loss.data.item()
        div_l1_total_loss += divL1Loss.data.item()
        if (divLTLambda > 0):
            div_lt_total_loss += divLTLoss.data.item()
        total_loss += loss_size.data.item()

        shuffled = True
        if shuffle_training and not training:
            shuffled = False
        if not shuffle_training and training:
            shuffled = False

        if training:
            # Run the backpropagation for all the losses.
            loss_size.backward()

            # Step the optimizer
            optimizer.step()

        n_batches += 1

        if training:
            # Print every 20th batch of an epoch
            if batch_idx % 20 == 0:
                if not remote:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)] \t'.format(
                        epoch, batch_idx * len(data), len(loader.dataset),
                        100. * batch_idx / len(loader)))
                else:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)] \t'.format(
                        epoch, batch_idx * len(data), 20480,
                        100. * batch_idx / (20480 / batch_size)))

    # Divide loss by dataset length
    p_l2_total_loss /= n_batches
    div_l2_total_loss /= n_batches
    p_l1_total_loss /= n_batches
    div_l1_total_loss /= n_batches
    div_lt_total_loss /= n_batches
    total_loss /= n_batches

    # Print for the whole dataset
    if training:
        sstring = 'Train'
    else:
        sstring = 'Validation'
    print('\n{} set: Avg total loss: {:.6f} (L2(p): {:.6f}; L2(div): {:.6f}; \
            L1(p): {:.6f}; L1(div): {:.6f}; LTDiv: {:.6f})'.format(
        sstring,
        total_loss, p_l2_total_loss, div_l2_total_loss,
        p_l1_total_loss, div_l1_total_loss, div_lt_total_loss))

    # Return loss scores
    return total_loss, p_l2_total_loss, div_l2_total_loss, \
        p_l1_total_loss, div_l1_total_loss, div_lt_total_loss

    # ********************************* Prepare saving files *****************

def save_checkpoint(state, is_best, save_path, filename):
    filename = glob.os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = glob.os.path.join(
            save_path, 'convModel_lastEpoch_best.pth')
        copyfile(filename, bestname)

# Create some arrays for recording results
train_loss_plot = np.empty((0, 7))
val_loss_plot = np.empty((0, 7))

# Save loss to disk
m_path = conf['modelDir']
model_save_path = glob.os.path.join(m_path, conf['modelFilename'])

# Save loss as numpy arrays to disk
p_path = conf['modelDir']
file_train = glob.os.path.join(p_path, 'train_loss')
file_val = glob.os.path.join(p_path, 'val_loss')

# Save mconf and conf to disk
file_conf = glob.os.path.join(m_path, conf['modelFilename'] + '_conf.pth')
file_mconf = glob.os.path.join(
    m_path, conf['modelFilename'] + '_mconf.pth')

# raw_input returns the empty string for "enter"
yes = {'yes', 'y', 'ye', ''}
no = {'no', 'n'}

if resume:
    start_epoch = state['epoch']
else:
    start_epoch = 1
    if ((not glob.os.path.exists(p_path)) and (
            not glob.os.path.exists(m_path))):
        if (p_path == m_path):
            glob.os.makedirs(p_path)
        else:
            glob.os.makedirs(m_path)
            glob.os.makedirs(p_path)

    # Here we are a bit barbaric, and we copy the whole model.py into the saved model
    # folder, so that we don't lose the network architecture.
    # We do that only if resuming training.
    path, last = glob.os.path.split(m_path)
    saved_model_name = glob.os.path.join(path, last, last + '_saved.py')
    copyfile('lib/model.py', saved_model_name)

    # Delete plot file if starting from scratch
    if (glob.os.path.isfile(file_train + '.npy')
            and glob.os.path.isfile(file_val + '.npy')):
        print(
            'Are you sure you want to delete existing files and start training from scratch. [y/n]')
        glob.os.remove(file_train + '.npy')
        glob.os.remove(file_val + '.npy')

# Save config dicts
torch.save(conf, file_conf)
torch.save(mconf, file_mconf)

# ********************************* Run epochs ***************************

n_epochs = max_epochs
if not resume:
    state = {}
    state['bestPerf'] = float('Inf')

print('')
print('==> Beginning simulation')
for epoch in range(start_epoch, n_epochs + 1):

    # Begin Training Timer:
    start_big = default_timer()

    # Train on training set and test on validation set
    train_loss, p_l2_tr, div_l2_tr, p_l1_tr, div_l1_tr, div_lt_tr = \
        run_epoch(epoch, train_loader, training=True)
    with torch.no_grad():
        val_loss, p_l2_val, div_l2_val, p_l1_val, div_l1_val, div_lt_val = \
            run_epoch(epoch, test_loader, training=False)

    # Step scheduler, will reduce LR if loss has plateaued
    scheduler.step(train_loss)

    # Store the elapsed time for the whole epoch:
    end_big = default_timer()

    # Store training loss function
    train_loss_plot = np.append(train_loss_plot, [[epoch, train_loss, p_l2_tr,
                                                    div_l2_tr, p_l1_tr, div_l1_tr, div_lt_tr]], axis=0)
    val_loss_plot = np.append(val_loss_plot, [[epoch, val_loss, p_l2_val,
                                                div_l2_val, p_l1_val, div_l1_val, div_lt_val]], axis=0)

    # Check if this is the best model so far and if so save to disk
    is_best = False
    state['epoch'] = epoch + 1
    state['state_dict'] = net.state_dict()
    state['optimizer'] = optimizer.state_dict()

    if val_loss < state['bestPerf']:
        is_best = True
        state['bestPerf'] = val_loss
    save_checkpoint(state, is_best, m_path, 'convModel_lastEpoch.pth')

    # Save loss to disk -- TODO: Check if there is a more efficient way, instead
    # of loading the whole file...
    if epoch % conf['freqToFile'] == 0:
        plot_train_file = file_train + '.npy'
        plot_val_file = file_val + '.npy'
        train_old = np.empty((0, 7))
        val_old = np.empty((0, 7))
        if (glob.os.path.isfile(plot_train_file)
                and glob.os.path.isfile(plot_val_file)):
            train_old = np.load(plot_train_file)
            val_old = np.load(plot_val_file)
        train_loss = np.append(train_old, train_loss_plot, axis=0)
        val_loss = np.append(val_old, val_loss_plot, axis=0)
        np.save(file_val, val_loss)
        np.save(file_train, train_loss)
        # Reset arrays
        train_loss_plot = np.empty((0, 7))
        val_loss_plot = np.empty((0, 7))


        # Delete model_saved_resume.py
        # print()
        # print('Deleting ' + temp_model)
        # glob.os.remove(temp_model)
