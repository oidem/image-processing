# module backprojector.py

import sys,math,cmath,os,copy
import numpy as np
from numpy import fft
import projector as p

def backrotate2D(data, rotdata, A3d, debug=False):

    if (data.shape[0] != rotdata.shape[0] or data.shape[1] != rotdata.shape[1]):
        print('Error: input and output array must have the same shape!')
        sys.exit()

    # define rotation matrix (inversede)
    # and correct the scaling by multiplying padding factor
    Ainv = A3d.T
    # Ainv *= int(factor)

    # 0 = z, 1 = y, 2 = x
    boxsize = data.shape

    if debug:
        print('boxsize: ', boxsize)

    radius = np.zeros(2, dtype=int)
    for i in range(2):
        radius[i] = math.floor(boxsize[i] / 2) - 1

    if debug:
        print('radius: ', radius)

    rmax_out: int = radius.min()

    rmax_ref: int = rmax_out # * int(factor)
    rmax_ref_2: int = rmax_ref * rmax_ref

    if (debug):
        print('rmax_ref_2: ', rmax_ref_2)
    
    AtA_xx  = Ainv[0,0] * Ainv[0,0] + Ainv[1,0] * Ainv[1,0]
    AtA_xy  = Ainv[0,0] * Ainv[0,1] + Ainv[1,0] * Ainv[1,1]
    AtA_yy  = Ainv[0,1] * Ainv[0,1] + Ainv[1,1] * Ainv[1,1]
    AtA_xy2 = AtA_xy * AtA_xy

    # define starting to access logical coordinates
    starting = np.array([1 - boxsize[0]/2, 1 - boxsize[1]/2], dtype=int)
    
    for i in range(data.shape[0]):
        y = i + starting[0]

        discr = AtA_xy2 * y*y - AtA_xx * (AtA_yy * y*y - rmax_ref_2)
        if (discr < 0.0):
            continue

        d = np.sqrt(discr) / AtA_xx
        q = - AtA_xy * y / AtA_xx
        
        for j in range(data.shape[1]):
            x = j + starting[1]

            myval = data[i,j]

            xu = x - 0.5
            yu = y - 0.5

            xp = Ainv[0,0] * xu + Ainv[0,1] * yu
            yp = Ainv[1,0] * xu + Ainv[1,1] * yu

            r_ref_2 = xp * xp + yp * yp
            if (r_ref_2 > rmax_ref_2):
                continue

            y0 = np.floor(yp + 0.5)
            fy = yp + 0.5 - y0
            y0 -= starting[0]
            y1:int = y0 + 1

            x0:int = np.floor(xp + 0.5)
            fx = xp + 0.5 - x0
            x0 -= starting[1]
            x1 = x0 + 1

            mfx = 1. - fx
            mfy = 1. - fy

            dd00 = mfy * mfx
            dd01 = mfy *  fx
            dd10 =  fy * mfx
            dd11 =  fy *  fx

            # Store slice in 3D weighted sum
            rotdata[int(y0),int(x0)] += dd00 * myval
            rotdata[int(y0),int(x1)] += dd01 * myval
            rotdata[int(y1),int(x0)] += dd10 * myval
            rotdata[int(y1),int(x1)] += dd11 * myval

    return

def backproject2Dto3D(f2d, f3d, weight, A3d, debug=False):
    Ainv = A3d.T

    # 0 = z, 1 = y, 2 = x
    boxsize = f3d.shape

    if debug:
        print('boxsize: ', boxsize)

    radius = np.zeros(3, dtype=int)
    for i in range(3):
        radius[i] = math.floor(boxsize[i] / 2) - 1

    if debug:
        print('radius: ', radius)

    rmax_out: int = radius.min()

    rmax_ref: int = rmax_out
    rmax_ref_2: int = rmax_ref * rmax_ref

    # define starting to access logical coordinates
    starting = np.array([1 - boxsize[0]/2, 1 - boxsize[1]/2, 1 - boxsize[2]/2], dtype=int)

    myweight = 1.0

    for i in range(boxsize[1]):
        y = i + starting[1]


        for j in range(boxsize[2]):
            x = j + starting[2]

            myval = f2d[i,j]

            xu = x - 0.5
            yu = y - 0.5

            xp = Ainv[0,0] * xu + Ainv[0,1] * yu
            yp = Ainv[1,0] * xu + Ainv[1,1] * yu
            zp = Ainv[2,0] * xu + Ainv[2,1] * yu

            r_ref_2 = xp * xp + yp * yp + zp * zp
            if (r_ref_2 > rmax_ref_2):
                continue

            x0:int = np.floor(xp + 0.5)
            fx = xp + 0.5 - x0
            x0 -= starting[2]
            x1:int = x0 + 1

            y0:int = np.floor(yp + 0.5)
            fy = yp + 0.5 - y0
            y0 -= starting[1]
            y1:int = y0 + 1

            z0:int = np.floor(zp + 0.5)
            fz = zp + 0.5 - z0
            z0 -= starting[0]
            z1:int = z0 + 1

            mfx = 1. - fx
            mfy = 1. - fy
            mfz = 1. - fz

            dd000 = mfz * mfy * mfx
            dd001 = mfz * mfy *  fx
            dd010 = mfz *  fy * mfx
            dd011 = mfz *  fy *  fx
            dd100 =  fz * mfy * mfx
            dd101 =  fz * mfy *  fx
            dd110 =  fz *  fy * mfx
            dd111 =  fz *  fy *  fx

            f3d[int(z0),int(y0),int(x0)] += dd000 * myval
            f3d[int(z0),int(y0),int(x1)] += dd001 * myval
            f3d[int(z0),int(y1),int(x0)] += dd010 * myval
            f3d[int(z0),int(y1),int(x1)] += dd011 * myval
            f3d[int(z1),int(y0),int(x0)] += dd100 * myval
            f3d[int(z1),int(y0),int(x1)] += dd101 * myval
            f3d[int(z1),int(y1),int(x0)] += dd110 * myval
            f3d[int(z1),int(y1),int(x1)] += dd111 * myval

            weight[int(z0),int(y0),int(x0)] += dd000 * myweight
            weight[int(z0),int(y0),int(x1)] += dd001 * myweight
            weight[int(z0),int(y1),int(x0)] += dd010 * myweight
            weight[int(z0),int(y1),int(x1)] += dd011 * myweight
            weight[int(z1),int(y0),int(x0)] += dd100 * myweight
            weight[int(z1),int(y0),int(x1)] += dd101 * myweight
            weight[int(z1),int(y1),int(x0)] += dd110 * myweight
            weight[int(z1),int(y1),int(x1)] += dd111 * myweight

    return

def reconstruct(f3d, weight, debug=False):
    # Fconv = fft.ifftshift(f3d)
    # Fweight = fft.ifftshift(weight)
    boxsize = f3d.shape
    radius = np.zeros(3, dtype=int)
    for i in range(3):
        radius[i] = math.floor(boxsize[i] / 2) - 1

    rmax_out: int = radius.min()
    rmax_ref: int = rmax_out
    rmax_ref_2: int = rmax_ref * rmax_ref
    starting = np.array([1 - boxsize[0]/2, 1 - boxsize[1]/2, 1 - boxsize[2]/2], dtype=int)

    radavg_weight = np.zeros(int(np.ceil(rmax_out)))
    counter = np.zeros(int(np.ceil(rmax_out)))
    round_rmax_ref_2 = np.round(rmax_ref_2)
    for k in range(boxsize[0]):
        kp = k + starting[0]
        for j in range(boxsize[1]):
            jp = j + starting[1]
            for i in range(boxsize[2]):
                ip = i + starting[2]

                r2 = (kp -0.5)*(kp -0.5) + (jp -0.5)*(jp -0.5) + (ip -0.5)*(ip -0.5)
                if (r2 < rmax_ref_2):
                    ires = np.floor(np.sqrt(r2))
                    if ires >= radavg_weight.shape[0]:
                        continue
                    radavg_weight[int(ires)] += weight[k,j,i]
                    counter[int(ires)] += 1.

    for i in range(radavg_weight.shape[0]):
        if counter[i] > 0 or radavg_weight[i] > 0:
            radavg_weight[i] /= 1000 * counter[i]

    for k in range(boxsize[0]):
        kp = k + starting[0]
        for j in range(boxsize[1]):
            jp = j + starting[1]
            for i in range(boxsize[2]):
                ip = i + starting[2]

                r2 = (kp -0.5)*(kp -0.5) + (jp -0.5)*(jp -0.5) + (ip -0.5)*(ip -0.5)
                ires = np.floor(np.sqrt(r2))
                iresp = ires
                if ires >= rmax_out:
                    iresp = rmax_out - 1
                myweight = np.max([weight[k,j,i], radavg_weight[int(iresp)]])

                if myweight != 0:
                    f3d[k,j,i] /= myweight

    return copy.deepcopy(f3d)
