# module projector.py

import sys,math,cmath,os,copy
import numpy as np
from numpy import fft

def linterp(frac, low, high):
    return low + (high - low) * frac

def deg2rad(deg):
    return deg * np.pi / 180.

def getRotationMatrix(A, rot=0, tilt=0, psi=0):
    rot  = deg2rad(rot)
    tilt = deg2rad(tilt)
    psi  = deg2rad(psi)

    ca = np.cos(rot)
    cb = np.cos(tilt)
    cg = np.cos(psi)
    sa = np.sin(rot)
    sb = np.sin(tilt)
    sg = np.sin(psi)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    A[0, 0] =  cg * cc - sg * sa
    A[0, 1] =  cg * cs + sg * ca
    A[0, 2] = -cg * sb
    A[1, 0] = -sg * cc - cg * sa
    A[1, 1] = -sg * cs + cg * ca
    A[1, 2] = sg * sb
    A[2, 0] = sc
    A[2, 1] = ss
    A[2, 2] = cb

    return

def centerFFT(image, forward):

    l = image.shape[1]
    aux = np.zeros(l)
    shift = int(l / 2)

    if not forward:
        shift *= -1

    for i in range(image.shape[0]):
        for j in range(l):
            jp = j + shift
            if jp < 0:
                jp += l
            elif jp >= l:
                jp -= l

            aux[int(jp)] = image[i,j]

        for j in range(l):
            image[i,j] = aux[j]

    l = image.shape[0]
    aux = np.zeros(l)
    shift = int(l / 2)

    if not forward:
        shift *= -1

    for j in range(image.shape[1]):
        for i in range(l):
            ip = i + shift
            if ip < 0:
                ip += l
            elif ip >= l:
                ip -= l

            aux[int(ip)] = image[i,j]

        for i in range(l):
            image[i,j] = aux[i]

    return

def rotate2D(data, rotdata, A3d, debug=False):
    if (data.shape[0] != rotdata.shape[0] or data.shape[1] != rotdata.shape[1]):
        print('Error: input and output array must have the same shape!')
        sys.exit()

    # define rotation matrix (inversede)
    # and correct the scaling by multiplying padding factor
    Ainv = A3d.T
    #Ainv *= int(factor)

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

    rmax_ref: int = rmax_out
    rmax_ref_2: int = rmax_ref * rmax_ref

    if (debug):
        print('rmax_ref_2: ', rmax_ref_2)

    # define starting to access logical coordinates
    starting = np.array([1 - boxsize[0]/2, 1 - boxsize[1]/2], dtype=int)
    
    for i in range(data.shape[0]):
        y = i + starting[0]

        for j in range(data.shape[1]):
            x = j + starting[1]

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

            d00 = data[int(y0),int(x0)]
            d01 = data[int(y0),int(x1)]
            d10 = data[int(y1),int(x0)]
            d11 = data[int(y1),int(x1)]

            dx0 = linterp(fx, d00, d01)
            dx1 = linterp(fx, d10, d11)

            rotdata[i,j] = linterp(fy, dx0, dx1)

    return

def project(vol, A3d, factor=1, debug=False):
    Ainv = A3d.T

    # 0 = z, 1 = y, 2 = x
    boxsize = vol.shape[0]
    padsize = boxsize * factor
    m = int((padsize - boxsize) / 2)

    vol_pad = np.pad(vol, (m,m))

    f3d = fft.fftshift(fft.fftn(fft.fftshift(vol_pad),s=(padsize,padsize,padsize)))
    f2d = np.zeros((padsize,padsize), dtype=complex)

    if debug:
        print('boxsize: ', boxsize)

    radius = np.zeros(3, dtype=int)
    for i in range(3):
        radius[i] = math.floor(padsize / 2) - 1

    if debug:
        print('radius: ', radius)

    rmax_out: int = radius.min()

    rmax_ref: int = rmax_out
    rmax_ref_2: int = rmax_ref * rmax_ref

    # define starting to access logical coordinates
    starting = np.array([1 - padsize/2, 1 - padsize/2, 1 - padsize/2], dtype=int)

    for i in range(f2d.shape[0]):
        y = i + starting[0]

        for j in range(f2d.shape[1]):
            x = j + starting[1]

            xu = x - 0.5
            yu = y - 0.5

            xp = Ainv[0,0] * xu + Ainv[0,1] * yu
            yp = Ainv[1,0] * xu + Ainv[1,1] * yu
            zp = Ainv[2,0] * xu + Ainv[2,1] * yu

            r_ref_2 = xp * xp + yp * yp + zp * zp
            if (r_ref_2 > rmax_ref_2):
                continue

            z0 = np.floor(zp + 0.5)
            fz = zp + 0.5 - z0
            z0 -= starting[0]
            z1:int = z0 + 1

            y0 = np.floor(yp + 0.5)
            fy = yp + 0.5 - y0
            y0 -= starting[1]
            y1:int = y0 + 1

            x0:int = np.floor(xp + 0.5)
            fx = xp + 0.5 - x0
            x0 -= starting[2]
            x1 = x0 + 1

            if x0 < 0 or x1 > f3d.shape[2] or y0 < 0 or y1 > f3d.shape[1] or z0 < 0 or z1 > f3d.shape[0]:
                continue

            d000 = f3d[int(z0),int(y0),int(x0)]
            d001 = f3d[int(z0),int(y0),int(x1)]
            d010 = f3d[int(z0),int(y1),int(x0)]
            d011 = f3d[int(z0),int(y1),int(x1)]
            d100 = f3d[int(z1),int(y0),int(x0)]
            d101 = f3d[int(z1),int(y0),int(x1)]
            d110 = f3d[int(z1),int(y1),int(x0)]
            d111 = f3d[int(z1),int(y1),int(x1)]

            dx00 = linterp(fx, d000, d001)
            dx01 = linterp(fx, d010, d011)
            dx10 = linterp(fx, d100, d101)
            dx11 = linterp(fx, d110, d111)
            dxy0 = linterp(fy, dx00, dx01)
            dxy1 = linterp(fy, dx10, dx11)

            f2d[i,j] = linterp(fz, dxy0, dxy1)

    img_pad = fft.ifftshift(fft.ifft2(fft.ifftshift(f2d))).real

    return copy.deepcopy(img_pad[m:boxsize+m,m:boxsize+m])

# projection in real-space
def rproject(vol, image, a3d):

    debug = False

    # inverse the input rotation matrix a3d (a3d^-1 = a3d^T)
    # 余裕のある人はなぜ逆行列にするか考えてみてください
    ainv = a3d.T

    if debug:
        print('a3d_inv:')
        print(ainv)
    
    # 0 = z, 1 = y, 2 = x
    boxsize = vol.shape

    # check x-y slice of vol and image have the same shape
    if (boxsize[1] != image.shape[0] or boxsize[2] != image.shape[1]):
        print('Error: output image has different shape from that of input volume')
        sys.exit()

    # output 2d image (image_out) will be always initialised
    image[:] = 0

    if debug:
        print('boxsize: ', boxsize)

    # boxの内接球の領域だけを計算する
    # そのための最大半径を定義
    _radius = np.zeros(3, dtype=int)
    for i in range(3):
        _radius[i] = math.floor(boxsize[i] / 2) - 1

    if debug:
        print('_radius: ', _radius)

    _rmax_out: int = _radius.min()
    _rmax_out_2: int = _rmax_out * _rmax_out;

    if debug:
        print('_rmax_out: ', _rmax_out)
        print('_rmax_out_2: ', _rmax_out_2)
    
    # loop for z-axis component
    for k in range(boxsize[0]):

        z: int = 0
        
        # boxsizeが奇数かどうかで場合分け
        if boxsize[0] % 2 == 0:
            z = k - _radius[0] if k <= _radius[0] else k - (_radius[0] + 1)
        else:
            z = k - (_radius[0] + 1)

        z2 = z * z

        if z2 > _rmax_out_2:
            continue
        
        # ymax depends on current z
        ymax = math.floor(math.sqrt(_rmax_out_2 - z2))

        # loop for y-axis component
        for j in range(boxsize[1]):

            y: int = 0
            
            # boxsizeが奇数かどうかで場合分け
            if boxsize[1] % 2 == 0:
                y = j - _radius[1] if j <= _radius[1] else j - (_radius[1] + 1)
            else:
                y = j - (_radius[1] + 1)

            y2 = y * y

            if (y2 > ymax * ymax):
                continue
        
            # xmax depends on current z and y
            xmax = math.floor(math.sqrt(_rmax_out_2 - z2 - y2)) 

            # loop for x-axis component
            for i in range(boxsize[2]):

                x: int = 0
            
                # boxsizeが奇数かどうかで場合分け
                if boxsize[2] % 2 == 0:
                    x = i - _radius[2] if i <= _radius[2] else i - (_radius[2] + 1)
                else:
                    x = i - (_radius[2] + 1)

                if x * x > xmax * xmax:
                    continue

                #  Get logical coordinates in the 3D map
                xp = ainv[0,0] * x + ainv[0,1] * y + ainv[0,2] * z
                yp = ainv[1,0] * x + ainv[1,1] * y + ainv[1,2] * z
                zp = ainv[2,0] * x + ainv[2,1] * y + ainv[2,2] * z

                # guarantee logical coordinates are always within rmax_out shell
                if (xp * xp + yp * yp + zp * zp) > _rmax_out_2:
                    continue

                # define nearest gridding coordinates
                x0 = math.floor(xp)
                fx = xp - x0
                if boxsize[0] % 2 == 0:
                    x0 += _radius[0]
                else:
                    x0 += _radius[0] + 1                    
                x1 = x0 + 1

                y0 = math.floor(yp)
                fy = yp - y0
                if boxsize[1] % 2 == 0:
                    y0 += _radius[1]
                else:
                    y0 += _radius[1] + 1
                y1 = y0 + 1

                z0 = math.floor(zp)
                fz = zp - z0
                if boxsize[2] % 2 == 0:
                    z0 += _radius[2]
                else:
                    z0 += _radius[2] + 1
                z1 = z0 + 1

                if x0 < 0 or x0+1 >= boxsize[0] or y0 < 0 or y0+1 >= boxsize[1] or z0 < 0 or z0+1 >= boxsize[2]:
                    continue

                v000 = vol[z0,y0,x0]
                v001 = vol[z0,y0,x1]
                v010 = vol[z0,y1,x0]
                v011 = vol[z0,y1,x1]
                v100 = vol[z1,y0,x0]
                v101 = vol[z1,y0,x1]
                v110 = vol[z1,y1,x0]
                v111 = vol[z1,y1,x1]

                # set the interpolated value in the 3D output array
                # x-axis
                vx00 = linterp(fx, v000, v001)
                vx01 = linterp(fx, v100, v101)
                vx10 = linterp(fx, v010, v011)
                vx11 = linterp(fx, v110, v111)

                # y-axis
                vxy0 = linterp(fy, vx00, vx10)
                vxy1 = linterp(fy, vx01, vx11)

                # z-axis (and final value)
                image[j,i] += linterp(fz, vxy0, vxy1)
    
    return
