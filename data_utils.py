import json
import numpy as np
import scipy.sparse.linalg as spsla
import scipy.sparse as sps


def get_stokes_solution(mats, Re, control, palpha=1, ct0=0):
    """"Compute Stokes solution."""
    J = mats['J']
    fp = mats['fp'] + mats['fp_div']
    NV = J.shape[1]
    NP = J.shape[0]

    if control == 'bc':
        fvstks = mats['fv'] + 1./Re*mats['fv_diff'] + ct0
        Astks = 1./Re*mats['A'] + 1./palpha*mats['Arob']
    else:
        fvstks = mats['fv'] + 1./Re*mats['fv_diff']
        Astks = 1./Re*mats['A']

    stksrhs = np.vstack([fvstks, fp])
    stksmat = sps.vstack([
                    sps.hstack([Astks, -J.T]),
                    sps.hstack([J, sps.csc_matrix((NP, NP))])
                ]).tocsc()
    stksvp = spsla.spsolve(stksmat, stksrhs).reshape((NV+NP, 1))
    stksv = stksvp[:NV].reshape((NV, 1))
    stksp = stksvp[NV:].reshape((NP, 1))
    return stksv, stksp


def solve_steadystate_nse(mats, Re, control, tol=1e-10, npicardstps=5, maxit=30,
                          palpha=1, uvec=np.array([[0], [0]]), v0=None):
    """Compute steady-state NSE solution."""
    J = mats['J']
    hmat = mats['H']
    fp = mats['fp'] + mats['fp_div']
    if control == 'dist':
        A = 1. / Re * mats['A'] + mats['L1'] + mats['L2']
        fv = mats['fv'] + 1./Re*mats['fv_diff'] + mats['fv_conv']
    else:
        A = 1./Re*mats['A'] + mats['L1'] + mats['L2'] + 1./palpha*mats['Arob']
        Brob = mats['Brob']
        fv = mats['fv'] + 1./Re*mats['fv_diff'] + mats['fv_conv'] \
                        + 1. / palpha*np.dot(Brob, uvec)

    updnorm = 1
    NV, NP = fv.shape[0], fp.shape[0]
    if v0 is None:
        curv = np.zeros((NV, 1))
    else:
        curv = v0
    stpcount = 0
    while updnorm > tol:
        picard = stpcount < npicardstps
        H1k, H2k = linearized_convection(hmat, curv, retparts=True)
        if picard:
            currhs = np.vstack([fv, fp])
            HL = H1k
        else:
            currhs = np.vstack([fv+eva_quadterm(hmat, curv), fp])
            HL = H1k + H2k
        cursysmat = sps.vstack([
                        sps.hstack([A+HL, -J.T]),
                        sps.hstack([J, sps.csc_matrix((NP, NP))])
                    ]).tocsc()
        nextvp = spsla.spsolve(cursysmat, currhs).reshape((NV+NP, 1))

        print('Iteration step {0} ({1})'.format(stpcount, 'Picard' if picard else 'Newton'))
        nextv = nextvp[:NV].reshape((NV, 1))
        nextp = nextvp[NV:].reshape((NP, 1))
        curnseres = A*nextv + eva_quadterm(hmat, nextv) - J.T*nextp - fv
        print('Norm of nse residual:   {0:e}'.format(np.linalg.norm(curnseres)))
        updnorm = np.linalg.norm(nextv - curv) / np.linalg.norm(nextv)
        print('Norm of current update: {0:e}'.format(updnorm))
        print('\n')
        curv = nextv
        stpcount += 1
        if stpcount > maxit:
            raise RuntimeError('Could not compute steady-state NSE solution.')

    return nextv, nextp


def linearized_convection(H, linv, retparts=False):
    """Compute linearized convection term."""
    nv = linv.size
    # H1 = np.zeros((nv, nv))
    # H2 = np.zeros((nv, nv))
    if retparts:
        H1 = H * (sps.kron(sps.eye(nv), linv))
        H2 = H * (sps.kron(linv, sps.eye(nv)))
        # for k in range(nv):
        #     H1[:, k] = (H[:, k*nv:k*nv + nv] @ linv)[:, 0]
        #     H2[:, k] = (H[:, [x+k for x in range(0, nv*nv, nv)]] @ linv)[:, 0]
        return H1, H2
    else:
        H1 = H * (sps.kron(sps.eye(nv), linv))
        H2 = H * (sps.kron(linv, sps.eye(nv)))
        # for k in range(nv):
        #     H[:, k] = ((H[:, k*nv:k*nv + nv]
        #             + H[:, [x+k for x in list(range(0, nv*nv, nv))]]) @ linv)[:, 0]
        return H1 + H2


def writevp_paraview(velvec=None, pvec=None, strtojson=None, visudict=None,
                     vfile='vel__.vtu', pfile='p__.vtu'):
    if visudict is None:
        jsfile = open(strtojson)
        visudict = json.load(jsfile)
        vaux = np.zeros((visudict['vdim'], 1))
        for bcdict in visudict['bclist']:
            intbcidx = [int(bci) for bci in bcdict.keys()]
            vaux[intbcidx, 0] = list(bcdict.values())
        vaux[visudict['invinds']] = velvec

    vxvtxdofs = visudict['vxvtxdofs']
    vyvtxdofs = visudict['vyvtxdofs']

    with open(vfile, 'w') as velfile:
        velfile.write(visudict['vtuheader_v'])
        for xvtx, yvtx in zip(vxvtxdofs, vyvtxdofs):
            velfile.write(u'{0} {1} {2} '.format(vaux[xvtx][0], vaux[yvtx][0], 0.))
        velfile.write(visudict['vtufooter_v'])

    if pvec is not None:
        pvtxdofs = visudict['pvtxdofs']
        with open(pfile, 'w') as prefile:
            prefile.write(visudict['vtuheader_p'])
            for pval in pvec[pvtxdofs, 0]:
                prefile.write(u'{0} '.format(pval))
            prefile.write(visudict['vtufooter_p'])


def collect_vtu_files(filelist, pvdfilestr):
    with open(pvdfilestr, 'w') as colfile:
        colfile.write(u'<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1"> <Collection>\n')
        for tsp, vtufile in enumerate(filelist):
            dtst = u'<DataSet timestep="{0}" part="0" file="{1}"/>'.format(tsp, vtufile)
            colfile.write(dtst)

        colfile.write(u'</Collection> </VTKFile>')


def eva_quadterm(H, v):
    ''' function to evaluate `H*kron(v, v)` without forming `kron(v, v)`

    Parameters:
    ---
    H : (nv, nv*nv) sparse array
        the tensor (as a matrix) that evaluates the convection term

    '''

    NV = v.size
    hvv = np.zeros((NV, 1))
    for k, vi in enumerate(v):
        hviv = H[:, k*NV:(k+1)*NV]*(vi[0]*v)
        hvv = hvv + hviv
    return np.array(hvv)
