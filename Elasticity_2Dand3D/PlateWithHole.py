'''
Implements the plate with hole (Kirsch) problem
See Example 4.1 in http://dx.doi.org/10.1016/j.cma.2017.08.032
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams.update({'font.size': 8})
import time
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
np.random.seed(1234)
tf.set_random_seed(1234)

from utils.Geom import Geometry2D
from utils.PINN import Elasticity2D

class PlateWHole(Geometry2D):
    '''
     Class for definining a plate with a hole domain in the 2nd quadrant   
            (C^0 parametrization with repeated knot in u direction)      
     Input: rad_int - radius of the hole
            lenSquare - length of the modeled plate
    '''
    def __init__(self, radInt, lenSquare):            
        
        geomData = dict()
         
        # Set degrees
        geomData['degree_u'] = 2
        geomData['degree_v'] = 1
        
        # Set control points
        geomData['ctrlpts_size_u'] = 5
        geomData['ctrlpts_size_v'] = 2
                
        geomData['ctrlpts'] = [[-1.*radInt,0.,0.],
           [-1.*lenSquare, 0., 0.],
           [-0.853553390593274*radInt, 0.353553390593274*radInt, 0.],
           [-1.*lenSquare, 0.5*lenSquare, 0.],
           [-0.603553390593274*radInt, 0.603553390593274*radInt, 0.],
           [-1.*lenSquare, 1.*lenSquare, 0.],
           [-0.353553390593274*radInt, 0.853553390593274*radInt, 0.],
           [-0.5*lenSquare, 1.*lenSquare, 0.],
           [0, 1.*radInt, 0],
           [0., 1.*lenSquare, 0.]] 
        
        geomData['weights'] = [1., 1., 0.853553390593274, 1., 0.853553390593274, 1., 
           0.853553390593274, 1., 1., 1.]
        
        # Set knot vectors
        geomData['knotvector_u'] = [0., 0., 0., 0.5, 0.5, 1., 1., 1.]
        geomData['knotvector_v'] = [0., 0., 1., 1.]              

        super().__init__(geomData)


class PINN_PH(Elasticity2D):
    '''
    Class including (symmetry) boundary conditions for the plate with hole problem
    '''       
    def __init__(self, model_data, model_pts, NN_param, scaler):
        #PhysicsInformedNN.__init__(self, model_data, model_pts, NN_param)
        super().__init__(model_data, model_pts, NN_param, scaler)
       
    def net_uv(self, x, y, scaler):

        X = tf.concat([x, y], 1)      

        uv = self.neural_net(X,self.weights,self.biases,scaler)

        u = x*uv[:, 0:1]
        v = y*uv[:, 1:2]

        return u, v

def cart2pol(x, y):
    rho = np.sqrt(np.array(x)**2 + np.array(y)**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
    
def getExactDisplacements(x,y,model):
    r, th = cart2pol(x,y)

    E = model['E']
    nu = model['nu']
    R = model['radInt']
    tx = model['P']
    
    u_exact = (1+nu)/E*tx*(1/(1+nu)*r*np.cos(th)+2*R**2/((1+nu)*r)*np.cos(th)+ \
        R**2/(2*r)*np.cos(3*th)-R**4/(2*r**3)*np.cos(3*th))
    v_exact = (1+nu)/E*tx*(-nu/(1+nu)*r*np.sin(th)-(1-nu)*R**2/((1+nu)*r)*np.sin(th)+ \
        R**2/(2*r)*np.sin(3*th)-R**4/(2*r**3)*np.sin(3*th))
                             
    return u_exact, v_exact

def getExactStresses(x,y,model):
    numPts = len(x)
    sigma_xx = np.zeros_like(x)
    sigma_yy = np.zeros_like(x)
    sigma_xy = np.zeros_like(x)
    for i in range(numPts):
        r, th = cart2pol(x[i],y[i])
    
        R = model['radInt']
        tx = model['P']
        
        stressrr = tx/2*(1-R**2/r**2)+tx/2*(1-4*R**2/r**2+3*R**4/r**4)*np.cos(2*th)
        stresstt = tx/2*(1+R**2/r**2)-tx/2*(1+3*R**4/r**4)*np.cos(2*th)
        stressrt = -tx/2*(1+2*R**2/r**2-3*R**4/r**4)*np.sin(2*th)
        
        A = np.array([[np.cos(th)**2, np.sin(th)**2, 2*np.sin(th)*np.cos(th)],
                       [np.sin(th)**2, np.cos(th)**2, -2*np.sin(th)*np.cos(th)],
                       [-np.sin(th)*np.cos(th), np.sin(th)*np.cos(th), 
                        np.cos(th)**2-np.sin(th)**2]])
                
        stress = np.linalg.solve(A,np.array([stressrr,stresstt,stressrt]))
        sigma_xx[i] = stress[0]
        sigma_yy[i] = stress[1]
        sigma_xy[i] = stress[2]
                             
    return sigma_xx, sigma_yy, sigma_xy

def getExactTraction(x, y, xNorm, yNorm, model):    
    sigma_xx, sigma_yy, sigma_xy = getExactStresses(x[:,0], y[:,0], model)
    sigma_xx = np.expand_dims(sigma_xx, axis=1)
    sigma_yy = np.expand_dims(sigma_yy, axis=1)
    sigma_xy = np.expand_dims(sigma_xy, axis=1)
    trac_x = xNorm[:,0:1]*sigma_xx + yNorm[:,0:1]*sigma_xy
    trac_y = xNorm[:,0:1]*sigma_xy + yNorm[:,0:1]*sigma_yy
    
    return trac_x, trac_y

def plotDeformedDisp(surfaceUx, surfaceUy, defShapeX, defShapeY):
    fig, axes = plt.subplots(ncols=2)
    cs1 = axes[0].contourf(defShapeX, defShapeY, surfaceUx, 255, cmap=plt.cm.jet)    
    cs2 = axes[1].contourf(defShapeX, defShapeY, surfaceUy, 255, cmap=plt.cm.jet)
    #plot equal colorbars as in 
    #https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    fig.colorbar(cs1, ax=axes[0], fraction=0.046, pad=0.04, format='%.2E')
    fig.colorbar(cs2, ax=axes[1], fraction=0.046, pad=0.04, format='%.2E')

    # axes[0].set_title("Displacement in x")
    # axes[1].set_title("Displacement in y")
    axes[0].set_title("u-displacement")
    axes[1].set_title("v-displacement")
    fig.tight_layout()
    for tax in axes:
        tax.set_xlabel('$x$')
        tax.set_ylabel('$y$')
        tax.set_aspect('equal')    
    plt.show()
    
def plotErrDisp(surfaceUx, surfaceUy, defShapeX, defShapeY):
    levels = np.linspace(0, 0.05, 10)
    fig, axes = plt.subplots(ncols=2)
    cs1 = axes[0].contourf(defShapeX, defShapeY, surfaceUx, 255, cmap=plt.cm.jet, levels = levels, extend = 'both')    
    cs2 = axes[1].contourf(defShapeX, defShapeY, surfaceUy, 255, cmap=plt.cm.jet, levels = levels, extend = 'both')
    #plot equal colorbars as in 
    #https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    fig.colorbar(cs1, ax=axes[0], fraction=0.046, pad=0.04, format='%.2E')
    fig.colorbar(cs2, ax=axes[1], fraction=0.046, pad=0.04, format='%.2E')

    # axes[0].set_title("Displacement in x")
    # axes[1].set_title("Displacement in y")
    axes[0].set_title("u-relative $L_1$ error")
    axes[1].set_title("v-relative $L_1$ error")
    fig.tight_layout()
    for tax in axes:
        tax.set_xlabel('$x$')
        tax.set_ylabel('$y$')
        tax.set_aspect('equal')  
    plt.show()        

def plotErrDisp_Origin(surfaceUx, surfaceUy, defShapeX, defShapeY):
    levels = np.linspace(0, 1, 10)
    fig, axes = plt.subplots(ncols=2)
    cs1 = axes[0].contourf(defShapeX, defShapeY, surfaceUx, 255, cmap=plt.cm.jet, levels = levels, extend = 'both')    
    cs2 = axes[1].contourf(defShapeX, defShapeY, surfaceUy, 255, cmap=plt.cm.jet, levels = levels, extend = 'both')
    
    # fig, axes = plt.subplots(ncols=2)
    # cs1 = axes[0].contourf(defShapeX, defShapeY, surfaceUx, 255, cmap=plt.cm.jet)    
    # cs2 = axes[1].contourf(defShapeX, defShapeY, surfaceUy, 255, cmap=plt.cm.jet)
    #plot equal colorbars as in 
    #https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    fig.colorbar(cs1, ax=axes[0], fraction=0.046, pad=0.04, format='%.2E')
    fig.colorbar(cs2, ax=axes[1], fraction=0.046, pad=0.04, format='%.2E')

    # axes[0].set_title("Displacement in x")
    # axes[1].set_title("Displacement in y")
    axes[0].set_title("u-relative $L_1$ error")
    axes[1].set_title("v-relative $L_1$ error")
    fig.tight_layout()
    for tax in axes:
        tax.set_xlabel('$x$')
        tax.set_ylabel('$y$')
        tax.set_aspect('equal')  
    plt.show()  
    
model_data = dict()
model_data['E'] = 1e5
model_data['nu'] = 0.3

model = dict()
model['E'] = model_data['E']
model['nu'] = model_data['nu']
model['radInt'] = 1.0
model['lenSquare'] = 4.0
model['P'] = 0.5

# Domain bounds
model['lb'] = np.array([-model['lenSquare'],0.]) #lower bound of the plate
model['ub'] = np.array([0.,model['lenSquare']]) # Upper bound of the plate

NN_param = dict()
NN_param['layers'] = [2, 30, 30, 30, 2]
NN_param['data_type'] = tf.float32

# Generating points inside the domain using Geometry class
myPlate = PlateWHole(model['radInt'], model['lenSquare'])

numElemU = 80
numElemV = 80
numGauss = 1

xPhys, yPhys, wgtsPhys = myPlate.getQuadIntPts(numElemU, numElemV, numGauss)
myPlate.plotKntSurf()
plt.scatter(xPhys, yPhys, s=0.5)
X_f = np.concatenate((xPhys,yPhys, wgtsPhys),axis=1)

# Generate the boundary points using Geometry class
numElemEdge = 80
numGaussEdge = 1
xEdge, yEdge, xNormEdge, yNormEdge, wgtsEdge = myPlate.getQuadEdgePts(numElemEdge,
                                                            numGaussEdge, 3)
plt.scatter(xEdge, yEdge, s=1, c='red', zorder=10 )
#plt.quiver(xEdge, yEdge, xNormEdge, yNormEdge)
plt.show()


trac_x, trac_y = getExactTraction(xEdge, yEdge, xNormEdge, yNormEdge, model)

xEdgePts = np.concatenate((xEdge, yEdge, wgtsEdge, trac_x, trac_y), axis=1)

model_pts = dict()
model_pts['X_int'] = X_f
model_pts['X_bnd'] = xEdgePts

time_run = 1
FileRec = 1
data_RE_fu = np.zeros((time_run, 4))
data_error = np.zeros((time_run, 6))
nPred = 40

scaler = np.float32([1e-5, 1e0*1.40/4.32*1e-5]) # np.float32([1e-5, 1e0*1.40/4.32*1e-5]) for nEPINN in all cases, np.float32([1, 1]) for EPINN

for t in range(time_run):
    modelNN = PINN_PH(model_data, model_pts, NN_param, scaler)
    
    start_time = time.time()
    num_train_its = 5000
    modelNN.train(num_train_its)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    
    withEdges = [1, 1, 1, 1]
    xGrid, yGrid = myPlate.getUnifIntPts(nPred, nPred, withEdges)
    Grid = np.concatenate((xGrid,yGrid),axis=1)
    
    u_pred, v_pred, energy_pred, sigma_x_pred, sigma_y_pred, tau_xy_pred = modelNN.predict(Grid)
    
    # Computing exact displacements
    u_exact, v_exact = getExactDisplacements(xGrid, yGrid, model)    
    
    errU = u_exact - u_pred
    errV = v_exact - v_pred
    errUsq = errU**2
    errUsq_norm = errUsq.sum()
    errVsq = errV**2
    errVsq_norm = errVsq.sum()
    exU = u_exact**2
    exU_norm = exU.sum()
    exV = v_exact**2
    exV_norm = exV.sum()
    error_u = np.sqrt(errUsq_norm/exU_norm)
    error_v = np.sqrt(errVsq_norm/exV_norm)
    print("Relative L2 error of u: ", error_u)
    print("Relative L2 error of v: ", error_v)
    
    errUsq = np.abs(errU)
    errUsq_l1norm = errUsq.sum()
    errVsq = np.abs(errV)
    errVsq_l1norm = errVsq.sum()
    exU = np.abs(u_exact)
    exU_l1norm = exU.sum()
    exV = np.abs(v_exact)
    exV_l1norm = exV.sum()
    error_u_l1 = np.sqrt(errUsq_l1norm/exU_l1norm)
    error_v_l1 = np.sqrt(errVsq_l1norm/exV_l1norm)
    print("Relative L1 error of u: ", error_u_l1)
    print("Relative L1 error of v: ", error_v_l1)
    
    exU_abs = np.absolute(u_exact)
    exU_normL1 = exU_abs.sum()
    exV_abs = np.absolute(v_exact)
    exV_normL1 = exV_abs.sum()
    print("L1 Norm of u: ", exU_normL1)
    print("L1 Norm of v: ", exV_normL1)

    #Compute the L2 and energy norm errors using integration
    u_pred_Int, v_pred_Int, energy_pred_Int, sigma_x_pred_Int, sigma_y_pred_Int, tau_xy_pred_Int = modelNN.predict(X_f)
    u_exact_Int, v_exact_Int = getExactDisplacements(X_f[:,0], X_f[:,1], model)
    err_l2 = np.sum(((u_exact_Int-u_pred_Int[:,0])**2 + (v_exact_Int-v_pred_Int[:,0])**2)*X_f[:,2])
    norm_l2 = np.sum((u_exact_Int**2 + v_exact_Int**2)*X_f[:,2])
    error_u_l2 = np.sqrt(err_l2/norm_l2)
    print("Relative L2 error (integration): ", error_u_l2)
    
    sigma_xx_exact, sigma_yy_exact, sigma_xy_exact = getExactStresses(X_f[:,0], X_f[:,1], model)
    sigma_xx_err = sigma_xx_exact - sigma_x_pred_Int[:,0]
    sigma_yy_err = sigma_yy_exact - sigma_y_pred_Int[:,0]
    sigma_xy_err = sigma_xy_exact - tau_xy_pred_Int[:,0]
    
    energy_err = 0
    energy_norm = 0
    numPts = numElemU*numElemV*numGauss

    C_mat = np.zeros((3,3))
    C_mat[0,0] = model['E']/(1-model['nu']**2)
    C_mat[1,1] = model['E']/(1-model['nu']**2)
    C_mat[0,1] = model['E']*model['nu']/(1-model['nu']**2)
    C_mat[1,0] = model['E']*model['nu']/(1-model['nu']**2)
    C_mat[2,2] = model['E']/(2*(1+model['nu']))
    C_inv = np.linalg.inv(C_mat)
    for i in range(numPts):
        err_pt = np.array([sigma_xx_err[i],sigma_yy_err[i],sigma_xy_err[i]])
        norm_pt = np.array([sigma_xx_exact[i],sigma_yy_exact[i],sigma_xy_exact[i]])
        energy_err = energy_err + err_pt@C_inv@err_pt.T*X_f[i,2]
        energy_norm = energy_norm + norm_pt@C_inv@norm_pt.T*X_f[i,2]
        
    print("Relative energy error (integration): ", np.sqrt(energy_err/energy_norm))
    print("Energy error (integration): ", np.sqrt(energy_err))
    
    EnergyInt_exact = 0
    EnergyInt_pred = 0
    
    for i in range(numPts):
        EnergyInt_exact = EnergyInt_exact + 0.5*(sigma_xx_exact[i]*u_exact_Int[i] + sigma_xx_exact[i]*v_exact_Int[i] + \
                        sigma_yy_exact[i]*u_exact_Int[i] + sigma_yy_exact[i]*v_exact_Int[i] + \
                        sigma_xy_exact[i]*u_exact_Int[i] + sigma_xy_exact[i]*v_exact_Int[i])*X_f[i,2]/model['lenSquare']
        EnergyInt_pred = EnergyInt_pred + 0.5*(sigma_x_pred_Int[i]*u_pred_Int[i] + sigma_x_pred_Int[i]*v_pred_Int[i] + \
                        sigma_y_pred_Int[i]*u_pred_Int[i] + sigma_y_pred_Int[i]*v_pred_Int[i] + \
                        tau_xy_pred_Int[i]*u_pred_Int[i] + tau_xy_pred_Int[i]*v_pred_Int[i])*X_f[i,2]/model['lenSquare']

    # TracInt_exact = 0
    # TracInt_pred = 0
    # for i in range(numElemEdge):
    #     TracInt_exact = TracInt_exact + (trac_x[i]*u_exact[i] + trac_y[i]*v_exact[i])*wgtsEdge[i]
    #     TracInt_pred = TracInt_pred + (trac_x[i]*u_pred[i] + trac_y[i]*v_pred[i])*wgtsEdge[i]
    
    # TPE_L1Err = np.abs(EnergyInt_pred  - TracInt_pred - EnergyInt_exact + TracInt_pred)
    
    TPE_L1Err = np.abs(EnergyInt_pred - EnergyInt_exact)
    print("Exact strain energy: ", EnergyInt_exact)
    print("Absolute strain energy error: ", TPE_L1Err)
    
    if FileRec == 1:
        data_RE_fu[t,0] = EnergyInt_exact
        data_RE_fu[t,1] = EnergyInt_pred
        data_RE_fu[t,2] = TPE_L1Err
        data_RE_fu[t,3] = np.abs(TPE_L1Err/EnergyInt_exact)
        data_error[t,0] = error_u_l1
        data_error[t,1] = error_u
        data_error[t,2] = error_v_l1
        data_error[t,3] = error_v
        data_error[t,4] = exU_normL1
        data_error[t,5] = exV_normL1
        df1 = pd.DataFrame(data_RE_fu)
        df2 = pd.DataFrame(data_error)
        with pd.ExcelWriter('PlatewHole_P1_DNN30x3.xlsx') as writer:
            df1.to_excel(writer, sheet_name='RE_Pi')
            df2.to_excel(writer, sheet_name='Sol_Errs')

# Plot results
# Magnification factors for plotting the deformed shape
x_fac = 2
y_fac = 2

# Compute the approximate displacements at plot points     
oShapeX = np.resize(xGrid, [nPred, nPred])
oShapeY = np.resize(yGrid, [nPred, nPred])
surfaceUx = np.resize(u_pred, [nPred, nPred])
surfaceUy = np.resize(v_pred, [nPred, nPred])
surfaceExUx = np.resize(u_exact, [nPred, nPred])
surfaceExUy = np.resize(v_exact, [nPred, nPred])

defShapeX = oShapeX + surfaceUx * x_fac
defShapeY = oShapeY + surfaceUy * y_fac
surfaceErrUx = np.absolute(surfaceExUx - surfaceUx)/np.absolute(surfaceExUx+1e-10)
surfaceErrUy = np.absolute(surfaceExUy - surfaceUy)/np.absolute(surfaceExUy+1e-10)

print("Deformation plots")
plotDeformedDisp(surfaceUx, surfaceUy, defShapeX, defShapeY)

print("Exact plots")
plotDeformedDisp(surfaceExUx, surfaceExUy, defShapeX, defShapeY)
print("Error plots")
plotErrDisp(surfaceErrUx, surfaceErrUy, oShapeX, oShapeY)
plotErrDisp_Origin(surfaceErrUx, surfaceErrUy, oShapeX, oShapeY)

# Plotting the strain energy densities        
sEnergy = np.resize(energy_pred, [nPred, nPred])
fig = plt.contourf(defShapeX, defShapeY, sEnergy, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Strain Energy Density")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('equal')
plt.show()

print('Loss convergence')    
range_adam = np.arange(1,num_train_its+1)
range_lbfgs = np.arange(num_train_its+2, num_train_its+2+len(modelNN.lbfgs_buffer))
ax0, = plt.plot(range_adam, modelNN.loss_adam_buff, label='Adam')
ax1, = plt.plot(range_lbfgs,  modelNN.lbfgs_buffer, label='L-BFGS')
plt.legend(handles=[ax0,ax1])
plt.ylim((min(modelNN.lbfgs_buffer),1))
plt.xlabel('Iteration')
plt.ylabel('Loss value')
plt.show()