"""
EPINNs for 1D linear Timoshenko beam bending problem
Author: Thang Le-Duc
Email: le.duc.thang0312@gmail.com; thangld@sju.ac.kr

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from pyDOE import lhs
import time
import pandas as pd
from matplotlib import rcParams
rcParams['text.usetex'] = True
# from openpyxl import Workbook
import statistics

# make figures bigger on HiDPI monitors
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

tf.reset_default_graph()  # To clear the defined variables and operations of the previous cell
np.random.seed(0)
tf.set_random_seed(0)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, lb, u_lb, rb, u_rb, X_f, layers,layers1,E,I,phi_left, phi_right,w_gau,scalerU,scalerPhi):

        self.lb = lb
        self.rb = rb

        self.x_lb = lb
        self.x_rb = rb
        self.u_lb = u_lb

        self.u_rb = u_rb
        self.w_gau = w_gau
        self.x_f = X_f
        self.phi_left = phi_left
        self.phi_right = phi_right
        # self.k = k
        self.E = E
        self.I = I
        # self.q = q
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        self.layers1 = layers1
        self.weights1, self.biases1 = self.initialize_NN(layers1)

        # tf Placeholders
        self.lb_tf = tf.placeholder(tf.float64)
        self.rb_tf = tf.placeholder(tf.float64)
        self.x_lb_tf = tf.placeholder(tf.float64)
        self.u_lb_tf = tf.placeholder(tf.float64)
        self.x_rb_tf = tf.placeholder(tf.float64)
        self.u_rb_tf = tf.placeholder(tf.float64)

        self.w_gau_tf = tf.placeholder(tf.float64)
        self.x_f_tf = tf.placeholder(tf.float64)
        self.phi_left_tf = tf.placeholder(tf.float64)
        self.phi_right_tf = tf.placeholder(tf.float64)
        
        # tf Graphs
        self.u_lb_pred, self.u_x_lb_pred = self.net_u(self.x_lb_tf,scalerU)
        self.u_rb_pred, self.u_x_rb_pred = self.net_u(self.x_rb_tf,scalerU)
        self.u_f_pred, self.u_x_f_pred = self.net_u(self.x_f_tf,scalerU)
        self.phi_lb_pred, self.phi_x_lb_pred = self.net_phi(self.x_lb_tf,scalerPhi)
        self.phi_rb_pred, self.phi_x_rb_pred = self.net_phi(self.x_rb_tf,scalerPhi)
        self.phi_f_pred, self.phi_x_f_pred = self.net_phi(self.x_f_tf,scalerPhi)

        self.f_u_pred, self.SE_u_pred = self.net_f_u(self.x_f_tf,scalerU,scalerPhi)
        # Loss phat #10e8
        self.loss = tf.reduce_sum(self.f_u_pred * self.w_gau_tf) * (self.rb_tf - self.lb_tf) * 0.5

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 10000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
        self.lbfgs_buffer = []
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64),
                           dtype=tf.float64)

    def neural_net(self, X, weights, biases, G):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.rb - self.lb) - 1.0
        # use tf.multiply for the first iteration/layers
        W = weights[0]
        b = biases[0]
        H = tf.nn.tanh(tf.add(tf.multiply(H, W), b))
        # H = tf.math.sin(tf.add(tf.multiply(H, W), b))
        # H = tf.sigmoid(tf.add(tf.multiply(H, W), b))
        # use tf.matmul for the remanining layers
        for l in range(1, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            # H = tf.math.sin(tf.add(tf.matmul(H, W), b))
            # H = tf.sigmoid(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        Y = tf.math.multiply(G,Y)
        return Y

    def net_u(self, x, scaler):
        uphi = self.neural_net(x, self.weights, self.biases, scaler)   # np.float64([1e0*4.88e-6/1.45e-6*1e-2])
        u = uphi
        u=x*u*(self.rb-x)
        u_x = tf.gradients(u, x)[0]
        # print('u',u)
        return u, u_x

    def net_phi(self, x, scaler):
        uphi = self.neural_net(x, self.weights1, self.biases1, scaler)
        phi = uphi
        phi=phi*x*(self.rb-x)
        phi_x = tf.gradients(phi, x)[0]
        # print('phi', phi)
        return phi, phi_x

    def net_f_u(self, x, scalerU, scalerPhi):
        u, u_x = self.net_u(x, scalerU)
        print('u', u)
        phi, phi_x=self.net_phi(x, scalerPhi)
        print('phi', phi)
        f_u=0.5*(k*G*A*(u_x+phi)**2+E*I*(phi_x)**2)-u*q
        SE_u=0.5*(k*G*A*(u_x+phi)**2+E*I*(phi_x)**2)
        return f_u, SE_u

    def callback(self, loss):
        self.lbfgs_buffer = np.append(self.lbfgs_buffer, loss)
        print('Loss:', loss)

    def train(self, nIter):
        self.loss_adam_buff = np.zeros(nIter)
        tf_dict = {self.x_lb_tf: self.x_lb,
                   self.u_lb_tf: self.u_lb,
                   self.x_rb_tf: self.x_rb,
                   self.u_rb_tf: self.u_rb,
                   self.x_f_tf: self.x_f,
                   self.phi_left_tf: self.phi_left,
                   self.phi_right_tf: self.phi_right,
                   self.w_gau_tf: self.w_gau,
                   self.lb_tf: self.lb,
                   self.rb_tf: self.rb
                   }

        start_time = time.time()
        writer = tf.summary.FileWriter('./graphs', self.sess.graph)
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            self.loss_adam_buff[it] = loss_value

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                scalar_summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss_value)])
                writer.add_summary(scalar_summary, it)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, X_star):

        tf_dict = {self.x_f_tf: X_star}
        u_star = self.sess.run(self.u_f_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star}
        u_x_star = self.sess.run(self.u_x_f_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star}
        phi_star = self.sess.run(self.phi_f_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star}
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        
        tf_dict = {self.x_f_tf: X_star}
        SE_u_star = self.sess.run(self.SE_u_pred, tf_dict)

        return u_star,u_x_star,phi_star, f_u_star, SE_u_star

    def getWeightsBiases(self):
        weights = self.sess.run(self.weights)
        biases = self.sess.run(self.biases)
        return weights, biases
    
    def plotConvergence(iter, adam_buff, lbfgs_buff):
        filename = "Loss_convergence_"
        plt.figure(figsize=(figWidth, figHeight))
        range_adam = np.arange(1, iter + 1)
        range_lbfgs = np.arange(iter + 2, iter + 2 + len(lbfgs_buff))
        plt.plot(range_adam, adam_buff,label='Adam', linewidth=2.0)
        plt.plot(range_lbfgs, lbfgs_buff, label='L-BFGS-B', linewidth=2.0)
        plt.legend()
        plt.xlabel('Iteration', fontweight='bold')
        plt.ylabel('Loss value', fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename + ".pdf", dpi=300, facecolor='w', edgecolor='w',
                transparent='true', bbox_inches='tight')

        plt.show()
        plt.close()

if __name__ == "__main__":
    figHeight = 5
    figWidth = 5
    noise = 0.0

    # Problem definition
    E = 50e6
    b = 1
    h = 0.5
    L = 30*h
    A = b*h
    I = (b * h ** 3) / 12
    q = 10      # uniform transverse load
    v = 0.2
    G = 0.5*E/(1+v)
    k = 5 / 6
    lb = 0.0    # left boundary
    rb = L      # right boundary
    
    # Dirichlet BC data
    u_left = 0
    u_right = 0
    phi_left=0
    phi_right=0

    # TPE integral setting
    N_f = 6     # number of interior points
    u, w_gauss = np.polynomial.legendre.leggauss(N_f)
    t = 0.5 * (u + 1) * (rb - lb) + lb
    X_f=np.reshape(t, (N_f,1))
    w_gauss=np.reshape(w_gauss, (N_f,1))
    
    # Training setting
    time_run = 1
    FileRec = 1
    nPred = 50
    epochs = 1000
    
    data_AE_u = np.zeros((time_run, nPred))
    data_RE_u = np.zeros((time_run, nPred))
    data_AE_phi = np.zeros((time_run, nPred))
    data_RE_phi = np.zeros((time_run, nPred))
    data_AE_fu = np.zeros((time_run, nPred))
    data_RE_fu = np.zeros((time_run, nPred))
    data_u = np.zeros((nPred, time_run + 1))
    data_phi = np.zeros((nPred, time_run + 1))
    data_fu = np.zeros((nPred, time_run + 1))
    data_error = np.zeros((time_run, 8))
    # data_fu = np.zeros((100, time_run))
    data_Pi = np.zeros((time_run, 2))
    
    # PINN model
    layers = [1, 500, 1]
    layers1 = [1, 500, 1]
    scalerU = np.float64([5e0*5.3e-6/2.7e-5*1e-2])  # np.float64([5e0*5.3e-6/2.7e-5*1e-2]) for nEPINN, np.float64([1]) for EPINN
    scalerPhi = np.float64([1e-2])                  # np.float64([1e-2]) for nEPINN, np.float64([1]) for EPINN
    model = PhysicsInformedNN(lb, u_left, rb, u_right, X_f, layers,layers1,E,I, phi_left,phi_right,w_gauss,scalerU,scalerPhi)
    
    for i in range(time_run):
        print('Time', i)

        start_time = time.time()
        model.train(epochs)
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))
        
        # GL quadrature rule
        u1, w_gauss1 = np.polynomial.legendre.leggauss(nPred)
        t1 = 0.5 * (u1 + 1) * (rb - lb) + lb
        X_star = np.reshape(t1, (nPred, 1))
        w_gauss1 = np.reshape(w_gauss1, (nPred, 1))
        
        # EPINN predictions
        u_pred, u_x_pred, phi_pred, f_u_pred, SE_u_pred = model.predict(X_star)

        # Extract exact solution
        u_exact = q * (X_star ** 4 - 2 * rb * X_star ** 3 + rb ** 2 * X_star ** 2) / (24 * E * I) + q * (
                    rb * X_star - X_star ** 2) / (2 * k * G * A)
        u_x_exact = q * (4 * X_star ** 3 - 6 * rb * X_star ** 2 + 2 * rb ** 2 * X_star ** 1) / (24 * E * I) + q * (
                rb - 2 * X_star ** 1) / (2 * k * G * A)
        phi_exact = -(q / (6 * E * I) * X_star ** 3 - q * rb / (4 * E * I) * X_star ** 2 - q * rb ** 2 / (
                    6 * E * I) * X_star + q * rb ** 2 / (4 * E * I) * X_star)
        phi_x_exact = -(q / (2 * E * I) * X_star ** 2 - q * rb / (2 * E * I) * X_star ** 1 - q * rb ** 2 / (
                6 * E * I) + q * rb ** 2 / (4 * E * I) )

        # Exact solution norms
        L1norm_u = np.linalg.norm(u_exact, 1)
        L2norm_u = np.linalg.norm(u_exact, 2)
        L1norm_phi = np.linalg.norm(phi_exact, 1)
        L2norm_phi = np.linalg.norm(phi_exact, 2)
        print('L1 norm of u: %e' % (L1norm_u))
        print('L1 norm of phi: %e' % (L1norm_phi))
        
        # Calculate EPINN prediction errors
        u_pred_err = u_exact - u_pred
        # f_u_pred = w_gauss1 * f_u_pred * (rb - lb) * 0.5
        f_u_exact = 0.5 * (k * G * A * (u_x_exact + phi_exact) ** 2 + E * I * phi_x_exact ** 2) - u_exact * q
        SE_u_exact = 0.5 * (k * G * A * (u_x_exact + phi_exact) ** 2 + E * I * phi_x_exact ** 2)

        error_u1 = (np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2))
        error_u2 = (np.linalg.norm(phi_exact - phi_pred, 2) / np.linalg.norm(phi_exact, 2))
        print('Rel. L1 error of u: %e' % (error_u1))
        print('Rel. L1 error of phi: %e' % (error_u2))

        Pi_pred = np.sum(w_gauss1 * f_u_pred * (rb - lb) * 0.5)
        Pi_exact = np.sum(w_gauss1 * f_u_exact * (rb - lb) * 0.5)
        print('f_u_pred', Pi_pred)
        print('f_u_exact', Pi_exact)
        print('Rel. L1 error of TPE: %e' % (abs(Pi_pred - Pi_exact)/ abs(Pi_exact)))
                
        if FileRec == 1:
            RE_fu = abs(Pi_pred - Pi_exact)/ abs(Pi_exact)
            AE_fu = abs(Pi_pred - Pi_exact)
            data_RE_fu[i,0] = RE_fu
            data_RE_fu[i,1] = AE_fu
            data_error[i,0] = error_u1
            data_error[i,1] = error_u2
            data_error[i,2] = L1norm_u
            data_error[i,3] = L2norm_u
            data_error[i,4] = L1norm_phi
            data_error[i,5] = L2norm_phi
            data_error[i,6] = 1/L1norm_u
            data_error[i,7] = 1/L1norm_phi
            df1 = pd.DataFrame(data_RE_fu)
            df2 = pd.DataFrame(data_error)
            with pd.ExcelWriter('TimoLinearCC_GL10_Q01_N500.xlsx') as writer:
                df1.to_excel(writer, sheet_name='RE_Pi')
                df2.to_excel(writer, sheet_name='Sol_Errs')


    # Potential energy distribution
    nColPred = 1000 
    XCol_star = np.linspace(lb, rb, num=nColPred)
    XCol_star = np.reshape(XCol_star, (nColPred, 1))
    uCol_pred, uCol_x_pred, phiCol_pred, PE_pred, SE_pred = model.predict(XCol_star)
    
    uCol_exact = q * (XCol_star ** 4 - 2 * rb * XCol_star ** 3 + rb ** 2 * XCol_star ** 2) / (24 * E * I) + q * (
                rb * XCol_star - XCol_star ** 2) / (2 * k * G * A)
    uCol_x_exact = q * (4 * XCol_star ** 3 - 6 * rb * XCol_star ** 2 + 2 * rb ** 2 * XCol_star ** 1) / (24 * E * I) + q * (
            rb - 2 * XCol_star ** 1) / (2 * k * G * A)
    phiCol_exact = -(q / (6 * E * I) * XCol_star ** 3 - q * rb / (4 * E * I) * XCol_star ** 2 - q * rb ** 2 / (
                6 * E * I) * XCol_star + q * rb ** 2 / (4 * E * I) * XCol_star)
    phiCol_x_exact = -(q / (2 * E * I) * XCol_star ** 2 - q * rb / (2 * E * I) * XCol_star ** 1 - q * rb ** 2 / (
            6 * E * I) + q * rb ** 2 / (4 * E * I) )
    
    PE_exact = 0.5 * (k * G * A * (uCol_x_exact + phiCol_exact) ** 2 + E * I * phiCol_x_exact ** 2) - uCol_exact * q
    print('f_u_pred', 1/nColPred*(rb-lb)*np.sum(PE_pred))
    print('f_u_exact', 1/nColPred*(rb-lb)*np.sum(PE_exact))
    
    plt.plot(X_star, u_pred, label='Predict', linewidth=2.5)
    plt.plot(X_star, u_exact, '--', label='Exact', linewidth=2.5)
#    plt.title('$u_{comp}$ and $u_{exact}$')
    plt.xlabel("x")
    plt.ylabel("w")
    plt.legend()
    plt.savefig("TimoLinearCC_w_GL10_Q10_N500.pdf")
    plt.show()
    #
    plt.plot(X_star, phi_pred, label='Predict', linewidth=2.5)
    plt.plot(X_star, phi_exact, '--', label='Exact', linewidth=2.5)
    # plt.title('$u_{comp}$ and $u_{exact}$')
    plt.xlabel("x")
    plt.ylabel("$\psi$")
    plt.legend()
    plt.savefig("TimoLinearCC_psi_GL10_Q10_N500.pdf")
    plt.show()

    plt.plot(X_star, f_u_pred, label='Predict', linewidth=2.5)
    plt.plot(X_star, f_u_exact, '--', label='Exact', linewidth=2.5)
    # plt.title('$u_{comp}$ and $u_{exact}$')
    plt.xlabel("x")
    plt.ylabel("$\Pi (x)$")
    plt.legend()
    plt.savefig("TimoLinearCC_TPE_GL10_Q10_N500.pdf")
    plt.show()
    
    plt.plot(X_star, SE_u_pred, label='Predict', linewidth=2.5)
    plt.plot(X_star, SE_u_exact, '--', label='Exact', linewidth=2.5)
    # plt.title('$u_{comp}$ and $u_{exact}$')
    plt.xlabel("x")
    plt.ylabel("$\Psi (x)$")
    plt.legend()
    plt.savefig("TimoLinearCC_Strain_GL10_Q10_N500.pdf")
    plt.show()
