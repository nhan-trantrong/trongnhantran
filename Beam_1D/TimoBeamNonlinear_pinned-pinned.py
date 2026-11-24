"""
EPINNs for 1D nonlinear Timoshenko beam bending problem
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
    def __init__(self, lb, u_lb, rb, u_rb, X_f, layers, layers1, layers2,E,I,phi_left, phi_right,w_gau,scalerU,scalerW,scalerPhi):

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

        self.layers = layers
        self.weights2, self.biases2 = self.initialize_NN(layers2)

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

        self.w_lb_pred, self.w_x_lb_pred = self.net_w(self.x_lb_tf,scalerW)
        self.w_rb_pred, self.w_x_rb_pred = self.net_w(self.x_rb_tf,scalerW)
        self.w_f_pred, self.w_x_f_pred = self.net_w(self.x_f_tf,scalerW)

        self.phi_lb_pred, self.phi_x_lb_pred = self.net_phi(self.x_lb_tf,scalerPhi)
        self.phi_rb_pred, self.phi_x_rb_pred = self.net_phi(self.x_rb_tf,scalerPhi)
        self.phi_f_pred, self.phi_x_f_pred = self.net_phi(self.x_f_tf,scalerPhi)

        self.f_u_pred = self.net_f_u(self.x_f_tf,scalerU,scalerW,scalerPhi)
        # Loss
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
        Y = tf.math.multiply(G, Y)
        return Y

    def net_u(self, x, scaler):
        uphi = self.neural_net(x, self.weights, self.biases, scaler)
        u = uphi
        u=u*x*(self.rb-x)
        u_x = tf.gradients(u, x)[0]
        return u, u_x
    def net_w(self, x, scaler):
        uphi = self.neural_net(x, self.weights1, self.biases1, scaler)
        w = uphi
        w=w*x*(self.rb-x)
        w_x = tf.gradients(w, x)[0]
        return w, w_x
    def net_phi(self, x, scaler):
        uphi = self.neural_net(x, self.weights2, self.biases2, scaler)
        phi = uphi
        phi_x = tf.gradients(phi, x)[0]
        return phi, phi_x

    def net_f_u(self, x, scalerU, scalerW, scalerPhi):
        u, u_x = self.net_u(x,scalerU)
        print('u', u)
        w, w_x = self.net_w(x,scalerW)
        phi, phi_x=self.net_phi(x,scalerPhi)
        # phi_xx= tf.gradients(phi_x, x)[0]
        S=(k*E*A)/(2*(1+v))
        f_u = E * A * 0.5 * (u_x + 0.5 * w_x ** 2) ** 2 + S * 0.5 * (w_x + phi) * (w_x + phi) + E * I * 0.5 * phi_x ** 2 - w * q
        return f_u

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
        w_star = self.sess.run(self.w_f_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star}
        w_x_star = self.sess.run(self.w_x_f_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star}
        phi_star = self.sess.run(self.phi_f_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star}
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)

        return u_star,u_x_star,w_star,w_x_star, phi_star, f_u_star

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
    
    # Problem definition
    E = 50e6
    b = 1
    h = 0.5
    I = b * (h ** 3) / 12
    A = b * h
    v = 0.2
    G = E / (2 * (1 + v))
    k = 5 / 6
    q = 1
    lb = 0.0  # left boundary
    rb = 15.0  # right boundary

    # Dirichlet BC data
    u_left = 0
    u_right = 0
    phi_left = 0
    phi_right = 0
    
    # TPE integral setting
    N_f = 10 # number of interior points
    u, w_gauss = np.polynomial.legendre.leggauss(N_f)
    t = 0.5 * (u + 1) * (rb - lb) + lb
    X_f = np.reshape(t, (N_f, 1))
    w_gauss = np.reshape(w_gauss, (N_f, 1))
    
    # Training setting
    time_run = 1
    FileRec = 1
    nPred = 101
    epochs = 1000
    
    # Extract exact solution
    df1 = pd.read_csv('./Data_Timo/W-P-P_F01.csv', header=None)
    w_exact = df1.values
    df2 = pd.read_csv('./Data_Timo/U-P-P_F01.csv', header=None)
    u_exact = df2.values
    df3 = pd.read_csv('./Data_Timo/DW-P-P_F01.csv', header=None)
    phi_exact = df3.values

    # Training setting
    data_AE_u = np.zeros((time_run, nPred))
    data_RE_u = np.zeros((time_run, nPred))
    data_AE_w = np.zeros((time_run, nPred))
    data_RE_w = np.zeros((time_run, nPred))
    data_AE_phi = np.zeros((time_run, nPred))
    data_RE_phi = np.zeros((time_run, nPred))
    data_u = np.zeros((time_run + 1, nPred))
    data_w = np.zeros((time_run + 1, nPred))
    data_phi = np.zeros((time_run + 1, nPred))
    data_error = np.zeros((time_run, 19))
    data_potential = np.zeros((time_run, 1))
    
    # PINN model
    scalerU = np.float64([1e0])     # np.float64([5e4*4.3358e-8/0.001267*5e-3]) for nEPINN, np.float64([1]) for EPINN
    scalerW = np.float64([1e0])                            # np.float64([1e-2]) for nEPINN, np.float64([1]) for EPINN
    scalerPhi = np.float64([1e0])    # np.float64([10e0*0.00027/0.001267*5e-3]) for nEPINN, np.float64([1]) for EPINN
    layers = [1, 2000, 1]
    layers1 = [1, 2000, 1]
    layers2 = [1, 2000, 1]
    model = PhysicsInformedNN(lb, u_left, rb, u_right, X_f, layers, layers1, layers2, E, I, phi_left, phi_right, w_gauss, scalerU, scalerW, scalerPhi)
    
    for i in range(time_run):
        print('Run time:', i)

        start_time = time.time()
        model.train(epochs)
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))

        X_star = np.linspace(lb, rb, nPred)[np.newaxis]
        X_star = X_star.T
        u_pred, u_x_pred, w_pred, w_x_pred, phi_pred, f_u_pred = model.predict(X_star)
        phi_pred = np.reshape(phi_pred, (nPred))
        phi_exact = np.reshape(phi_exact, (nPred))
        u_pred = np.reshape(u_pred, (nPred))
        u_exact = np.reshape(u_exact, (nPred))
        w_pred = np.reshape(w_pred, (nPred))
        w_exact = np.reshape(w_exact, (nPred))
        
        # Exact solution norms
        L1norm_u = np.linalg.norm(u_exact, 1)
        print('L1 norm of u: %e' % (L1norm_u))
        L2norm_u = np.linalg.norm(u_exact, 2)
        print('L2 norm of u: %e' % (L2norm_u))
        L1norm_w = np.linalg.norm(w_exact, 1)
        print('L1 norm of w: %e' % (L1norm_w))
        L2norm_w = np.linalg.norm(w_exact, 2)
        print('L2 norm of w: %e' % (L2norm_w))
        L1norm_phi = np.linalg.norm(phi_exact, 1)
        print('L1 norm of phi: %e' % (L1norm_phi))
        L2norm_phi = np.linalg.norm(phi_exact, 2)
        print('L2 norm of phi: %e' % (L2norm_phi))
        
        # EPINN prediction errors
        error_u1 = (np.linalg.norm(u_exact - u_pred, 1) / np.linalg.norm(u_exact, 1))
        print('Rel. L1 error of u: %e' % (error_u1))
        error_u2 = (np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2))
        print('Rel. L2 error of u: %e' % (error_u2))

        error_w1 = (np.linalg.norm(w_exact - w_pred, 1) / np.linalg.norm(w_exact, 1))
        print('Rel. L1 error of w: %e' % (error_w1))
        error_w2 = (np.linalg.norm(w_exact - w_pred, 2) / np.linalg.norm(w_exact, 2))
        print('Rel. L2 error of w: %e' % (error_w2))

        error_phi1 = (np.linalg.norm(phi_exact - phi_pred, 1) / np.linalg.norm(phi_exact, 1))
        print('Rel. L1 error of phi: %e' % (error_phi1))
        error_phi2 = (np.linalg.norm(phi_exact - phi_pred, 2) / np.linalg.norm(phi_exact, 2))
        print('Rel. L2 error of phi: %e' % (error_phi2))
        
        L1norm_delta_u = np.linalg.norm(u_exact - u_pred, 1)
        L1norm_delta_w = np.linalg.norm(w_exact - w_pred, 1)
        L1norm_delta_phi = np.linalg.norm(phi_exact - phi_pred, 1)
        alpha_u = L1norm_delta_u/(L1norm_delta_u + L1norm_delta_w + L1norm_delta_phi)
        alpha_w = L1norm_delta_w/(L1norm_delta_u + L1norm_delta_w + L1norm_delta_phi)
        alpha_phi = L1norm_delta_phi/(L1norm_delta_u + L1norm_delta_w + L1norm_delta_phi)
                
        if FileRec == 1:
            data_error[i,0] = error_u1
            data_error[i,1] = error_u2
            data_error[i,2] = error_w1
            data_error[i,3] = error_w2
            data_error[i,4] = error_phi1
            data_error[i,5] = error_phi2
            data_error[i,6] = L1norm_u
            data_error[i,7] = L2norm_u
            data_error[i,8] = L1norm_w
            data_error[i,9] = L2norm_w
            data_error[i,10] = L1norm_phi
            data_error[i,11] = L2norm_phi
            data_error[i,12] = model.lbfgs_buffer[-1]
            data_error[i,13] = np.linalg.norm(u_pred, 2)
            data_error[i,14] = np.linalg.norm(w_pred, 2)
            data_error[i,15] = np.linalg.norm(phi_pred, 2)
            data_error[i,16] = alpha_u
            data_error[i,17] = alpha_w
            data_error[i,18] = alpha_phi
            df1 = pd.DataFrame(data_error)
            with pd.ExcelWriter('TimoNonlinearPP_GL10_Q10_DNN1000.xlsx') as writer:
                df1.to_excel(writer, sheet_name='Sol_Errs')
    
    # Plot displacements
    w_exact = np.reshape(w_exact, (nPred, 1))
    plt.plot(X_star, w_pred, label='Predict', linewidth=2.5)
    plt.plot(X_star, w_exact, '--', label='exact', linewidth=2.5)
    plt.xlabel("x")
    plt.ylabel("w")
    plt.legend()
    plt.savefig("TimoNonlinearPP_w_GL10_Q10_DNN1000.pdf")
    plt.show()
    
    u_exact = np.reshape(u_exact, (nPred, 1))
    plt.plot(X_star, u_pred, label='Predict', linewidth=2.5)
    plt.plot(X_star, u_exact, '--', label='exact', linewidth=2.5)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.savefig("TimoNonlinearPP_u_GL10_Q10_DNN1000.pdf")
    plt.show()
    
    phi_exact = np.reshape(phi_exact, (nPred, 1))
    plt.plot(X_star, phi_pred, label='Predict', linewidth=2.5)
    plt.plot(X_star, phi_exact, '--', label='exact', linewidth=2.5)
    plt.xlabel("x")
    plt.ylabel("$\psi$")
    plt.legend()
    plt.savefig("TimoNonlinearPP_psi_GL10_Q10_DNN1000.pdf")
    plt.show()
