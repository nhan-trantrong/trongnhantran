"""
EPINNs for 1D linear Euler-Bernoulli beam bending problem
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
# import statistics

# make figures bigger on HiDPI monitors
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "times new roman"

tf.reset_default_graph()  # To clear the defined variables and operations of the previous cell
np.random.seed(0)
tf.set_random_seed(0)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, lb, u_lb, rb, u_rb, X_f, layers,E,I,u_x_left, u_x_right,w_gau,scaler):

        self.lb = lb
        self.rb = rb

        self.x_lb = lb
        self.x_rb = rb
        self.u_lb = u_lb

        self.u_rb = u_rb
        self.w_gau = w_gau
        self.x_f = X_f
        self.u_x_left = u_x_left
        self.u_x_right = u_x_right
        # self.k = k
        self.E=E
        self.I = I
        # self.q = q
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # tf Placeholders
        self.lb_tf = tf.placeholder(tf.float64)
        self.rb_tf = tf.placeholder(tf.float64)
        self.x_lb_tf = tf.placeholder(tf.float64)
        self.u_lb_tf = tf.placeholder(tf.float64)
        self.x_rb_tf = tf.placeholder(tf.float64)
        self.u_rb_tf = tf.placeholder(tf.float64)

        self.w_gau_tf = tf.placeholder(tf.float64)
        self.x_f_tf = tf.placeholder(tf.float64)
        self.u_x_left_tf = tf.placeholder(tf.float64)
        self.u_x_right_tf = tf.placeholder(tf.float64)
        
        # tf Graphs
        self.u_lb_pred, self.u_x_lb_pred = self.net_u(self.x_lb_tf,scaler)
        self.u_rb_pred, self.u_x_rb_pred = self.net_u(self.x_rb_tf,scaler)
        self.u_f_pred, self.u_x_f_pred = self.net_u(self.x_f_tf,scaler)
        self.f_u_pred, self.SE_u_pred = self.net_f_u(self.x_f_tf,scaler)

        # Loss function
        self.loss_energy= tf.reduce_sum(self.f_u_pred*self.w_gau_tf)*(self.rb_tf-self.lb_tf)*0.5
        self.loss = self.loss_energy

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 10000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        # choose optimizer
        self.lbfgs_buffer = []
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001)
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
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64),
                           dtype=tf.float64)

    def neural_net(self, X, weights, biases, G):
        num_layers = len(weights) + 1
        H = (X - self.lb) / (self.rb - self.lb) # normalize into (0,1)
        
        # use tf.multiply for the first iteration/layers
        W = weights[0]
        b = biases[0]
        # choose activation funtion
        H = tf.nn.tanh(tf.add(tf.multiply(H, W), b))
        # H = tf.sigmoid(tf.add(tf.multiply(H, W), b))
        # H = tf.math.sin(tf.add(tf.multiply(H, W), b))
        
        # use tf.matmul for the remanining layers
        for l in range(1, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            # H = tf.sigmoid(tf.add(tf.matmul(H, W), b))
            # H = tf.math.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        Y = tf.math.multiply(G,Y)
        return Y

    def net_u(self, x, scaler):
        # scaler = np.float64([1e0]) # np.float64([1e-3])
        u = self.neural_net(x, self.weights, self.biases, scaler)
        u = u * x**2*(self.rb-x)**2 # hard constraint for essential BCs
        u_x = tf.gradients(u, x)[0]
        return u, u_x

    def net_f_u(self, x, scaler):
        u, u_x = self.net_u(x, scaler)
        u_xx = tf.gradients(u_x, x)[0]
        f_u = 0.5 * E * I * (u_xx) ** 2 - q * u
        SE_u = 0.5 * E * I * (u_xx) ** 2
        return f_u, SE_u
    def callback(self, loss):
        self.lbfgs_buffer = np.append(self.lbfgs_buffer, loss)
        # writer = tf.summary.FileWriter('./graphs', self.sess.graph)
        # scalar_summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss)])
        # writer.add_summary(scalar_summary)
        print('Loss:', loss)

    def train(self, nIter):
        self.loss_adam_buff = np.zeros(nIter)
        tf_dict = {self.x_lb_tf: self.x_lb,
                   self.u_lb_tf: self.u_lb,
                   self.x_rb_tf: self.x_rb,
                   self.u_rb_tf: self.u_rb,
                   self.x_f_tf: self.x_f,
                   self.u_x_left_tf: self.u_x_left,
                   self.u_x_right_tf: self.u_x_right,
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
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        
        tf_dict = {self.x_f_tf: X_star}
        SE_u_star = self.sess.run(self.SE_u_pred, tf_dict)

        return u_star,u_x_star, f_u_star, SE_u_star

    def getWeightsBiases(self):
        weights = self.sess.run(self.weights)
        biases = self.sess.run(self.biases)
        return weights, biases

def plotConvergence(iter, adam_buff, lbfgs_buff):
    # filename = "Loss_convergence_"
    plt.figure(figsize=(figWidth, figHeight))
    range_adam = np.arange(1, iter + 1)
    range_lbfgs = np.arange(iter + 2, iter + 2 + len(lbfgs_buff))
    plt.plot(range_adam, adam_buff,label='Adam', linewidth=2.0)
    plt.plot(range_lbfgs, lbfgs_buff, label='L-BFGS-B', linewidth=2.0)
    plt.legend()
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.ylim(-0.06, 0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')

if __name__ == "__main__":
    figHeight = 5
    figWidth = 5
    
    # Problem definition
    E = 50e6
    b = 1
    h = 0.5
    L = 30*h
    A = b*h
    I = (b * h ** 3) / 12
    v = 0.2
    q = 1      # uniform transverse load
    lb = 0.0    # left boundary
    rb = L      # right boundary

    # Dirichlet BC data
    u_left = 0
    u_right = 0
    u_x_left = 0
    u_x_right = 0

    # TPE integral setting
    N_f = 7 # number of integral points
    u, w_gauss = np.polynomial.legendre.leggauss(N_f)
    t = 0.5 * (u + 1) * (rb - lb) + lb
    X_f = np.reshape(t, (N_f, 1))
    w_gauss = np.reshape(w_gauss, (N_f, 1))

    # Training setting
    time_run = 1
    FileRec = 1
    nGLPred = 50
    
    data_AE_u = np.zeros((time_run, nGLPred))
    data_RE_u = np.zeros((time_run, nGLPred))
    data_AE_fu = np.zeros((time_run, nGLPred))
    data_RE_fu = np.zeros((time_run, nGLPred))
    data_u = np.zeros((time_run+1, nGLPred))
    data_fu = np.zeros(( time_run+1, nGLPred))
    data_error = np.zeros((time_run, 5))
    data_Pi=np.zeros((time_run, 2))
    data_time = np.zeros((time_run, 1))
    epochs = 1000
    
    # PINN model
    layers = [1, 500, 1] # MLP configuration
    scaler = np.float64([1e-2]) # np.float64([1e-2]) for nEPINN, np.float64([1]) for EPINN
    model = PhysicsInformedNN(lb, u_left, rb, u_right, X_f, layers, E, I, u_x_left, u_x_right, w_gauss,scaler)


    for i in range(time_run):

        print('Time', i)

        start_time = time.time()
        model.train(epochs)
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))
        
        # GL quadrature rule
        u1, w_gauss1 = np.polynomial.legendre.leggauss(nGLPred)
        t1 = 0.5 * (u1 + 1) * (rb - lb) + lb
        XGL_star = np.reshape(t1, (nGLPred, 1))
        w_gauss1 = np.reshape(w_gauss1, (nGLPred, 1))
        
        # EPINN predictions
        u_pred, u_x_pred, f_u_pred, SE_u_pred = model.predict(XGL_star)
        f_u_pred = w_gauss1 * f_u_pred * (rb - lb) * 0.5
        
        # Extract exact solution
        u_exact = q * (XGL_star ** 4 - 2 * rb * XGL_star ** 3 + rb ** 2 * XGL_star ** 2) / (24 * E * I);
        u_x_exact = q * (4 * XGL_star ** 3 - 6 * rb * XGL_star ** 2 + 2 * rb ** 2 * XGL_star ** 1) / (24 * E * I)
        u_xx_exact = q * (12 * XGL_star ** 2 - 12 * rb * XGL_star + 2 * rb ** 2) / (24 * E * I)
        # u_xxx_exact = q * (24 * XGL_star - 12 * rb) / (24 * E * I)
        f_u_exact = w_gauss1 * (0.5 * E * I * u_xx_exact ** 2 - u_exact * q) * (rb - lb) * 0.5
        
        # Exact solution norms
        L1norm_u = np.linalg.norm(u_exact, 1)
        L2norm_u = np.linalg.norm(u_exact, 2)
        print('L2 norm of u: %e' % (L2norm_u))
        
        # Calculate EPINN prediction errors
        u_pred_err = u_exact - u_pred
        error_u1 = (np.linalg.norm(u_exact - u_pred, 1) / np.linalg.norm(u_exact, 1))
        error_u2 = (np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2))
        print('Rel. L1 error of u: %e' % (error_u1))
        print('Rel. L2 error of u: %e' % (error_u2))

        u_exact = np.reshape(u_exact, (nGLPred))
        u_pred = np.reshape(u_pred, (nGLPred))
        f_u_pred = np.reshape(f_u_pred, (nGLPred))
        f_u_exact = np.reshape(f_u_exact, (nGLPred))
        Pi_pred = np.sum(f_u_pred)
        Pi_exact = np.sum(f_u_exact)
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
            data_error[i,4] = 1/L1norm_u
            data_time[i, 0] = elapsed
            df1 = pd.DataFrame(data_RE_fu)
            df2 = pd.DataFrame(data_error)
            df3 = pd.DataFrame(data_time)
            with pd.ExcelWriter('EBLinearCC_GL03_Q1_DNN1000.xlsx') as writer:
                df1.to_excel(writer, sheet_name='RE_Pi')
                df2.to_excel(writer, sheet_name='Sol_Errs')
                df3.to_excel(writer, sheet_name='time')


    # Potential energy distribution
    nPred = 1000
    X_star = np.linspace(lb, rb, num=nPred)
    X_star = np.reshape(X_star, (nPred, 1))
    uCol_pred, uCol_x_pred, PE_pred, SE_pred = model.predict(X_star)
    
    uCol_exact = q * (X_star ** 4 - 2 * rb * X_star ** 3 + rb ** 2 * X_star ** 2) / (24 * E * I);
    uCol_x_exact = q * (4 * X_star ** 3 - 6 * rb * X_star ** 2 + 2 * rb ** 2 * X_star ** 1) / (24 * E * I)
    uCol_xx_exact = q * (12 * X_star ** 2 - 12 * rb * X_star + 2 * rb ** 2) / (24 * E * I)
    
    PE_exact = 0.5 * E * I * uCol_xx_exact ** 2 - uCol_exact * q    # Potential energy distribution
    SE_exact = 0.5 * E * I * uCol_xx_exact ** 2                     # Strain energy distribution
    
    # Plot displacement, potential energy and strain energy distributions
    plt.rcParams["font.family"] = "times new roman"
    plt.plot(X_star, uCol_pred, label='Predict', linewidth=2.5)
    plt.plot(X_star, uCol_exact, '--', label='Exact', linewidth=2.5)
    plt.xlabel("x")
    plt.ylabel("w")
    plt.legend()
    plt.savefig("EBLinearCC_w_GL10_Q1_DNN1000.pdf")
    plt.show()
    
    plt.plot(X_star, PE_pred, label='Predict', linewidth=2.5)
    plt.plot(X_star, PE_exact, '--', label='Exact', linewidth=2.5)
    plt.xlabel("x")
    plt.ylabel("$\Pi (x)$")
    plt.legend()
    plt.savefig("EBLinearCC_PE_GL10_Q1_DNN1000.pdf")
    plt.show()
    
    plt.plot(X_star, SE_pred, label='Predict', linewidth=2.5)
    plt.plot(X_star, SE_exact, '--', label='Exact', linewidth=2.5)
    plt.xlabel("x")
    plt.ylabel("$\Psi (x)$")
    plt.legend()
    plt.savefig("EBLinearCC_Strain_GL10_Q1_DNN1000.pdf")
    plt.show()