from scipy.integrate import quad
import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde
import seaborn as sns
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.stats import wasserstein_distance
from functools import partial
from tqdm import tqdm 
from torch.utils.data import DataLoader


def drawTrajectory(dfs, y = 'disp', df_labels = ['ERM', 'DP', 'EI', 'BE', 'ER'], 
                   ylabelpad = 0, legend_loc = (0.58, 0.52), title = r'$A$-conscious',
                  colors = ['#073B4C','#FFD166','#06D6A0','#118AB2', '#DD3497', '#AE017E', '#7A0177', '#49006A'], 
                  markers = ['v', 'o', 'x', '1', 's', '>', '.'], zero_line = False, n_iters = 20, file_name=None):
    width = 5
    height = width

    plt.rc('font', family='serif', serif='times new roman')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    plt.rc('axes', labelsize=28)
    plt.rc('axes', linewidth=1)
    mpl.rcParams['patch.linewidth']=0.5 #width of the boundary of legend

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True) #plot two subgraphs in one figure, 1 by 2 size, sharing y-axis
    fig.subplots_adjust(left=.13, bottom=.25, right=0.97, top=0.88, wspace=0.05) #margin of the figure
    fig.set_size_inches(width, height) #exact size of the figure
    
    for i in range(len(dfs)):
        dfs[i][:n_iters].plot.line(y = y, ax = ax, label = df_labels[i], color = colors[i], marker = markers[i])
    ax.set_xlabel('')
    ax.set_title(title, fontsize = 25)
    ax.legend(fontsize = 16, bbox_to_anchor=legend_loc)
    
    if zero_line: 
        xmin, xmax = ax.get_xlim()
        ax.hlines(y = 0, color = 'gray', linestyle = '--', xmin = xmin, xmax = xmax)

    #a nice trick to add common x_label and y_label
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel("Iteration")
    if y == 'disp':
        ylabel = 'Underlying DP Disp'
    elif y == 'wd':
        ylabel = "Distance"
    elif y == 'er':
        ylabel = "Error Rate"
    elif y == 'cdp':
        ylabel = 'Classification DP Disp'
    elif y == 'cef':
        ylabel = 'Classification Effort-encourging Unfairness'

    plt.ylabel(ylabel, labelpad = ylabelpad)
    # lines, labels = ax1.get_legend_handles_labels()
    # fig.legend(lines, labels, loc = 'upper center')
    if file_name: 
        plt.savefig('figures/%s.pdf' % file_name)
        print('File saved in figures/%s.pdf!' % file_name)

class populationDynamics_gaussian(object):
    def __init__(self, gaussians = [(0, 1), (1, 0.9)], frac=None, eps=.6, lazy_pos = True, group0_frac = [0.5, 0.5]):
        # a list of lists, where each list contains parameters 
        # of gaussians given a value of a sensitive feature 
        self.init_data = []
        for gaussian in gaussians:
            self.init_data.append({'mean': gaussian[0], 'std': gaussian[1]})    
                
        
        self.frac, self.eps = frac, eps
        self.lazy_pos = lazy_pos
        self.err_thres = frac / 2
        self.group_frac = np.column_stack((group0_frac, 1-np.array(group0_frac)))
        
    def effort(self, x, boundary, effort_pos = 0, eps = None):
        #see how it's used in update
        if eps is None: eps = self.eps 
        if x >= boundary:
            return effort_pos
        else:
            return 1/(boundary-x + self.eps)**2
    
    def WassersteinDistance(self, data):
        #returns a distance for every sensitive feature 
        ws_distance = np.zeros(len(data)//2)
        for i in range(len(data)//2):
        # https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/
            mu0, sigma0, mu1, sigma1 = data[i * 2]['mean'], data[i * 2]['std'], \
                                        data[i * 2 + 1]['mean'], data[i * 2 + 1]['std']
            ws_distance[i] = (mu0 - mu1)**2 + (sigma0 - sigma1)**2
        return ws_distance
    
    def distance(self, data):
        distances = np.zeros(len(data//2))
        for i in range(len(data)//2): 
            mu0, sigma0, mu1, sigma1 = data[i * 2]['mean'], data[i * 2]['std'], \
                                        data[i * 2 + 1]['mean'], data[i * 2 + 1]['std']
            distances[i] = quad(lambda x: abs(norm.pdf(x, mu0, sigma0) - norm.pdf(x, mu1, sigma1)), -np.inf, np.inf)[0]/2
        return distances

    def errorRate(self, data, truebs, bs):
        error_rate = 0
        for i in len(data)//2: 
            mu0, sigma0, mu1, sigma1 = data[i * 2]['mean'], data[i * 2]['std'], \
                                        data[i * 2 + 1]['mean'], data[i * 2 + 1]['std']
            tb = truebs[i][0]
            e0 = abs(norm.cdf(tb, mu0, sigma0) - norm.cdf(bs[i][0], mu0, sigma0))
            e1 = abs(norm.cdf(tb, mu1, sigma1) - norm.cdf(bs[i][1], mu1, sigma1))
            error_rate += e0*self.group_frac[i][0]+e1*self.group_frac[i][1]
        return error_rate
        
    def trueBoundary(self, data, thres = 1e-3):
        #can be improved later by returning an arr instead of arr of arr (check)
        tb = np.zeros((len(data)//2, 2))
        for i in range(len(data)//2):  
            if self.frac:
                mu0, sigma0, mu1, sigma1 = data[i * 2]['mean'], data[i * 2]['std'], \
                                            data[i * 2 + 1]['mean'], data[i * 2 + 1]['std']
                # lower bound and upper bound for binary search
                l, u = min(mu0-3*sigma0, mu1-3*sigma1), max(mu0+3*sigma0, mu1+3*sigma1)
                while True:
                    p = (l+u)/2
                    frac = 1-(norm.cdf(p, mu0, sigma0) + norm.cdf(p, mu1, sigma1))/2
                    if abs(frac - self.frac) < thres or (u-l) < 1e-10: break
                    if frac > self.frac:
                        l = p
                    else:
                        u = p
                tb[i] = np.ones(2) * p
            else:
                tb[i] = np.zeros(2)
        return tb
        
    def dpDisp(self, data, bs, is_abs = True):
        #returns an array of disparities(1 for each sens feature)
        dp_disparities = np.zeros(len(data)//2)
        for i in range(len(data)//2): 
            pos = []
            for a in range(2):
                mu, sigma = data[i * 2 + a]['mean'], data[i * 2 + a]['std']
                pos.append(1-norm.cdf(bs[i][a], mu, sigma))
            dp_disparities[i] = abs(pos[1] - pos[0]) if is_abs else pos[1] - pos[0]
        return dp_disparities
        
    def efDisp(self, data, bs, delta, is_abs = True):
        #returns an array of disparities(1 for each sens feature)
        ef_disparities = np.zeros(len(data)//2)
        for i in range(len(data)//2): 
            pos = []
            for a in range(2):
                mu, sigma = data[i *2 + a]['mean'], data[i * 2 + a]['std']
                pos.append((norm.cdf(bs[i][a], mu + delta, sigma)-norm.cdf(bs[i][a], mu, sigma)) / norm.cdf(bs[i][a], mu, sigma))
            ef_disparities[i] = abs(pos[1] - pos[0]) if is_abs else pos[1] - pos[0]
        return ef_disparities

    def beDisp(self, data, bs, delta, is_abs = True):
        be_disparities = np.zeros(len(data)//2)
        for i in range(len(data)//2): 
            pos = []
            for a in range(2):
                mu, sigma = data[i * 2 + a]['mean'], data[i * 2 + a]['std']
                pos.append(norm.cdf(bs[i][a], mu + delta, sigma)-norm.cdf(bs[i][a], mu, sigma))
            be_disparities[i] = abs(pos[1] - pos[0]) if is_abs else pos[1] - pos[0]
        return be_disparities
    
    def erDisp(self, data, bs, delta, is_abs = True):
        er_disparities = np.zeros(len(data)//2)
        for i in range(len(data)//2): 
            efforts = []
            for a in range(2):
                mu, sigma = data[i * 2 + a]['mean'], data[i * 2 + a]['std']
                effort = quad(lambda x: (bs[i][a]-x) * norm.pdf(x, mu, sigma), -np.inf, bs[i][a])[0]
                efforts.append(effort / norm.cdf(bs[a], mu, sigma))
            er_disparities[i] = abs(efforts[0] - efforts[1]) if is_abs else efforts[1] - efforts[0]
        return er_disparities
    
    def ilerDisp(self, data, bs, delta, is_abs = True):
        iler_disparities = np.zeros(len(data)//2)
        for i in range(len(data)//2): 
            efforts1, efforts2 = [], []
            u0 = (bs[i][0] - data[i * 2]['mean'])/data[i * 2 ]['std']
            u1 = (bs[i][1] - data[i * 2 + 1]['mean'])/data[i * 2 + 1]['std']
            u = min(u0, u1)
            for a in range(2):
                mu, sigma = data[i * 2 + a]['mean'], data[a]['std']
                effort1 = bs[i][a] - (data[i * 2 + a]['mean'] - 3*data[i * 2 + a]['std'])/data[i * 2 + a]['std']
                effort2 = bs[i][a] - (data[i * 2 + a]['mean'] - u*data[i * 2 + a]['std'])/data[i * 2 + a]['std']
                efforts1.append(effort1 / norm.cdf(bs[i][a], mu, sigma))
                efforts2.append(effort2 / norm.cdf(bs[i][a], mu, sigma))
            iler_disparities[i] = max(abs(efforts1[0] - efforts1[1]), abs(efforts2[0] - efforts2[1]))
        return iler_disparities
        
    def DPBoundary(self, data, truebs, c = 0, thres = 1e-3):
        #check the bounds, sum and input data
        bnds = []
        for i in range(len(data)//2):
            #tb = truebs[sens_feature_idx][0]
            mu0, sigma0, mu1, sigma1 = data[i * 2]['mean'], data[i * 2]['std'], \
                                        data[i * 2 + 1]['mean'], data[i * 2 + 1]['std']
            # cons = ({'type': 'ineq', 'fun': lambda x: c-self.dpDisp(data, x)})
            bnds.append((mu0-3*sigma0, mu0+3*sigma0))
            bnds.append((mu1-3*sigma1, mu1+3*sigma1))
        func = lambda x: np.sum(self.dpDisp(data, x), dtype=np.float32)
        cons = ({'type': 'ineq', 'fun': lambda x: self.err_thres-self.errorRate(data, truebs, x)})
        res = minimize(func, truebs, method = 'SLSQP', bounds = bnds, constraints = cons)
        return res.x 

    def EFBoundary(self, data, truebs, delta, c = 0, thres = 1e-3):
        #tb = truebs[0]
        bnds = []
        for i in range(len(data)//2):
            mu0, sigma0, mu1, sigma1 = data[i * 2]['mean'], data[i * 2]['std'], \
                                        data[i * 2 + 1]['mean'], data[i * 2 + 1]['std']
            bnds.append((mu0-3*sigma0, mu0+3*sigma0))
            bnds.append((mu1-3*sigma1, mu1+3*sigma1))
        func = lambda x: np.sum(self.efDisp(data, x, delta), dtype=np.float32)
        cons = ({'type': 'ineq', 'fun': lambda x: self.err_thres-self.errorRate(data, truebs, x)})
        res = minimize(func, truebs, method = 'SLSQP', bounds = bnds, constraints = cons)
        return res.x

    def BEBoundary(self, data, truebs, delta, c = 0, thres = 1e-3):
        #tb = truebs[0]
        bnds = []
        for i in range(len(data)//2):
            mu0, sigma0, mu1, sigma1 = data[i * 2]['mean'], data[i * 2]['std'], \
                                        data[i * 2 + 1]['mean'], data[i * 2 + 1]['std']
            bnds.append((mu0-3*sigma0, mu0+3*sigma0))
            bnds.append((mu1-3*sigma1, mu1+3*sigma1))
        
        func = lambda x: self.beDisp(data, x, delta)
        cons = ({'type': 'ineq', 'fun': lambda x: self.err_thres-self.errorRate(data, truebs, x)})
        res = minimize(func, truebs, method = 'SLSQP', bounds = bnds, constraints = cons)
        return res.x

    def ERBoundary(self, data, truebs, delta, c = 0, thres = 1e-3):
        #tb = truebs[0]
        bnds = []
        for i in range(len(data)//2):
            mu0, sigma0, mu1, sigma1 = data[i * 2]['mean'], data[i * 2]['std'], \
                                        data[i * 2 + 1]['mean'], data[i * 2 + 1]['std']
            bnds.append((mu0-3*sigma0, mu0+3*sigma0))
            bnds.append((mu1-3*sigma1, mu1+3*sigma1))
        func = lambda x: self.erDisp(data, x, delta)
        cons = ({'type': 'ineq', 'fun': lambda x: self.err_thres-self.errorRate(data, truebs, x)})
        res = minimize(func, truebs, method = 'SLSQP', bounds = bnds, constraints = cons)
        return res.x

    def ILERBoundary(self, data, truebs, delta, c = 0, thres = 1e-3):
        #tb = truebs[0]
        bnds = []
        for i in range(len(data)//2):
            mu0, sigma0, mu1, sigma1 = data[i * 2]['mean'], data[i * 2]['std'], \
                                        data[i * 2 + 1]['mean'], data[i * 2 + 1]['std']
            bnds.append((mu0-3*sigma0, mu0+3*sigma0))
            bnds.append((mu1-3*sigma1, mu1+3*sigma1))
        func = lambda x: self.ilerDisp(data, x, delta)
        cons = ({'type': 'ineq', 'fun': lambda x: self.err_thres-self.errorRate(data, truebs, x)})
        res = minimize(func, truebs, method = 'SLSQP', bounds = bnds, constraints = cons)
        return res.x

    def update(self, data, bs, lazy_pos = True):
        for i in range(len(data)//2):
            mean_efforts, updated_var = np.zeros(2), np.zeros(2)
            for a in range(2):
                if lazy_pos:
                    mean_efforts[a] = quad(lambda x: norm.pdf(x, data[i * 2 + a]['mean'], data[i * 2 + a]['std']) \
                                    * self.effort(x, bs[i][a], eps = 1/data[i * 2 + a]['std'] ** .5), -np.inf, np.inf)[0] 
                    updated_var[a] = quad(lambda x: norm.pdf(x, data[i * 2 + a]['mean'], data[i * 2 + a]['std']) \
                                    * (x + self.effort(x, bs[i][a], eps = 1/data[i * 2 + a]['std'] ** .5) - data[i * 2 + a]['mean'] - mean_efforts[a]) ** 2, \
                                    -np.inf, np.inf)[0]
                else:
                    effort_neg = quad(lambda x: norm.pdf(x, data[i * 2 + a]['mean'], data[i * 2 + a]['std']) \
                                        * self.effort(x, bs[i][a], eps = 1/data[i * 2 + a]['std'] ** .5), -np.inf, bs[i][a])[0] \
                                        /norm.cdf(bs[i][a], data[i * 2 + a]['mean'], data[i * 2 + a]['std'])
                    effort_pos = effort_neg
                    mean_efforts[a] = effort_neg
                    updated_var[a] = quad(lambda x: norm.pdf(x, data[i * 2 + a]['mean'], data[i * 2 + a]['std']) \
                                        * (x+self.effort(x, bs[i][a], effort_pos, eps = 1/data[i * 2 + a]['std'] ** .5) - data[i * 2 + a]['mean'] - mean_efforts[a]) ** 2, \
                                        -np.inf, np.inf)[0]
                data[i * 2 + a] = {'mean': data[i * 2 + a]['mean'] + mean_efforts[a], 'std': updated_var[a]**.5}
        return data

    def selectDelta(self, data, bs, mode = 'median'):
        if mode == 'median':
            mu0, sigma0, mu1, sigma1 = data[0]['mean'], data[0]['std'], data[1]['mean'], data[1]['std']
            neg_frac = norm.cdf(bs[0], mu0, sigma0) + norm.cdf(bs[1], mu1, sigma1)
            func = lambda x: norm.cdf(x, mu0-bs[0], sigma0) + norm.cdf(x, mu1-bs[1], sigma1) / neg_frac - 0.5
            cons = ({'type': 'eq', 'fun': func})
            bnds = [(None, 0)]
            res = minimize(func, 0, method = 'SLSQP', bounds = bnds, constraints = cons)
            return self.effort(res.x, 0)
            
        elif mode == 'mean':
            mean_efforts, neg_frac = np.zeros(2), np.zeros(2)
            for a in range(2):
                mean_efforts[a] = quad(lambda x: norm.pdf(x, data[a]['mean'], data[a]['std']) \
                                    * self.effort(x, bs[a], eps = 1/data[a]['std'] ** .5), -np.inf, bs[a])[0] 
                neg_frac[a] = norm.cdf(bs[a], data[a]['mean'], data[a]['std'])
            # print((mean_efforts[0] * neg_frac[0] + mean_efforts[1] * neg_frac[1]) / (neg_frac[0] + neg_frac[1]))
            return (mean_efforts[0] * neg_frac[0] + mean_efforts[1] * neg_frac[1]) / (neg_frac[0] + neg_frac[1])
        else:
            raise ValueError("Unexpected delta select mode %s! Supported mode: \"mean\" and \"median\"." % mode)
    
    def dataPrint(self, data):
        mu0, sigma0, mu1, sigma1 = data[0]['mean'], data[0]['std'], data[1]['mean'], data[1]['std']
        return (r'$\mu^{(0)}=$%.1f, $\sigma^{(0)}=$%.1f' % (mu0, sigma0), r'$\mu^{(1)}=$%.1f, $\sigma^{(1)}=$%.1f' % (mu1, sigma1))

    def plot(self, data, truebs, bs = None, title = None):
        fig, ax = plt.subplots(nrows = 1, ncols = 2, sharey = True, sharex = True)
        labels = self.dataPrint(data)
        for a in range(2):
            mu, sigma = data[a]['mean'], data[a]['std']
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            ax[a].plot(x, norm.pdf(x, mu, sigma))
            ax[a].set_xlabel(labels[a], fontsize = 18)
            ymin, ymax = ax[a].get_ylim()
            ax[a].vlines(x = truebs[a], color = 'g', linestyle='-.', ymin = ymin, ymax = ymax)
            if not bs is None: ax[a].vlines(x = bs[a], color = 'm', linestyle = '-', ymin = ymin, ymax = ymax)
        plt.title(title)

    def run(self, mode = 'true', n_iter = 20, delta = 0.5, c = 0, thres = .001, select_delta = False, plot = True):
        if mode == 'true':
            data = copy.deepcopy(self.init_data)
            disp, wd, cdp, cef = np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1)
            for i in range(n_iter+1):
                truebs = self.trueBoundary(data, thres)
                disp[i], wd[i], cdp[i], cef[i] = self.dpDisp(data, truebs), self.distance(data), self.dpDisp(data, truebs), self.efDisp(data, truebs, delta)
                text = 'Iteration %d: Underlying DP Disp: %.3f, Classification DP Disp: %.3f, Classification EF Disp: %.3f, Wasserstein Distance: %.3f' % (
                        i, disp[i], cdp[i], cef[i], wd[i])
                if plot: self.plot(data, truebs, title = text)
                data = self.update(data, truebs)
            return pd.DataFrame({'disp':disp, 'wd':wd, 'cdp': cdp, 'cef': cef})
                
        elif mode == 'dp':
            data = copy.deepcopy(self.init_data)
            disp, wd, er, cdp, cef = np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1)
            for i in range(n_iter+1):
                truebs = self.trueBoundary(data, thres)
                bs = self.DPBoundary(data, truebs, c, thres)
                disp[i], wd[i], er[i], cdp[i], cef[i] = self.dpDisp(data, truebs), self.distance(data), self.errorRate(data, truebs, bs), self.dpDisp(data, bs), self.efDisp(data, bs, delta)
                text = 'Iteration %d: Underlying DP Disp: %.3f, Classification DP Disp: %.3f, Classification EF Disp: %.3f, Wasserstein Distance: %.3f, Error Rate: %.3f' % (
                        i, disp[i], cdp[i], cef[i] , wd[i], er[i])
                if plot: self.plot(data, truebs, bs, title = text)
                data = self.update(data, bs)
            return pd.DataFrame({'disp':disp, 'wd':wd, 'er':er, 'cdp':cdp, 'cef':cef})

        elif mode == 'ef':
            data = copy.deepcopy(self.init_data)
            disp, wd, er, cdp, cef = np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1)
            for i in range(n_iter+1):
                truebs = self.trueBoundary(data, thres)
                if select_delta: delta = self.selectDelta(data, truebs, select_delta)
                bs = self.EFBoundary(data, truebs, delta, c, thres)
                disp[i], wd[i], er[i], cdp[i], cef[i] = self.dpDisp(data, truebs), self.distance(data), self.errorRate(data, truebs, bs), self.dpDisp(data, bs), self.efDisp(data, bs, delta)
                text = 'Iteration %d: Underlying DP Disp: %.3f, Classification DP Disp: %.3f, Classification EF Disp: %.3f, Wasserstein Distance: %.3f, Error Rate: %.3f' % (
                        i, disp[i], cdp[i], cef[i] , wd[i], er[i])
                if plot: self.plot(data, truebs, bs, title = text)
                data = self.update(data, bs)
            return pd.DataFrame({'disp':disp, 'wd':wd, 'er':er, 'cdp':cdp, 'cef':cef})

        elif mode == 'be':
            data = copy.deepcopy(self.init_data)
            disp, wd, er, cdp, cef, cbe = np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1)
            for i in range(n_iter+1):
                truebs = self.trueBoundary(data, thres)
                if select_delta: delta = self.selectDelta(data, truebs, select_delta)
                bs = self.BEBoundary(data, truebs, delta, c, thres)
                disp[i], wd[i], er[i], cdp[i], cef[i], cbe[i] = self.dpDisp(data, truebs), self.distance(data), self.errorRate(data, truebs, bs), self.dpDisp(data, bs), self.efDisp(data, bs, delta), self.beDisp(data, bs, delta)
                text = 'Iteration %d: Underlying DP Disp: %.3f, Classification DP Disp: %.3f, Classification EF Disp: %.3f, Classification BE Disp: %.3f, Wasserstein Distance: %.3f, Error Rate: %.3f' % (
                        i, disp[i], cdp[i], cef[i], cbe[i], wd[i], er[i])
                if plot: self.plot(data, truebs, bs, title = text)
                data = self.update(data, bs)
            return pd.DataFrame({'disp':disp, 'wd':wd, 'er':er, 'cdp':cdp, 'cef':cef, 'cbe': cbe})

        elif mode == 'er':
            data = copy.deepcopy(self.init_data)
            disp, wd, er, cdp, cef, cer = np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1)
            for i in range(n_iter+1):
                truebs = self.trueBoundary(data, thres)
                if select_delta: delta = self.selectDelta(data, truebs, select_delta)
                bs = self.ERBoundary(data, truebs, delta, c, thres)
                disp[i], wd[i], er[i], cdp[i], cef[i], cer[i] = self.dpDisp(data, truebs), self.distance(data), self.errorRate(data, truebs, bs), self.dpDisp(data, bs), self.efDisp(data, bs, delta), self.erDisp(data, bs, delta)
                text = 'Iteration %d: Underlying DP Disp: %.3f, Classification DP Disp: %.3f, Classification EF Disp: %.3f, Classification ER Disp: %.3f, Wasserstein Distance: %.3f, Error Rate: %.3f' % (
                        i, disp[i], cdp[i], cef[i], cer[i], wd[i], er[i])
                if plot: self.plot(data, truebs, bs, title = text)
                data = self.update(data, bs)
            return pd.DataFrame({'disp':disp, 'wd':wd, 'er':er, 'cdp':cdp, 'cef':cef, 'cer': cer})

        elif mode == 'iler':
            data = copy.deepcopy(self.init_data)
            disp, wd, er, cdp, cef, cer = np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1)
            for i in range(n_iter+1):
                truebs = self.trueBoundary(data, thres)
                if select_delta: delta = self.selectDelta(data, truebs, select_delta)
                bs = self.ILERBoundary(data, truebs, delta, c, thres)
                disp[i], wd[i], er[i], cdp[i], cef[i], cer[i] = self.dpDisp(data, truebs), self.distance(data), self.errorRate(data, truebs, bs), self.dpDisp(data, bs), self.efDisp(data, bs, delta), self.ilerDisp(data, bs, delta)
                text = 'Iteration %d: Underlying DP Disp: %.3f, Classification DP Disp: %.3f, Classification EF Disp: %.3f, Classification ER Disp: %.3f, Wasserstein Distance: %.3f, Error Rate: %.3f' % (
                        i, disp[i], cdp[i], cef[i], cer[i], wd[i], er[i])
                if plot: self.plot(data, truebs, bs, title = text)
                data = self.update(data, bs)
            return pd.DataFrame({'disp':disp, 'wd':wd, 'er':er, 'cdp':cdp, 'cef':cef, 'ciler': cer})

        else:
            raise ValueError("Unexpected mode %s! Supported mode: \"true\", \"dp\", and \"ef\"." % mode)
