{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -U nolds matplotlib numpy pandas mpld3 statsmodels scikit-learn scipy git+https://github.com/manu-mannattil/nolitsa.git"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Using matplotlib backend: Qt5Agg\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'d:\\\\timeserieslab'"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# %matplotlib inline\n",
    "import mpld3\n",
    "mpld3.enable_notebook()\n",
    "%matplotlib auto\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import os\n",
    "import numpy as np\n",
    "from scipy.spatial import KDTree\n",
    "from nolitsa import data, delay, dimension, d2, utils\n",
    "from mpl_toolkits import mplot3d\n",
    "from statsmodels.tsa.arima_process import arma_generate_sample, arma_acf\n",
    "\n",
    "os.getcwd()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "def linearfitnrmse(xV, m, Tmax=1, show=False):\n",
    "    ''' \n",
    "    % LINEARFITNRMSE fits an AR model and computes the fitting error\n",
    "    % for T-step ahead.\n",
    "    % INPUTS:\n",
    "    %  xV      : vector of the scalar time series\n",
    "    %  m       : the embedding dimension.\n",
    "    %  Tmax    : the prediction horizon, the fit is made for T=1...Tmax steps\n",
    "    %            ahead.\n",
    "    %  tittxt  : string to be displayed in the title of the figure \n",
    "    %            if not specified, no plot is made\n",
    "    % OUTPUT: \n",
    "    %  nrmseV  : vector of length Tmax, the nrmse of the fit for T-mappings, T=1...Tmax.\n",
    "    %  phiV    : the coefficients of the estimated AR time series (of length (m+1)\n",
    "    %            with phi(0) as first component\n",
    "    '''\n",
    "    from statsmodels.api import OLS\n",
    "\n",
    "    n = xV.shape[0]\n",
    "    mx = np.mean(xV[:n-Tmax+1])\n",
    "    yV = xV[:n-Tmax+1] - mx\n",
    "    nvec = n - m - 1 - Tmax\n",
    "    yM = np.full(shape=(nvec-1, m), fill_value=np.nan)\n",
    "    for j in np.arange(m):\n",
    "        yM[:, [m-j-1]] = yV[j:nvec+j-1]\n",
    "    rV = yV[j+1:nvec+j]\n",
    "    # np.linalg.lstsq(yM, rV)\n",
    "    ols = OLS(endog=rV, exog=yM).fit()\n",
    "    aV = ols.params\n",
    "    a0 = (1 - np.sum(aV)) * mx\n",
    "    phiV = np.r_[a0, aV]\n",
    "    preM = np.full(shape=(n+Tmax-1, Tmax), fill_value=np.nan)\n",
    "    for i in np.arange(m, n):\n",
    "        preV = np.full(shape=(m+Tmax, 1), fill_value=np.nan)\n",
    "        preV[:m] = xV[i-m: i] - mx\n",
    "        for T in np.arange(1, Tmax+1):\n",
    "            preV[m + T - 1] = np.dot(aV, (preV[T-1:m+T-1][::-1]))\n",
    "            preM[i + T - 1, [T-1]] = preV[m + T - 1]\n",
    "    preM = preM + mx\n",
    "    nrmseV = np.ones(shape=(Tmax, 1))\n",
    "    for T in np.arange(1, Tmax+1):\n",
    "        nrmseV[T-1] = nrmse(xV[m + T - 1:n], preM[m + T - 1: n, [T-1]])\n",
    "    if show:\n",
    "        fig, ax = plt.subplots(1, 1)\n",
    "        ax.plot(np.arange(1, Tmax+1), nrmseV, marker='x')\n",
    "        ax.set_xlabel('prediction time T')\n",
    "        ax.set_ylabel('NRMSE(T)')\n",
    "    return nrmseV, phiV\n",
    "\n",
    "def localfitnrmse(xV, tau, m, Tmax, nnei, q, show=''):\n",
    "    '''\n",
    "     LOCALFITNRMSE makes fitting using a local model of zeroth order (average \n",
    "    % mapping or nearest neighbor mappings if only one neighbor is chosen) or a \n",
    "    % local linear model and computes the fitting error for T-step ahead. For \n",
    "    % the search for neighboring points it uses the Matlab k-d-tree search.\n",
    "    % The fitting here means that predictions are made for all the points in\n",
    "    % the data set (in-sample prediction). The prediction error statistic \n",
    "    % (NRMSE measure) for the T-step ahead predictions is the goodness-of-fit \n",
    "    % statistic. \n",
    "    % The state space reconstruction is done with the method of delays having \n",
    "    % as parameters the embedding dimension 'm' and the delay time 'tau'. \n",
    "    % The local prediction model is one of the following:\n",
    "    % Ordinary Least Squares, OLS (standard local linear model): if the \n",
    "    % truncation parameter q >= m\n",
    "    % Principal Component Regression, PCR, project the parameter space of the \n",
    "    % model to only q of the m principal axes: if 0<q<m\n",
    "    % Local Average Mapping, LAM: if q=0.\n",
    "    % The local region is determined by the number of neighbours 'nnei'. \n",
    "    % The k-d-tree data structure is utilized to speed up computation time in \n",
    "    % the search of neighboring points and the implementation of Matlab is \n",
    "    % used. \n",
    "    % INPUTS:\n",
    "    %  xV      : vector of the scalar time series\n",
    "    %  tau     : the delay time (usually set to 1).\n",
    "    %  m       : the embedding dimension.\n",
    "    %  Tmax    : the prediction horizon, the fit is made for T=1...Tmax steps\n",
    "    %            ahead.\n",
    "    %  nnei    : number of nearest neighbors to be used in the local model. \n",
    "    %            If k=1,the nearest neighbor mapping is the fitted value. \n",
    "    %            If k>1, the model as defined by the input patameter 'q' is\n",
    "    %            used. \n",
    "    %  q       : the truncation parameter for a normalization of the local\n",
    "    %            linear model if specified (to project the parameter space of\n",
    "    %            the model, using Principal Component Regression, PCR, locally).\n",
    "    %            if q>=m -> Ordinary Least Squares, OLS (standard local linear\n",
    "    %                       model, no projection)\n",
    "    %            if 0<q<m -> PCR(q)\n",
    "    %            if q=0 -> local average model (if in addition nnei=1 ->\n",
    "    %            then the zeroth order model is applied)\n",
    "    %  tittxt  : string to be displayed in the title of the figure \n",
    "    %            if not specified, no plot is made\n",
    "    % OUTPUT: \n",
    "    %  nrmseV  : vector of length Tmax, the nrmse of the fit for T-mappings, \n",
    "    %            T=1...Tmax.\n",
    "    %  preM    : the matrix of size nvec x (1+Tmax) having the fit (in-sample\n",
    "    %            predictions) for T=1,...,Tmax, for each of the nvec \n",
    "    %            reconstructed points from the whole time series. The first\n",
    "    %            column has the time of the target point and the rest Tmax\n",
    "    %            columns the fits for T=1,...,Tmax time steps ahead.\n",
    "    '''\n",
    "    if q > m:\n",
    "        q = int(m)\n",
    "    n = xV.shape[0]\n",
    "\n",
    "    if n < 2 * (m-1)*tau - Tmax:\n",
    "        print('too short timeseries')\n",
    "        return\n",
    "\n",
    "    nvec = n - (m-1)*tau - Tmax\n",
    "    xM = np.full(shape=(nvec, m), fill_value=np.nan)\n",
    "\n",
    "    for j in np.arange(m):\n",
    "        xM[:, [m-j-1]] = xV[j*tau:nvec+j*tau]\n",
    "    from scipy.spatial import KDTree\n",
    "    kdtreeS = KDTree(xM)\n",
    "    preM = np.full(shape=(nvec, Tmax), fill_value=np.nan)\n",
    "    _, nneiindM = kdtreeS.query(xM, k=nnei+1, p=2)\n",
    "    nneiindM = nneiindM[:, 1:]\n",
    "    for i in np.arange(nvec):\n",
    "        neiM = xM[nneiindM[i]]\n",
    "        yV = xV[nneiindM[i] + m*tau]\n",
    "        if q == 0 or nnei == 1:\n",
    "            preM[i, 0] = np.mean(yV)\n",
    "        else:\n",
    "            mneiV = np.mean(neiM, axis=0)\n",
    "            my = np.mean(yV)\n",
    "            zM = neiM - mneiV\n",
    "            [Ux, Sx, Vx] = np.linalg.svd(zM, full_matrices=False)\n",
    "            Sx = np.diag(Sx)\n",
    "            Vx = Vx.T\n",
    "            tmpM = Vx[:, :q] @ (np.linalg.inv(Sx[:q, :q]) @ Ux[:, :q].T)\n",
    "            lsbV = tmpM @ (yV - my)\n",
    "            preM[i] = my + (xM[i, ] - mneiV) @ lsbV\n",
    "    if Tmax > 1:\n",
    "        winnowM = np.full(shape=(nvec, (m - 1) * tau + 1), fill_value=np.nan)\n",
    "        for i in np.arange(m*tau):\n",
    "            winnowM[:, [i]] = xV[i:nvec+i]\n",
    "        for T in np.arange(2, Tmax+1):\n",
    "            winnowM = np.concatenate([winnowM, preM[:, [T-2]]], axis=1)\n",
    "            targM = winnowM[:, :-(m+1)*tau:-tau]\n",
    "            _, nneiindM = kdtreeS.query(targM, k=nnei, p=2)\n",
    "\n",
    "            for i in np.arange(nvec):\n",
    "                neiM = xM[nneiindM[i], :]\n",
    "                yV = xV[nneiindM[i] + (m-1)*tau+1]\n",
    "                if q == 0 or nnei == 1:\n",
    "                    preM[i, T-1] = np.mean(yV)\n",
    "                else:\n",
    "                    mneiV = np.mean(neiM, axis=0)\n",
    "                    my = np.mean(yV)\n",
    "                    zM = neiM - mneiV\n",
    "                    [Ux, Sx, Vx] = np.linalg.svd(zM, full_matrices=False)\n",
    "                    Sx = np.diag(Sx)\n",
    "                    Vx = Vx.T\n",
    "                    tmpM = Vx[:, :q] @ (np.linalg.inv(Sx[:q, :q]) @ Ux[:, :q].T)\n",
    "                    lsbV = tmpM @ (yV - my)\n",
    "                    preM[i, T-1] = my + (targM[i, :] - mneiV) @ lsbV\n",
    "\n",
    "    nrmseV = np.full(shape=(Tmax, 1), fill_value=np.nan)\n",
    "    idx = (np.arange(nvec) + (m-1)*tau).astype(np.int)\n",
    "    for t_idx in np.arange(1, Tmax+1):\n",
    "        nrmseV[t_idx-1] = nrmse(trueV=xV[idx + t_idx, ], predictedV=preM[:, [t_idx-1]])\n",
    "    if show:\n",
    "        fig, ax = plt.subplots(1, 1)\n",
    "        ax.plot(np.arange(1, Tmax+1), nrmseV, marker='x')\n",
    "        ax.set_xlabel('prediction time T')\n",
    "        ax.set_ylabel('NRMSE(T)')\n",
    "    return nrmseV, preM\n",
    "\n",
    "def plot_timeseries(xV, get_histogram=False, title='', savepath=''):\n",
    "    # #plot timeseries\n",
    "    fig, ax = plt.subplots(1, 1, figsize=(14, 8))\n",
    "    ax.plot(xV, marker='x', linestyle='--', linewidth=2);\n",
    "    ax.set_xlabel('time')\n",
    "    ax.set_ylabel('value')\n",
    "    if len(title) > 0:\n",
    "        ax.set_title(title, x=0.5, y=1.0);\n",
    "    plt.tight_layout()\n",
    "    if len(savepath) > 0:\n",
    "        plt.savefig(f'{savepath}/{title}_xM.jpeg')\n",
    "    # #plot histogram\n",
    "    if get_histogram:\n",
    "        fig, ax = plt.subplots(1, 1, figsize=(14, 8))\n",
    "        ax.hist(xV, alpha=0.8, rwidth=0.9);\n",
    "        ax.set_xlabel('value')\n",
    "        ax.set_title('Histogram')\n",
    "        plt.tight_layout()\n",
    "        if len(title) > 0:\n",
    "            ax.set_title(title, x=0.5, y=1.0);\n",
    "        plt.tight_layout()\n",
    "        if len(savepath) > 0:\n",
    "            plt.savefig(f'{savepath}/{title}_hist.jpeg')\n",
    "            \n",
    "def ANN(X, k):\n",
    "    '''\n",
    "    helper func\n",
    "    '''\n",
    "    tree = KDTree(X, leaf_size=1, metric='chebyshev')\n",
    "    dists, nnidx = tree.query(X, k=k)\n",
    "    del tree\n",
    "    return nnidx, dists\n",
    "\n",
    "def ANNR(X, rV):\n",
    "    '''\n",
    "    helper func\n",
    "    '''\n",
    "    tree = KDTree(X, leaf_size=1, metric='chebyshev')\n",
    "    nnnidx = tree.query_radius(X, r=rV, count_only=True)\n",
    "    return nnnidx\n",
    "\n",
    "def nneighforgivenr(X, rV):\n",
    "    '''\n",
    "    helper func\n",
    "    '''\n",
    "    npV = ANNR(X, rV)\n",
    "    npV[npV == 0] = 1\n",
    "    return npV\n",
    "\n",
    "\n",
    "def mi_estimator_ksg1(xV, yV, nnei=5, normalize=False):\n",
    "    '''\n",
    "    calculates I(X;Y) using KSG algorithm1 (with max-norms squares)\n",
    "    '''\n",
    "    from scipy.special import psi\n",
    "\n",
    "    n = xV.shape[0]\n",
    "    psi_nnei = psi(nnei)\n",
    "    psi_n = psi(n)\n",
    "\n",
    "    if normalize:\n",
    "        xV = (xV - np.min(xV)) / np.ptp(xV)\n",
    "        yV = (yV - np.min(yV)) / np.ptp(yV)\n",
    "\n",
    "    xembM = np.concatenate((xV, yV), axis=1)\n",
    "    _, distsM = ANN(xembM, nnei + 1)\n",
    "    maxdistV = distsM[:, -1]\n",
    "    n_x = nneighforgivenr(X=xV, rV=maxdistV - np.ones(n) * 10 ** (-10))\n",
    "    n_y = nneighforgivenr(X=yV, rV=maxdistV - np.ones(n) * 10 ** (-10))\n",
    "    psibothM = psi(np.concatenate((n_x.reshape(-1, 1), n_y.reshape(-1, 1)), axis=1))\n",
    "    #     # I(X;Y) = ψ(k) + ψ(Ν) - <ψ(Nx + 1) + ψ(Ny + 1)>\n",
    "    mi = psi_nnei + psi_n - np.mean(np.sum(psibothM, axis=1))\n",
    "    return mi\n",
    "\n",
    "def falsenearestneighbors(xV, m_max=10, tau=1, show=False):\n",
    "    dim = np.arange(1, m_max + 1)\n",
    "    f1, _, _ = dimension.fnn(xV, tau=tau, dim=dim, window=10, metric='cityblock')\n",
    "    if show:\n",
    "        fig, ax = plt.subplots(1, 1, figsize=(14, 8))\n",
    "        ax.scatter(dim, f1)\n",
    "        ax.axhline(0.01, linestyle='--', color='red', label='1% threshold')\n",
    "        ax.set_xlabel(f'm')\n",
    "        ax.set_title(f'FNN ({m_max})')\n",
    "        ax.set_xticks(dim)\n",
    "        ax.legend()\n",
    "        plt.show()\n",
    "    return f1\n",
    "\n",
    "def correlationdimension(xV, tau, mmax, fac=4, logrmin=-1e6, logrmax=1e6, show=False):\n",
    "    m_all = np.arange(1, m_max + 1)\n",
    "    corrdimV = []\n",
    "    logrM = []\n",
    "    logCrM = []\n",
    "    polyM = []\n",
    "\n",
    "    for m in m_all:\n",
    "        corrdim, *corrData = nolds.corr_dim(xV, m, debug_data=True)\n",
    "        corrdimV.append(corrdim)\n",
    "        logrM.append(corrData[0][0])\n",
    "        logCrM.append(corrData[0][1])\n",
    "        polyM.append(corrData[0][2])\n",
    "    if show:\n",
    "        fig, ax = plt.subplots(1, 1, figsize=(14, 8))\n",
    "        ax.plot(m_all, corrdimV, marker='x', linestyle='.-')\n",
    "        ax.set_xlabel('m')\n",
    "        ax.set_xticks(m_all)\n",
    "        ax.set_ylabel('v')\n",
    "        ax.set_title('Corr Dim vs m')\n",
    "        \n",
    "        \n",
    "    return corrdimV, logrM, logCrM, polyM\n",
    "       \n",
    "def split2train_testset(xV, test_proportion):\n",
    "    n = np.int(len(xV) * test_proportion)\n",
    "    return xV[:n], xV[n:]\n",
    "\n",
    "    \n",
    "def plot_3d_attractor(xM):\n",
    "    fig = plt.figure(figsize=(14, 8))\n",
    "    ax = fig.add_subplot(111, projection='3d')\n",
    "    ax.scatter(xM[:, 0], xM[:, 1], xM[:, 2])\n",
    "    ax.plot(xM[:, 0], xM[:, 1], xM[:, 2], linestyle='--')\n",
    "    plt.show()\n",
    "\n",
    "def embed_data(x, order=3, delay=1):\n",
    "    \"\"\"Time-delay embedding.\n",
    "    Parameters\n",
    "    ----------\n",
    "    x : 1d-array, shape (n_times)\n",
    "        Time series\n",
    "    order : int\n",
    "        Embedding dimension (order)\n",
    "    delay : int\n",
    "        Delay.\n",
    "    Returns\n",
    "    -------\n",
    "    embedded : ndarray, shape (n_times - (order - 1) * delay, order)\n",
    "        Embedded time-series.\n",
    "    \"\"\"\n",
    "    N = len(x)\n",
    "    Y = np.empty((order, N - (order - 1) * delay))\n",
    "    for i in range(order):\n",
    "        Y[i] = x[i * delay:i * delay + Y.shape[1]]\n",
    "    return Y.T\n",
    "\n",
    "def logisticmap(n=1024, r=3., x0=None):\n",
    "    ntrans = 10\n",
    "    xV = np.full(shape=(n+ntrans, 1), fill_value=np.nan)\n",
    "    if x0 is None:\n",
    "        xV[0] = np.random.uniform(low=0, high=2.9)\n",
    "    else:\n",
    "        xV[0] = x0\n",
    "    for t in np.arange(1, n+ntrans):\n",
    "        xV[t] = r * xV[t-1] * (1 - xV[t-1])\n",
    "    xV = xV[ntrans:, [0]]\n",
    "    return xV.reshape(-1, )\n",
    "\n",
    "def generate_arma_ts(phiV, thetaV, n, sdnoise=1):\n",
    "    '''\n",
    "    Generate an ARMA(p,q) time series of length 'n' with Gaussian input noise.\n",
    "    Note that phiV = [phi(0) phi(1) ... phi(p)]' and phi(0) is the constant\n",
    "    term, and thetaV = [theta(1) ... theta(q)]'.\n",
    "    sdnoise is the SD of the input noise (if left out then sdnoise=1).\n",
    "    The generating ARMA(p,q) process reads\n",
    "    x(t) = phi(0) + phi(1)*x(t-1) + ... + phi(p)*x(t-p) +\n",
    "            +z(t) - theta(1)*z(t-1) + ... - theta(q)*z(t-p),\n",
    "    z(t) ~ WN(0,sdnoise^2)\n",
    "    '''\n",
    "    phiV = np.array(phiV)\n",
    "    thetaV = np.array(thetaV)\n",
    "    ar_params = np.r_[1, -phiV[:]]  # add zero lag\n",
    "    ma_params = np.r_[1, thetaV[:]]  # add zero lag\n",
    "    xV = arma_generate_sample(ar=ar_params, ma=ma_params, nsample=n, scale=sdnoise, burnin=100)\n",
    "    return xV\n",
    "   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "n = 100\n",
    "x0 = 0.51\n",
    "r = 3.9\n",
    "xV = logisticmap(n=n, r=r, x0=x0)\n",
    "plt.figure()\n",
    "plt.plot(xV)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "embedded = embed_data(xV, order=2, delay=1)\n",
    "# embedded;\n",
    "plt.scatter(xV[:-1], xV[1:])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "xV = generate_arma_ts([0.8], [0], n)\n",
    "plt.plot(xV);\n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "embedded = embed_data(xV, order=2, delay=1)\n",
    "# embedded;\n",
    "plt.scatter(xV[:-1], xV[1:])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "xM = data.henon();\n",
    "xM, xM.shape\n",
    "plt.plot(xM);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(14, 8))\n",
    "xM = xM[:, 0]\n",
    "plt.plot(xM);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "embedded = embed_data(xM, order=3, delay=1)\n",
    "embedded\n",
    "plot_3d_attractor(embedded)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## REAL DATA ATTRACTORS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "os.chdir('d:/timeserieslab/')\n",
    "df = pd.read_csv('./data/BTCUSDT.csv')\n",
    "df.set_index(pd.to_datetime(df['time']), inplace=True)\n",
    "df.drop('time', axis=1, inplace=True)\n",
    "df = df['01-01-2020':'12-31-2021']\n",
    "df.plot();\n",
    "logreturns = df.apply(lambda x: np.log(x)).diff().bfill()\n",
    "logreturns.plot();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "embedded = embed_data(logreturns.values.reshape(-1,), order=2, delay=1)\n",
    "plt.scatter(embedded[:-1], embedded[1:]);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "eeg = pd.read_csv('./data/epileeg.dat')\n",
    "# eeg.plot();\n",
    "embedded = embed_data(eeg.values.reshape(-1,), order=3, delay=1)\n",
    "plot_3d_attractor(embedded)\n",
    "#plt.scatter(embedded[:-1], embedded[1:], linestyle='--', marker='x');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Check for stationarity (Visual Inspection + statistical test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "xM = xM[2900:3000]\n",
    "plt.figure(figsize=(14, 8))\n",
    "plt.plot(xM);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from statsmodels.tsa.stattools import adfuller\n",
    "adf = adfuller(xM, maxlag=None)\n",
    "print(adf)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Compute autocorrelation and delayed mutual information.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(dir(delay))\n",
    "help(delay.acorr)\n",
    "xM.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "maxtau = 10\n",
    "lag = np.arange(maxtau)\n",
    "r = delay.acorr(xM, maxtau=maxtau)\n",
    "i = delay.dmi(xM, maxtau=maxtau)\n",
    "r_delay = np.argmax(r < 1.0 / np.e)\n",
    "\n",
    "plt.figure(1, figsize=(14, 8))\n",
    "\n",
    "plt.subplot(211)\n",
    "plt.title(r'Delay estimation for Henon map')\n",
    "plt.ylabel(r'Delayed mutual information')\n",
    "plt.plot(lag, i, marker='o')\n",
    "\n",
    "plt.subplot(212)\n",
    "plt.xlabel(r'Time delay $\\tau$')\n",
    "plt.ylabel(r'Autocorrelation')\n",
    "plt.plot(lag, r, r_delay, r[r_delay], 'o')\n",
    "plt.axhline(1.0 / np.e, linestyle='--', alpha=0.7, color='red' )\n",
    "\n",
    "\n",
    "plt.figure(2, figsize=(14, 8))\n",
    "plt.subplot(111)\n",
    "plt.title(r'Time delay = %d' % r_delay)\n",
    "plt.xlabel(r'$x(t)$')\n",
    "plt.ylabel(r'$x(t + \\tau)$')\n",
    "plt.plot(xM[:-r_delay], xM[r_delay:], '.')\n",
    "\n",
    "plt.show()\n",
    "print(r)\n",
    "print(r'Autocorrelation time = %d' % r_delay)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dim = np.arange(1, 10 + 1)\n",
    "f1, f2, f3 = dimension.fnn(xM, tau=1, dim=dim, window=10, metric='cityblock')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.title(r'FNN for Henon map')\n",
    "plt.xlabel(r'Embedding dimension $d$')\n",
    "plt.ylabel(r'FNN (%)')\n",
    "plt.plot(dim, 100 * f1, 'bo--', label=r'Test I')\n",
    "plt.plot(dim, 100 * f2, 'g^--', label=r'Test II')\n",
    "plt.plot(dim, 100 * f3, 'rs-', label=r'Test I + II')\n",
    "plt.legend()\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Correlation Sum"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(14, 8))\n",
    "plt.title('Local $D_2$ vs $r$ for Henon map')\n",
    "plt.xlabel(r'Distance $r$')\n",
    "plt.ylabel(r'Local $D_2$')\n",
    "theiler_window = 10\n",
    "tau = 1\n",
    "dim = np.arange(1, 10 + 1)\n",
    "\n",
    "for r, c in d2.c2_embed(xM, tau=tau, dim=dim, window=theiler_window,\n",
    "                        r=utils.gprange(0.001, 1.0, 100)):\n",
    "    plt.semilogx(r[3:-3], d2.d2(r, c), color='#4682B4')\n",
    "\n",
    "plt.semilogx(utils.gprange(0.001, 1.0, 100), 1.220 * np.ones(100),\n",
    "             color='#000000')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "r = n_utils.gprange(0.001, 1.0, 100)\n",
    "corr_dim, debug_data = nolds.corr_dim(xM, emb_dim=2, rvals=r, debug_data=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rvals = debug_data[0] #values used for log(r)\n",
    "csums = debug_data[1] #the corresponding log(C(r))\n",
    "poly = debug_data[2] #line coefficients ([slope, intercept])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = plt.figure(figsize=(14, 8))\n",
    "ax = fig.add_subplot(111)\n",
    "ax.plot(rvals, csums)\n",
    "ax.set_xlabel('log(r)')\n",
    "ax.set_ylabel('log(C(r))')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Predictions\n",
    "- LAP\n",
    "- LLP"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "n = xM.shape[0]\n",
    "test_prop = 0.3\n",
    "split_point = int(n * (1 - test_prop))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_xM = xM[:split_point]\n",
    "test_xM = xM[split_point:]\n",
    "\n",
    "plt.figure(figsize=(16, 8))\n",
    "plt.plot(np.arange(split_point), train_xM, color='blue', alpha=0.7, label='train set');\n",
    "plt.plot(np.arange(split_point, xM.shape[0]), test_xM, color='red', linestyle='--', alpha=0.7, label='test set');\n",
    "plt.legend()\n",
    "plt.xlabel('Time');\n",
    "plt.ylabel('Value');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KDTree\n",
    "embed_train_data = embed_data(train_xM, 2, 1)\n",
    "embed_test_data = embed_data(test_xM, 2, 1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "knn = 5\n",
    "tree = KDTree(embed_train_data, leaf_size=1, metric='chebyshev')\n",
    "neighbors_idx = []\n",
    "for i, state in enumerate(embed_test_data):\n",
    "    dist, neigh_idx = tree.query(state.reshape(1, -1), k=knn)\n",
    "    neighbors_idx.append(tuple([i, neigh_idx[0]]))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(16, 8))\n",
    "plt.plot(xM)\n",
    "plt.xlabel('time')\n",
    "plt.ylabel('xM')\n",
    "plt.axvline(split_point, linestyle='--', color='black', alpha=0.5)\n",
    "# #get neighbors of first test datapoint  \n",
    "check_test_point = 1\n",
    "neigh = neighbors_idx[check_test_point]\n",
    "print(neigh)\n",
    "test_state_idx = neigh[0]\n",
    "neighs_idx = neigh[1]\n",
    "plt.plot([neighs_idx, neighs_idx+1], [xM[neighs_idx], xM[neighs_idx+1]], linestyle='--', color='orange' )\n",
    "plt.scatter([neighs_idx, neighs_idx+1], [xM[neighs_idx], xM[neighs_idx+1]], linestyle='--', color='orange')\n",
    "# #test set state\n",
    "plt.plot([test_state_idx+split_point, test_state_idx+split_point+1], [xM[test_state_idx+split_point], xM[test_state_idx+split_point+1]], linestyle='--', color='red' )\n",
    "plt.scatter([test_state_idx+split_point, test_state_idx+split_point+1], [xM[test_state_idx+split_point], xM[test_state_idx+split_point+1]], linestyle='--', color='red')\n",
    "    \n",
    "plt.figure(figsize=(16, 8))\n",
    "plt.scatter(embed_train_data[:, 0], embed_train_data[:, 1], alpha=0.3)\n",
    "plt.scatter(embed_train_data[neighs_idx][:, 0], embed_train_data[neighs_idx][:, 1], label=f'neighbors in train data')\n",
    "plt.scatter(embed_test_data[test_state_idx, 0], embed_test_data[test_state_idx, 1], label=f'test point:{test_state_idx}')\n",
    "plt.xlabel('x(t)')\n",
    "plt.ylabel('x(t+1)')\n",
    "plt.legend()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "T = 1;\n",
    "lap_predictions = []\n",
    "for neigh in neighbors_idx:\n",
    "    test_state_idx = neigh[0]\n",
    "    neighs_idx = neigh[1]\n",
    "    images_idx = neighs_idx + T\n",
    "    images = xM[images_idx]\n",
    "    lap = np.sum(images) / len(images)\n",
    "    lap_predictions.append(lap)\n",
    "plt.figure(figsize=(14, 8))\n",
    "plt.plot(np.arange(split_point+1, xM.shape[0]), lap_predictions, label='LAP prediction', alpha=0.9, linestyle='-.')\n",
    "plt.plot(np.arange(split_point+1, xM.shape[0]), test_xM[1:,], label='True values', alpha=0.9, linestyle='--')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.sqrt(np.mean((np.array(lap_predictions) -  test_xM[1:,])**2))/np.std(test_xM[1:,])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X, y = [], []\n",
    "for neigh in neigh_idx:\n",
    "    x_ =  [xM[neigh], xM[neigh+1]]\n",
    "    y_ = [xM[neigh+1+T]]\n",
    "    X.append(x_)\n",
    "    y.append(y_)\n",
    "import statsmodels.api as sm\n",
    "X = sm.add_constant(X)\n",
    "X = np.asarray(X)\n",
    "y = np.asarray(y)\n",
    "ols = sm.OLS(endog=y, exog=X).fit()\n",
    "llp = np.dot(ols.params, [1, xM[i+split_point], xM[i+split_point+1]])\n",
    "ols.summary()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Real Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('epileeg.dat', 'r') as file:\n",
    "    lines = file.readlines()\n",
    "xM = np.full(shape=len(lines), fill_value=np.nan)\n",
    "for i, line in enumerate(lines):\n",
    "    point = line.rstrip().lstrip()\n",
    "    xM[i] = point\n",
    "xM = np.array(xM)\n",
    "xM\n",
    "plt.plot(xM)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
