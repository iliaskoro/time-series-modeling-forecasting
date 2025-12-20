# Imports
import nolds
import numpy as np
import matplotlib
import pandas as pd
import mpld3
import statsmodels
import sklearn
import pmdarima as pm
import os

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_process import arma_generate_sample, arma_acf
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARIMA

os.getcwd()
def plot_timeseries(xV, get_histogram=False, title='', savepath=''):

# plot timeseries
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(xV, marker='x', linestyle='--', linewidth=2)
    ax.set_xlabel('time')
    ax.set_ylabel('value')

    if len(title) > 0:
        ax.set_title(title, x=0.5, y=1.0)
        plt.tight_layout()

    if len(savepath) > 0:
        plt.savefig(f'{savepath}/{title}_xM.jpeg')

# plot histogram
    if get_histogram:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.hist(xV, alpha=0.8, rwidth=0.9)
        ax.set_xlabel('value')
        ax.set_title('Histogram')
        plt.tight_layout()

        if len(title) > 0:
            ax.set_title(title, x=0.5, y=1.0)
            plt.tight_layout()

        if len(savepath) > 0:
            plt.savefig(f'{savepath}/{title}_hist.jpeg')

    def read_datfile(path):
    xV = np.loadtxt(path)
    return xV

#     "def rolling_window(xV, window):\n",
#     "    '''\n",
#     "    returns moving average of a time series xV \n",
#     "    with length of window\n",
#     "    '''\n",
#     "    xV = xV.flatten()\n",
#     "    return np.convolve(xV, np.ones(window)/window, mode='same') \n",
#
#     "def polynomial_fit(xV, p):\n",
#     "    '''\n",
#     "    fit to a given time series with a polynomial of a given order.\n",
#     "    :param xV: vector of length 'n' of the time series\n",
#     "    :param p: the order of the polynomial to be fitted\n",
#     "    :return: vector of length 'n' of the fitted time series\n",
#
#     "    n = xV.shape[0]\n",
#     "    xV = xV[:]\n",
#     "    if p > 1:\n",
#     "        tV = np.arange(n)\n",
#     "        bV = np.polyfit(x=tV, y=xV, deg=p)\n",
#     "        muV = np.polyval(p=bV, x=tV)\n",
#     "    else:\n",
#     "        muV = np.full(shape=n, fill_value=np.nan)\n",
#     "    return muV\n",
#     "\n",
#     "\n",
#     "def seasonal_components(xV, period):\n",
#     "    '''\n",
#     "    computes the periodic time series comprised of repetetive\n",
#     "    patterns of seasonal components given a time series and the season\n",
#     "    (period).\n",
#     "    '''\n",
#     "    n = xV.shape[0]\n",
#     "    sV = np.full(shape=(n,), fill_value=np.nan)\n",
#     "    monV = np.full(shape=(period,), fill_value=np.nan)\n",
#     "    for i in np.arange(period):\n",
#     "        monV[i] = np.mean(xV[i:n:period])\n",
#     "    monV = monV - np.mean(monV)\n",
#     "    for i in np.arange(period):\n",
#     "         sV[i:n:period] = monV[i] * np.ones(shape=len(np.arange(i, n, period)))\n",
#     "    return sV\n",
#     "\n",
#     "\n",
#     "\n",
#     "def generate_arma_ts(phiV, thetaV, n, sdnoise=1):\n",
#     "    '''\n",
#     "    Generate an ARMA(p,q) time series of length 'n' with Gaussian input noise.\n",
#     "    Note that phiV = [phi(0) phi(1) ... phi(p)]' and phi(0) is the constant\n",
#     "    term, and thetaV = [theta(1) ... theta(q)]'.\n",
#     "    sdnoise is the SD of the input noise (if left out then sdnoise=1).\n",
#     "    The generating ARMA(p,q) process reads\n",
#     "    x(t) = phi(0) + phi(1)*x(t-1) + ... + phi(p)*x(t-p) +\n",
#     "            +z(t) - theta(1)*z(t-1) + ... - theta(q)*z(t-p),\n",
#     "    z(t) ~ WN(0,sdnoise^2)\n",
#     "    '''\n",
#     "    phiV = np.array(phiV)\n",
#     "    thetaV = np.array(thetaV)\n",
#     "    ar_params = np.r_[1, -phiV[:]]  # add zero lag\n",
#     "    ma_params = np.r_[1, thetaV[:]]  # add zero lag\n",
#     "    xV = arma_generate_sample(ar=ar_params, ma=ma_params, nsample=n, scale=sdnoise, burnin=100)\n",
#     "    return xV\n",
#     "    # q = len(thetaV)\n",
#     "    # p = len(phiV) - 1\n",
#     "    # pq = np.max(p, q)\n",
#     "    # ntrans = 100 + pq\n",
#     "    # phiV = phiV[:]\n",
#     "    # thetaV = thetaV[:]\n",
#     "    # if p > 0:\n",
#     "    #     root_arV = np.roots(np.r_[1, -phiV[1:]])\n",
#     "    #     if np.any(np.abs(root_arV) >= 1):\n",
#     "    #         print(f'The AR({p}) part of the process is not stationary.\\n')\n",
#     "    # if q > 0:\n",
#     "    #     root_maV = np.roots(np.r_[1, -thetaV[1:]])\n",
#     "    #     if np.any(np.abs(root_maV) >= 1):\n",
#     "    #         print(f'The MA({p}) part of the process is not stationary.\\n')\n",
#     "    # x0V = sdnoise * np.random.normal(size=(pq, 1))\n",
#     "    # zV = sdnoise * np.random.normal(size=(n+ntrans, 1))\n",
#     "    # xV = np.full(shape=(n+ntrans, 1), fill_value=np.nan)\n",
#     "    # xV[:pq] = x0V\n",
#     "    # if p == 0:\n",
#     "    #     for i in np.arange(pq+1, n+ntrans):\n",
#     "    #         xV[i] = phiV[0] + zV[i] - thetaV * np.flipud(zV[i - q:i])\n",
#     "    # elif q == 0:\n",
#     "    #     for i in np.arange(pq+1, n+ntrans):\n",
#     "    #         xV[i] = phiV[0] + phiV[1: p+1] * np.flipud(xV[i-p:i-1]) + zV[i]\n",
#     "    # else:\n",
#     "    #     for i in np.arange(pq+1, n+ntrans):\n",
#     "    #         xV[i] = phiV[0] + phiV[1:p+1] * np.flipud(xV[i-p:i-1]) + zV[i] - thetaV * np.flipud(zV[i-q:i-1])\n",
#     "    # xV = xV[ntrans + 1:]\n",
#     "    # return xV\n",
#     "\n",
#     "def add_stochastic_trend(xV):\n",
#     "    '''\n",
#     "    adds a stochastic trend to a given time series (for\n",
#     "    simulating purposes). The time series of stochastic trend is generated by\n",
#     "    simulating a smoothed random walk time series of the same length as that of the\n",
#     "    given time series.\n",
#     "    :param xV: vector of length 'n' of the given time series\n",
#     "    :return: vector of length 'n' of the sum of the given time series and a stochastic trend\n",
#     "    '''\n",
#     "    xV = xV[:]\n",
#     "    n = xV.shape[0]\n",
#     "    maorder = np.round(n // 5)\n",
#     "    x_std = np.std(xV)\n",
#     "    zV = 0.1 * x_std * np.random.normal(0, 1, n)\n",
#     "    zV = np.cumsum(zV)\n",
#     "    wV = rolling_window(zV, window=maorder)\n",
#     "    yV = xV + wV\n",
#     "    return yV\n",
#     "\n",
#     "def add_seasonality(xV, period):\n",
#     "    '''\n",
#     "    adds a seasonal component to a given time series (for\n",
#     "    simulating purposes). The time series of seasonality is generated by\n",
#     "    cosine function of a given period 'per' and amplitude equal to the\n",
#     "    standard deviation of the given time series.\n",
#     "    '''\n",
#     "    n = xV.shape[0]\n",
#     "    xV = xV[:]\n",
#     "    x_sd = np.std(xV)\n",
#     "    zV = x_sd * np.cos(2*np.pi*np.arange(n)/period)\n",
#     "    return xV + zV\n",
#     "\n",
#     "\n",
#     "def armacoefs2autocorr(phiV, thetaV, lags=10):\n",
#     "    '''\n",
#     "    Theoretical autocorrelation function of an ARMA process.\n",
#     "    phiV: The coefficients for autoregressive lag polynomial, not including zero lag.\n",
#     "    thetaV : array_like, 1d\n",
#     "        The coefficients for moving-average lag polynomial, not including zero lag.\n",
#     "    '''\n",
#     "    phiV, thetaV = np.array(phiV), np.array(thetaV)\n",
#     "    phiV = np.r_[1, -phiV] # add zero lag\n",
#     "    thetaV = np.r_[1, thetaV] # #add zero lag\n",
#     "    acf_ = arma_acf(phiV, thetaV, lags=lags)\n",
#     "    fig, ax = plt.subplots(1, 1, figsize=(14, 8))\n",
#     "    ax.scatter(np.arange(1, lags), acf_[1:], marker='o')\n",
#     "    ax.set_xlabel('Lags')\n",
#     "    ax.set_xticks(np.arange(1, lags))\n",
#     "    ax.set_yticks(np.arange(-1, 1, 0.1))\n",
#     "    ax.set_title('ACF', fontsize=14)\n",
#     "    ax.grid(linestyle='--', linewidth=0.5, alpha=0.15)\n",
#     "    plt.show()\n",
#     "#     for t in np.arange(lags):\n",
#     "#         ax.axvline(t, ymax=acf_[t], color='red', alpha=0.3);\n",
#     "\n",
#     "# def macoef2autocorr(phiV, thetaV, lags=10):\n",
#     "#     from statsmodels.tsa.arima_process import arma_pacf\n",
#     "#     pacf_ = arma_pacf(phiV, thetaV, lags=10)\n",
#     "#     fig, ax = plt.subplots(1, 1)\n",
#     "#     ax.scatter(np.arange(lags), pacf_, marker='o');\n",
#     "#     for t in np.arange(lags):\n",
#     "#         ax.axvline(t, ymax=pacf_[t], color='red', alpha=0.3);\n",
#     "        \n",
#     "\n",
#     "def plot_3d_attractor(xM):\n",
#     "    '''\n",
#     "    plot 3d attractor\n",
#     "    '''\n",
#     "    fig = plt.figure(figsize=(14, 8))\n",
#     "    ax = fig.add_subplot(111, projection='3d')\n",
#     "    ax.scatter(xM[:, [0]], xM[:, [1]], xM[:, [2]])\n",
#     "    plt.show()\n",
#     "\n",
#     "def embed_data(xV, m=3, tau=1):\n",
#     "    \"\"\"Time-delay embedding.\n",
#     "    Parameters\n",
#     "    ----------\n",
#     "    x : 1d-array, shape (n_times)\n",
#     "        Time series\n",
#     "    m : int\n",
#     "        Embedding dimension (order)\n",
#     "    tau : int\n",
#     "        Delay.\n",
#     "    Returns\n",
#     "    -------\n",
#     "    embedded : ndarray, shape (n_times - (order - 1) * delay, order)\n",
#     "        Embedded time-series.\n",
#     "    \"\"\"\n",
#     "    N = len(xV)\n",
#     "    nvec = N - (m-1)*tau\n",
#     "    xM = np.zeros(shape=(nvec, m))\n",
#     "    for i in np.arange(m):\n",
#     "        xM[:, m-i-1] = xV[i*tau:nvec+i*tau]\n",
#     "    return xM\n",
#     "\n",
#     "def get_acf(xV, lags=10, alpha=0.05, show=True):\n",
#     "    '''\n",
#     "    calculate acf of timeseries xV to lag (lags) and show\n",
#     "    figure with confidence interval with (alpha)\n",
#     "    '''\n",
#     "    acfV = acf(xV, nlags=lags)[1:]\n",
#     "    z_inv = norm.ppf(1-alpha/2)\n",
#     "    upperbound95 = z_inv / np.sqrt(xV.shape[0])\n",
#     "    lowerbound95 = -upperbound95\n",
#     "    if show:\n",
#     "        fig, ax = plt.subplots(1, 1, figsize=(14,8))\n",
#     "        ax.plot(np.arange(1, lags+1), acfV, marker='o')\n",
#     "        ax.axhline(upperbound95, linestyle='--', color='red', label=f'Conf. Int {(1-alpha)*100}%')\n",
#     "        ax.axhline(lowerbound95, linestyle='--', color='red')\n",
#     "        ax.set_title('ACF')\n",
#     "        ax.set_xlabel('Lag')\n",
#     "        ax.set_xticks(np.arange(1, lags+1))\n",
#     "        ax.grid(linestyle='--', linewidth=0.5, alpha=0.15)\n",
#     "        ax.legend()\n",
#     "    return acfV  \n",
#     "\t\n",
#     "def get_pacf(xV, lags=10, alpha=0.05, show=True):\n",
#     "    '''\n",
#     "    calculate pacf of timeseries xV to lag (lags) and show\n",
#     "    figure with confidence interval with (alpha)\n",
#     "    '''\n",
#     "    pacfV = pacf(xV, nlags=lags)[1:]\n",
#     "    z_inv = norm.ppf(1-alpha/2)\n",
#     "    upperbound95 = z_inv / np.sqrt(xV.shape[0])\n",
#     "    lowerbound95 = -upperbound95\n",
#     "    if show:\n",
#     "        fig, ax = plt.subplots(1, 1, figsize=(14,8))\n",
#     "        ax.plot(np.arange(1, lags+1), pacfV, marker='o')\n",
#     "        ax.axhline(upperbound95, linestyle='--', color='red', label=f'Conf. Int {(1-alpha)*100}%')\n",
#     "        ax.axhline(lowerbound95, linestyle='--', color='red')\n",
#     "        ax.set_title('PACF')\n",
#     "        ax.set_xlabel('Lag')\n",
#     "        ax.set_xticks(np.arange(1, lags+1))\n",
#     "        ax.grid(linestyle='--', linewidth=0.5, alpha=0.15)\n",
#     "        ax.legend()\n",
#     "    return pacfV  \n",
#     "\n",
#     "def portmanteau_test(xV, maxtau, show=False):\n",
#     "    '''\n",
#     "    PORTMANTEAULB hypothesis test (H0) for independence of time series:\n",
#     "    tests jointly that several autocorrelations are zero.\n",
#     "    It computes the Ljung-Box statistic of the modified sum of \n",
#     "    autocorrelations up to a maximum lag, for maximum lags \n",
#     "    1,2,...,maxtau. \n",
#     "    '''\n",
#     "    ljung_val, ljung_pval = acorr_ljungbox(xV, lags=maxtau)\n",
#     "    if show:\n",
#     "        fig, ax = plt.subplots(1, 1)\n",
#     "        ax.scatter(np.arange(len(ljung_pval)), ljung_pval)\n",
#     "        ax.axhline(0.05, linestyle='--', color='r')\n",
#     "        ax.set_title('Ljung-Box Portmanteau test')\n",
#     "        ax.set_yticks(np.arange(0, 1.1))\n",
#     "        plt.show()\n",
#     "    return ljung_val, ljung_pval\n",
#     "\n",
#     "def fit_arima_model(xV, p, q, d=0, show=False):\n",
#     "    '''\n",
#     "    fit ARIMA(p, d, q) in xV\n",
#     "    returns: summary (table), fittedvalues, residuals, model, AIC\n",
#     "    '''\n",
#     "    model = ARIMA(xV, order=(p, d, q)).fit()\n",
#     "    summary = model.summary()\n",
#     "    fittedvalues = model.fittedvalues\n",
#     "    fittedvalues = np.array(fittedvalues).reshape(-1, 1)\n",
#     "    resid = model.resid\n",
#     "    if show:\n",
#     "        fig, ax = plt.subplots(1, 1, figsize=(14, 8))\n",
#     "        ax.plot(xV, label='Original', color='blue')\n",
#     "        ax.plot(fittedvalues, label='FittedValues', color='red', linestyle='--', alpha=0.9)\n",
#     "        ax.legend()\n",
#     "        ax.set_title(f'ARIMA({p}, {d}, {q})')\n",
#     "        fig, ax = plt.subplots(2, 1, figsize=(14, 8))\n",
#     "        ax[0].hist(resid, label='Residual')\n",
#     "        ax[1].scatter(np.arange(len(resid)),resid)\n",
#     "        plt.title('Residuals')\n",
#     "        plt.legend()\n",
#     "    return summary, fittedvalues, resid, model, model.aic\n",
#     "\n",
#     "def calculate_fitting_error(xV, model, Tmax=20, show=False):\n",
#     "    '''\n",
#     "    calculate fitting error with NRMSE for given model in timeseries xV\n",
#     "    till prediction horizon Tmax\n",
#     "    returns:\n",
#     "    nrmseV\n",
#     "    preds: for timesteps T=1, 2, 3\n",
#     "    '''\n",
#     "    nrmseV = np.full(shape=Tmax, fill_value=np.nan)\n",
#     "    nobs = len(xV)\n",
#     "    xV_std = np.std(xV)\n",
#     "    vartar = np.sum((xV - np.mean(xV)) ** 2)\n",
#     "    predM = []\n",
#     "    tmin = np.max([len(model.arparams), len(model.maparams), 1]) # start prediction after getting all lags needed from model\n",
#     "    for T in np.arange(1, Tmax):\n",
#     "        errors = []\n",
#     "        predV = np.full(shape=nobs, fill_value=np.nan)\n",
#     "        for t in np.arange(tmin, nobs-T):\n",
#     "            pred_ = model.predict(start=t, end=t+T-1, dynamic=True)\n",
#     "            # predV.append(pred_[-1])\n",
#     "            ytrue = xV[t+T-1]\n",
#     "            predV[t+T-1] = pred_[-1]\n",
#     "            error = pred_[-1] - ytrue\n",
#     "            errors.append(error)\n",
#     "        predM.append(predV)\n",
#     "        errors = np.array(errors)\n",
#     "        mse = np.mean(np.power(errors, 2))\n",
#     "        rmse = np.sqrt(mse)\n",
#     "        nrmseV[T] = (rmse / xV_std)\n",
#     "        # nrmseV[T] = (np.sum(errors**2) / vartar)\n",
#     "    if show:\n",
#     "        fig, ax = plt.subplots(1, 1, figsize=(14, 8))\n",
#     "        ax.plot(np.arange(1, Tmax), nrmseV[1:], marker='x', label='NRMSE');\n",
#     "        ax.axhline(1, color='red', linestyle='--');\n",
#     "        ax.set_title('Fitting Error')\n",
#     "        ax.legend()\n",
#     "        ax.set_xlabel('T')\n",
#     "        ax.set_xticks(np.arange(1, Tmax))\n",
#     "        plt.show()\n",
#     "        # #plot multistep prediction for T=1, 2, 3\n",
#     "        fig, ax = plt.subplots(1, 1, figsize=(14,8))\n",
#     "        ax.plot(xV, label='original')\n",
#     "        colors = ['red', 'green', 'black']\n",
#     "        for i, preds in enumerate(predM[:3]):\n",
#     "            ax.plot(preds, color=colors[i], linestyle='--', label=f'T={i+1}', alpha=0.7)\n",
#     "        ax.legend(loc='best')\n",
#     "        plt.show()\n",
#     "    return nrmseV, predM\n",
#     "\n",
#     "\n",
#     "def predict_multistep(model, Tmax=10, show=False):\n",
#     "    tmin = np.max([len(model.arparams), len(model.maparams), 1]) # start prediction after getting all lags needed from model\n",
#     "    preds = model.predict(start=tmin, end=Tmax, dynamic=True)\n",
#     "    if show:\n",
#     "        fig, ax = plt.subplots(1, 1, figsize=(14, 8))\n",
#     "        ax.plot(preds)\n",
#     "        ax.set_title('Multistep prediction')\n",
#     "        ax.set_xlabel('T')\n",
#     "        plt.show\n",
#     "    return preds\n",
#     "\n",
#     "def gaussianisation(data):\n",
#     "    '''\n",
#     "    transform a variable of any distribution \n",
#     "    into normal\n",
#     "    '''\n",
#     "    sort_ind = np.argsort(data)\n",
#     "    gaussian_data = np.random.normal(0, 1, size=data.shape[0])\n",
#     "    gaussian_data_ind = np.argsort(gaussian_data)\n",
#     "    g_d_sorted = gaussian_data[gaussian_data_ind]\n",
#     "    y = np.zeros(shape=data.shape[0])\n",
#     "    for i in np.arange(data.shape[0]):\n",
#     "        y[i] = g_d_sorted[sort_ind[i]]\n",
#     "    return y\n",
#     "\n",
#     "def get_nrmse(target, predicted):\n",
#     "    se = (target - predicted)**2\n",
#     "    mse = np.mean(se)\n",
#     "    rmse = np.sqrt(mse)\n",
#     "    return rmse/np.std(target)"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "## GENERATE DATA"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "n = 1000\n",
#     "sd_noise = 1\n",
#     "mux = 0\n",
#     "###white noise\n",
#     "# xV = np.random.normal(0, sd_noise, n) + mux\n",
#     "###random walk\n",
#     "# xV = np.cumsum(xV)\n",
#     "###AR(1)\n",
#     "xV = generate_arma_ts(phiV=[0.9], thetaV=[0.], n=n)\n",
#     "plot_timeseries(xV)\n"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "_ = get_acf(xV)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "plot_acf(xV, zero=False, lags=10);\n",
#     "plot_pacf(xV, zero=False, lags=10);\n"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "# #add stochastic trend\n",
#     "xV = add_stochastic_trend(xV)\n",
#     "plot_timeseries(xV)\n",
#     "plot_acf(xV, zero=False, lags=10);\n"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "# #add seasonality\n",
#     "perseason = 12\n",
#     "xV = add_seasonality(xV, period=perseason)\n",
#     "plot_timeseries(xV)\n",
#     "plot_acf(xV, zero=False, lags=10);"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "## REMOVE TREND"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "window = 15\n",
#     "ma = rolling_window(xV=xV, window=window)\n",
#     "plt.figure()\n",
#     "plt.plot(ma, linestyle='--')\n",
#     "plt.plot(xV, alpha=0.5)\n",
#     "plt.plot(xV-ma, alpha=0.5)\n",
#     "plt.legend([f'MA', 'Original', 'Detrended'])"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "xV_df = pd.DataFrame(xV)\n",
#     "xV_df.rolling(window=window, min_periods=1).mean().plot()\n"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "###polynomial fit\n",
#     "p = 3;\n",
#     "pol = polynomial_fit(xV, p=p)\n",
#     "plt.plot(pol)\n",
#     "plt.plot(xV, alpha=0.5)\n",
#     "plt.plot(xV-pol, alpha=0.5)\n",
#     "plt.legend([f'Pol({p})', 'Original', 'Detrended'])"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "detrended = xV - ma;\n",
#     "plt.plot(detrended)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "plot_acf(detrended, zero=False);"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "###first diffs\n",
#     "fd = np.diff(xV)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "plt.plot(fd)"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "## REAL DATA"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "os.chdir('d:/timeserieslab/')\n",
#     "df = pd.read_csv('./data/BTCUSDT.csv')"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "df.head()"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "df.set_index(pd.to_datetime(df['time']), inplace=True)\n",
#     "df.drop('time', axis=1, inplace=True)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "df.plot()"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "df = df['01-01-2020':'12-31-2020']\n",
#     "df.plot();"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "xV = df.values"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "get_acf(xV, );"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "xV = np.log(df).diff().bfill()"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "xV.plot();"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "xV = xV.values"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "plot_acf(xV, zero=False);"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "plot_pacf(xV, zero=False);"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "ljung_val, ljung_pval = portmanteau_test(xV, maxtau=10, show=True)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "summary, fittedvalues, resid, model, aic = fit_arima_model(xV=xV, p=1, q=1, d=0, show=True)\n",
#     "aic"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "best_aic = np.inf\n",
#     "best_p = None\n",
#     "best_q = None\n",
#     "for p in np.arange(1, 6, dtype=np.int):\n",
#     "    for q in np.arange(0, 6, dtype=np.int):\n",
#     "        try:\n",
#     "            _, _, _, _, aic = fit_arima_model(xV=xV, p=p, q=q, d=0, show=False)\n",
#     "        except ValueError as err:\n",
#     "            print(f'p:{p} - q:{q} - err:{err}')\n",
#     "            continue\n",
#     "        print(f'p:{p} - q:{q} - aic:{aic}')\n",
#     "        if aic < best_aic:\n",
#     "            best_p = p\n",
#     "            best_q = q\n",
#     "            best_aic = aic\n",
#     "print(f'AR order:{best_p}')\n",
#     "print(f'MA order:{best_q}')\n",
#     "print(f'Best AIC:{best_aic}')"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "summary, fittedvalues, resid, model, aic = fit_arima_model(xV=xV, p=best_p, q=best_q, d=0, show=True)\n",
#     "summary"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "nrmseV, predM = calculate_fitting_error(xV, model, Tmax=10, show=True)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "xV = np.log(df).diff().bfill()\n",
#     "xV_sq = xV ** 2\n",
#     "xV_sq.plot();"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "plot_acf(xV_sq, zero=False);\n",
#     "# utils.get_pacf(xV_sq)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "plot_pacf(xV_sq, zero=False);"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "## GAUSSIANIZE DATA"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "temp = np.random.uniform(-1, 1, 5000)\n",
#     "temp_gaussian = gaussianisation(temp)\n",
#     "fig, ax = plt.subplots(1, 2)\n",
#     "ax[0].hist(temp);\n",
#     "ax[1].hist(temp_gaussian);"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "xV_gaussian = gaussianisation(xV.values.reshape(-1,))\n",
#     "plt.hist(xV_gaussian)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "plot_acf(xV_gaussian, zero=False);"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "xVsq_gaussian = gaussianisation(xV_sq.values.reshape(-1,))\n",
#     "plt.hist(xVsq_gaussian);"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "plot_acf(xVsq_gaussian, zero=False);"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "plot_pacf(xVsq_gaussian, zero=False);"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "## RESHUFFLE"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "xV_gaussian_rp = xV_gaussian[np.random.permutation(np.arange(xV_gaussian.shape[0]))]\n",
#     "plot_acf(xV_gaussian_rp, zero=False);"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "## PORTMANTEAU TEST"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "ljung_val, ljung_pval = portmanteau_test(xV, maxtau=10, show=True)"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "## FIT MODEL"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "summary, fittedvalues, resid, model, aic = fit_arima_model(xV=xV, p=1, q=0, d=0, show=True)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "nrmseV, predM = calculate_fitting_error(xV, model, Tmax=10, show=True)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "prop = 0.7\n",
#     "split_point = int(prop*xV.shape[0]) \n",
#     "train_xV, test_xV = xV[:split_point], xV[split_point:]\n",
#     "summary, fittedvalues, resid, model, aic = fit_arima_model(xV=train_xV, p=1, q=0, d=0, show=True)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "Tmax = 50\n",
#     "forecast, sterror, confint = model.forecast(Tmax)\n",
#     "plt.plot(forecast, label='Forecast')\n",
#     "plt.fill_between(np.arange(Tmax),confint[:,0], confint[:,1], alpha=0.3, color='c', label='Conf.Int')\n",
#     "plt.plot(test_xV[:Tmax], linestyle='--', color='b', label='Original')\n",
#     "plt.legend()"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "## Out of sample predictions using pm.auto_arima"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "prop = 0.7\n",
#     "split_point = int(prop*xV.shape[0]) \n",
#     "train_xV, test_xV = xV[:split_point], xV[split_point:]\n",
#     "model = pm.auto_arima(train_xV, \n",
#     "                          start_p=0, start_q=0,d=0, max_p=5, \n",
#     "                          max_q=5, start_P=0, D=None, start_Q=0, max_P=5, \n",
#     "                          max_D=1, max_Q=5,stepwise=True,seasonal=False)\n",
#     "print(model.summary())\n",
#     "plt.hist(model.resid());"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "## multistep oos prediction"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "def predict_oos_multistep(model, Tmax=10, return_conf_int=True, alpha=0.05, show=True):\n",
#     "    '''\n",
#     "    out of sample predictions starting from last train values\n",
#     "    '''\n",
#     "    if return_conf_int:\n",
#     "        preds, conf_bounds = model.predict(n_periods=Tmax, return_conf_int=return_conf_int, alpha=alpha)\n",
#     "    else:\n",
#     "        preds = model.predict(n_periods=Tmax, return_conf_int=return_conf_int, alpha=alpha)\n",
#     "        conf_bounds = []\n",
#     "    if show:\n",
#     "        fig, ax = plt.subplots(1, 1)\n",
#     "        ax.plot(np.arange(1, Tmax+1), preds)\n",
#     "    return preds, conf_bounds\n",
#     "model."
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "in_preds = model.predict_in_sample(dynamic=False)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "Tmax = 30\n",
#     "return_conf_int = True\n",
#     "alpha = 0.05\n",
#     "preds, conf_bounds = predict_oos_multistep(model, Tmax=Tmax, return_conf_int=return_conf_int, alpha=alpha, show=False)\n",
#     "plt.figure();\n",
#     "plt.plot(np.arange(1, Tmax+1),preds, label='predictions');\n",
#     "plt.plot(np.arange(1, Tmax+1),test_xV[:Tmax], label='original');\n",
#     "if return_conf_int:\n",
#     "    plt.fill_between(np.arange(1, Tmax+1), conf_bounds[:, 0], conf_bounds[:, 1], color='green', alpha=0.3)\n",
#     "plt.legend();"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "## rolling oos prediction"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "preds = []\n",
#     "bounds = []\n",
#     "return_conf_int = True\n",
#     "alpha = 0.05\n",
#     "for i in test_xV:\n",
#     "    prediction, conf_bounds = model.predict(n_periods=1, return_conf_int=return_conf_int, alpha=alpha)\n",
#     "    model.update(i)\n",
#     "    preds.append(prediction[0])\n",
#     "    bounds.append(conf_bounds[0])\n",
#     "plt.figure();\n",
#     "plt.plot(preds, label='predictions', linestyle='--', alpha=0.3);\n",
#     "plt.plot(test_xV, label='original', alpha=0.7);\n",
#     "if return_conf_int:\n",
#     "    bounds = np.array(bounds)\n",
#     "    plt.fill_between(np.arange(len(test_xV)), bounds[:, 0], bounds[:, 1], alpha=0.3, color='green')\n",
#     "plt.legend();"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "## BTCUSDT price prediction\n"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "os.chdir('d:/timeserieslab/')\n",
#     "df = pd.read_csv('./data/BTCUSDT.csv')\n",
#     "df.set_index(pd.to_datetime(df['time']), inplace=True)\n",
#     "df.drop('time', axis=1, inplace=True)\n",
#     "df = df['01-01-2020':'12-31-2021']\n",
#     "df.plot();"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "logreturns = df.apply(lambda x: np.log(x)).diff().bfill()\n",
#     "logreturns.plot();\n",
#     "plot_acf(logreturns.values, zero=False)\n",
#     "plot_pacf(logreturns.values, zero=False)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "train_split = '12-01-2020'\n",
#     "\n",
#     "train_xV, test_xV = logreturns[:train_split], logreturns[train_split:]\n",
#     "model = pm.auto_arima(train_xV, \n",
#     "                          start_p=0, start_q=0,d=0, max_p=5, \n",
#     "                          max_q=5, start_P=0, D=None, start_Q=0, max_P=5, \n",
#     "                          max_D=1, max_Q=5,stepwise=True,seasonal=False)\n",
#     "print(model.summary())\n",
#     "\n",
#     "plt.figure()\n",
#     "plt.hist(model.resid());\n",
#     "plt.figure()\n",
#     "plt.plot(train_xV, label='train data');\n",
#     "insample_preds = model.predict_in_sample()\n",
#     "insample_preds_df = pd.DataFrame(index=train_xV.index, data=insample_preds)\n",
#     "plt.plot(insample_preds_df, label='fitted values');\n",
#     "plt.legend();\n"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "preds = []\n",
#     "for i in test_xV.values:\n",
#     "    prediction = model.predict(n_periods=1)[0]\n",
#     "    model.update(i)\n",
#     "    preds.append(prediction)\n",
#     "preds = np.array(preds)\n",
#     "preds_df = pd.DataFrame(index=test_xV.index, data=preds)\n",
#     "\n",
#     "plt.figure();\n",
#     "plt.plot(preds_df, label='predictions', linestyle='--', alpha=0.3);\n",
#     "plt.plot(test_xV, label='original', alpha=0.7);\n",
#     "plt.legend();"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "results = pd.concat([preds_df, test_xV], axis=1)\n",
#     "results.columns = ['pred', 'true']\n",
#     "results['hit'] = (results['pred'] > 0) == (results['true'] > 0)\n",
#     "results.head()"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "print(results['hit'].mean())"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "results.resample('M').agg({'hit':'mean'})"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": []
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "get_nrmse(target=test_xV.values, predicted=preds)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": []
#   }
#  ],
#  "metadata": {
#   "kernelspec": {
#    "display_name": "Python 3",
#    "language": "python",
#    "name": "python3"
#   },
#   "language_info": {
#    "codemirror_mode": {
#     "name": "ipython",
#     "version": 3
#    },
#    "file_extension": ".py",
#    "mimetype": "text/x-python",
#    "name": "python",
#    "nbconvert_exporter": "python",
#    "pygments_lexer": "ipython3",
#    "version": "3.7.3"
#   }
#  },
#  "nbformat": 4,
#  "nbformat_minor": 4
# }
