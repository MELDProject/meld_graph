import pandas as pd
import numpy as np
import patsy
import math

def biweight_midvar(data, center=None, axis=None):
    if center is None:
        center = np.median(data, axis=axis, keepdims=True)
        
    mad = np.median(abs(data - center), axis=axis, keepdims=True)
    
    d = data - center
    u = d/(9*mad)
    
    indic = np.abs(u) < 1
    
    num = d * d * (1. - u**2)**4
    num[~indic] = 0.
    num = np.sum(num, axis=axis)
    
    dem = (1. - u**2) * (1. - 5.*u**2)
    dem[~indic] = 0.
    dem = np.abs(np.sum(dem, axis=axis))**2
    
    n = np.sum(np.ones(data.shape), axis=axis)
    return n * num/dem

def betaNA(yy, designn):
    # designn <- designn[!is.na(yy),]
    # yy <- yy[!is.na(yy)]
    # B <- solve(crossprod(designn), crossprod(designn, yy))
    designn = designn.dropna()
    yy = yy[~yy.isna()]
    B = np.linalg.lstsq(designn, yy, rcond=None)[0]
    return B


def aprior(delta_hat):
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat, ddof=1)
    return (2 * s2 + m ** 2) / float(s2)


def bprior(delta_hat):
    m = delta_hat.mean()
    s2 = np.var(delta_hat, ddof=1)
    return (m * s2 + m ** 3) / s2


def apriorMat(delta_hat):
    m = delta_hat.mean(axis=1)
    s2 = delta_hat.var(ddof=1, axis=1)
    out = (m * s2 + m ** 3) / s2
    return out


def bpriorMat(delta_hat):
    m = delta_hat.mean(axis=1)
    s2 = delta_hat.var(ddof=1, axis=1)
    out = (m * s2 + m ** 3) / s2
    return out


def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001, robust=False):
    n = (1 - np.isnan(sdat)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()
    ones = np.ones((1, sdat.shape[1]))

    change = 1
    change_old = 1
    count = 0
    while change > conv:
        g_new = np.array(postmean(g_hat, g_bar, n, d_old, t2))
        
        if robust:
            sum2 = n*biweight_midvar(sdat,center=g_new.reshape((g_new.shape[0], 1)),axis=1)
            # sum2 = n*(1.482602218505602*np.median(abs(sdat - np.dot(g_new.reshape((g_new.shape[0], 1)), ones)), axis = 1)) ** 2
        else:
            sum2 = ((sdat - np.dot(g_new.reshape((g_new.shape[0], 1)), ones)) ** 2).sum(
                axis=1
            )
        
        d_new = postvar(sum2, n, a, b)
        # change = max(
        #     (abs(g_new - g_old.item()) / g_old.item()).max(),
        #     (abs(d_new - d_old) / d_old).max(),
        # )
        change = max(max(abs(g_new - g_old) / g_old), max(abs(d_new - d_old) / d_old))
        # print(max(abs(g_new - g_old) / g_old), "," ,max(abs(d_new - d_old) / d_old))
        
        if count > 30:
            if change > change_old:
                print('[neuroCombat WARNING] Empirical Bayes step failed to converge after 30 iterations, using estimate before change between iterations increases.')
                break

        g_old = g_new  # .copy()
        d_old = d_new  # .copy()
        change_old = change
        count = count + 1
    adjust = (g_new, d_new)
    return adjust


def int_eprior(sdat, g_hat, d_hat):
    r = sdat.shape[0]
    gamma_star, delta_star = [], []
    for i in range(0, r, 1):
        g = np.delete(g_hat, i)
        d = np.delete(d_hat, i)
        x = sdat[i, :]
        n = x.shape[0]
        j = np.repeat(1, n)
        A = np.repeat(x, g.shape[0])
        A = A.reshape(n, g.shape[0])
        A = np.transpose(A)
        B = np.repeat(g, n)
        B = B.reshape(g.shape[0], n)
        resid2 = np.square(A - B)
        sum2 = resid2.dot(j)
        LH = 1 / (2 * math.pi * d) ** (n / 2) * np.exp(-sum2 / (2 * d))
        LH = np.nan_to_num(LH)
        gamma_star.append(sum(g * LH) / sum(LH))
        delta_star.append(sum(d * LH) / sum(LH))
    adjust = (gamma_star, delta_star)
    return adjust


def getDataDictDC(batch, mod, verbose, mean_only, ref_batch=None):
    nbatch = len(batch.cat.categories)
    batches = []
    n_batches = []
    for x in batch.cat.categories:
        indices = np.where(batch == x)
        batches.append(indices)
        n_batches.append(len(indices[0]))
    n_array = np.array(n_batches).sum()
    batchmod = patsy.dmatrix("~-1+batch", batch, return_type="dataframe")
    if verbose:
        print("[combat] Found", nbatch, "batches")
    if not mean_only and np.all(n_batches == 1):
        raise ValueError(
            "Found site with only one sample; consider using mean_only=True"
        )
    ref = None
    if ref_batch is not None:
        if ref_batch not in batch.cat.categories:
            raise ValueError("Reference batch not in batch list")
        if verbose:
            print("[combat] Using batch=%s as a reference batch" % ref_batch)
        ref = np.where(np.any(batch.cat.categories == ref_batch))[0][
            0
        ]  # find the reference
        batchmod.iloc[:, ref] = 1
    # combine batch variable and covariates
    design = pd.concat([batchmod, mod], axis=1)
    # design = pd.concat([batchmod.reset_index(drop=True), mod.reset_index(drop=True)], axis=1)
    n_covariates = design.shape[1] - batchmod.shape[1]
    if verbose:
        print(
            "[combat] Adjusting for ",
            n_covariates,
            " covariate(s) or covariate level(s)",
        )
    out = {}
    out["batch"] = batch
    out["batches"] = batches
    out["n_batch"] = nbatch
    out["n_batches"] = n_batches
    out["n_array"] = n_array
    out["n_covariates"] = n_covariates
    out["design"] = design
    out["batch_design"] = design.iloc[:, :nbatch]
    out["ref"] = ref
    out["ref_batch"] = ref_batch
    return out


def getSigmaSummary(dat, data_dict, design, hasNAs, central_out):
    batches = data_dict["batches"]
    nbatches = data_dict["n_batches"]
    narray = data_dict["n_array"]
    nbatch = data_dict["n_batch"]
    ref_batch = data_dict["ref_batch"]
    ref = data_dict["ref"]
    Bhat = central_out["B_hat"]
    stand_mean = central_out["stand_mean"][:, 0:narray]

    if not hasNAs:
        if ref_batch is not None:
            ref_dat = dat.iloc[:, batches[ref][0]]
            factors = nbatches[ref] / (nbatches[ref] - 1)
            var_pooled = (
                np.cov(dat - np.matmul(design.iloc[batches[ref][0]], Bhat).transpose())
                / factors
            )
        else:
            factors = narray / (narray - 1)
            var_pooled = np.cov(dat - np.matmul(design, Bhat).transpose()) / factors
    else:
        if ref_batch is not None:
            ref_dat = dat.iloc[:, batches[ref][0]]
            ns = ref_dat.isna().sum()
            factors = nbatches[ref] / (nbatches[ref] - 1)
            var_pooled = (
                np.cov(dat - np.matmul(design.iloc[batches[ref][0]], Bhat).transpose())
                / factors
            )
        else:
            ns = dat.isna().sum()
            factors = ns / (ns - 1)
            var_pooled = np.cov(dat - np.matmul(design, Bhat).transpose()) / factors
    # todo: note sure why I had to do this
    var_pooled = np.diagonal(var_pooled)
    return var_pooled


def getStandardizedDataDC(dat, data_dict, design, hasNAs, central_out):
    batches = data_dict["batches"]
    nbatches = data_dict["n_batches"]
    narray = data_dict["n_array"]
    nbatch = data_dict["n_batch"]
    ref_batch = data_dict["ref_batch"]
    ref = data_dict["ref"]
    Bhat = central_out["B_hat"]
    stand_mean = central_out["stand_mean"]
    var_pooled = central_out["var_pooled"]

    if design is not None:
        tmp = design
        tmp.iloc[:, :nbatch] = 0
        mod_mean = np.matmul(tmp, Bhat).transpose()
    else:
        mod_mean = np.zeros(narray)
    # todo check stand_mean
    stand_mean = stand_mean[:, 0:narray]
    # s_data = (dat - stand_mean - mod_mean) / np.matmul(np.sqrt(var_pooled), np.ones(narray))
    s_data = (dat - stand_mean - mod_mean) / np.tile(
        np.sqrt(var_pooled), (narray, 1)
    ).transpose()
    return {
        "s_data": s_data,
        "stand_mean": stand_mean,
        "mod_mean": mod_mean,
        "var_pooled": var_pooled,
        "beta_hat": Bhat,
    }


def getNaiveEstimators(s_data, data_dict, hasNAs, mean_only, robust):
    # todo double check this
    batch_design = data_dict["batch_design"]
    batches = data_dict["batches"]
    
    if robust:
        gamma_hat = []
        for i in batches:
            gamma_hat.append(np.median(s_data.iloc[:, i[0]], axis = 1))
        gamma_hat = np.array(gamma_hat)
    else:
        if not hasNAs:
            gamma_hat = np.matmul(np.linalg.inv(np.matmul(batch_design.transpose(), batch_design)), batch_design.transpose(),)
            gamma_hat = np.matmul(gamma_hat, s_data.transpose())
        else:
            gamma_hat = s_data.apply(betaNA, axis=0, args=(batch_design,))
    
    delta_hat = []
    for i in batches:
        if mean_only:
            delta_hat.append(np.ones(s_data.shape[1]))
        elif robust:
            delta_hat.append(biweight_midvar(s_data.iloc[:, i[0]],axis=1))
        else:
            delta_hat.append(np.var(s_data.iloc[:, i[0]],axis=1,ddof=1))
    delta_hat = np.array(delta_hat)
    return {"gamma_hat": gamma_hat, "delta_hat": delta_hat}


def getEbEstimators(
    naiveEstimators, s_data, data_dict, parametric=True, mean_only=False, robust = False
):
#     gamma_hat = (
#         naiveEstimators["gamma_hat"]
#         .to_numpy()
#         .reshape(naiveEstimators["gamma_hat"].shape[1])
#     )
    
    gamma_hat = naiveEstimators["gamma_hat"]
    delta_hat = naiveEstimators["delta_hat"]
    # pd.DataFrame(naiveEstimators["delta_hat"].reshape(1, len(naiveEstimators["delta_hat"])))
    batches = data_dict["batches"]
    nbatch = data_dict["n_batch"]
    ref_batch = data_dict["ref_batch"]
    ref = data_dict["ref"]

    def getParametricEstimators():
        gamma_star = delta_star = []
        for i in range(nbatch):
            if mean_only:
                gamma_star.append(postmean(gamma_hat[i,:], gamma_bar[i], 1, 1, t2[i]))
                delta_star.append(np.ones(len(s_data)))
            elif robust:
                #gamma_star.append(postmean(gamma_hat, gamma_bar, 1, delta_hat, t2))
                #delta_star.append(delta_hat)
                
                temp = it_sol(
                    s_data.iloc[:, batches[i][0]],
                    gamma_hat[i],
                    delta_hat[i],
                    gamma_bar[i],
                    t2[i],
                    a_prior[i],
                    b_prior[i],
                    robust=robust
                )
                gamma_star.append(temp[0])
                delta_star.append(temp[1])
            else:
                temp = it_sol(
                    s_data.iloc[:, batches[i][0]],
                    gamma_hat[i],
                    delta_hat[i],
                    gamma_bar[i],
                    t2[i],
                    a_prior[i],
                    b_prior[i],
                )
                gamma_star.append(temp[0])
                delta_star.append(temp[1])
        return gamma_star[0], delta_star[1]

    def getNonParametricEstimators():
        gamma_star = delta_star = []
        for i in range(nbatch):
            if mean_only:
                delta_hat[i] = 1
            else:
                temp = int_eprior(s_data[batches[i]], gamma_hat[i], delta_hat[i])
                gamma_star.append(temp[0])
                delta_star.append(temp[1])
        return gamma_star, delta_star
    
    gamma_bar = gamma_hat.mean(axis=1)
    t2 = gamma_hat.var(ddof=1, axis=1)
    a_prior = apriorMat(delta_hat)
    b_prior = bpriorMat(delta_hat)
    
    tmp = getParametricEstimators() if parametric else getNonParametricEstimators()
    if ref_batch is not None:
        # set reference batch mean equal to 0
        tmp[0][ref] = 0
        # set reference batch variance equal to 1
        tmp[1][ref] = 1
    out = {}
    out["gamma_star"] = tmp[0]
    out["delta_star"] = tmp[1]
    out["gamma_bar"] = gamma_bar
    out["t2"] = t2
    out["a_prior"] = a_prior
    out["b_prior"] = b_prior
    return out


def getNonEbEstimators(naiveEstimators, data_dict):
    batches = data_dict["batches"]
    nbatch = data_dict["n_batch"]
    ref_batch = data_dict["ref_batch"]
    ref = data_dict["ref"]
    
    gamma_hat = naiveEstimators["gamma_hat"]
    delta_hat = naiveEstimators["delta_hat"]
    
    out = {}
    out["gamma_star"] = gamma_hat
    out["delta_star"] = delta_hat
    out["gamma_bar"] = None
    out["t2"] = None
    out["a_prior"] = None
    out["b_prior"] = None
    if ref_batch is not None:
        # set reference batch mean equal to 0
        out["gamma_star"][batches[ref]] = 0
        # set reference batch variance equal to 1
        out["delta_star"][batches[ref]] = 1
    return out


def getCorrectedData(
    dat, s_data, data_dict, estimators, naive_estimators, std_objects, eb=True
):
    # todo this is not correct
    var_pooled = std_objects["var_pooled"]
    stand_mean = std_objects["stand_mean"]
    mod_mean = std_objects["mod_mean"]
    batches = data_dict["batches"]
    batch_design = data_dict["batch_design"]
    n_batches = data_dict["n_batches"]
    n_array = data_dict["n_array"]
    ref_batch = data_dict["ref_batch"]
    ref = data_dict["ref"]

    if eb:
        gamma_star = estimators["gamma_star"]
        delta_star = estimators["delta_star"]
    else:
        gamma_star = naive_estimators["gamma_hat"]
        delta_star = naive_estimators["delta_hat"]
    gamma_star = gamma_star.reshape(1, len(gamma_star))

    bayesdata = s_data.copy()
    j = 0
    for i in batches:
        # einsum: https://stackoverflow.com/a/33641428/2624391
        top = (
            bayesdata.iloc[:, i[0]]
            - np.einsum(
                "ij,i->ij",
                gamma_star,
                batch_design.iloc[i[0], :].to_numpy().flatten(),
            ).transpose()
        )
        bottom = np.einsum("i,j->ij", np.sqrt(delta_star), np.ones(n_batches[j]))
        bayesdata.iloc[:, i[0]] = top / bottom
        j += 1
    bayesdata = (
        (
            bayesdata
            * np.einsum("i,j->ij", np.sqrt(var_pooled), np.ones(n_array).transpose())
        )
        + stand_mean
        + mod_mean
    )
    if ref_batch is not None:
        bayesdata.iloc[:, batches[ref][0]] = dat.iloc[:, batches[ref][0]]
    return bayesdata