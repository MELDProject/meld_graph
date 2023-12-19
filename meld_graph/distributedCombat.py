import pandas as pd
import numpy as np
import pickle
import meld_graph.distributedCombat_helpers as helpers

#' Distributed ComBat step at each site
#'
#' @param dat A \emph{p x n} matrix (or object coercible by
#'   \link[base]{as.matrix} to a numeric matrix) of observations where \emph{p}
#'   is the number of features and \emph{n} is the number of subjects.
#' @param batch Factor indicating batch. Needs to have the same levels across
#'   all individual sites, but can have multiple batches per site (i.e.
#'   multiple levels in each site)
#' @param mod Optional design matrix of covariates to preserve, usually from
#'    \link[stats]{model.matrix}. This matrix needs to have the same columns
#'    across sites. The rows must be in the same order as the data columns.
#' @param central.out Output list from \code{distributedCombat_central}. Output
#'   of \code{distributedCombat_site} will depend on the values of
#'   \code{central.out}. If \code{NULL}, then the output will be sufficient for
#'   estimation of \code{B_hat}. If \code{B_hat} is provided, then the output
#'   will be sufficient for estimation of \code{sigma} or for harmonization if
#'   \code{mean_only} is \code{TRUE}. If \code{sigma} is provided, then
#'   harmonization will be performed.
#' @param eb If \code{TRUE}, the empirical Bayes step is used to pool
#'   information across features, as per the original ComBat methodology. If
#'   \code{FALSE}, adjustments are made for each feature individually.
#'   Recommended left as \code{TRUE}.
#' @param parametric If \code{TRUE}, parametric priors are used for the
#'   empirical Bayes step, otherwise non-parametric priors are used. See
#'   neuroComBat package for more details.
#' @param mean_only If \code{TRUE}, distributed ComBat does not harmonize the
#'   variance of features.
#' @param verbose If \code{TRUE}, print progress updates to the console.
#' @param file File name of .pickle file to export
#'
def distributedCombat_site(
    dat,
    batch,
    mod=None,
    ref_batch=None,
    central_out=None,
    eb=True,
    parametric=True,
    mean_only=False,
    robust=False,
    verbose=False,
    file=None,
):
    if file is None:
        file = "distributedCombat_site.pickle"
        print(
            "Must specify filename to output results as a file. Currently saving output to current workspace only."
        )
    if isinstance(central_out, str):
        central_out = pd.read_pickle(central_out)
    hasNAs = np.isnan(dat).any(axis=None)
    if verbose and hasNAs:
        print("[neuroCombat] WARNING: NaNs detected in data")
    if mean_only:
        print("[neuroCombat] Performing ComBat with mean only")

    ##################### Getting design ############################
    data_dict = helpers.getDataDictDC(
        batch, mod, verbose=verbose, mean_only=mean_only, ref_batch=ref_batch
    )

    design = data_dict["design"].copy()
    #################################################################

    ############### Site matrices for standardization ###############
    # W^T W used in LS estimation
    ls_site = []
    ls_site.append(np.dot(design.transpose(), design))
    ls_site.append(np.dot(design.transpose(), dat.transpose()))

    data_dict_out = data_dict.copy()
    data_dict_out["design"] = None

    # new data_dict with batches within current site
    incl_bat = [x > 0 for x in data_dict["n_batches"]]
    data_dict_site = data_dict.copy()
    data_dict_site["batches"] = [
        data_dict["batches"][i] for i in range(len(data_dict["batches"])) if incl_bat[i]
    ]
    data_dict_site["n_batch"] = incl_bat.count(True)
    data_dict_site["n_batches"] = [
        data_dict["n_batches"][i]
        for i in range(len(data_dict["n_batches"]))
        if incl_bat[i]
    ]
    data_dict_site["batch_design"] = data_dict["batch_design"].loc[:, incl_bat]

    # remove reference batch information if reference batch is not in site
    if ref_batch is not None:
        if data_dict_site["ref"] in data_dict_site["batch"].unique():
            data_dict_site["ref"] = np.where(
                np.any(data_dict_site["batch"] == ref_batch)
            )[0][0]
        else:
            data_dict_site["ref"] = None
            data_dict_site["ref_batch"] = None

    if central_out is None:
        site_out = {
            "ls_site": ls_site,
            "data_dict": data_dict,
            "sigma_site": None,
        }
        with open(file, "wb") as handle:
            pickle.dump(site_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return site_out

    # If beta.estimates given, get summary statistics for sigma estimation

    if "var_pooled" not in central_out or central_out["var_pooled"] is None:
        sigma_site = helpers.getSigmaSummary(dat, data_dict, design, hasNAs, central_out)
        site_out = {
            "ls_site": ls_site,
            "data_dict": data_dict,
            "sigma_site": sigma_site,
        }
        with open(file, "wb") as handle:
            pickle.dump(site_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return site_out

    stdObjects = helpers.getStandardizedDataDC(
        dat=dat,
        data_dict=data_dict,
        design=design,
        hasNAs=hasNAs,
        central_out=central_out,
    )
    s_data = stdObjects["s_data"]

    ##################### Getting L/S estimates #######################
    if verbose:
        print("[distributedCombat] Fitting L/S model and finding priors")
    naiveEstimators = helpers.getNaiveEstimators(
        s_data=s_data,
        data_dict=data_dict_site,
        hasNAs=hasNAs,
        mean_only=mean_only,
        robust = robust
    )
    ####################################################################
    ########################### Getting final estimators ###############
    if eb:
        if verbose:
            print(
                "[distributedCombat] Finding ",
                ("" if parametric else "non-"),
                "parametric adjustments",
                sep="",
            )
        estimators = helpers.getEbEstimators(
            naiveEstimators=naiveEstimators,
            s_data=s_data,
            data_dict=data_dict_site,
            parametric=parametric,
            mean_only=mean_only,
            robust=robust
        )
    else:
        estimators = helpers.getNonEbEstimators(
            naiveEstimators=naiveEstimators, data_dict=data_dict_site
        )

    ######################### Correct data #############################
    if verbose:
        print("[distributedCombat] Adjusting the Data")
    bayesdata = helpers.getCorrectedData(
        dat=dat,
        s_data=s_data,
        data_dict=data_dict_site,
        estimators=estimators,
        naive_estimators=naiveEstimators,
        std_objects=stdObjects,
        eb=eb,
    )

    # List of estimates:
    estimates = {
        "gamma_hat": naiveEstimators["gamma_hat"],
        "delta_hat": naiveEstimators["delta_hat"],
        "gamma_star": estimators["gamma_star"],
        "delta_star": estimators["delta_star"],
        "gamma_bar": estimators["gamma_bar"],
        "t2": estimators["t2"],
        "a_prior": estimators["a_prior"],
        "b_prior": estimators["b_prior"],
        "stand_mean": stdObjects["stand_mean"],
        "mod_mean": stdObjects["mod_mean"],
        "var_pooled": stdObjects["var_pooled"],
        "beta_hat": stdObjects["beta_hat"],
        "mod": mod,
        "batch": batch,
        "ref_batch": ref_batch,
        "eb": eb,
        "parametric": parametric,
        "mean_only": mean_only,
    }
    site_out = {"dat_combat": bayesdata, "estimates": estimates}
    with open(file, "wb") as handle:
        pickle.dump(site_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return site_out


#' Distributed ComBat step at analysis core
#'
#' @param site.outs List of filenames containing site outputs.
#' @param file File name of .pickle file to export
def distributedCombat_central(site_outs, ref_batch=None, verbose=False, file=None):
    if file is None:
        print(
            "Must specify filename to output results as a file. Currently saving output to current workspace only."
        )
        file = None
    site_outs = [pickle.load(open(site_out, "rb")) for site_out in site_outs]
    m = len(site_outs)  # number of sites
    # get n.batches and n.array from sites
    batch_levels = site_outs[0]["data_dict"]["batch"].cat.categories
    n_batches = np.cumsum(
        [site_out["data_dict"]["n_batches"] for site_out in site_outs], axis=0
    )[-1]
    n_batch = len(n_batches)
    n_array = np.array(n_batches).sum()
    n_arrays = [site_out["data_dict"]["n_array"] for site_out in site_outs]

    # get reference batch if specified
    ref = None
    if ref_batch is not None:
        batch = site_outs[0]["data_dict"]["batch"]
        if ref_batch not in batch.cat.categories:
            raise ValueError("ref_batch not in batch.cat.categories")
        if verbose:
            print("[combat] Using batch=%s as a reference batch" % ref_batch)
        ref = np.where(batch_levels == ref_batch)

    # check if beta estimates have been given to sites
    step1s = np.array([site_out["sigma_site"] is None for site_out in site_outs])
    if len(np.unique(step1s)) > 1:
        raise ValueError(
            "Not all sites are at the same step, please confirm with each site."
        )
    step1 = np.all(step1s)

    #### Step 1: Get LS estimate across sites ####
    ls1 = np.array([x["ls_site"][0] for x in site_outs]).astype(np.float64)
    ls2 = np.array([x["ls_site"][1] for x in site_outs]).astype(np.float64)
    ls1 = np.cumsum(ls1, axis=0)[-1]
    ls2 = np.cumsum(ls2, axis=0)[-1]
    B_hat = np.matmul(np.transpose(np.linalg.inv(ls1)), ls2)

    if ref_batch is not None:
        grand_mean = B_hat[ref].transpose()
    else:
        grand_mean = np.matmul(
            np.transpose(n_batches / n_array), B_hat[range(n_batch), :]
        )
    grand_mean = np.reshape(grand_mean, (1, len(grand_mean)))
    stand_mean = np.matmul(
        np.transpose(grand_mean), np.transpose(np.ones(n_array)).reshape(1, n_array)
    )

    if step1:
        central_out = {"B_hat": B_hat, "stand_mean": stand_mean, "var_pooled": None}
        if file is not None:
            with open(file, "wb") as handle:
                pickle.dump(central_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return central_out

    # #### Step 2: Get standardization parameters ####
    vars = list(map(lambda x: x["sigma_site"], site_outs))

    # if ref_batch specified, use estimated variance from reference site
    if ref_batch is not None:
        var_pooled = vars[ref[0][0]]
    else:
        var_pooled = np.zeros(len(vars[0]))
        for i in range(m):
            var_pooled += n_arrays[i] * np.array(vars[i])
        var_pooled = var_pooled / n_array

    central_out = {"B_hat": B_hat, "stand_mean": stand_mean, "var_pooled": var_pooled}
    if file is not None:
        with open(file, "wb") as handle:
            pickle.dump(central_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return central_out