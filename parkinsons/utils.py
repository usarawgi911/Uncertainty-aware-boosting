from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import forestci as fci
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

use_ci = True
mod0_rmse = []
mod1_rmse = []


def picp(y_true, y_pred, y_std, sigma=1):
    cnt = 0
    if use_ci:
        y_std = y_std / 2
    for i in range(len(y_true)):
        if (y_true[i] - sigma*y_std[i] <= y_pred[i]) and (y_pred[i] <= y_true[i] + sigma*y_std[i]):
            cnt = cnt + 1
    return 100 * cnt / (len(y_true))


def mpiw(y_std):
    return np.mean(y_std)


def train_rfg(X_train, y_train, X_test, y_test, sample_weight=None, uncertainty=False):
    rfg = RandomForestRegressor(n_estimators=300, random_state=0).fit(X_train, y_train, sample_weight)
    preds = [rfg.predict(X_train), rfg.predict(X_test)]
    
    variance_tr = fci.random_forest_error(rfg, X_train, X_train)
    variance_te = fci.random_forest_error(rfg, X_train, X_test)
    
    if uncertainty:
        sw_tr = variance_tr
        sw_te = variance_te
    else:
        sw_tr = (preds[0]-y_train)**2
        sw_te = (preds[1]-y_test)**2
    variance = [variance_tr, variance_te]
    sws = [sw_tr, sw_te]
    # print("Train rmse: ", mean_squared_error(preds[0], y_train, squared=False))
    # print("Test rmse: ", mean_squared_error(preds[1], y_test, squared=False))
    
    return preds, variance, sws

def boost_ensemble(X_train, y_train, X_test, y_test, boosting=False, uncertainty=False):
    sample_weight = None
    results = [0]*len(X_train)
    all_train_preds = []
    all_test_preds = []
    sample_weights_tr = [np.asarray([1]*len(X_train[0]))]
    sample_weights_te = [np.asarray([1]*len(X_test[0]))]
    
    variance_tr = []
    variance_te = []
    for i in range(len(X_train)):
        # print("Modality ", i)
        Xf_train = X_train[i]
        Xf_test = X_test[i]
        if not boosting:
            preds, variances, sws = train_rfg(Xf_train, y_train, Xf_test, y_test, sample_weight=None, uncertainty=uncertainty)
        else:
            preds, variances, sws = train_rfg(Xf_train, y_train, Xf_test, y_test, sample_weight=sample_weights_tr[-1], uncertainty=uncertainty)
        all_train_preds.append(preds[0])
        all_test_preds.append(preds[1])
        if i==0:
            mod0_rmse.append(mean_squared_error(preds[1], y_test, squared=False))
        else:
            mod1_rmse.append(mean_squared_error(preds[1], y_test, squared=False))
        sample_weights_tr.append(sws[0])
        sample_weights_te.append(sws[1])
        
        variance_tr.append(variances[0])
        variance_te.append(variances[1])
        # print("-"*30)
    return np.asarray(all_train_preds), np.asarray(all_test_preds), np.asarray(variance_tr), np.asarray(variance_te)

def pprint(p, curr):
    m = np.mean(curr, axis=0)
    s = np.std(curr, axis=0)
    print_ans = ''
    for a, b in zip(m, s):
        print_ans+="{:.3f} +/- {:.3f}, ".format(a, b)
    print(p+print_ans)

def train_ensemble(X, y, cols, boosting=True, uncertainty=True):
    
    uw_tr_rmse = []
    uw_te_rmse = []
    w_tr_rmse = []
    w_te_rmse = []
    
    mod0_mpiw=[]
    mod1_mpiw=[]

    mod0_picp=[]
    mod1_picp=[]

    mod0_picps_uw = []
    mod0_picps_w = []
    mod1_picps_uw = []
    mod1_picps_w = []

    all_y_te = []
    all_y_pred_uw = []
    all_y_pred_w = []
    all_y_std0 = []
    all_y_std1 = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    if boosting==True and uncertainty==True:
        print("*"*20, "UA ENSEMBLE", "*"*20)
    if boosting==True and uncertainty==False:
        print("*"*20, "VANILLA ENSEMBLE", "*"*20)

    fold=1
    for train_index, test_index in kf.split(X):
        print("Fold ", fold)

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = np.asarray(y[train_index]), np.asarray(y[test_index])

        X_train_ens = []
        X_test_ens = []

        for i in range(len(cols)):
            X_train_ens.append(np.asarray(X_train[cols[i]]))
            X_test_ens.append(np.asarray(X_test[cols[i]]))

        tr, te, sw_tr, sw_te = boost_ensemble(X_train_ens, y_train, X_test_ens, y_test, boosting=boosting, uncertainty=uncertainty)
        
        all_y_te.extend(y_test)
        all_y_std0.extend(sw_te[0])
        all_y_std1.extend(sw_te[1])

        mod0_mpiw.append(mpiw(np.sqrt(np.abs(sw_te[0]))))
        mod1_mpiw.append(mpiw(np.sqrt(np.abs(sw_te[1]))))
        
        tmp0=[]
        tmp1=[]
        for sig in range(1,4):
            tmp0.append(picp(y_test, te[0, :], np.sqrt(sw_te)[0, :], sig))
            tmp1.append(picp(y_test, te[1, :], np.sqrt(sw_te)[1, :], sig))
        
        mod0_picp.append(tmp0)
        mod1_picp.append(tmp1)

        sw_tr_ = 1/np.asarray(sw_tr)
        sw_te_ = 1/np.asarray(sw_te)
        w_tr = sw_tr_/np.sum(sw_tr_, axis=0)
        w_te = sw_te_/np.sum(sw_te_, axis=0)

        uw_tr_rmse.append(mean_squared_error(np.mean(tr, axis=0), y_train, squared=False))
        uw_te_rmse.append(mean_squared_error(np.mean(te, axis=0), y_test, squared=False))

        w_tr_rmse.append(mean_squared_error(np.sum(tr*w_tr, axis=0), y_train, squared=False))
        w_te_rmse.append(mean_squared_error(np.sum(te*w_te, axis=0), y_test, squared=False))
        
        all_y_pred_uw.extend(np.mean(te, axis=0))
        all_y_pred_w.extend(np.sum(te*w_te, axis=0))

        tmp0=[]
        tmp1=[]
        for sig in range(1,4):
            tmp0.append(picp(y_test, np.mean(te, axis=0), np.sqrt(sw_te)[0, :], sig))
            tmp1.append(picp(y_test, np.mean(te, axis=0), np.sqrt(sw_te)[1, :], sig))
        
        mod0_picps_uw.append(tmp0)
        mod1_picps_uw.append(tmp1)

        tmp0=[]
        tmp1=[]
        for sig in range(1,4):
            tmp0.append(picp(y_test, np.sum(te*w_te, axis=0), np.sqrt(sw_te)[0, :], sig))
            tmp1.append(picp(y_test, np.sum(te*w_te, axis=0), np.sqrt(sw_te)[1, :], sig))
        
        mod0_picps_w.append(tmp0)
        mod1_picps_w.append(tmp1)

        fold+=1
        print("*"*50)
    
    print("*"*20,"Results", "*"*20)
    print("Mod 0 Test RMSE: {:.3f} +/- {:.3f}".format(np.mean(mod0_rmse), np.std(mod0_rmse)))
    print("Mod 1 Test RMSE: {:.3f} +/- {:.3f}".format(np.mean(mod1_rmse), np.std(mod1_rmse)))
    
    print("Mod 0 MPIW: {:.3f} +/- {:.3f}, ".format(np.mean(mod0_mpiw), np.std(mod0_mpiw)))
    print("Mod 1 MPIW: {:.3f} +/- {:.3f}, ".format(np.mean(mod1_mpiw), np.std(mod1_mpiw)))

    pprint("Mod 0 PICP: ", mod0_picp)
    pprint("Mod 1 PICP: ", mod1_picp)
    print("*"*50)
    print("Unweighted")

    print("Train ensemble RMSE: {:.3f} +/- {:.3f}".format(np.mean(uw_tr_rmse), np.std(uw_tr_rmse)))
    print("Test ensemble RMSE: {:.3f} +/- {:.3f}".format(np.mean(uw_te_rmse), np.std(uw_te_rmse)))
    print("Train ensemble RMSEs: {}".format(uw_tr_rmse))
    print("Test ensemble RMSEs: {}".format(uw_te_rmse))
    pprint("Mod 0 PICP: ", mod0_picps_uw)
    pprint("Mod 1 PICP: ", mod1_picps_uw)
    
    print("*"*50)
    print("Weighted")

    print("Train ensemble RMSE: {:.3f} +/- {:.3f}".format(np.mean(w_tr_rmse), np.std(w_tr_rmse)))
    print("Test ensemble RMSE: {:.3f} +/- {:.3f}".format(np.mean(w_te_rmse), np.std(w_te_rmse)))
    print("Train ensemble RMSEs: {}".format(w_tr_rmse))
    print("Test ensemble RMSEs: {}".format(w_te_rmse))
    pprint("Mod 0 PICP: ", mod0_picps_w)
    pprint("Mod 1 PICP: ", mod1_picps_w)
    
    print("\n")
    return all_y_te, all_y_pred_uw, all_y_pred_w, np.sqrt(all_y_std0), np.sqrt(all_y_std1)

def plot_empirical_rule(mus, sigmas, true_values, model_name):
    thresholds = [0.12566, 0.25335, 0.38532, 0.52440, 0.67339, 0.84162, 1.03643, 1.28155, 1.64485]
    
    values = [[] for i in range(len(sigmas))]
    threshold_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    fig, ax = plt.subplots()
    ideal = [i for i in range(10,100,10)]

    if 'vanilla' in model_name:
        model_name_ = 'Vanilla Ensemble'
    elif 'weighted' in model_name:
        model_name_ = 'UA Ensemble(weighted)'
    else:
        model_name_ = 'UA Ensemble'
    plt.plot([], [], ' ', label=model_name_, color='white')
    plt.plot(ideal, ideal, label='Ideal Calibration', linewidth=2, color='black', linestyle=':')


    for cluster_id in range(len(sigmas)):
        cluster_id = 1-cluster_id
        print('Cluster {}'.format(cluster_id+1))
        for t in thresholds:
            count = 0
            for i in range(len(mus)):
                if np.abs(mus[i] - true_values[i])<= t* sigmas[cluster_id, i]:
                    count+=1 
            values[cluster_id].append(count)

        values[cluster_id] = np.array(values[cluster_id])*100/len(mus)
        
        plt.scatter(threshold_values, values[cluster_id], s=96)

        if cluster_id==1:
            plt.plot(threshold_values, values[cluster_id], label='Amplitude', linewidth=3)
        else:
            plt.plot(threshold_values, values[cluster_id], label='Frequency', linewidth=3)

        plt.ylabel('% of True Values inside Interval', fontsize=18)

    plt.xlabel('% of Prediction Interval around Mean', fontsize=18)


    plt.xticks(range(10, 100, 10))
    plt.yticks(range(10, 100, 10))
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    plt.legend(fontsize=15, loc='lower right')
    plt.tight_layout(pad=0)
    plt.savefig(model_name, dpi=300)
    plt.clf()
    # plt.show()
    plt.close()
