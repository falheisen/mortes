import numpy as np
from numpy import matlib
from scipy import stats
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from catboost import CatBoostClassifier
from sklearn.metrics import top_k_accuracy_score
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

np.random.seed(12227)

data_arrays = np.load(
    '../data/data_top_causes_all_features.npz', allow_pickle=True)
X_test = data_arrays['X_test']
X_train = data_arrays['X_train']
y_test = data_arrays['y_test']
y_train = data_arrays['y_train']

n_classes = len(np.unique(y_train))
n_samples = len(X_train)
n_features = len(X_train[0])
print(f"n_features={n_features}")
print(f"n_samples={n_samples}")

#####################################################################################################
##
# CLASSES
##
#####################################################################################################


class infFS_U():
    """.

    Parameters
    ----------
    copy : bool, (default=True)
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.

    Attributes
    ----------

    References
    ----------
    ""G. Roffo et al A Graph-based Feature Filtering Approach. In the IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). In PAMI, 2020"""

    def __init__(self, alpha=0.5, top_n=10, verbose=False, copy=True):
        self.__name__ = 'Infinite Feature Selection Unsupervised'
        self.copy = copy
        self.verbose = verbose
        self.alpha = alpha
        self.top_n = top_n

    def bsxfun(self, func, A, B):
        dim = A.shape[0]
        output = np.zeros((dim, dim))
        for i in range(0, dim):
            for j in range(0, dim):
                output[i][j] = func(A[i], B[j])

        return output

    def fit(self, X, Y=None):
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)

        # 1) Standard Deviation over the samples

        corr_ij, _ = stats.spearmanr(X)
        corr_ij[np.where(np.isnan(corr_ij))[0]] = 0  # remove NaN
        corr_ij[np.where(np.isinf(corr_ij))[0]] = 0  # remove inf
        corr_ij = 1-np.abs(corr_ij)

        STD = np.std(X, axis=0, ddof=1)
        STDMatrix = self.bsxfun(max, STD, STD.T)
        STDMatrix = STDMatrix - np.min(STDMatrix)
        sigma_ij = STDMatrix / np.max(STDMatrix)
        sigma_ij[np.where(np.isnan(sigma_ij))[0]] = 0  # remove NaN
        sigma_ij[np.where(np.isinf(sigma_ij))[0]] = 0  # remove inf

        # 2) Building the graph G = <V,E>
        if self.verbose:
            print('2) Building the graph G = <V,E>')

        N = X.shape[1]
        eps = 5e-06 * N
        factor = 1 - eps  # shrinking

        A = (self.alpha*sigma_ij + (1-self.alpha)*corr_ij)

        rho = np.max(np.sum(A, axis=1))

        # Substochastic Rescaling
        A = A / (np.max(np.sum(A, axis=1))+eps)

        # Letting paths tend to infinite: Inf-FS Core
        I = np.eye((A.shape[0]))  # Identity Matrix

        r = factor/rho

        y = I - (r * A)

        S = np.linalg.inv(y)

        # 5) Estimating energy scores
        self.WEIGHT = np.sum(S, axis=1)  # prob. scores s(i)

        # 6) Ranking features according to s
        self.RANKED = np.argsort(self.WEIGHT, axis=-1)[::-1]

        e = np.ones((N))
        t = np.dot(S, e)

        nbins = int(0.5 * N)
        counts = np.histogram(t, nbins)[0]

        thr = np.mean(counts)
        size_sub = np.sum(counts > thr)
        self.subset = self.RANKED[0:size_sub]
        return self

    def transform(self, X, Y=None, top_n=None):
        # Update the how many features we wanna w.r.t its weights
        if top_n is not None:
            self.top_n = top_n

        return X[:, self.RANKED[0:self.top_n]]


class infFS_S():
    """Inf-FS Supervised (2020).

    Parameters
    ----------
    copy : bool, (default=True)
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.

    Attributes
    ----------

    References
    ----------
    ""G. Roffo et al A Graph-based Feature Filtering Approach. In the IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). In PAMI, 2020"""

    def __init__(self, alpha=[0.5, 0.5, 0.5], top_n=10, verbose=False, copy=True):
        self.__name__ = 'Infinite Feature Selection Supervised'
        self.copy = copy
        self.verbose = verbose
        self.top_n = top_n
        self.alpha = alpha

    def bsxfun(self, func, A, B):
        dim = A.shape[0]
        output = np.zeros((dim, dim))
        for i in range(0, dim):
            for j in range(0, dim):
                output[i][j] = func(A[i], B[j])

        return output

    def muteinf(self, A, Y):
        n = A.shape[0]
        Z = np.column_stack((A, Y[:, 0]))
        if n/10 > 20:
            nbins = 20
        else:
            nbins = int(max(np.floor(n/10), 10))

        pA = np.histogram(A, nbins)[0]
        pA = pA/n
        pA[np.where(pA == 0)[0]] = 0.00001
        od = Y.shape[1]
        c_l = od
        if od == 1:
            pY = np.array([len(np.where(Y == +1.0)[0]),
                          len(np.where(Y == +-1.0)[0])])/n
            c_l = 2
        else:
            pY = np.zeros((1, od))
            for i in range(0, od):
                pY[i] = len(np.where(Y == +1.0)[0])
            pY = pY / n

        p = np.zeros((c_l, nbins))
        rx = np.abs(np.max(A)-np.min(A))/nbins

        for i in range(0, c_l):
            x_1 = np.min(A)
            for j in range(0, nbins):
                if i == 1 and od == 1:
                    interval = ((x_1 <= Z[:, 0]) & (Z[:, 1] == -1.0))*1
                else:
                    interval = ((x_1 <= Z[:, 0]) & (Z[:, 1] == +1.0))*1
                if j < nbins-1:
                    interval = interval & (Z[:, 0] < x_1 + rx)

                p[i, j] = len(np.where(interval == 1)[0])

                if p[i, j] == 0:
                    p[i, j] = 0.00001

                x_1 = x_1 + rx

        HA = -np.sum(pA * np.log(pA))
        HY = -np.sum(pY * np.log(pY))
        pA = matlib.repmat(pA, c_l, 1)
        pY = matlib.repmat(np.reshape(pY, (1, 2)).T, 1, nbins)
        p = p/n
        info = np.sum(np.sum(p * np.log(p / (pA * pY)), axis=0))
        info = 2 * info / (HA + HY)
        return info

    def getGraphWeights(self, train_x, train_y, alpha, eps):

        # Metric 1: Mutual Information
        mi_s = []

        for i in range(0, train_x.shape[1]):
            mi_s.append(self.muteinf(train_x[:, i], train_y))
        mi_s = np.array(mi_s)
        mi_s[np.where(np.isnan(mi_s))[0]] = 0  # remove NaN
        mi_s[np.where(np.isinf(mi_s))[0]] = 0  # remove inf

        # Zero-Max norm
        mi_s = mi_s - np.min(np.min(mi_s))
        mi_s = mi_s / np.max(np.max(mi_s))

        # Metric 2: class separation
        fi_s = np.array([np.mean(train_x[np.where(train_y == +1.0)[0], :], axis=0) -
                        np.mean(train_x[np.where(train_y == -1.0)[0], :], axis=0)])**2
        st = np.std(train_x[np.where(train_y == +1.0)
                    [0], :], axis=0, ddof=1)**2
        st = st + \
            np.std(train_x[np.where(train_y == -1.0)[0], :], axis=0, ddof=1)**2
        st[np.where(st == 0)[0]] = 10000  # remove ones where nothing occurs
        fi_s = fi_s / st

        fi_s[np.where(np.isnan(fi_s))[0]] = 0  # remove NaN
        fi_s[np.where(np.isinf(fi_s))[0]] = 0  # remove inf

        # Zero-Max norm
        fi_s = fi_s - np.min(np.min(fi_s))
        fi_s = fi_s / np.max(np.max(fi_s))

        std_s = np.std(train_x, axis=0, ddof=1)
        std_s[np.where(np.isnan(std_s))[0]] = 0  # remove NaN
        std_s[np.where(np.isinf(std_s))[0]] = 0  # remove inf

        SD = self.bsxfun(max, std_s, std_s.T)
        SD = SD - np.min(np.min(SD))
        SD = SD / np.max(np.max(SD))

        mi_s = np.reshape(mi_s, (1, mi_s.shape[0]))
        MI = np.matlib.repmat(mi_s, mi_s.shape[1], 1)
        FI = np.matlib.repmat(fi_s, fi_s.shape[1], 1)

        G = alpha[0] * ((MI + FI.T)/2) + alpha[1] * \
            ((MI+SD.T)/2) + alpha[2] * ((SD+FI.T)/2)

        rho = np.max(np.sum(G, axis=1))

        # Substochastic Rescaling
        G = G / (np.max(np.sum(G, axis=1)) + eps)
        return G, rho

    def fit(self, X, Y):
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)
        Y = check_array(Y, copy=self.copy, dtype=FLOAT_DTYPES, ensure_2d=False)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        Y[np.where(Y == 0)[0]] = -1

        N = X.shape[1]

        eps = 5e-06 * N

        factor = 1 - eps  # shrinking

        A, rho = self.getGraphWeights(X, Y, self.alpha, eps)

        # Letting paths tend to infinite: Inf-FS Core
        I = np.eye((A.shape[0]))

        r = factor / rho  # Set a meaningful value for r: = 0 < r < 1 / rho

        y = I - (r*A)

        S = np.linalg.inv(y)

        # 5) Estimating energy scores
        self.WEIGHT = np.sum(S, axis=1)  # prob.scores s(i)

        # 6) Ranking features according to s
        self.RANKED = np.argsort(self.WEIGHT, axis=-1)[::-1]

        e = np.ones((N))
        t = np.dot(S, e)

        nbins = int(0.5 * N)
        counts = np.histogram(t, nbins)[0]

        thr = np.mean(counts)
        size_sub = np.sum(counts > thr)
        self.subset = self.RANKED[0:size_sub]

        return self

    def transform(self, X, Y=None, top_n=None):
        # Update the how many features we wanna w.r.t its weights
        if top_n is not None:
            self.top_n = top_n

        return X[:, self.RANKED[0:self.top_n]]

#####################################################################################################
##
# REALIZAR A SELEÇÃO DE PARÂMETROS
##
#####################################################################################################


k = 3
percentages = range(10, 101, 10)  # Cria uma sequência de 10% a 100%
features_array = [int(n_features * pct / 100) for pct in percentages]

fss = infFS_S()
fss.fit(X_train, y_train)

fsu = infFS_U()
fsu.fit(X_train, y_train)

catboost_accuracies_fss = []
catboost_top_k_accuracies_fss = []
catboost_accuracies_fsu = []
catboost_top_k_accuracies_fsu = []

for i in range(len(features_array)):

    X_train_top_features_fss = fss.transform(X_train, top_n=features_array[i])
    X_test_top_features_fss = fss.transform(X_test, top_n=features_array[i])

    catboost_classifier_fss = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.1, loss_function='MultiClass', custom_metric=['Accuracy'])
    catboost_classifier_fss.fit(X_train_top_features_fss, y_train)
    catboost_accuracy_fss = catboost_classifier_fss.score(
        X_test_top_features_fss, y_test)
    catboost_y_pred_proba_fss = catboost_classifier_fss.predict_proba(
        X_test_top_features_fss)
    catboost_top_k_accuracy_fss = top_k_accuracy_score(
        y_test, catboost_y_pred_proba_fss, k=k)
    catboost_accuracies_fss.append(catboost_accuracy_fss)
    catboost_top_k_accuracies_fss.append(catboost_top_k_accuracy_fss)

    X_train_top_features_fsu = fsu.transform(X_train, top_n=features_array[i])
    X_test_top_features_fsu = fsu.transform(X_test, top_n=features_array[i])

    catboost_classifier_fsu = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.1, loss_function='MultiClass', custom_metric=['Accuracy'])
    catboost_classifier_fsu.fit(X_train_top_features_fsu, y_train)
    catboost_accuracy_fsu = catboost_classifier_fsu.score(
        X_test_top_features_fsu, y_test)
    catboost_y_pred_proba_fsu = catboost_classifier_fsu.predict_proba(
        X_test_top_features_fsu)
    catboost_top_k_accuracy_fsu = top_k_accuracy_score(
        y_test, catboost_y_pred_proba_fsu, k=k)
    catboost_accuracies_fsu.append(catboost_accuracy_fsu)
    catboost_top_k_accuracies_fsu.append(catboost_top_k_accuracy_fsu)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(percentages, [acc * 100 for acc in catboost_accuracies_fss],
         'b--', label='Accuracy')  # Convertendo para porcentagem
plt.plot(percentages, [top_k_acc * 100 for top_k_acc in catboost_top_k_accuracies_fss],
         'ro--', label='Top-3 Accuracy')  # Convertendo para porcentagem
plt.axhline(y=100/11, color='gray', linestyle='--',
            label='Random Guess (1/11)')  # Linha para sorteio aleatório
# plt.xticks(range(1, 12))  # Exibindo todos os valores de k no eixo x
plt.yticks(range(0, 71, 10))  # Escala do eixo y em intervalos de 10%
plt.xlabel('Top features selected (%)')
plt.ylabel('Accuracy (%)')  # Alterando o rótulo do eixo y para porcentagem
plt.title('XGBoost Accuracy and Top-3 Accuracy with Feature Selection (Inf-FS_S)')
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.gca().xaxis.set_major_formatter(PercentFormatter())
plt.legend()
plt.grid(True)
plt.savefig('../02. Relatorio/feature-selection-fss.png',
            format='png', dpi=300)
plt.show()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(percentages, [acc * 100 for acc in catboost_accuracies_fss],
         'b--', label='Accuracy')  # Convertendo para porcentagem
plt.plot(percentages, [top_k_acc * 100 for top_k_acc in catboost_top_k_accuracies_fss],
         'ro--', label='Top-3 Accuracy')  # Convertendo para porcentagem
plt.axhline(y=100/11, color='gray', linestyle='--',
            label='Random Guess (1/11)')  # Linha para sorteio aleatório
# plt.xticks(range(1, 12))  # Exibindo todos os valores de k no eixo x
plt.yticks(range(0, 71, 10))  # Escala do eixo y em intervalos de 10%
plt.xlabel('Top features selected (%)')
plt.ylabel('Accuracy (%)')  # Alterando o rótulo do eixo y para porcentagem
plt.title('XGBoost Accuracy and Top-3 Accuracy with Feature Selection (Inf-FS_U)')
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.gca().xaxis.set_major_formatter(PercentFormatter())
plt.legend()
plt.grid(True)
plt.savefig('../02. Relatorio/feature-selection-fsu.png',
            format='png', dpi=300)
plt.show()

for element in fss.RANKED:
    print(element)

for element in fsu.RANKED:
    print(element)
