from sklearn.preprocessing import StandardScaler
def mandera_detect(gradients, true_poisoned_workers=None):
    """
    Improved MANDERA detection for MedMNIST experiments

    Returns
    -------
    predict_poi : list
        predicted malicious nodes
    precision : float
    recall : float
    f1 : float
    feats : DataFrame
    """

    if type(gradients) == pd.DataFrame:

        ranks = gradients.rank(axis=0, method="average")

        vars = ranks.var(axis=1).pow(0.5)
        mus = ranks.mean(axis=1)

        feats = pd.concat([mus, vars], axis=1)
        feats.columns = ["e_i", "v_i"]

    elif type(gradients) == list:

        flat_grad = flatten_grads(gradients)

        ranks = pd.DataFrame(flat_grad).rank(axis=0, method="average")

        vars = ranks.var(axis=1).pow(0.5)
        mus = ranks.mean(axis=1)

        feats = pd.concat([mus, vars], axis=1)
        feats.columns = ["e_i", "v_i"]

    else:
        raise ValueError(
            "gradients must be DataFrame or list"
        )

    # =====================================================
    # Feature normalization
    # =====================================================

    scaler = StandardScaler()

    X = scaler.fit_transform(feats.values)

    # =====================================================
    # KMeans clustering
    # =====================================================

    model = KMeans(
        n_clusters=2,
        random_state=0,
        n_init=20
    )

    group = model.fit_predict(X)

    # =====================================================
    # Choose malicious cluster
    #
    # Original paper:
    # use duplicate-count heuristic
    #
    # New:
    # malicious cluster usually has
    # smaller within-cluster variance
    # =====================================================

    cluster0 = X[group == 0]
    cluster1 = X[group == 1]

    var0 = np.mean(np.var(cluster0, axis=0))
    var1 = np.mean(np.var(cluster1, axis=0))

    size0 = len(cluster0)
    size1 = len(cluster1)

    # smaller & tighter cluster -> malicious

    score0 = var0 * size0
    score1 = var1 * size1

    if score0 < score1:
        bad_label = 0
    else:
        bad_label = 1

    predict_poi = [
        idx
        for idx, label in enumerate(group)
        if label == bad_label
    ]

    precision = 0.0
    recall = 0.0
    f1 = 0.0

    if true_poisoned_workers is not None:

        tp = len(
            set(predict_poi)
            &
            set(true_poisoned_workers)
        )

        fp = len(
            set(predict_poi)
            -
            set(true_poisoned_workers)
        )

        fn = len(
            set(true_poisoned_workers)
            -
            set(predict_poi)
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    return predict_poi, precision, recall, f1, feats