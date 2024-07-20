# InstaCart-
InstaCart Market Basket Analysis




class CustomStackingClassifier:
    def __init__(self, estimators, random_state, params, nround, 
                 version, loop=3,
                 valid_size=0.05, stratify=True, verbose=1,
                 early_stopping=60, use_probas=True):
        self.clf = estimators
        self.mod=cpickle
        self.loop = loop
        self.params = params
        self.nround = nround    
        self.version = version
        self.valid_size = valid_size
        self.verbose = verbose
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.models = []


    def split_build_valid(self, train_user, X_train, y_train):
        train_user['is_valid'] = np.random.choice(
            [0,1],
            size=len(train_user),
            p=[1-self.valid_size, self.valid_size])

        valid_n = train_user['is_valid'].sum()
        build_n = (train_user.shape[0] - valid_n)
        
        print('build user:{}, valid user:{}'.format(build_n, valid_n))
        valid_user = train_user[train_user['is_valid']==1].user_id
        is_valid = X_train.user_id.isin(valid_user)
        
        dbuild = lgb.Dataset(X_train[~is_valid].drop('user_id', axis=1),
                             y_train[~is_valid],
                             categorical_feature=['product_id', 'aisle_id', 'department_id'])
        dvalid = lgb.Dataset(X_train[is_valid].drop('user_id', axis=1),
                             label=y_train[is_valid],
                             categorical_feature=['product_id', 'aisle_id', 'department_id'])
        watchlist_set = [dbuild, dvalid]
        watchlist_name = ['build', 'valid']
        
        print('FINAL SHAPE')
        print('dbuild.shape:{}  dvalid.shape:{}\n'.format(
            dbuild.data.shape,
            dvalid.data.shape))
        return dbuild, dvalid, watchlist_set, watchlist_name

    def fit(self, x, y):
        np.random.seed(self.random_state)
        train_user = x[['user_id']].drop_duplicates()

        for i in range(self.loop):
            dbuild, dvalid, watchlist_set, watchlist_name = self.split_build_valid(train_user, x, y)
            gc.collect();

            # Train models
            model = lgb.train(
                self.params,
                dbuild,
                self.nround,
                watchlist_set,
                watchlist_name,
                early_stopping_rounds=self.early_stopping,
                categorical_feature=['product_id', 'aisle_id', 'department_id'],
                verbose_eval=5)
            joblib.dump(model, "lgb_models/lgb_trained_{}_{}".format(self.version, i))
            self.models.append(model)
            del [dbuild, dvalid, watchlist_set, watchlist_name];
            gc.collect();
        del train_user;
        gc.collect()
        return self


    def predict(self, x, test_data):
        sub_test = test_data[['order_id', 'product_id']]
        sub_test['yhat'] = 0
        for model in self.models:
            sub_test['yhat'] += model.predict(x)
        sub_test['yhat'] /= self.loop
        return sub_test
