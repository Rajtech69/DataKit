"""Tests for machine learning models module src/datakit/ml/models.py."""
import pandas as pd
import pytest

from datakit import DataKit
from datakit.core.results import ModelResult, PlotResult


class TestFitModel:
    def test_fit_random_forest_classification(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="smoker", model="rf", random_state=42)

        assert isinstance(res, ModelResult)
        assert res.task == "classification"
        assert "accuracy" in res.metrics
        assert "f1" in res.metrics
        assert res.feature_importances is not None
        assert "=== DataKit Model Training Report" in res.summary()

        plot_imp = res.plot_importance()
        assert isinstance(plot_imp, PlotResult)

        plot_eval = res.plot_evaluation()
        assert isinstance(plot_eval, PlotResult)

        preds = res.predict(res.X_test)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(res.y_test)

    def test_fit_random_forest_regression(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="charges", model="rf", task="regression", random_state=42)

        assert isinstance(res, ModelResult)
        assert res.task == "regression"
        assert "r2" in res.metrics
        assert "mae" in res.metrics
        assert "rmse" in res.metrics

        plot_eval = res.plot_evaluation()
        assert isinstance(plot_eval, PlotResult)

    def test_fit_logistic_regression(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="smoker", model="logistic", random_state=42, C=0.5, max_iter=500)

        assert isinstance(res, ModelResult)
        assert res.model_name == "LogisticRegression"
        assert res.task == "classification"
        assert "accuracy" in res.metrics
        assert res.feature_importances is not None
        assert isinstance(res.feature_importances, pd.Series)
        assert not res.feature_importances.empty

        plot_imp = res.plot_importance()
        assert isinstance(plot_imp, PlotResult)

        # Test uniform kwarg alias: alpha -> C
        res_alpha = cleaned.fit(target="smoker", model="logistic", alpha=2.0, random_state=42)
        assert res_alpha.model.C == pytest.approx(0.5)
        assert res_alpha.feature_importances is not None

    def test_fit_linear_regression(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="charges", model="linear", task="regression", random_state=42)

        assert isinstance(res, ModelResult)
        assert res.model_name == "LinearRegression"
        assert res.task == "regression"
        assert "r2" in res.metrics
        assert res.feature_importances is not None
        assert isinstance(res.feature_importances, pd.Series)
        assert not res.feature_importances.empty

        plot_imp = res.plot_importance()
        assert isinstance(plot_imp, PlotResult)

        preds = res.predict(res.X_test)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(res.y_test)

    def test_fit_ridge_regression_and_classification(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)

        # Ridge Regression
        res_reg = cleaned.fit(target="charges", model="ridge", task="regression", alpha=0.5, random_state=42)
        assert isinstance(res_reg, ModelResult)
        assert res_reg.model_name == "Ridge"
        assert res_reg.model.alpha == 0.5
        assert res_reg.task == "regression"
        assert "r2" in res_reg.metrics
        assert res_reg.feature_importances is not None
        assert isinstance(res_reg.feature_importances, pd.Series)
        assert isinstance(res_reg.plot_importance(), PlotResult)

        # Test uniform kwarg alias: C -> alpha
        res_c = cleaned.fit(target="charges", model="ridge", task="regression", C=2.0, random_state=42)
        assert res_c.model.alpha == pytest.approx(0.5)

        # Ridge Classification
        res_cls = cleaned.fit(target="smoker", model="ridge", task="classification", random_state=42)
        assert isinstance(res_cls, ModelResult)
        assert res_cls.model_name == "RidgeClassifier"
        assert res_cls.task == "classification"
        assert "accuracy" in res_cls.metrics
        assert res_cls.feature_importances is not None
        assert isinstance(res_cls.plot_importance(), PlotResult)

    def test_fit_lasso_regression(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(
            target="charges",
            model="lasso",
            task="regression",
            alpha=0.01,
            max_iter=2000,
            random_state=42,
        )

        assert isinstance(res, ModelResult)
        assert res.model_name == "Lasso"
        assert res.model.alpha == 0.01
        assert res.model.max_iter == 2000
        assert res.task == "regression"
        assert "r2" in res.metrics
        assert res.feature_importances is not None
        assert isinstance(res.feature_importances, pd.Series)
        assert isinstance(res.plot_importance(), PlotResult)

        # Test uniform kwarg alias: C -> alpha
        res_c = cleaned.fit(target="charges", model="lasso", task="regression", C=10.0, random_state=42)
        assert res_c.model.alpha == pytest.approx(0.1)

    def test_fit_knn(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="smoker", model="knn", n_neighbors=3, weights="distance", metric="euclidean")

        assert isinstance(res, ModelResult)
        assert res.model_name == "KNeighborsClassifier"
        assert res.model.n_neighbors == 3
        assert res.model.weights == "distance"
        assert res.model.metric == "euclidean"
        assert "accuracy" in res.metrics

        preds = res.predict(res.X_test)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(res.y_test)

        plot_eval = res.plot_evaluation()
        assert isinstance(plot_eval, PlotResult)

        with pytest.raises(ValueError, match="does not support feature importances"):
            res.plot_importance()

    def test_fit_knn_regressor_and_k_alias(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="charges", model="knn", k=7, task="regression")

        assert isinstance(res, ModelResult)
        assert res.model_name == "KNeighborsRegressor"
        assert res.model.n_neighbors == 7
        assert res.task == "regression"
        assert "r2" in res.metrics
        assert "rmse" in res.metrics

    def test_fit_svc_classification_and_kwargs(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="smoker", model="svc", C=2.0, kernel="rbf", gamma="scale")

        assert isinstance(res, ModelResult)
        assert res.model_name == "SVC"
        assert res.model.C == 2.0
        assert res.model.kernel == "rbf"
        assert "accuracy" in res.metrics

        preds = res.predict(res.X_test)
        assert isinstance(preds, pd.Series)

        with pytest.raises(ValueError, match="does not support feature importances"):
            res.plot_importance()

    def test_fit_svm_c_alias(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="smoker", model="svm", c=0.5)

        assert isinstance(res, ModelResult)
        assert res.model_name == "SVC"
        assert res.model.C == 0.5

    def test_fit_svr_regression_and_kwargs(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="charges", model="svr", C=10.0, kernel="rbf", task="regression")

        assert isinstance(res, ModelResult)
        assert res.model_name == "SVR"
        assert res.model.C == 10.0
        assert res.task == "regression"
        assert "r2" in res.metrics
        assert "mae" in res.metrics

        with pytest.raises(ValueError, match="does not support feature importances"):
            res.plot_importance()

    def test_fit_naive_bayes_classification_and_kwargs(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="smoker", model="naive_bayes", smoothing=1e-8)

        assert isinstance(res, ModelResult)
        assert res.model_name == "GaussianNB"
        assert res.model.var_smoothing == 1e-8
        assert "accuracy" in res.metrics

        res_nb_alias = cleaned.fit(target="smoker", model="nb")
        assert isinstance(res_nb_alias, ModelResult)
        assert res_nb_alias.model_name == "GaussianNB"

        with pytest.raises(ValueError, match="does not support feature importances"):
            res.plot_importance()

    def test_fit_unsupported_model_raises(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        with pytest.raises(ValueError, match="Unsupported classification model"):
            cleaned.fit(target="smoker", model="nonexistent_algorithm")

    def test_fit_small_dataset_auto_task(self):
        data = DataKit({
            "age": [19, 28, 33, 32, 31, 46, 54, 37],
            "sex": ["female", "male", "male", "male", "female", "female", "female", "male"],
            "bmi": [27.9, 33.77, 22.7, 28.88, 25.74, 33.44, 30.8, 27.74],
            "smoker": ["yes", "no", "no", "no", "no", "no", "no", "no"],
            "charges": [16884.92, 1725.55, 4449.46, 21984.47, 3866.86, 8240.58, 11299.34, 7281.50]
        })
        res = data.fit(target="charges", model="linear")
        assert isinstance(res, ModelResult)
        assert res.task == "regression"
        assert res.model_name == "LinearRegression"
        assert "r2" in res.metrics

    def test_fit_gradient_boosting_classification(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(
            target="smoker",
            model="gb",
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=4,
            random_state=42,
        )

        assert isinstance(res, ModelResult)
        assert res.model_name == "GradientBoostingClassifier"
        assert res.task == "classification"
        assert "accuracy" in res.metrics
        assert "f1" in res.metrics
        assert res.feature_importances is not None
        assert isinstance(res.feature_importances, pd.Series)
        assert not res.feature_importances.empty

        # Test alias 'gradient_boosting'
        res_alias = cleaned.fit(target="smoker", model="gradient_boosting", n_estimators=20)
        assert isinstance(res_alias, ModelResult)
        assert res_alias.model_name == "GradientBoostingClassifier"

    def test_fit_gradient_boosting_regression(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(
            target="charges",
            model="gb",
            task="regression",
            n_estimators=60,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=3,
            random_state=42,
        )

        assert isinstance(res, ModelResult)
        assert res.model_name == "GradientBoostingRegressor"
        assert res.task == "regression"
        assert "r2" in res.metrics
        assert "mae" in res.metrics
        assert "rmse" in res.metrics
        assert res.feature_importances is not None
        assert isinstance(res.feature_importances, pd.Series)

    def test_fit_extra_trees_classification(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(
            target="smoker",
            model="extra_trees",
            n_estimators=40,
            max_depth=5,
            min_samples_split=3,
            random_state=42,
        )

        assert isinstance(res, ModelResult)
        assert res.model_name == "ExtraTreesClassifier"
        assert res.task == "classification"
        assert "accuracy" in res.metrics
        assert res.feature_importances is not None
        assert isinstance(res.feature_importances, pd.Series)

        # Test shortcut alias 'et'
        res_alias = cleaned.fit(target="smoker", model="et", n_estimators=20)
        assert isinstance(res_alias, ModelResult)
        assert res_alias.model_name == "ExtraTreesClassifier"

    def test_fit_extra_trees_regression(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(
            target="charges",
            model="extra_trees",
            task="regression",
            n_estimators=30,
            max_depth=6,
            min_samples_split=4,
            random_state=42,
        )

        assert isinstance(res, ModelResult)
        assert res.model_name == "ExtraTreesRegressor"
        assert res.task == "regression"
        assert "r2" in res.metrics
        assert res.feature_importances is not None
        assert isinstance(res.feature_importances, pd.Series)

    def test_fit_decision_tree_classification(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        # Pass uniform tree kwargs including n_estimators to test safe handling
        res = cleaned.fit(
            target="smoker",
            model="tree",
            n_estimators=100,  # Should be safely handled/ignored for single tree
            max_depth=4,
            min_samples_split=3,
            random_state=42,
        )

        assert isinstance(res, ModelResult)
        assert res.model_name == "DecisionTreeClassifier"
        assert res.task == "classification"
        assert "accuracy" in res.metrics
        assert res.feature_importances is not None
        assert isinstance(res.feature_importances, pd.Series)

        # Test alias 'decision_tree'
        res_alias = cleaned.fit(target="smoker", model="decision_tree", max_depth=3)
        assert isinstance(res_alias, ModelResult)
        assert res_alias.model_name == "DecisionTreeClassifier"

    def test_fit_decision_tree_regression(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(
            target="charges",
            model="decision_tree",
            task="regression",
            n_estimators=50,  # Should be safely handled/ignored for single tree
            max_depth=5,
            min_samples_split=2,
            random_state=42,
        )

        assert isinstance(res, ModelResult)
        assert res.model_name == "DecisionTreeRegressor"
        assert res.task == "regression"
        assert "r2" in res.metrics
        assert res.feature_importances is not None
        assert isinstance(res.feature_importances, pd.Series)

    def test_tree_models_feature_importances_format(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        for tree_model in ["rf", "gb", "extra_trees", "tree"]:
            res = cleaned.fit(target="smoker", model=tree_model, random_state=42)
            assert res.feature_importances is not None
            assert isinstance(res.feature_importances, pd.Series)
            assert len(res.feature_importances) > 0
            # Values should be sorted descending
            vals = res.feature_importances.values
            assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_predict_raw_dataframe(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="charges", model="rf", random_state=42)

        # Pass a raw DataFrame with un-encoded string categories
        raw_input = pd.DataFrame({
            "age": [30, 45],
            "sex": ["female", "male"],
            "bmi": [25.5, 31.0],
            "children": [1, 2],
            "smoker": ["no", "yes"],
            "region": ["southwest", "northeast"],
            "plan_type": ["basic", "basic"],
            "id": [101, 102]
        })

        preds = res.predict(raw_input)
        assert isinstance(preds, pd.Series)
        assert len(preds) == 2

    def test_classification_roc_and_pr_curves(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="smoker", model="rf", random_state=42)

        # Test predict_proba
        probs = res.predict_proba(res.X_test)
        assert isinstance(probs, pd.DataFrame)

        # Test plot_roc_curve and plot_pr_curve
        roc_plot = res.plot_roc_curve()
        assert isinstance(roc_plot, PlotResult)

        pr_plot = res.plot_pr_curve()
        assert isinstance(pr_plot, PlotResult)

    def test_tune_model_hyperparameter_search(self, insurance_datakit):
        from datakit.core.results import TuneResult
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)

        tune_res = cleaned.tune(target="charges", model="rf", n_iter=3, random_state=42)
        assert isinstance(tune_res, TuneResult)
        assert tune_res.model_name == "RandomForestRegressor"
        assert "=== DataKit Hyperparameter Tuning Report" in tune_res.summary()
        assert isinstance(tune_res.cv_results, pd.DataFrame)
        assert isinstance(tune_res.best_model, ModelResult)


