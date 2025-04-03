import pytest
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score

# Define a minimal model class within the test file for self-containment
class SimpleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
        # No is_fitted_ attribute initially

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        # Dummy fitting logic: store the number of features and classes
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.is_fitted_ = True # Standard scikit-learn attribute indicating fitted state
        # Return the classifier
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        # Dummy prediction logic: predict the first class for all samples
        return np.full(X.shape[0], self.classes_[0])

    def score(self, X, y):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# Fixture for dummy data
@pytest.fixture
def dummy_data():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    return X, y

# Fixture for an initialized model
@pytest.fixture
def initialized_model():
    return SimpleModel(C=1.0)

# Fixture for a trained model
@pytest.fixture
def trained_model(initialized_model, dummy_data):
    X, y = dummy_data
    return initialized_model.fit(X, y)

# Test 1: Model Initialization
def test_model_initialization(initialized_model):
    assert initialized_model is not None
    assert isinstance(initialized_model, SimpleModel)
    assert initialized_model.C == 1.0
    # Check that fit attributes are not present before fitting
    assert not hasattr(initialized_model, 'is_fitted_')
    assert not hasattr(initialized_model, 'classes_')
    assert not hasattr(initialized_model, 'n_features_in_')

# Test 2: Model Training
def test_model_training(initialized_model, dummy_data):
    X, y = dummy_data
    model = initialized_model.fit(X, y)
    # Check standard sklearn fitted attribute
    assert hasattr(model, 'is_fitted_') and model.is_fitted_
    # Check attributes learned during fit
    assert hasattr(model, 'classes_')
    assert hasattr(model, 'n_features_in_')
    assert model.n_features_in_ == X.shape[1]
    np.testing.assert_array_equal(model.classes_, np.unique(y))
    # Check fit returns self
    assert isinstance(model, SimpleModel)

# Test 3: Model Prediction
def test_model_prediction(trained_model, dummy_data):
    X, y = dummy_data
    predictions = trained_model.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (X.shape[0],)
    # Check if predictions are within the expected classes
    assert np.all(np.isin(predictions, trained_model.classes_))

    # Test prediction on unseen data (different shape/values)
    X_new = np.array([[9.0, 10.0], [11.0, 12.0]])
    predictions_new = trained_model.predict(X_new)
    assert predictions_new.shape == (X_new.shape[0],)
    assert np.all(np.isin(predictions_new, trained_model.classes_))

# Test 4: Model Evaluation
def test_model_evaluation(trained_model, dummy_data):
    X, y = dummy_data
    score = trained_model.score(X, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0 # Assuming score is accuracy or similar metric

# Test edge cases: using model before fitting
def test_predict_before_fit(initialized_model, dummy_data):
    X, _ = dummy_data
    with pytest.raises(NotFittedError):
        initialized_model.predict(X)

def test_evaluate_before_fit(initialized_model, dummy_data):
    X, y = dummy_data
    with pytest.raises(NotFittedError):
        initialized_model.score(X, y)

# Test initialization with different parameters
def test_model_initialization_custom_params():
    model = SimpleModel(C=10.0)
    assert model.C == 10.0