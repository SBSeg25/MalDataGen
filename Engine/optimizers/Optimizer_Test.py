import unittest

from AdaDelta import AdaDelta
from Adam import Adam
from FTRL import FTRL
from Nadam import NADAM
from Optimizers import Optimizers
from RSMProp import RMSProp
from SGD import SGD


class MockArguments:
    """Mock arguments class to initialize optimizers"""
# This implementation is adapted from the original Keras source code,
# available at: https://github.com/keras-team/keras
# It has been modified for customization and integration into this specific context.

    def __init__(self):
        # Adam parameters
        self.adam_optimizer_beta_1 = 0.9
        self.adam_optimizer_beta_2 = 0.999
        self.adam_optimizer_epsilon = 1e-7
        self.adam_optimizer_amsgrad = False
        self.adam_optimizer_learning_rate = 0.001

        # AdaDelta parameters
        self.ada_delta_optimizer_rho = 0.95
        self.ada_delta_optimizer_epsilon = 1e-7
        self.ada_delta_optimizer_use_ema = False
        self.ada_delta_optimizer_ema_momentum = 0.99
        self.ada_delta_optimizer_learning_rate = 0.001

        # NAdam parameters
        self.nadam_optimizer_beta_1 = 0.9
        self.nadam_optimizer_beta_2 = 0.999
        self.nadam_optimizer_epsilon = 1e-7
        self.nadam_optimizer_use_ema = False
        self.nadam_optimizer_ema_momentum = 0.99
        self.nadam_optimizer_learning_rate = 0.001

        # RMSProp parameters
        self.rsmprop_optimizer_rho = 0.9
        self.rsmprop_optimizer_epsilon = 1e-7
        self.rsmprop_optimizer_use_ema = False
        self.rsmprop_optimizer_momentum = 0.0
        self.rsmprop_optimizer_ema_momentum = 0.99
        self.rsmprop_optimizer_learning_rate = 0.001

        # SGD parameters
        self.sgd_optimizer_use_ema = False
        self.sgd_optimizer_momentum = 0.0
        self.sgd_optimizer_nesterov = False
        self.sgd_optimizer_ema_momentum = 0.99
        self.sgd_optimizer_learning_rate = 0.01

        # FTRL parameters
        self.ftrl_optimizer_beta = 0.0
        self.ftrl_optimizer_use_ema = False
        self.ftrl_optimizer_ema_momentum = 0.99
        self.ftrl_optimizer_learning_rate = 0.001
        self.ftrl_optimizer_learning_rate_power = -0.5
        self.ftrl_optimizer_initial_accumulator_value = 0.1
        self.ftrl_optimizer_l1_regularization_strength = 0.0
        self.ftrl_optimizer_l2_regularization_strength = 0.0
        self.ftrl_optimizer_l2_shrinkage_regularization_strength = 0.0


class TestOptimizers(unittest.TestCase):

    def setUp(self):
        """Initialize optimizers instance with mock arguments"""
        self.args = MockArguments()
        self.optimizers = Optimizers(self.args)

    def test_initialization(self):
        """Test that all parameters are correctly initialized"""
        self.assertEqual(self.optimizers._adam_optimizer_beta_1, 0.9)
        self.assertEqual(self.optimizers._ada_delta_optimizer_rho, 0.95)
        self.assertEqual(self.optimizers._nadam_optimizer_learning_rate, 0.001)
        self.assertEqual(self.optimizers._rsmprop_optimizer_momentum, 0.0)
        self.assertEqual(self.optimizers._sgd_optimizer_nesterov, False)
        self.assertEqual(self.optimizers._ftrl_optimizer_beta, 0.0)

    def test_get_optimizer_adam(self):
        """Test Adam optimizer creation"""
        optimizer = self.optimizers.get_optimizer_adam(
            learning_rate=0.002,
            beta_1=0.8,
            beta_2=0.888,
            epsilon=1e-8,
            amsgrad=True
        )
        self.assertIsInstance(optimizer, Adam)

        # Test invalid parameters
        with self.assertRaises(ValueError):
            self.optimizers.get_optimizer_adam(learning_rate=-0.001)
        with self.assertRaises(ValueError):
            self.optimizers.get_optimizer_adam(beta_1=1.1)
        with self.assertRaises(TypeError):
            self.optimizers.get_optimizer_adam(amsgrad="not_a_boolean")

    def test_get_optimizer_ada_delta(self):
        """Test AdaDelta optimizer creation"""
        optimizer = self.optimizers.get_optimizer_ada_delta(
            learning_rate=0.1,
            rho=0.9,
            epsilon=1e-6,
            use_ema=True,
            ema_momentum=0.95
        )
        self.assertIsInstance(optimizer, AdaDelta)

        # Test invalid parameters
        with self.assertRaises(ValueError):
            self.optimizers.get_optimizer_ada_delta(rho=1.1)
        with self.assertRaises(ValueError):
            self.optimizers.get_optimizer_ada_delta(ema_momentum=1.1)

    def test_get_optimizer_ftrl(self):
        """Test FTRL optimizer creation"""
        optimizer = self.optimizers.get_optimizer_ftrl(
            learning_rate=0.01,
            learning_rate_power=-0.1,
            initial_accumulator_value=0.5,
            l1_regularization_strength=0.1,
            l2_regularization_strength=0.1,
            beta=0.2,
            use_ema=True
        )
        self.assertIsInstance(optimizer, FTRL)

        # Test invalid parameters
        with self.assertRaises(ValueError):
            self.optimizers.get_optimizer_ftrl(learning_rate_power=0.1)
        with self.assertRaises(ValueError):
            self.optimizers.get_optimizer_ftrl(l1_regularization_strength=-0.1)

    def test_get_optimizer_nadam(self):
        """Test NAdam optimizer creation"""
        optimizer = self.optimizers.get_optimizer_nadam(
            learning_rate=0.005,
            beta_1=0.95,
            beta_2=0.9999,
            epsilon=1e-8,
            use_ema=True
        )
        self.assertIsInstance(optimizer, NADAM)

        # Test invalid parameters
        with self.assertRaises(ValueError):
            self.optimizers.get_optimizer_nadam(beta_2=1.1)

    def test_get_optimizer_rsmprop(self):
        """Test RMSProp optimizer creation"""
        optimizer = self.optimizers.get_optimizer_rsmprop(
            learning_rate=0.01,
            rho=0.85,
            momentum=0.1,
            epsilon=1e-6,
            centered=True
        )
        self.assertIsInstance(optimizer, RMSProp)

        # Test invalid parameters
        with self.assertRaises(ValueError):
            self.optimizers.get_optimizer_rsmprop(momentum=-0.1)
        with self.assertRaises(TypeError):
            self.optimizers.get_optimizer_rsmprop(centered="not_a_boolean")

    def test_get_optimizer_sgd(self):
        """Test SGD optimizer creation"""
        optimizer = self.optimizers.get_optimizer_sgd(
            learning_rate=0.1,
            momentum=0.9,
            nesterov=True
        )
        self.assertIsInstance(optimizer, SGD)

        # Test invalid parameters
        with self.assertRaises(ValueError):
            self.optimizers.get_optimizer_sgd(momentum=-0.1)
        with self.assertRaises(TypeError):
            self.optimizers.get_optimizer_sgd(nesterov="not_a_boolean")

    def test_get_optimizer_factory_method(self):
        """Test the main factory method with all optimizer types"""
        # Test each optimizer type
        for opt_name in ['adam', 'adadelta', 'ftrl', 'nadam', 'rsmprop', 'sgd']:
            optimizer = self.optimizers.get_optimizer(opt_name)
            self.assertIsNotNone(optimizer)

        # Test case insensitivity
        optimizer = self.optimizers.get_optimizer('Adam')
        self.assertIsInstance(optimizer, Adam)

        # Test unsupported optimizer
        with self.assertRaises(ValueError):
            self.optimizers.get_optimizer('unsupported_optimizer')


if __name__ == '__main__':
    unittest.main()
