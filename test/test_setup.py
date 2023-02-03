import sys
import os
import unittest
sys.path.append("../src")
from setup import Setup


class TestSetup(unittest.TestCase):
    """
    Test class for the Setup class
    """
    def setUp(self):
        """
        Set up the class with test fixtures

        :param cfg_file: str, name to the config file
        """
        self.cfg_file = "config.yaml"
        self.setup = Setup(self.cfg_file)

    def test_read_config(self):
        """
        Test the read_config method
        """
