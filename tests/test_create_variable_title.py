import unittest
import pandas as pd

from src.features.build_features import create_variable_title

class TestCreateVariableTitle(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
                     'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
                     'Allen, Mr. William Henry', 'Moran, Mr. James',
                     'McCarthy, Mr. Timothy J', 'Palsson, Master. Gosta Leonard',
                     'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)',
                     'Nasser, Mrs. Nicholas (Adele Achem)'],
            'Age': [22, 38, 26, 35, 35, 27, 54, 2, 27, 14],
            'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
        })

    def test_create_variable_title_default_variable_name(self):
        expected_result = pd.DataFrame({
            'Title': ['Mr.', 'Mrs.', 'Miss.', 'Mrs.', 'Mr.', 'Mr.', 'Mr.', 'Master.', 'Mrs.', 'Mrs.'],
            'Age': [22, 38, 26, 35, 35, 27, 54, 2, 27, 14],
            'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
        })
        result = create_variable_title(self.data)
        self.assertTrue(result['Title'].equals(expected_result['Title']))

    def test_create_variable_title_custom_variable_name(self):
        expected_result = pd.DataFrame({
            'Title': ['Mr.', 'Mrs.', 'Miss.', 'Mrs.', 'Mr.', 'Mr.', 'Mr.', 'Master.', 'Mrs.', 'Mrs.'],
            'Age': [22, 38, 26, 35, 35, 27, 54, 2, 27, 14],
            'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
        })
        result = create_variable_title(self.data, variable_name='Name')
        self.assertTrue(result['Title'].equals(expected_result['Title']))

    def test_create_variable_title_replace_dona(self):
        data = pd.DataFrame({
            'Name': ['Mrs, Dona. Anna', 'Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)'],
            'Age': [39, 22, 38],
            'Survived': [1, 0, 1]
        })
        expected_result = pd.DataFrame({
            'Title': ['Mrs.', 'Mr.', 'Mrs.'],
            'Age': [39, 22, 38],
            'Survived': [1, 0, 1]
        })
        result = create_variable_title(data, variable_name='Name')
        self.assertTrue(result['Title'].equals(expected_result['Title']))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)