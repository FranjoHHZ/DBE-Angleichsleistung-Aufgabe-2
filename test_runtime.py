import unittest
import pandas as pd
import numpy as np
from main_runtime import TheAlgorithm, X_train, y_train

class TestAlgorithm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Lade Testdaten aus CSV-Dateien
        cls.X_test = pd.read_csv('test_data.csv')
        cls.y_test = pd.read_csv('test_labels.csv')

        # Initialisiere die Algorithmus-Klasse mit den geladenen Testdaten
        cls.algo = TheAlgorithm(X_train, y_train, cls.X_test, cls.y_test)
        
        # Lade gespeicherte Referenzwerte (Test Accuracy aus reference_accuracies.txt)
        with open('reference_accuracies.txt', 'r') as f:
            lines = f.readlines()
            cls.reference_test_accuracy = float(lines[1].split(':')[1].strip())
        
        # Lade gespeicherte Referenzkonfusionsmatrix
        cls.reference_confusion_matrix = pd.read_csv('reference_confusion_matrix.csv').values

        # Führe die fit()-Funktion einmal aus und speichere die repräsentative Laufzeit
        cls.representative_runtime = cls.algo.fit()

    def test_predict_accuracy(self):
        """Test, ob die Accuracy und die Konfusionsmatrix korrekt sind."""
        test_accuracy = self.algo.predict()
        self.assertAlmostEqual(
            test_accuracy, self.reference_test_accuracy, delta=0.5,
            msg="Die Accuracy der predict()-Funktion weicht von der Referenz ab."
        )
        self.assertTrue(
            np.array_equal(self.algo.test_confusion_matrix, self.reference_confusion_matrix),
            "Die Konfusionsmatrix der predict()-Funktion weicht von der Referenz ab."
        )

    def test_fit_runtime(self):
        """Test, ob die Laufzeit der fit()-Funktion innerhalb von 120% der repräsentativen Laufzeit liegt."""
        import time
        start_time = time.time()
        self.algo.fit()
        elapsed_time = time.time() - start_time
        max_allowed_time = cls.representative_runtime * 1.2
        self.assertLessEqual(
            elapsed_time, max_allowed_time,
            "Die Laufzeit der fit()-Funktion überschreitet den Grenzwert."
        )

# Führe die Unit-Tests aus
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
