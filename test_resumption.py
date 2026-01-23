import unittest
import json
import os
import shutil
from unittest.mock import MagicMock
import automl

# Mock the train function to avoid actual training
original_train = automl.train
automl.train = MagicMock(return_value={"best_val_loss": 0.5, "best_bleu": 25.0, "experiment_dir": "mock_dir"})

class TestAutoMLResumption(unittest.TestCase):
    def setUp(self):
        self.output_file = "test_automl_results.json"
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
            
        self.search_space = {
            "learning_rate": [0.001, 0.01],
            "batch_size": [16]
        }
        # Total combinations: 2

    def tearDown(self):
        # if os.path.exists(self.output_file):
        #     os.remove(self.output_file)
        automl.train.reset_mock()

    def test_resumption(self):
        # Run 1: Run only 1 trial
        print("--- Running Batch 1 (1 trial) ---")
        runner1 = automl.AutoML(self.search_space, max_trials=1, output_file=self.output_file)
        runner1.run(method="grid")
        
        # Verify 1 result exists
        with open(self.output_file, 'r') as f:
            results1 = json.load(f)
        self.assertEqual(len(results1), 1)
        self.assertEqual(automl.train.call_count, 1)
        
        # Run 2: Run "2" trials (should be 2 total, so 1 new one)
        print("--- Running Batch 2 (Should skip 1, run 1) ---")
        runner2 = automl.AutoML(self.search_space, max_trials=2, output_file=self.output_file)
        runner2.run(method="grid")
        
        # Verify 2 results total
        with open(self.output_file, 'r') as f:
            results2 = json.load(f)
        self.assertEqual(len(results2), 2)
        
        # Train should have been called exactly 1 more time (total 2)
        self.assertEqual(automl.train.call_count, 2)
        
        # Check that we have both configs
        lrs = [r['config']['learning_rate'] for r in results2]
        self.assertIn(0.001, lrs)
        self.assertIn(0.01, lrs)

if __name__ == '__main__':
    unittest.main()
