import itertools
import random
import json
import os
import torch
from train import train

class AutoML:
    def __init__(self, search_space, max_trials=10, output_file="automl_results.json"):
        self.search_space = search_space
        self.max_trials = max_trials
        self.output_file = output_file
        self.results = []
        
        # Load existing results if available
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r") as f:
                    self.results = json.load(f)
                print(f"Loaded {len(self.results)} existing trials from {self.output_file}")
            except Exception as e:
                print(f"Could not load existing results: {e}. Starting fresh.")
                self.results = []

    def get_random_config(self):
        """
        Samples a random configuration from the search space.
        """
        config = {}
        for key, values in self.search_space.items():
            config[key] = random.choice(values)
        return config

    def generate_grid(self):
        """
        Generates all possible combinations (Grid Search).
        """
        keys = self.search_space.keys()
        values = self.search_space.values()
        combinations = list(itertools.product(*values))
        
        configs = []
        for combo in combinations:
            config = dict(zip(keys, combo))
            configs.append(config)
        return configs

    def is_config_completed(self, config):
        """
        Checks if the configuration has already been run.
        Compares only the keys present in the search space.
        """
        for result in self.results:
            past_config = result["config"]
            match = True
            for key in self.search_space.keys():
                # We convert to string for comparison to avoid type mismatches or float precision issues if loaded from JSON
                if str(past_config.get(key)) != str(config.get(key)):
                    match = False
                    break
            if match:
                return True
        return False

    def run(self, method="random"):
        print(f"Starting AutoML with method: {method}")
        
        configs_to_run = []
        if method == "grid":
            all_configs = self.generate_grid()
            print(f"Grid Search: Generated {len(all_configs)} total combinations.")
            configs_to_run = all_configs
        else:
            # For random, we just generate max_trials configs
            # Note: Random might generate duplicates or already run configs. 
            # Ideally we'd check, but for now we'll stick to basic random generation 
            # and maybe just filter doubles if they happen to match exactly.
            configs_to_run = [self.get_random_config() for _ in range(self.max_trials)]

        # Filter out already completed configs
        pending_configs = []
        for config in configs_to_run:
            if not self.is_config_completed(config):
                pending_configs.append(config)
        
        print(f"Found {len(self.results)} completed trials. {len(pending_configs)} pending trials to run.")
        
        # Limit if max_trials is set (only for grid, or if we want to stop early)
        if method == "grid" and self.max_trials < len(pending_configs):
             print(f"Limiting to first {self.max_trials} pending trials.")
             pending_configs = pending_configs[:self.max_trials]

        best_score = float("-inf") 
        best_config = None
        
        # Determine current best from loaded results
        for res in self.results:
            score = res.get("score", float("-inf"))
            if score > best_score:
                best_score = score
                best_config = res["config"]

        for i, config in enumerate(pending_configs):
            trial_num = len(self.results) + 1
            print(f"\n\n{'='*50}")
            print(f"Trial {trial_num} (Pending {i+1}/{len(pending_configs)})")
            print(f"Config: {json.dumps(config, indent=2)}")
            print(f"{'='*50}\n")
            
            # Generate a unique experiment name
            optimizer_name = config.get('optimizer', 'Adam')
            rnn_name = config.get('rnn_type', 'LSTM')
            config["experiment_name"] = f"AutoML_Trial_{trial_num}_{optimizer_name}_{rnn_name}"
            
            # Enforce some defaults for AutoML
            config["num_epochs"] = 50 
            config["patience"] = 3     
            config["save_model"] = False 
            config["tqdm_disable"] = True 

            try:
                # Run Training
                result = train(config)
                
                # Check result
                score = result["best_bleu"]
                metric_name = "BLEU"
                if score == 0:
                     score = -result["best_val_loss"]
                     metric_name = "-ValLoss"
                
                record = {
                    "trial": trial_num,
                    "config": config,
                    "result": result,
                    "score": score,
                    "metric": metric_name
                }
                self.results.append(record)
                
                # Save results incrementally (overwrite file with full updated list)
                with open(self.output_file, "w") as f:
                    json.dump(self.results, f, indent=4)
                
                print(f"Trial {trial_num} Finished. Score ({metric_name}): {score}")
                
                if score > best_score:
                    best_score = score
                    best_config = config
                    print(f"!!! New Best Model Found !!!")
                    
            except Exception as e:
                print(f"[ERROR] Trial {trial_num} failed: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n\n{'='*80}")
        print("AutoML Complete.")
        print(f"Best Score: {best_score}")
        print(f"Best Config: {json.dumps(best_config, indent=2)}")
        print(f"{'='*80}")

if __name__ == "__main__":
    # Define your search space here
    search_space = {
        "learning_rate": [3e-4, 1e-4],
        "batch_size": [32, 64],
        "embed_size": [256, 512],
        "hidden_size": [512],
        "num_layers": [1], 
        "dropout": [0.5],
        "rnn_type": ["LSTM"],
        "optimizer": ["Adam"],
        "model_type": ["attention"],
        "attention_dim": [256, 512]
    }
    
    # max_trials set very high to ensure we cover the whole grid if needed
    automl = AutoML(search_space, max_trials=10000, output_file="automl_results_grid.json")
    
    # Use "grid" to try every single config
    automl.run(method="grid")
