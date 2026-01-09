from .motion_planning import planning_batch_inference




class PlanningAgent:
    def __init__(self, model_name="", verbose=True) -> None:
        self.verbose = verbose
    
    
    def run_batch(self, data_samples, data_path, save_path, args=None):
        inference_result = planning_batch_inference(
            data_samples=data_samples, 
            data_path=data_path, 
            save_path=save_path,
            args=args,
            verbose=self.verbose
        )
        return inference_result