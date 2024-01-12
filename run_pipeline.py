from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

if __name__ =="__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
<<<<<<< HEAD
    training_pipeline(data_path="data/data.xlsx")
=======
    training_pipeline(data_path="data\Banking Case - Data.xlsx")
>>>>>>> 657c1bf529173177957a36bef2e11ad6eca77ac2

