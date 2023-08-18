from zenml import step, pipeline

@step
def method():
    print("Method 1")

@pipeline
def train():
    method()

train()