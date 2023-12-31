from pipelines.deployment_pipleine import deployment_pipeline, inference_pipeline
import click

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy and predict"
@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY,PREDICT,DEPLOY_AND_PREDICT])
    DEFAULT=DEPLOY_AND_PREDICT,
    help="Optionally you can choose to only run the deployment"
    "pipeline to train and deploy a model ('deploy'), or to"
    "only run a prediction against the deployed model"
    "('predict).By default both will be run"
    "('deploy_and_predict').",
)

@click.option(
    "--min-accuracy",
    default=0.92,
    help="Minimum accuracy required to deploy the model",
)

def run_deployment(confi:str,min_accuracy:float):
    if deploy:
        deployment_pipeline(min_accuracy)
    if predict:
        inference_pipeline(min_accuracy)