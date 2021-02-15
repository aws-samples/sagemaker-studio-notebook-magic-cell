# SageMaker custom studio image providing magic cells for a simplified experience in a notebook.


## Overview

This solution provides a magic cell for a Jupyter Notebook to simplify the SageMaker training and SageMaker processing function by customizing a kernel and using Bring-Your-Own Image in SageMaker Studio. This solution was built in collaboration with Leo Blokhin (lblokhin@provectus.com
) from Provectus.

Look at the ./examples folder for Notebook examples for Tensorflow and Pytorch for MNIST training job.

Currently the functions supported are the following.

# Pytorch magic command.
```
  %pytorch [--estimator_name ESTIMATOR_NAME] [--entry_point ENTRY_POINT]
               [--source_dir SOURCE_DIR] [--role ROLE]
               [--framework_version FRAMEWORK_VERSION]
               [--py_version PY_VERSION] [--instance_type INSTANCE_TYPE]
               [--instance_count INSTANCE_COUNT] [--output_path OUTPUT_PATH]
               [--hyperparameters FOO:1,BAR:0.555,BAZ:ABC | 'FOO : 1, BAR : 0.555, BAZ : ABC']
               [--channel_training CHANNEL_TRAINING]
               [--channel_testing CHANNEL_TESTING]
               [--use_spot_instances [USE_SPOT_INSTANCES]]
               [--max_wait MAX_WAIT]
               [--enable_sagemaker_metrics [ENABLE_SAGEMAKER_METRICS]]
               [--metric_definitions ['Name: loss, Regex: Loss = .*?);' ['Name: loss, Regex: Loss = (.*?;' ...]]]
               [--name_contains NAME_CONTAINS] [--max_result MAX_RESULT]
               {submit,list,status,logs,delete}
```


# Tensorflow magic command.
```
  %tfjob [--estimator_name ESTIMATOR_NAME] [--entry_point ENTRY_POINT]
             [--source_dir SOURCE_DIR] [--role ROLE]
             [--framework_version FRAMEWORK_VERSION] [--py_version PY_VERSION]
             [--instance_type INSTANCE_TYPE] [--instance_count INSTANCE_COUNT]
             [--output_path OUTPUT_PATH]
             [--hyperparameters FOO:1,BAR:0.555,BAZ:ABC | 'FOO : 1, BAR : 0.555, BAZ : ABC']
             [--channel_training CHANNEL_TRAINING]
             [--channel_testing CHANNEL_TESTING]
             [--use_spot_instances [USE_SPOT_INSTANCES]] [--max_wait MAX_WAIT]
             [--enable_sagemaker_metrics [ENABLE_SAGEMAKER_METRICS]]
             [--metric_definitions ['Name: ganloss, Regex: GAN_loss=.*?);' ['Name: ganloss, Regex: GAN_loss=(.*?;' ...]]]
             [--distribution {parameter_server,horovod}]
             [--mpi_processes_per_host MPI_PROCESSES_PER_HOST]
             [--mpi_custom_mpi_options MPI_CUSTOM_MPI_OPTIONS]
             [--name_contains NAME_CONTAINS] [--max_result MAX_RESULT]
             {submit,list,status,logs,delete}
```

# Pyspark processor magic command
```
  %pyspark [--base_job_name BASE_JOB_NAME] [--submit_app SUBMIT_APP]
               [--framework_version FRAMEWORK_VERSION]
               [--instance_type INSTANCE_TYPE]
               [--instance_count INSTANCE_COUNT]
               [--max_runtime_in_seconds MAX_RUNTIME_IN_SECONDS]
               [--submit_py_files [SUBMIT_PY_FILES [SUBMIT_PY_FILES ...]]]
               [--submit_jars [SUBMIT_JARS [SUBMIT_JARS ...]]]
               [--submit_files [SUBMIT_FILES [SUBMIT_FILES ...]]]
               [--arguments '--foo bar --baz 123']
               [--spark_event_logs_s3_uri SPARK_EVENT_LOGS_S3_URI]
               [--logs [LOGS]] [--name_contains NAME_CONTAINS]
               [--max_result MAX_RESULT]
               {submit,list,status,delete}
```




# Building the image

You can follow the instructions below or run the `Makefile` by modifying the environment variables and running one of the following command.
```
make login
make build
make push
make create-version
make all 
```


Build the Docker image and push to Amazon ECR.
```
# Modify these as required. The Docker registry endpoint can be tuned based on your current region from https://docs.aws.amazon.com/general/latest/gr/ecr.html#ecr-docker-endpoints
REGION=<aws-region>
ACCOUNT_ID=<account-id>


# Build the image
IMAGE_NAME=smmagiccell
aws --region ${REGION} ecr get-login-password | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/smstudio-custom
docker build . -t ${IMAGE_NAME} -t ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/smstudio-custom:${IMAGE_NAME}
```

```
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/smstudio-custom:${IMAGE_NAME}
```

### Using with SageMaker Studio

Create a SageMaker Image with the image in ECR.

```
# Role in your account to be used for the SageMaker Image
ROLE_ARN=<role-arn>

aws --region ${REGION} sagemaker create-image \
    --image-name ${IMAGE_NAME} \
    --role-arn ${ROLE_ARN}

aws --region ${REGION} sagemaker create-image-version \
    --image-name ${IMAGE_NAME} \
    --base-image "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/smstudio-custom:${IMAGE_NAME}"

# Verify the image-version is created successfully. Do NOT proceed if image-version is in CREATE_FAILED state or in any other state apart from CREATED.
aws --region ${REGION} sagemaker describe-image-version --image-name ${IMAGE_NAME}
```

Create a AppImageConfig for this image

```
aws --region ${REGION} sagemaker create-app-image-config --cli-input-json file://app-image-config-input.json

```

Create a Domain, providing the SageMaker Image and AppImageConfig in the Domain creation. Replace the placeholders for VPC ID, Subnet IDs, and Execution Role in `create-domain-input.json`

```
aws --region ${REGION} sagemaker create-domain --cli-input-json file://create-domain-input.json
```

If you have an existing Domain, you can also use the `update-domain`

```
aws --region ${REGION} sagemaker update-domain --cli-input-json file://update-domain-input.json

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

