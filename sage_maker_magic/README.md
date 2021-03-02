## Jupyter Kernel Extensions

### sm kernel

This kernel provides support for interactive Tensorflow, Pytorch integration in SageMaker Studio.

See `sage_maker_magic/sage_maker_kernel/kernelmagics.py` for the definitions of all cell magics.

+ Distributed Pytorch in AWS Sagemaker Studio [Example Notebook](examples/PyTorch_Magic.ipynb)
+ Distributed Tensorflow in AWS Sagemaker Studio [Example Notebook](examples/TF_Magic.ipynb)
+ Distributed Scikit Learn in SageMaker with Magic [Example Notebook](examples/SKLearn_Magic.ipynb)
+ SageMaker Python SDK, boto3, AWS CLI examples [Example Notebook](examples/SDK_sm_boto3_AWS_CLI.ipynb)
+ PySpark Processing in AWS Sagemaker Studio [Example Notebook](examples/Spark_Magic.ipynb)


### %%pytorch?

```
Docstring:
::

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
               {submit,list,status,logs,delete,show_defaults}

Pytorch magic command.

methods:
  {submit,list,status,logs,delete,show_defaults}

submit:
  --estimator_name ESTIMATOR_NAME
                        estimator shell variable name
  --entry_point ENTRY_POINT
                        notebook local code file
  --source_dir SOURCE_DIR
                        notebook local code src, may contain requirements.txt
  --role ROLE           An AWS IAM role (either name or full ARN). The Amazon
                        SageMaker training jobs and APIs that create Amazon
                        SageMaker endpoints use this role to access training
                        data and model artifacts. After the endpoint is
                        created, the inference code might use the IAM role, if
                        it needs to access an AWS resource.
  --framework_version FRAMEWORK_VERSION
                        PyTorch version
  --py_version PY_VERSION
                        Python version
  --instance_type INSTANCE_TYPE
                        Type of EC2 instance to use for training, for example,
                        ‘ml.c4.xlarge’.
  --instance_count INSTANCE_COUNT
                        Number of Amazon EC2 instances to use for training.
  --output_path OUTPUT_PATH
                        S3 location for saving the training result (model
                        artifacts and output files). If not specified, results
                        are stored to a default bucket. If the bucket with the
                        specific name does not exist, the estimator creates
                        the bucket during the fit() method execution.
  --hyperparameters <FOO:1,BAR:0.555,BAZ:ABC | 'FOO : 1, BAR : 0.555, BAZ : ABC'>
                        Hyperparameters are passed to your script as arguments
                        and can be retrieved with an argparse.
  --channel_training CHANNEL_TRAINING
                        A string that represents the path to the directory
                        that contains the input data for the training channel.
  --channel_testing CHANNEL_TESTING
                        A string that represents the path to the directory
                        that contains the input data for the testing channel.

submit-spot:
  --use_spot_instances <[USE_SPOT_INSTANCES]>
                        Specifies whether to use SageMaker Managed Spot
                        instances for training. If enabled then the max_wait
                        arg should also be set. More information:
                        https://docs.aws.amazon.com/sagemaker/latest/dg/model-
                        managed-spot-training.html
  --max_wait MAX_WAIT   Timeout in seconds waiting for spot training instances
                        (default: None). After this amount of time Amazon
                        SageMaker will stop waiting for Spot instances to
                        become available (default: None).

submit-metrics:
  --enable_sagemaker_metrics <[ENABLE_SAGEMAKER_METRICS]>
                        Enables SageMaker Metrics Time Series. For more
                        information see: https://docs.aws.amazon.com/sagemaker
                        /latest/dg/API_AlgorithmSpecification.html# SageMaker-
                        Type-AlgorithmSpecification-
                        EnableSageMakerMetricsTimeSeries
  --metric_definitions <['Name: loss, Regex: Loss = (.*?);' ['Name: loss, Regex: Loss = (.*?);' ...]]>
                        A list of dictionaries that defines the metric(s) used
                        to evaluate the training jobs. Each dictionary
                        contains two keys: ‘Name’ for the name of the metric,
                        and ‘Regex’ for the regular expression used to extract
                        the metric from the logs. This should be defined only
                        for jobs that don’t use an Amazon algorithm.

list:
  --name_contains NAME_CONTAINS
  --max_result MAX_RESULT
File:      sage_maker_kernel/kernelmagics.py
```

### %%tfjob?

```
Docstring:
::

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
             {submit,list,status,logs,delete,show_defaults}

Tensorflow magic command.

methods:
  {submit,list,status,logs,delete,show_defaults}

submit:
  --estimator_name ESTIMATOR_NAME
                        estimator shell variable name
  --entry_point ENTRY_POINT
                        notebook local code file
  --source_dir SOURCE_DIR
                        notebook local code src, may contain requirements.txt
  --role ROLE           An AWS IAM role (either name or full ARN). The Amazon
                        SageMaker training jobs and APIs that create Amazon
                        SageMaker endpoints use this role to access training
                        data and model artifacts. After the endpoint is
                        created, the inference code might use the IAM role, if
                        it needs to access an AWS resource.
  --framework_version FRAMEWORK_VERSION
                        TensorFlow version
  --py_version PY_VERSION
                        Python version
  --instance_type INSTANCE_TYPE
                        Type of EC2 instance to use for training, for example,
                        ‘ml.c4.xlarge’.
  --instance_count INSTANCE_COUNT
                        Number of Amazon EC2 instances to use for training.
  --output_path OUTPUT_PATH
                        S3 location for saving the training result (model
                        artifacts and output files). If not specified, results
                        are stored to a default bucket. If the bucket with the
                        specific name does not exist, the estimator creates
                        the bucket during the fit() method execution.
  --hyperparameters <FOO:1,BAR:0.555,BAZ:ABC | 'FOO : 1, BAR : 0.555, BAZ : ABC'>
                        Hyperparameters are passed to your script as arguments
                        and can be retrieved with an argparse.
  --channel_training CHANNEL_TRAINING
                        A string that represents the path to the directory
                        that contains the input data for the training channel.
  --channel_testing CHANNEL_TESTING
                        A string that represents the path to the directory
                        that contains the input data for the testing channel.

submit-spot:
  --use_spot_instances <[USE_SPOT_INSTANCES]>
                        Specifies whether to use SageMaker Managed Spot
                        instances for training. If enabled then the max_wait
                        arg should also be set. More information:
                        https://docs.aws.amazon.com/sagemaker/latest/dg/model-
                        managed-spot-training.html
  --max_wait MAX_WAIT   Timeout in seconds waiting for spot training instances
                        (default: None). After this amount of time Amazon
                        SageMaker will stop waiting for Spot instances to
                        become available (default: None).

submit-metrics:
  --enable_sagemaker_metrics <[ENABLE_SAGEMAKER_METRICS]>
                        Enables SageMaker Metrics Time Series. For more
                        information see: https://docs.aws.amazon.com/sagemaker
                        /latest/dg/API_AlgorithmSpecification.html# SageMaker-
                        Type-AlgorithmSpecification-
                        EnableSageMakerMetricsTimeSeries
  --metric_definitions <['Name: ganloss, Regex: GAN_loss=(.*?);' ['Name: ganloss, Regex: GAN_loss=(.*?);' ...]]>
                        A list of dictionaries that defines the metric(s) used
                        to evaluate the training jobs. Each dictionary
                        contains two keys: ‘Name’ for the name of the metric,
                        and ‘Regex’ for the regular expression used to extract
                        the metric from the logs. This should be defined only
                        for jobs that don’t use an Amazon algorithm.

submit-distribution:
  --distribution <{parameter_server,horovod}>
                        To run your training job with multiple instances in a
                        distributed fashion, set instance_count to a number
                        larger than 1. We support two different types of
                        distributed training, parameter server and Horovod.
                        The distribution parameter is used to configure which
                        distributed training strategy to use.
  --mpi_processes_per_host MPI_PROCESSES_PER_HOST
                        horovod mpi_processes_per_host
  --mpi_custom_mpi_options MPI_CUSTOM_MPI_OPTIONS
                        horovod custom_mpi_options

list:
  --name_contains NAME_CONTAINS
  --max_result MAX_RESULT
File:      sage_maker_kernel/kernelmagics.py
```
### %%%%sklearn?
```
Docstring:
::

  %sklearn [--estimator_name ESTIMATOR_NAME] [--entry_point ENTRY_POINT]
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
               {submit,list,status,logs,delete,show_defaults}

SKLearn magic command.

methods:
  {submit,list,status,logs,delete,show_defaults}

submit:
  --estimator_name ESTIMATOR_NAME
                        estimator shell variable name
  --entry_point ENTRY_POINT
                        notebook local code file
  --source_dir SOURCE_DIR
                        Path (absolute, relative or an S3 URI) to a directory
                        with any other training source code dependencies aside
                        from the entry point file (default: None). If
                        source_dir is an S3 URI, it must point to a tar.gz
                        file. Structure within this directory are preserved
                        when training on Amazon SageMaker.
  --role ROLE           An AWS IAM role (either name or full ARN). The Amazon
                        SageMaker training jobs and APIs that create Amazon
                        SageMaker endpoints use this role to access training
                        data and model artifacts. After the endpoint is
                        created, the inference code might use the IAM role, if
                        it needs to access an AWS resource.
  --framework_version FRAMEWORK_VERSION
                        Scikit-learn version you want to use for executing
                        your model training code. List of supported versions:
                        https://github.com/aws/sagemaker-python-sdk#sklearn-
                        sagemaker-estimators
  --py_version PY_VERSION
                        Python version
  --instance_type INSTANCE_TYPE
                        Type of EC2 instance to use for training, for example,
                        ‘ml.c4.xlarge’.
  --instance_count INSTANCE_COUNT
                        Number of Amazon EC2 instances to use for training.
  --output_path OUTPUT_PATH
                        S3 location for saving the training result (model
                        artifacts and output files). If not specified, results
                        are stored to a default bucket. If the bucket with the
                        specific name does not exist, the estimator creates
                        the bucket during the fit() method execution.
  --hyperparameters <FOO:1,BAR:0.555,BAZ:ABC | 'FOO : 1, BAR : 0.555, BAZ : ABC'>
                        Hyperparameters are passed to your script as arguments
                        and can be retrieved with an argparse.
  --channel_training CHANNEL_TRAINING
                        A string that represents the path to the directory
                        that contains the input data for the training channel.
  --channel_testing CHANNEL_TESTING
                        A string that represents the path to the directory
                        that contains the input data for the testing channel.

submit-spot:
  --use_spot_instances <[USE_SPOT_INSTANCES]>
                        Specifies whether to use SageMaker Managed Spot
                        instances for training. If enabled then the max_wait
                        arg should also be set. More information:
                        https://docs.aws.amazon.com/sagemaker/latest/dg/model-
                        managed-spot-training.html
  --max_wait MAX_WAIT   Timeout in seconds waiting for spot training instances
                        (default: None). After this amount of time Amazon
                        SageMaker will stop waiting for Spot instances to
                        become available (default: None).

submit-metrics:
  --enable_sagemaker_metrics <[ENABLE_SAGEMAKER_METRICS]>
                        Enables SageMaker Metrics Time Series. For more
                        information see: https://docs.aws.amazon.com/sagemaker
                        /latest/dg/API_AlgorithmSpecification.html# SageMaker-
                        Type-AlgorithmSpecification-
                        EnableSageMakerMetricsTimeSeries
  --metric_definitions <['Name: loss, Regex: Loss = (.*?);' ['Name: loss, Regex: Loss = (.*?);' ...]]>
                        A list of dictionaries that defines the metric(s) used
                        to evaluate the training jobs. Each dictionary
                        contains two keys: ‘Name’ for the name of the metric,
                        and ‘Regex’ for the regular expression used to extract
                        the metric from the logs. This should be defined only
                        for jobs that don’t use an Amazon algorithm.

list:
  --name_contains NAME_CONTAINS
  --max_result MAX_RESULT
```

### %%pyspark?
```
Docstring:
::

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
               {submit,list,status,delete,show_defaults}

Pyspark processor magic command

positional arguments:
  {submit,list,status,delete,show_defaults}

processor:
  --base_job_name BASE_JOB_NAME
                        Prefix for processing name. If not specified, the
                        processor generates a default job name, based on the
                        training image name and current timestamp.
  --submit_app SUBMIT_APP
                        Path (local or S3) to Python file to submit to Spark
                        as the primary application
  --framework_version FRAMEWORK_VERSION
                        The version of SageMaker PySpark.
  --instance_type INSTANCE_TYPE
                        Type of EC2 instance to use for processing, for
                        example, ‘ml.c4.xlarge’.
  --instance_count INSTANCE_COUNT
                        Number of Amazon EC2 instances to use for processing.
  --max_runtime_in_seconds MAX_RUNTIME_IN_SECONDS
                        Timeout in seconds. After this amount of time Amazon
                        SageMaker terminates the job regardless of its current
                        status.
  --submit_py_files <[SUBMIT_PY_FILES [SUBMIT_PY_FILES ...]]>
                        You can specify any python dependencies or files that
                        your script depends on
  --submit_jars <[SUBMIT_JARS [SUBMIT_JARS ...]]>
                        You can specify any jar dependencies or files that
                        your script depends on
  --submit_files <[SUBMIT_FILES [SUBMIT_FILES ...]]>
                        List of .zip, .egg, or .py files to place on the
                        PYTHONPATH for Python apps.
  --arguments <'--foo bar --baz 123'>
                        A list of string arguments to be passed to a
                        processing job
  --spark_event_logs_s3_uri SPARK_EVENT_LOGS_S3_URI
                        S3 path where spark application events will be
                        published to.
  --logs <[LOGS]>       Whether to show the logs produced by the job.

list:
  --name_contains NAME_CONTAINS
  --max_result MAX_RESULT
```