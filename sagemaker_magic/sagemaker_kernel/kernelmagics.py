# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from __future__ import print_function
from IPython.core.magic import Magics, magics_class
from IPython.core.magic import needs_local_scope, cell_magic, line_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring, defaults, argument_group

import uuid
import re
import json

## Sage Maker

from sagemaker import get_execution_role, Session
from sagemaker.pytorch import PyTorch
from sagemaker.tensorflow import TensorFlow
from sagemaker import get_execution_role
import boto3


def hyperparameters(string):
    return dict(re.findall(r"([a-zA-Z_][\w\-]*)\s*:\s*([\w\.\-]+)", string))

def metric_definitions(string):
    return dict({
        "Name":re.findall(r"^\'[Name]+\s*:\s*([^\,]+)", string)[0],
        "Regex":re.findall(r".+[\,]\s*[Regex]+\s*:\s*(.+)'$", string)[0]
    })

class CommonSagemakerMagics(Magics):
    """
        Common SageMaker magic class.
    """
    def __init__(self, shell, data=None):
        super(CommonSagemakerMagics, self).__init__(shell)
        self.args = {}
        self.Estimator = None
        self.cell = None
        self.method_matcher = {
            'submit': self._submit,
            'status': self._status,
            'delete': self._delete,
            'list': self._list,
            'logs': self._logs
        }

    def _get_latest_training_job_name(self):
        return self.shell.user_ns.get('___{}_latest_training_job_name'.format(self.Estimator.__name__), None)

    def _fit(self):
        est = self.Estimator(**self.args)
        channels = {
            "training": self.args.get('channel_training'),
            "testing": self.args.get('channel_testing')
        }
        est.fit(inputs=channels, wait=False)
        self.shell.user_ns['___{}_latest_training_job_name'.format(self.Estimator.__name__)] = est.latest_training_job.name
        self.shell.user_ns[self.args['estimator_name']] = est


    def _full_fill_args(self):
        filtered = {k: v for k, v in self.args.items() if v is not None}
        self.args.clear()
        self.args.update(filtered)
        self.args['entry_point'] = self.args.get('entry_point', self.upload_content(self.cell))
        self.args['role'] = self.args.get('role', get_execution_role())
        self.args['estimator_name'] = self.args.get('estimator_name', '___{}_estimator'.format(self.Estimator.__name__))

    def _submit(self):
        self._full_fill_args()

        print('submit:\n', json.dumps(self.args, sort_keys=True, indent=4, default=str))

        self._fit()

        return {
            '___{}_latest_training_job_name'.format(self.Estimator.__name__): self._get_latest_training_job_name(),
            'estimator_variable': self.args['estimator_name']
        }

    def _process_latest(self, func):
        if self._get_latest_training_job_name():
            return func(self._get_latest_training_job_name())
        else:
            return "please submit at least one job"

    def _status(self):
        return self._process_latest(Session().describe_training_job)

    def _logs(self):
        return self._process_latest(Session().logs_for_job)

    def _delete(self):
        self._process_latest(Session().stop_training_job)
        return self._process_latest(Session().describe_training_job)

    def _list(self):
        return boto3.client('sagemaker').list_training_jobs(NameContains=self.args.get('name_contains'),MaxResults=self.args.get('max_result'))

    @staticmethod
    def upload_content(content, path=None):
        if path is None:
            local_file = '/tmp/tmp-%s.py' % str(uuid.uuid4())
        else:
            local_file = path
        with open(local_file, 'w') as f:
            f.write(content)
        return local_file

@magics_class
class TensorFlowMagic(CommonSagemakerMagics):
    """
            Tensorflow magic class.
    """
    def __init__(self, shell, data=None):
        super(TensorFlowMagic, self).__init__(shell)
        self.Estimator = TensorFlow


    def tf_distribution(self, choise):
        distribution = {
            "parameter_server": {"parameter_server": {"enabled": True}},
            "horovod": {
                "mpi": {
                    "enabled": True,
                    "processes_per_host": self.args['mpi_processes_per_host'],
                    "custom_mpi_options": self.args['mpi_custom_mpi_options']
                },
            }
        }
        return distribution.get(choise, None)


    @magic_arguments()
    @argument_group(title='methods', description=None)
    @argument('method', type=str, choices=['submit', 'list', 'status', 'logs', 'delete'])
    @argument_group(title='submit', description=None)
    @argument('--estimator_name', type=str, help='estimator shell variable name')
    @argument('--entry_point', type=str, help='notebook local code file')
    @argument('--source_dir', type=str, help='notebook local code src, may contain requirements.txt')
    @argument('--role', type=str, help='An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs that create Amazon SageMaker endpoints use this role to access training data and model artifacts. After the endpoint is created, the inference code might use the IAM role, if it needs to access an AWS resource.')
    @argument('--framework_version', type=str, help='TensorFlow version', default='2.3.0')
    @argument('--py_version', type=str, help='Python version', default='py37')
    @argument('--instance_type', type=str, help='Type of EC2 instance to use for training, for example, ‘ml.c4.xlarge’.', default='ml.c4.xlarge')
    @argument('--instance_count', type=int, help='Number of Amazon EC2 instances to use for training.', default=1)
    @argument('--output_path', type=str, help='S3 location for saving the training result (model artifacts and output files). If not specified, results are stored to a default bucket. If the bucket with the specific name does not exist, the estimator creates the bucket during the fit() method execution.')
    @argument('--hyperparameters', type=hyperparameters, help='Hyperparameters are passed to your script as arguments and can be retrieved with an argparse.', metavar='FOO:1,BAR:0.555,BAZ:ABC | \'FOO : 1, BAR : 0.555, BAZ : ABC\'')
    @argument('--channel_training', type=str, help='A string that represents the path to the directory that contains the input data for the training channel. ')
    @argument('--channel_testing', type=str, help='A string that represents the path to the directory that contains the input data for the testing channel. ')
    @argument_group(title='submit-spot', description=None)
    @argument('--use_spot_instances', type=bool, help='Specifies whether to use SageMaker Managed Spot instances for training. If enabled then the max_wait arg should also be set. More information: https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html ', default=False, nargs='?', const=True)
    @argument('--max_wait', type=int, help='Timeout in seconds waiting for spot training instances (default: None). After this amount of time Amazon SageMaker will stop waiting for Spot instances to become available (default: None).')
    @argument_group(title='submit-metrics', description=None)
    @argument('--enable_sagemaker_metrics', type=bool, help='Enables SageMaker Metrics Time Series. For more information see: https://docs.aws.amazon.com/sagemaker/latest/dg/API_AlgorithmSpecification.html# SageMaker-Type-AlgorithmSpecification-EnableSageMakerMetricsTimeSeries ', default=False, nargs='?', const=True)
    @argument('--metric_definitions', type=metric_definitions, nargs='*', help='A list of dictionaries that defines the metric(s) used to evaluate the training jobs. Each dictionary contains two keys: ‘Name’ for the name of the metric, and ‘Regex’ for the regular expression used to extract the metric from the logs. This should be defined only for jobs that don’t use an Amazon algorithm.', metavar="\'Name: ganloss, Regex: GAN_loss=(.*?);\'")
    @argument_group(title='submit-distribution', description=None)
    @argument('--distribution', type=str, choices=['parameter_server', 'horovod'], help='To run your training job with multiple instances in a distributed fashion, set instance_count to a number larger than 1. We support two different types of distributed training, parameter server and Horovod. The distribution parameter is used to configure which distributed training strategy to use.')
    @argument('--mpi_processes_per_host', type=int, help="horovod mpi_processes_per_host", default=4)
    @argument('--mpi_custom_mpi_options', type=str, help="horovod custom_mpi_options", default="--NCCL_DEBUG INFO")
    @argument_group(title='list', description=None)
    @argument('--name_contains', type=str, help='', default='tensorflow')
    @argument('--max_result', type=str, help='', default=10)
    @cell_magic
    @line_magic
    @needs_local_scope
    def tfjob(self, line, cell="", local_ns=None):
        """
            Tensorflow magic command.
        """
        self.cell = cell
        self.args = vars(parse_argstring(self.tfjob, line))
        self.args['distribution'] = self.tf_distribution(self.args['distribution'])
        print(json.dumps(self.method_matcher[self.args.pop('method')](), sort_keys=True, indent=4, default=str))

@magics_class
class PyTorchMagic(CommonSagemakerMagics):
    """
        PyTorch magic class.
    """
    def __init__(self, shell, data=None):
        super(PyTorchMagic, self).__init__(shell)
        self.Estimator = PyTorch

    @magic_arguments()
    @argument_group(title='methods', description=None)
    @argument('method', type=str, choices=['submit', 'list', 'status', 'logs', 'delete'])
    @argument_group(title='submit', description=None)
    @argument('--estimator_name', type=str, help='estimator shell variable name')
    @argument('--entry_point', type=str, help='notebook local code file')
    @argument('--source_dir', type=str, help='notebook local code src, may contain requirements.txt')
    @argument('--role', type=str, help='An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs that create Amazon SageMaker endpoints use this role to access training data and model artifacts. After the endpoint is created, the inference code might use the IAM role, if it needs to access an AWS resource.')
    @argument('--framework_version', type=str, help='PyTorch version', default='1.5.0')
    @argument('--py_version', type=str, help='Python version', default='py3')
    @argument('--instance_type', type=str, help='Type of EC2 instance to use for training, for example, ‘ml.c4.xlarge’.', default='ml.c4.xlarge')
    @argument('--instance_count', type=int, help='Number of Amazon EC2 instances to use for training.', default=1)
    @argument('--output_path', type=str, help='S3 location for saving the training result (model artifacts and output files). If not specified, results are stored to a default bucket. If the bucket with the specific name does not exist, the estimator creates the bucket during the fit() method execution.')
    @argument('--hyperparameters', type=hyperparameters, help='Hyperparameters are passed to your script as arguments and can be retrieved with an argparse.', metavar='FOO:1,BAR:0.555,BAZ:ABC | \'FOO : 1, BAR : 0.555, BAZ : ABC\'')
    @argument('--channel_training', type=str, help='A string that represents the path to the directory that contains the input data for the training channel. ')
    @argument('--channel_testing', type=str, help='A string that represents the path to the directory that contains the input data for the testing channel. ')
    @argument_group(title='submit-spot', description=None)
    @argument('--use_spot_instances', type=bool, help='Specifies whether to use SageMaker Managed Spot instances for training. If enabled then the max_wait arg should also be set. More information: https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html ', default=False, nargs='?', const=True)
    @argument('--max_wait', type=int, help='Timeout in seconds waiting for spot training instances (default: None). After this amount of time Amazon SageMaker will stop waiting for Spot instances to become available (default: None).')
    @argument_group(title='submit-metrics', description=None)
    @argument('--enable_sagemaker_metrics', type=bool, help='Enables SageMaker Metrics Time Series. For more information see: https://docs.aws.amazon.com/sagemaker/latest/dg/API_AlgorithmSpecification.html# SageMaker-Type-AlgorithmSpecification-EnableSageMakerMetricsTimeSeries ', default=False, nargs='?', const=True)
    @argument('--metric_definitions', type=metric_definitions, nargs='*', help='A list of dictionaries that defines the metric(s) used to evaluate the training jobs. Each dictionary contains two keys: ‘Name’ for the name of the metric, and ‘Regex’ for the regular expression used to extract the metric from the logs. This should be defined only for jobs that don’t use an Amazon algorithm.', metavar="\'Name: loss, Regex: Loss = (.*?);\'")
    @argument_group(title='list', description=None)
    @argument('--name_contains', type=str, help='', default='pytorch')
    @argument('--max_result', type=str, help='', default=10)
    @cell_magic
    @line_magic
    @needs_local_scope
    def pytorch(self, line, cell="", local_ns=None):
        """
            Pytorch magic command.
        """
        self.cell = cell
        self.args = vars(parse_argstring(self.pytorch, line))
        print(json.dumps(self.method_matcher[self.args.pop('method')](), sort_keys=True, indent=4, default=str))


def load_ipython_extension(ipython):
    ipython.register_magics(TensorFlowMagic)
    ipython.register_magics(PyTorchMagic)
