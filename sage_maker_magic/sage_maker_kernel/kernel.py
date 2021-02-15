# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import sys
import logging
sys.path.append('/usr/local/share/extensions/')
try:
    from asyncio import Future
except ImportError:
    class Future(object):
        """A class nothing will use."""

from ipykernel.ipkernel import IPythonKernel


class UserCodeParser(object):
    def get_code_to_run(self, code):
        return code


class KernelBase(IPythonKernel):
    def __init__(self, implementation, implementation_version, language, language_version, language_info,
                 session_language, user_code_parser=None, **kwargs):
        # Required by Jupyter - Override
        self.implementation = implementation
        self.implementation_version = implementation_version
        self.language = language
        self.language_version = language_version
        self.language_info = language_info
        self._fatal_error = False

        # Override
        self.session_language = session_language

        super(KernelBase, self).__init__(**kwargs)

        if user_code_parser is None:
            self.user_code_parser = UserCodeParser()
        else:
            self.user_code_parser = user_code_parser

        # Disable warnings for test env in HDI
        # requests.packages.urllib3.disable_warnings()

        self._load_aliases_extension()
        self._load_magics_extension()


    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False):
        return self._do_execute(code, silent, store_history, user_expressions, allow_stdin)

    def _do_execute(self, code, silent, store_history, user_expressions, allow_stdin):
        code_to_run = self.user_code_parser.get_code_to_run(code)
        res = self._execute_cell(code_to_run, silent, store_history, user_expressions, allow_stdin)
        return res

    def _execute_cell(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False,
                      shutdown_if_error=False, log_if_error=None):
        reply_content = self._execute_cell_for_user(code, silent, store_history, user_expressions, allow_stdin)
        if shutdown_if_error and reply_content[u"status"] == u"error":
            error_from_reply = reply_content[u"evalue"]
            if log_if_error is not None:
                message = "{}\nException details:\n\t\"{}\"".format(log_if_error, error_from_reply)
                return self._abort_with_fatal_error(message)
        return reply_content

    def _execute_cell_for_user(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False):
        result = super(KernelBase, self).do_execute(code, silent, store_history, user_expressions, allow_stdin)
        if isinstance(result, Future):
            result = result.result()
        return result

    def _complete_cell(self):
        """A method that runs a cell with no effect. Call this and return the value it
        returns when there's some sort of error preventing the user's cell from executing; this
        will register the cell from the Jupyter UI as being completed."""
        return self._execute_cell("None", False, True, None, False)

    def _show_user_error(self, message):
        print(message)
        #self.ipython_display.send_error(message)

    def _queue_fatal_error(self, message):
        """Queues up a fatal error to be thrown when the next cell is executed; does not
        raise an error immediately. We use this for errors that happen on kernel startup,
        since IPython crashes if we throw an exception in the __init__ method."""
        self._fatal_error = message

    def _abort_with_fatal_error(self, message):
        """Queues up a fatal error and throws it immediately."""
        self._queue_fatal_error(message)
        return self._repeat_fatal_error()

    def _repeat_fatal_error(self):
        """Throws an error that has already been queued."""
        error = self._fatal_error
        print('Fatal error: %s' % error)
        #self.ipython_display.send_error(error)
        return self._complete_cell()

    def _load_aliases_extension(self):
        list_training_jobs = "alias list_training_jobs aws sagemaker list-training-jobs"
        self._execute_cell(list_training_jobs, True, False, shutdown_if_error=False,
                           log_if_error="Failed to alias list_training_jobs.")
        stop_training_job = "alias stop_training_job aws sagemaker stop-training-job --training-job-name"
        self._execute_cell(stop_training_job, True, False, shutdown_if_error=False,
                           log_if_error="Failed to alias stop_training_job.")
        describe_training_job = "alias describe_training_job aws sagemaker describe-training-job --training-job-name"
        self._execute_cell(describe_training_job, True, False, shutdown_if_error=False,
                           log_if_error="Failed to alias describe_training_job.")

    def _load_magics_extension(self):
        register_magics_code = "%load_ext sage_maker_kernel.kernelmagics"
        self._execute_cell(register_magics_code, True, False, shutdown_if_error=False,
                           log_if_error="Failed to load the SageMaker magics.")

        print('Loaded magics.')


class SageMakerKernel(KernelBase):
    def __init__(self, **kwargs):
        implementation = 'sm_kernel'
        implementation_version = '0.1'
        language = 'python'
        language_version = '3'
        language_info = {
            'name': 'sm_kernel',
            'mimetype': 'text/x-python',
            'codemirror_mode': {'name': 'python', 'version': 3},
            'pygments_lexer': 'python'
        }

        session_language = 'python'

        super(SageMakerKernel, self).__init__(implementation, implementation_version, language, language_version,
                                            language_info, session_language, **kwargs)
