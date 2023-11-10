import datetime
import os
import sys


class Logger:
    def __init__(self, summaries_directory, projector_config_file_path=None):
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'wt')
        self.summaries_directory = summaries_directory
        self.projector_config_file_path = projector_config_file_path

    def log(self, text):
        log(text)


class Progress:
    def __init__(self, name, is_background_job, size, logger):
        self.previous_progress = None
        self.name = name
        self.is_background_job = is_background_job
        self.size = size
        self.logger = logger
        self.start_datetime = datetime.datetime.now()

    def update(self, index):
        current_progress = int(float(index) / self.size * 100)

        if current_progress != self.previous_progress:
            current_datetime = datetime.datetime.now()
            diff = current_datetime - self.start_datetime

            if not self.is_background_job:
                if index != 0:
                    print("\r", end="")

                print("[%s - %s Iteration progress: %2d%%]" %
                      (self.name, diff, current_progress), end="")
                sys.stdout.flush()
            else:
                if self.logger is not None:
                    self.logger.log("[%s - %s Iteration progress: %2d%%]" %
                                    (self.name, diff, current_progress))
                else:
                    print("[%s - %s Iteration progress: %2d%%]" %
                          (self.name, diff, current_progress))

        self.previous_progress = current_progress

    def finish(self):
        if not self.is_background_job:
            print("\r", end="")
            sys.stdout.flush()


def log(text):
    timestamp = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    print("[%s] %s" % (timestamp, text))
