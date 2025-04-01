import logging
from model.student_performance import StudentPerformance

class Logger:
    def __init__(self):
        self.student_performance = StudentPerformance()
        # clear log
        with open('output.log', 'w'):
            pass

        # initiate logging
        logging.basicConfig(
            filename='output.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info('Application started')

    def write_log(msg, log_display, category):
        if log_display != None:
            log_display.value +=  msg + '\n'
        logging.info(category + ': ' +msg)

    def add_action(self, action, value):
        self.student_performance.addAction(action, value)
    
    def get_result(self):
        return str(self.student_performance)

