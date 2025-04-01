import logging

class Logger:
    def write_log(msg, log_display, category):
        if log_display != None:
            log_display.value +=  msg + '\n'
        logging.info(category + ': ' +msg)


    def __init__(self, controller):
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

        if controller.get_online_version():
            from model.google_drive import GoogleDriveModel
            self.drive = GoogleDriveModel()

