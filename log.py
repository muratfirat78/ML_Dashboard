import logging

def write_log(msg, log_display, category):
    log_display.value +=  msg + '\n'
    logging.info(category + ': ' +msg)

def write_to_drive():
    from model.google_drive import GoogleDriveModel
    drive = GoogleDriveModel()
    drive.upload_file( 'output.log', '/123/output.log')
    logging.info('write drive')

def initialize_logging():
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


initialize_logging()