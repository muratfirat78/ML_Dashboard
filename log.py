import logging

def write_log(msg, log_display, category):
    log_display.value +=  msg + '\n'
    logging.info(category + ': ' +msg)


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