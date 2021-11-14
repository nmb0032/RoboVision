import logging

def getLog(nm, loglevel="DEBUG"):
    #Creating custom logger
    logger=logging.getLogger(nm)
    #reading contents from properties file
    if loglevel=="ERROR":
        logger.setLevel(logging.ERROR)
    elif loglevel=="DEBUG":
        logger.setLevel(logging.DEBUG)
    #Creating Formatters    
    formatter=logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    #Creating Handlers
    stream_handler=logging.StreamHandler()
    #Adding Formatters to Handlers
    stream_handler.setFormatter(formatter)
    #Adding Handlers to logger
    logger.addHandler(stream_handler)
    return logger