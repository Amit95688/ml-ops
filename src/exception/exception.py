import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.logger.logger import logging

def error_message(error,message:str):
    _,_,exc_tb=error_details= sys.exc_info()     
    filename= exc_tb.tb_frame.f_code.co_filename
    error_message= f"Error occurred in script: {filename} at line number: {exc_tb.tb_lineno} error message: {str(message)}"
    return error_message


class CustomException(Exception):
    def __init__(self,error,message):
        super().__init__(message)
        self.error_message= error_message(error,message)

    def __str__(self):
        return self.error_message
    
if __name__=="__main__":
    try:
        logging.info("Attempting division by zero")
        a=1/0
    except Exception as e:
        logging.error(f"Exception caught: {str(e)}")
        raise CustomException(e,"Division by zero error")