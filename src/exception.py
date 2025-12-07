import os, sys
from src.logger import logging

def error_message_detail(error, error_details: sys):
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno

    error_message = f"Error occurred in python script [{file_name}] line number [{line_no}] error message [{error}]"
    return error_message

class CustomException(Exception):
    """
    Custom exception class for the project.
    Automatically captures filename, line number, and error message,
    and logs it using the project logger.
    """
    def __init__(self, error_message, error_details: sys):
        
        # Call the base Exception constructor
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details = error_details)

    def __str__(self):
        return self.error_message