import sys
import datetime
from typing import Optional
from types import ModuleType



class AppException(Exception):
    """
    Base class for all custom exceptions in the web application.
    """
    def __init__(self, error_message: Exception, error_detail: Optional[ModuleType]=None): 
        """
        error_message: The error message to be displayed.
        error_detail: The sys module's error detail.
        """
        super().__init__(error_message)
        
        self.exc_info = error_detail.exc_info() if error_detail else sys.exc_info() 

        self.error_message = AppException.error_message_detail(error_message)

    @staticmethod
    def error_message_detail(error: Exception) -> str: # type: ignore
        """
        
        """
       
        #getting the current timestamp

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        _,_, exc_tb = sys.exc_info()
        #extracting file name from exception traceback
        file_name = exc_tb.tb_frame.f_code.co_filename # type: ignore

        #preparing error message
        error_message = (
            f"\n================== ❌ ERROR TRACE ❌ ==================\n"
            f"🕒 Time     : {timestamp}\n"
            f"📂 File     : {file_name}\n"
            f"📌 Line     : {exc_tb.tb_lineno}\n" # type: ignore
            f"💥 Message  : {error}\n"
            f"========================================================\n"
        )
        return error_message
    
    def __repr__(self) -> str:
        return AppException.__name__.__str__()

    def __str__(self) -> str:
        return self.error_message
    