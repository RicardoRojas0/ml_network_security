import sys


class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error occurred in file: {self.filename} at line {self.lineno} with message: {self.error_message}"


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        raise NetworkSecurityException(e, sys)
