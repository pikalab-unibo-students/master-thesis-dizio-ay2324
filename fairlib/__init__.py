import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('fairlib')

# this is the initial module of your app
# this is executed whenever some client-code is calling `import fairlib` or `from fairlib import ...`
# put your main classes here, eg:
class MyClass:
    def my_method(self):
        return "Hello World"


def main():
    # this is the main module of your app
    # it is only required if your project must be runnable
    # this is the script to be executed whenever some users writes `python -m fairlib` on the command line, eg.
    x = MyClass().my_method()
    print(x)


# let this be the last line of this file
logger.info("fairlib loaded")
