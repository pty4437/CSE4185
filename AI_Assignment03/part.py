import argparse
import sys

TOLERANCE = 1e-4  # For measuring whether two floats are equal
BASIC_MODE = 'basic'
AUTO_MODE = 'auto'
ALL_MODE = 'all'

def isCollection(x):
    return isinstance(x, list) or isinstance(x, tuple)
# Return whether two answers are equal.
def isEqual(trueAnswer, predAnswer, tolerance = TOLERANCE):
    # Handle floats specially
    if isinstance(trueAnswer, float) or isinstance(predAnswer, float):
        return abs(trueAnswer - predAnswer) < tolerance
    # Recurse on collections to deal with floats inside them
    if isCollection(trueAnswer) and isCollection(predAnswer) and len(trueAnswer) == len(predAnswer):
        for a, b in zip(trueAnswer, predAnswer):
            if not isEqual(a, b): return False
        return True
    if isinstance(trueAnswer, dict) and isinstance(predAnswer, dict):
        if len(trueAnswer) != len(predAnswer): return False
        for k, v in list(trueAnswer.items()):
            if not isEqual(predAnswer.get(k), v): return False
        return True

    # Numpy array comparison
    if type(trueAnswer).__name__ == 'ndarray':
        import numpy as np
        if isinstance(trueAnswer, np.ndarray) and isinstance(predAnswer, np.ndarray):
            if trueAnswer.shape != predAnswer.shape:
                return False
            for a, b in zip(trueAnswer, predAnswer):
                if not isEqual(a, b): return False
            return True

    # Do normal comparison
    return trueAnswer == predAnswer

class CallFunction:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        result = self.function(*args)
        return result

class Part:
    def __init__(self, number, func, description, basic):
        if not isinstance(number, str):
            raise Exception("Invalid number: %s" % number)
        if func != None and not callable(func):
            raise Exception("Invalid Func: %s" % func)
        if not description:
            print('ERROR: description required for part {}'.format(number))
        self.number = number
        self.description = description
        self.func = func
        self.basic = basic

        self.messages = []
        self.failed = False

    def fail(self):
        self.failed = True

    def is_basic(self):
        return self.func is not None and self.basic
    def is_auto(self):
        return self.func is not None
    def is_manual(self):
        return self.func is None

class Test:
    def __init__(self, args=sys.argv):
        self.parts = []  # Parts to be added
        self.useSolution = False

        parser = argparse.ArgumentParser()
        parser.add_argument('--summary', action='store_true', help='Don\'t actually run code')
        parser.add_argument('remainder', nargs=argparse.REMAINDER)
        self.params = parser.parse_args(args[1:])

        args = self.params.remainder
        if len(args) < 1:
            self.mode = AUTO_MODE
            self.selectedPartName = None
        else:
            if args[0] in [BASIC_MODE, AUTO_MODE, ALL_MODE]:
                self.mode = args[0]
                self.selectedPartName = None
            else:
                self.mode = AUTO_MODE
                self.selectedPartName = args[0]

        self.messages = []  # General messages
        self.currentPart = None  # Which part we're checking
        self.fatalError = False  # Set this if we should just stop immediately

    def addBasicPart(self, number, Func, description=""):
        self.assertNewNumber(number)
        part = Part(number, Func, description, basic=True)
        self.parts.append(part)

    def assertNewNumber(self, number):
        if number in [part.number for part in self.parts]:
            raise Exception("Part number %s already exists" % number)

    # Try to load the module
    def load(self, moduleName):
        try:
            return __import__(moduleName)
        except Exception as e:
            self.fail("Threw exception when importing '%s': %s" % (moduleName, e))
            self.fatalError = True
            return None
        except:
            self.fail("Threw exception when importing '%s'" % moduleName)
            self.fatalError = True
            return None

    def checkPart(self, part):
        print()
        print('----- START TEST %s -----' % (part.description))
        self.currentPart = part

        CallFunction(part.func)()  # Call the part's function

    def getSelectedParts(self):
        parts = []
        for part in self.parts:
            if self.selectedPartName is not None and self.selectedPartName != part.number:
                continue
            if self.mode == BASIC_MODE:
                if part.is_basic():
                    parts.append(part)

            elif self.mode == AUTO_MODE:
                if part.is_auto():
                    parts.append(part)
            elif self.mode == ALL_MODE:
                parts.append(part)

            else:
                raise Exception("Invalid mode: {}".format(self.mode))
        return parts

    def start(self):
        parts = self.getSelectedParts()

        # Check it!
        if not self.params.summary and not self.fatalError:
            print()
            print('======================================== START Formula Check =========================================')
            for part in parts:
                self.checkPart(part)

    def fail(self, message):
        self.addMessage(message)
        if self.currentPart:
            self.currentPart.points = 0
            self.currentPart.fail()
        return False

    def truncateString(self, string, length=200):
        if len(string) <= length:
            return string
        else:
            return string[:length] + '...'

    def assignFullCredit(self):
        if not self.currentPart.failed:
            self.currentPart.points = 2
        return True

    def requireIsEqual(self, trueAnswer, predAnswer, tolerance = TOLERANCE):
        if isEqual(trueAnswer, predAnswer, tolerance):
            return self.assignFullCredit()
        else:
            return self.fail("Expected '%s', but got '%s'" % (self.truncateString(str(trueAnswer)), self.truncateString(str(predAnswer))))

    def addMessage(self, message):
        if not self.useSolution:
            print(message)
        if self.currentPart:
            self.currentPart.messages.append(message)
        else:
            self.messages.append(message)