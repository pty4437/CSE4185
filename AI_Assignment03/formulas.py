#!/usr/bin/env python3

from logic import *

import pickle, gzip, os, random
import part

test = part.Test()
hw03 = test.load('hw03')

def checkFormula(name, predForm, preconditionForm=None):
    filename = os.path.join('tests', name + '.pklz')
    objects, targetModels = pickle.load(gzip.open(filename))

    preconditionPredForm = And(preconditionForm, predForm) if preconditionForm else predForm
    predModels = performModelChecking([preconditionPredForm], findAll=True, objects=objects)
    ok = True
    def hashkey(model): return tuple(sorted(str(atom) for atom in model))
    targetModelSet = set(hashkey(model) for model in targetModels)
    predModelSet = set(hashkey(model) for model in predModels)

    ## If formula is not correct
    for model in targetModels:
        if hashkey(model) not in predModelSet:
            test.fail("Your formula (%s) says the following model is FALSE, but it should be TRUE:" % predForm)
            ok = False
            printModel(model)
            test.fail("-------------------------------------------> Not Correct")
            return
    for model in predModels:
        if hashkey(model) not in targetModelSet:
            test.fail("Your formula (%s) says the following model is TRUE, but it should be FALSE:" % predForm)
            ok = False
            printModel(model)
            test.fail("-------------------------------------------> Not Correct")
            return

    ## If formula is correct
    test.addMessage('You matched the %d tests' % len(targetModels))
    test.addMessage('Example model: %s' % rstr(random.choice(targetModels)))
    test.addMessage('----------------------> Correct Formula!')


# name: name of this formula set (used to load the tests)
# predForms: formulas predicted in the hw03
# predQuery: query formula predicted in the hw03
def addParts(name, numForms, predictionFunc):
    # part is either an individual formula (0:numForms), all (combine everything)
    def check(part):
        predForms, predQuery = predictionFunc()
        if len(predForms) < numForms:
            test.fail("Wanted %d formulas, but got %d formulas:" % (numForms, len(predForms)))
            for form in predForms: print(('-', form))
            return

        if part == 'all':
            checkFormula(name + '-all', AndList(predForms))

        else:  # Check the part-th formula
            checkFormula(name + '-' + str(part), predForms[part])

    def createCheck(part): return lambda : check(part)  # To create closure

    for part in list(range(numForms)):
        description = 'test of statement %s for %s' % (part+1, name)
        test.addBasicPart(name + '-' + str(part), createCheck(part), description=description)


if __name__=='__main__':

    test.addBasicPart('01-1', lambda: checkFormula('01-1', hw03.logic01_01()), description='test of statement 1 for 01')
    test.addBasicPart('01-2', lambda: checkFormula('01-2', hw03.logic01_02()), description='test of statement 2 for 01')
    test.addBasicPart('01-3', lambda: checkFormula('01-3', hw03.logic01_03()), description='test of statement 3 for 01')

    formula2a_precondition = AntiReflexive('Mother')
    formula2b_precondition = AntiReflexive('Child')
    formula2c_precondition = AntiReflexive('Child')
    formula2d_precondition = AntiReflexive('Parent')

    test.addBasicPart('02-1', lambda: checkFormula('02-1', hw03.logic02_01(), formula2a_precondition), description='test of statement 1 for 02')
    test.addBasicPart('02-2', lambda: checkFormula('02-2', hw03.logic02_02(), formula2b_precondition), description='test of statement 2 for 02')
    test.addBasicPart('02-3', lambda: checkFormula('02-3', hw03.logic02_03(), formula2c_precondition), description='test of statement 3 for 02')
    test.addBasicPart('02-4', lambda: checkFormula('02-4', hw03.logic02_04(), formula2d_precondition), description='test of statement 4 for 02')

    addParts('03', 6, hw03.suspect)
    addParts('04', 6, hw03.number_theorem)

    test.start()

