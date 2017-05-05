import sys, os, re
from textwrap import dedent
from teafacto.util import argprun
from collections import OrderedDict


def main(scriptname="testrunscript.py",
         modelfilepattern="testmodelfile{}.txt",
         modelfile="none",
         numtestcans="5,10,400",
         multiprune="0,1",
         mode="concat,seq,multi,multic"):
    if not os.path.exists("alleval"):
        os.makedirs("alleval")
    loc = locals()
    griddict = OrderedDict({x: loc[x].split(",") for x in "numtestcans multiprune mode".split()})
    #print griddict
    if modelfile == "none":
        for filename in os.listdir("."):
            m = re.match("^{}$".format(modelfilepattern.format("(\d{0,4}\.?(\d{0,3}ep)?)")), filename)
            if m:
                modelname = m.group(1)
                print filename, modelname
    else:
        print modelfile
    if modelfile == "none":
        for filename in os.listdir("."):
            m = re.match("^{}$".format(modelfilepattern.format("(\d{0,4}\.?(\d{0,3}ep)?)")), filename)
            if m:
                modelname = m.group(1)
                runstuff(modelname, griddict, scriptname)
    else:
        modelname = modelfile
        runstuff(modelname, griddict, scriptname)


def runstuff(modelname, griddict, scriptname):
    for i in range(reduce(lambda x, y: x * y, map(len, griddict.values()))):
        indexes = OrderedDict()
        for k, v in griddict.items():
            indexes[k] = i % len(v)
            i //= len(v)
        #print indexes
        options = "".join(["-{} {} ".format(x, griddict[x][indexes[x]]) for x in griddict.keys()])
        cmd = """python {}
                    -loadmodel {}
                    {}"""\
            .format(scriptname,
                    modelname,
                    options
                    )
        cmd = re.sub("\n", "", cmd)
        cmd = re.sub("\s{2,}", " ", cmd)
        print cmd
        targetname = "alleval/{}.out".format(re.sub("\s", "_", cmd))
        os.system("echo {} > {}".format(cmd, targetname))
        os.system("{} >> {} 2>&1".format(cmd, targetname))


if __name__ == "__main__":
    argprun(main)

