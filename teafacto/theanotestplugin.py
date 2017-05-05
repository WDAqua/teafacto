import logging, theano, os

from nose.plugins import Plugin

log = logging.getLogger('nose.plugins.theano')

class TheanoConfigNosePlugin(Plugin):
    name = "theanoinit"

    def begin(self):
        theano.config.floatX = "float32"
        log.log(30, theano.config.floatX)