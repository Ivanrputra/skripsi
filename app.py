from flask import Flask, render_template
from datetime import datetime

import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as iread
import tensorflow as tf
from PIL import Image
import numpy as np
from random import shuffle
from random import randint
import urllib.request
from io import BytesIO
import trainnew

app = Flask(__name__)
@app.route('/')
def homepage():
    a = trainnew.class1('asdad')
    # a = trainnew.b()
    return a
    # the_time = datetime.now().strftime("%A, %d %b %Y %l:%M %p")

    # return """
    # <h1>Hello heroku</h1>
    # <p>It is currently {time}.</p>

    # <img src="http://loremflickr.com/600/400" />
    # """.format(time=the_time)
@app.route('/<test>')
def homeepage(test):
    return trainnew.class1("asdasd")
    # return render_template('hello.html', hasil=trainnew.class1("asdasd"))

@app.route('/class/<cl>')
def homeepager(cl):
    return trainnew.classi(cl)
    # return render_template('class.html', hasil=trainnew.classi(cl))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)