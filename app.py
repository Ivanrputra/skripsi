from flask import Flask, render_template
from datetime import datetime
app = Flask(__name__)
import trainnew

@app.route('/')
def homepage():
	a = trainnew.ivan('asdad')
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
	return render_template('hello.html', hasil=test)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)

