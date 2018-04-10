from flask import Flask, render_template
from datetime import datetime
app = Flask(__name__)
import trainnew

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
	return render_template('hello.html', hasil=trainnew.class1("asdasd"))
    
@app.route('/class/<cl>')
def homeepager(cl):
    return render_template('class.html', hasil=trainnew.class(cl))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
