import os
from flask import Flask
from flask import render_template
from flask import request
<<<<<<< f4c27b76a2891d219c8525ced200995dbc78f43b

=======
import trainnew
>>>>>>> Base Celery and flask application
import task

app = Flask(__name__)

@app.route("/")
def hello():
    name = request.args.get('name', 'John doe')
    result = task.hello.delay(name)
    result.wait()
    return render_template('index.html', celery=result)

<<<<<<< f4c27b76a2891d219c8525ced200995dbc78f43b
=======
# @app.route("/")
# def hello():
#     name = request.args.get('name', 'John doe')
#     result = task.hello.delay(name)
#     result.wait()
#     return render_template('index.html', celery=result)


>>>>>>> Base Celery and flask application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)