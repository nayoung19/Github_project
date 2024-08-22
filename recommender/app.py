from flask import Flask, render_template, request

from recommender.main.main import Main

app = Flask(__name__)


@app.route("/home")
def root():
    """
    主页
    :return: recommender.html
    """
    return render_template('recommender.html')


@app.route("/result", methods=['POST'])
def form():
    language = request.form['lan']
    issue = request.form['topic'] + " " + request.form['title'] + " " + request.form['body']
    m = Main()
    m.get_similar_issue(issue, language)
    return render_template('recommender.html', result=m.user_info_list)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)
