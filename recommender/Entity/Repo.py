class Repo:
    def __init__(self, repoName, language, readme, topic, description, contributors: list):
        self.repoName = repoName
        self.language = language
        self.readme = readme
        self.topic = topic
        self.description = description
        self.contributors = contributors

