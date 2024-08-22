from recommender.util.util import Util


class User:
    user_list = []
    util = Util()

    # def __init__(self, userid, login, followers, repos, email, location, labels: list, relation):
    #     self.userid = userid
    #     self.login = login
    #     self.followers = followers
    #     self.repos = repos
    #     self.email = email
    #     self.location = location
    #     self.labels = labels
    #     self.relation = relation

    def __init__(self, userid, login):
        self.userid = userid
        self.login = login

    def clearList(self):
        self.user_list = []
