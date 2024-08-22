from recommender.Entity.Repo import Repo
from recommender.Entity.User import User


class IssueEntity:
    def __init__(self, num, title, body, user: User, closed: User, assignees: list, labels: list,
                 repo: Repo):
        self.num = num
        self.title = title
        self.body = body
        self.user = user
        self.assignees = assignees
        self.closed = closed
        self.labels = labels
        self.repo = repo
