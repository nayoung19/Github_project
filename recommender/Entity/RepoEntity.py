import threading
from recommender.util.util import Util

util = Util()


class RepoEntity:
    repo_name_list = []
    repo_issue_num = []
    repo_title_list = []
    repo_body_list = []
    repo_userid_list = []
    repo_userLogin_list = []
    repo_id_list = []
    repo_login_list = []
    repo_assignees_list = []
    repo_state_list = []
    repo_comments_list = []
    repo_labels_list = []
    repo_readme_list = []
    repo_language_list = []
    repo_topic_list = []
    repo_descriptions_list = []
    repo_contributors_list = []
    index = 1
    df_dict = {}
    index_list = ["repo_name", "num", "title", "body", "userId", "userLogin",  "closeId", "closeLogin", "assignees",
                  "state", "comments", "labels", "readme", "language", "topics", "description", "contributors"]
    times = 0

    def __init__(self):
        self.clear_list()
        self.lock = threading.RLock()
        # 单次数据库数据增量
        self.list_size = util.get_batch_num(is_repo=True)

    # 添加一条数据，如果列表中保存的数据量超过1000就会向文件中做一次增量保存
    def __add_repo(self, user_tuple: dict):
        self.repo_name_list.append(user_tuple["repo_name"])
        self.repo_issue_num.append(user_tuple["num"])
        self.repo_title_list.append(user_tuple["title"])
        self.repo_body_list.append(user_tuple["body"])
        self.repo_userid_list.append(user_tuple["user_id"])
        self.repo_userLogin_list.append(user_tuple["user_login"])
        self.repo_id_list.append(user_tuple["id"])
        self.repo_login_list.append(user_tuple["login"])
        self.repo_assignees_list.append(user_tuple["assignees"])
        self.repo_state_list.append(user_tuple["state"])
        self.repo_comments_list.append(user_tuple["comments"])
        self.repo_labels_list.append(user_tuple["labels"])
        self.repo_readme_list.append(user_tuple["readme"])
        self.repo_language_list.append(user_tuple["language"])
        self.repo_topic_list.append(user_tuple["topics"])
        self.repo_descriptions_list.append(user_tuple["description"])
        self.repo_contributors_list.append(user_tuple["contributors"])

    # print("已获取用户：{uuid}的信息".format(uuid=user_dict["login"]))

    def add_repo(self, repo_tuple: list):
        self.lock.acquire()
        try:
            for j in repo_tuple:
                for k in j["issue"]:
                    issue = {"repo_name": j["repo_name"], "num": k["num"], "title": k["title"], "body": k["body"],
                             "id": k["id"], "login": k["login"],
                             "user_id": k["user_id"], "user_login": k["user_login"], "assignees": k["assignees"],
                             "state": k["state"], "comments": k["comments"], "labels": k["labels"],
                             "readme": j["readme"], "language": j["language"], "topics": j["topics"],
                             "description": j["description"], "contributors": j["contributors"]}
                    if self.index >= self.list_size:
                        self.times += 1
                        self.__add_repo(issue)
                        for i in range(0, len(self.index_list)):
                            self.df_dict[self.index_list[i]] = list(self.__dict__.values())[i]
                        util.save(self.df_dict, is_repo=True)
                        log_info = '\n' + f"已进行{self.times}数据次保存，单次数据保存量为：{self.list_size}"
                        print(log_info, flush=True)
                        print(self.df_dict[self.index_list[0]])
                        self.index = 1
                        self.clear_list()
                    else:
                        self.__add_repo(issue)
                        self.index += 1
        finally:
            self.lock.release()

    # 清空各个列表
    def clear_list(self):
        self.repo_name_list = []
        self.repo_issue_num = []
        self.repo_title_list = []
        self.repo_body_list = []
        self.repo_userid_list = []
        self.repo_userLogin_list = []
        self.repo_id_list = []
        self.repo_login_list = []
        self.repo_assignees_list = []
        self.repo_state_list = []
        self.repo_comments_list = []
        self.repo_labels_list = []
        self.repo_readme_list = []
        self.repo_language_list = []
        self.repo_topic_list = []
        self.repo_descriptions_list = []
        self.repo_contributors_list = []
