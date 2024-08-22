import json

import parsel

from recommender.Entity.RepoEntity import RepoEntity
from recommender.Requester.BaseRequester import BaseRequester
from recommender.Parser.Parser import Parser
from recommender.util.util import Util
from recommender.threadpool.ThreadPool import ThreadPool
from recommender.Requester.UserRequester import UserRequester
import requests
import os
import time

util = Util()
thread = ThreadPool(util.get_thread_num())
user_requester = UserRequester()


class ReposRequester(BaseRequester):
    def __init__(self):
        BaseRequester.__init__(self)
        self.result_path = os.path.dirname(os.getcwd()) + os.sep + "result" + os.sep

    def get_repo_contributors(self, username: str, repo_name: str):
        url = f"https://api.github.com/repos/{username}/{repo_name}/contributors"
        response_json = requests.get(url, headers=self._random_header()).json()
        contributors_list = Parser.parser_contributors(response_json, True)
        return contributors_list

    def get_languages(self, user_name: str, repo_name: str):
        url = f"https://api.github.com/repos/{user_name}/{repo_name}/languages"
        response_json = requests.get(url, headers=self._random_header()).text
        languages_list = eval(response_json)
        return languages_list

    def get_fork_commit(self, username: str):
        fork_repo = user_requester.get_user_fork(username)
        star = 0
        commit = 0
        reponum = len(fork_repo)
        if not fork_repo:
            fork_repo = {"fork_num": reponum, "commit": 0, "star": 0}
        else:
            for i in fork_repo:
                count_url = f"https://api.github.com/repos/{username}/{i}"
                json_result = requests.get(count_url, headers=self._random_header()).json()
                star += json_result["stargazers_count"]
                commit += json_result["network_count"]
            fork_repo = {"fork_num": reponum, "commit": commit, "star": star}
        return fork_repo

    def get_repo_details(self, username: str, repo_name: str):
        if repo_name == '':
            repo_dict = {"repo_name": username + "/" + repo_name, "file": 0,
                         "code": 0, "language": {},
                         "description": 0, "stargazers_count": 0,
                         "commit_num": 0,
                         "contributors": 0}
            return repo_dict
        url = f"https://github.com/{username}/{repo_name}/"
        response_text = requests.get(url, headers=self._random_header()).text
        commit_num_str = parsel.Selector(response_text).xpath("//a/span['d-none d-sm-inline']/strong/text()").get()
        if commit_num_str is None:
            repo_dict = {"repo_name": username + "/" + repo_name, "file": 0,
                         "code": 0, "language": {},
                         "description": 0, "stargazers_count": 0,
                         "commit_num": 0,
                         "contributors": 0}
            return repo_dict
        count_url = f"https://api.github.com/repos/{username}/{repo_name}"
        json_result = requests.get(count_url, headers=self._random_header()).json()
        stargazers_count = json_result["stargazers_count"]
        description = json_result["description"]
        contributors = self.get_repo_contributors(username, repo_name)
        commit = self.get_repo_commit(username, repo_name)
        total = commit["total"]
        code = commit["additions"] - commit["deletions"]
        commit_num = commit["commit_num"]
        language = self.get_languages(username, repo_name)
        repo_dict = {"repo_name": username + "/" + repo_name, "file": total,
                     "code": code, "language": language,
                     "description": description, "stargazers_count": stargazers_count,
                     "commit_num": commit_num,
                     "contributors": contributors}
        return repo_dict

    def get_repo_issue(self, username: str, repo_name: str):
        repo_entity = RepoEntity()
        issue_list = []
        count_list = []
        break_count = 0
        repo_dict = {"repo_name": username + "/" + repo_name, "issue": issue_list}
        issue_count = 4675
        if issue_count == 0:
            return repo_dict
        elif issue_count < 100:
            page = 1
        else:
            page = issue_count // 100 + 1
        for i in range(1, page):
            url = f"https://api.github.com/repos/{username}/{repo_name}/issues?state=all&page={i}&per_page=100"
            self._test()
            json_result = requests.get(url, headers=self._random_header()).json()
            issue_num_list = Parser.parser_issues(json_result, Type=True)
            for j in range(len(issue_num_list)):
                # thread.run(func=self.__get_single_issue,
                #            args=(username, repo_name, issue_num_list[j], issue_list, count_list),
                #            callback=self.callback)
                self.__get_single_issue(username, repo_name, issue_num_list[j], issue_list, count_list)
            while True:
                util.process_bar(percent=len(issue_list) / issue_count,
                                 start_str=f"对库{username}/{repo_name}的issue的爬取进度：", end_str="100%",
                                 total_length=50)
                # if len(issue_list) < issue_count:
                if len(issue_list) == len(issue_num_list):
                    repo_dict["issue"] = issue_list
                    repo_issue = [repo_dict]
                    repo_entity.add_repo(repo_issue)
                    issue_list = []
                    break
                else:
                    # time.sleep(1)
                    last_count = len(issue_list)
                    time.sleep(1)
                    if last_count == len(issue_list):
                        break_count += 1
                    else:
                        break_count = 0
                    if break_count >= 60 and self._get_limit_count() < 4999:
                        print("\n该repo的commit的获取时间卡了超过60秒，已放弃该repo")
                        break
        thread.close()
        return repo_dict

    def __get_single_issue(self, username, repo_name, issue_num, issue_list: list, count_list: list):
        issue_url = f"https://api.github.com/repos/{username}/{repo_name}/issues/{issue_num}"
        result = requests.get(issue_url, headers=self._random_header()).text
        comment_url = f"https://api.github.com/repos/{username}/{repo_name}/issues/{issue_num}/comments"
        comments = requests.get(comment_url, headers=self._random_header()).text
        json_result = json.loads(result)
        json_comment = json.loads(comments)
        self._test()
        issue_dict = Parser.parser_issues(json_result)
        issue_dict["num"] = issue_num
        comments_list = Parser.parser_comments(json_comment, issue_dict["user_id"])
        issue_dict["comments"] = comments_list
        issue_list.append(issue_dict)
        count_list.append(issue_dict)

    def __get_commit_num(self, username: str, repo_name: str):
        url = f"https://github.com/{username}/{repo_name}/"
        response_text = requests.get(url, headers=self._random_header()).text
        commit_num_str = parsel.Selector(response_text).xpath("//a/span['d-none d-sm-inline']/strong/text()").get()
        if commit_num_str == None:
            # 此时库为空库
            commit_num = 0
        else:
            commit_num = int(commit_num_str.replace(",", ""))
        return commit_num

    def get_repo_commit(self, username: str, repo_name: str, is_save=False):
        # 如果break_count大于20，证明一个问题的获取以超出了20秒，那么此时放弃这个问题用户的获取
        break_count = 0
        count_list = []
        commit_num = self.__get_commit_num(username, repo_name)
        if commit_num == 0:
            repo_commit_dict = {'total': 0, 'additions': 0, 'deletions': 0, "commitnum": 0}
            return repo_commit_dict
        elif commit_num < 100:
            page = 1
        else:
            page = commit_num // 100 + 1
        # 用来存放所有的commit的相关信息，其中包括每次commit的sha，changed files，additions，deletions
        commit_list = []
        # 用来存放一个库所欲commit数据变化的总值，但无法精确到每次commit的sha
        repo_commit_dict = {}
        for i in range(page):
            url = f"https://api.github.com/repos/{username}/{repo_name}/commits?page={i}&per_page=100"
            self._test()
            json_result = requests.get(url, headers=self._random_header()).json()
            commit_sha_list = Parser.parser_commit(json_result, is_json=True)
            for j in range(len(commit_sha_list)):
                thread.run(func=self.__get_single_commit,
                           args=(username, repo_name, commit_sha_list[j], commit_list, count_list),
                           callback=self.callback)
            while True:
                util.process_bar(percent=len(commit_list) / commit_num,
                                 start_str=f"对库{username}/{repo_name}的commit的爬取进度：", end_str="100%",
                                 total_length=50)
                if len(commit_list) < commit_num:
                    if len(count_list) == len(commit_sha_list):
                        count_list = []
                        break
                    else:
                        # time.sleep(1)
                        last_count = len(commit_list)
                        time.sleep(1)
                        if last_count == len(commit_list):
                            break_count += 1
                        else:
                            break_count = 0
                        if break_count >= 60 and self._get_limit_count() < 4999:
                            print("\n该repo的commit的获取时间卡了超过60秒，已放弃该repo")
                            break
                else:
                    break
        if len(commit_list) == 0:
            repo_commit_dict = {'total': 0, 'additions': 0, 'deletions': 0, "commitnum": 0}
            return repo_commit_dict
        for i in commit_list:
            if i is None:
                pass
            else:
                if not repo_commit_dict.__contains__("total"):
                    repo_commit_dict["total"] = i["total"]
                repo_commit_dict["total"] += i["total"]
                if not repo_commit_dict.__contains__("additions"):
                    repo_commit_dict["additions"] = i["total"]
                repo_commit_dict["additions"] += i["additions"]
                if not repo_commit_dict.__contains__("deletions"):
                    repo_commit_dict["deletions"] = i["total"]
                repo_commit_dict["deletions"] += i["deletions"]
        repo_commit_dict["commit_num"] = commit_num
        thread.close()
        return repo_commit_dict

    def __get_single_commit(self, username, repo_name, commit_sha: str, commit_list: list, count_list: list):
        try:
            commit_dict = {}
            commit_url = f"https://github.com/{username}/{repo_name}/commit/{commit_sha}"
            response_text = requests.get(commit_url, headers=self._random_header()).text
            change_num_list = Parser.parser_commit(response_text)
            commit_dict["sha"] = commit_sha
            # 这里的total指的是修改的文件的数量，因为总代吗修改行数本身是没有的，
            # 如果想要硬算的话，其实就是把additions和deletions做了个减法，我感觉没必要，还不如总修改文件数量来的实在
            commit_dict["total"] = change_num_list[0]
            commit_dict["additions"] = change_num_list[1]
            commit_dict["deletions"] = change_num_list[2]
        except:
            commit_dict = None
        commit_list.append(commit_dict)
        count_list.append(commit_dict)

    def callback(self, status, result):
        pass

    def get_repos(self, user, is_save=False):
        repo_detail = {}
        user_repo_list = []
        # 如果传入的user是一个字符串，此时模式会切换为根据用户名，爬取该用户所有的库
        if type(user) == str:
            user_dict = user_requester.get_single_user_info(username=user)
            if user_dict is None:
                pass
            if user_dict["repos"] is None:
                repo_detail = {"repo_name": [], "open_issues": 0, "issue": [],
                               "description": [], "watchers": 0, "readme": 0, "fork": 0,
                               "language": [], "subscribers_count": 0, "topics": [],
                               "contributors": []}
            else:
                for i in range(len(user_dict["repos"])):
                    repo_name = str(user_dict["repos"][i].split("/")[-1])
                    repo_detail = self.get_repo_details(user, repo_name)

            user_repo_list.append(repo_detail)
            return user_repo_list
        # 如果传入的一个list，那么证明需要大量爬取用户库了，那么此时会将传入的list进行解析，然后批量爬取
        elif type(user) == list:
            for i in range(len(user)):
                if user[i] is None:
                    pass
                else:
                    username = user[i]["login"]
                    repos = user[i]["repos"]
                    if repos is None:
                        repo_dict = {"repo_name": None, "open_issues": 0, "issue": None,
                                     "description": None, "watchers": 0, "readme": 0, "fork": 0,
                                     "language": None, "subscribers_count": 0, "topics": None,
                                     "contributors": None}
                        return repo_dict
                    else:
                        for j in range(len(repos)):
                            repo_name = str(repos[j].split("/")[-1])
                            repo_detail = self.get_repo_details(username, repo_name)
                            user_repo_list.append(repo_detail)
                        return user_repo_list
        else:
            print("请输入正确的user：1.用户名（str）\t2.用户列表(list[dict])【如没有，请使用user_requester.get_users()方法获取】")


if __name__ == '__main__':
    r = ReposRequester()
    print(r.get_fork_commit("adam-james-v"))
    # repolist = []
    # star = 0
    # com = 0
    # fork_n = len(repolist)
    # for u in repolist:
    #     dic = r.get_fork_commit("MarkVillacampa", u)
    #     star += dic['star']
    #     com += dic['commit']
    # print(fork_n, com, star)
