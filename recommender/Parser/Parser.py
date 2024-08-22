import json
import parsel

import requests

from recommender.util.util import Util


class Parser(object):
    @staticmethod
    def parser_user(result_json):
        single_user_info_dict = {}
        all_user_list = []
        user_repos_list = []
        if type(result_json) == list:
            for i in result_json:
                single_user = dict()
                single_user["login"] = i["login"]
                single_user["id"] = i["id"]
                all_user_list.append(single_user)
            return all_user_list
        elif type(result_json) == str:
            repo = parsel.Selector(result_json).xpath("//*[@id=\"user-repositories-list\"]//*/*/*/h3/a/text()").getall()
            for i in repo:
                user_repos_list.append(i.strip())
            new_links = parsel.Selector(result_json).xpath('//*[@id="user-repositories-list"]/div[2]/div/a/@href') \
                .extract()
            while new_links and len(new_links) > 0:
                next_page = new_links[0]
                util = Util()
                headers = {"user-agent": util.get_user_agent(), "Authorization": "token " + util.get_api_token()}
                result = requests.get(url=next_page, headers=headers).text
                repo = parsel.Selector(result).xpath("//*[@id=\"user-repositories-list\"]//*/*/*/h3/a/text()").getall()
                for i in repo:
                    user_repos_list.append(i.strip())
                new_links = parsel.Selector(result).xpath('//*[@id="user-repositories-list"]/div[2]/div/a[2]/@href') \
                    .extract()
            return user_repos_list
        else:
            single_user_info_dict["login"] = result_json["login"]
            single_user_info_dict["id"] = result_json["id"]
            single_user_info_dict["email"] = result_json["email"]
            single_user_info_dict["location"] = result_json["location"]
            single_user_info_dict["hireable"] = result_json["hireable"]
            single_user_info_dict["public_repos"] = result_json["public_repos"]
            single_user_info_dict["followers"] = result_json["followers"]
            single_user_info_dict["following"] = result_json["following"]
            return single_user_info_dict

    @staticmethod
    def parser_repos(json_result):
        repos_dict = dict()
        repos_dict["fork"] = json_result["network_count"]
        repos_dict["description"] = json_result["description"]
        repos_dict["watchers"] = json_result["watchers"]
        repos_dict["open_issues"] = json_result["open_issues"]
        repos_dict["subscribers_count"] = json_result["subscribers_count"]
        repos_dict["description"] = json_result["description"]
        return repos_dict

    @staticmethod
    def parser_topics(json_result):
        topic = ""
        for i in json_result["names"]:
            topic += i
        return topic

    @staticmethod
    def parser_issues(result, Type=False):
        if Type:
            issue_list_number = list()
            for i in result:
                issue_list_number.append(i["number"])
            return issue_list_number
        else:
            issue_dict = dict()
            label_list = list()
            issue_dict["title"] = result["title"]
            issue_dict["body"] = result["body"]
            if result["assignees"]:
                assignees_list = list()
                for i in result["assignees"]:
                    assignees_list.append({"login": i["login"]})
                issue_dict["assignees"] = assignees_list
            else:
                issue_dict["assignees"] = []
            issue_dict["user_id"] = result["user"]["id"]
            issue_dict["user_login"] = result["user"]["login"]
            if result["labels"]:
                for i in result["labels"]:
                    label_list.append(i["name"])
            issue_dict["labels"] = label_list
            issue_dict["state"] = result["state"]
            if issue_dict["state"] == "closed":
                issue_dict["id"] = result["closed_by"]["id"]
                issue_dict["login"] = result["closed_by"]["login"]
            else:
                issue_dict["id"] = []
                issue_dict["login"] = []
            return issue_dict

    @staticmethod
    def parser_comments(result, user_id):
        comments_list = list()
        if result:
            for i in result:
                comments_dict = dict()
                comments_dict["id"] = i["user"]["id"]
                comments_dict["login"] = i["user"]["login"]
                comments_dict["body"] = i["body"]
                comments_dict["author_association"] = i["author_association"]
                if comments_dict["id"] != user_id:
                    comments_list.append(comments_dict)
        return comments_list

    @staticmethod
    def parser_contributors(result, Type=False):
        if type(result) == list:
            all_user_list = []
            for i in result:
                contributors_dict = dict()
                contributors_dict["login"] = i["login"]
                all_user_list.append(contributors_dict)
            return all_user_list

    @staticmethod
    def parser_commit(result, is_json=False):
        if is_json:
            commit_list_sha = list()
            for i in result:
                commit_list_sha.append(i["sha"])
            return commit_list_sha
        else:
            num_list = parsel.Selector(result).xpath('//*[@id="toc"]/div[2]//*/text()').getall()
            # print(num_list)
            change_num_list = []
            for i in num_list:
                if str(i).strip() != "":
                    change_num_list.append(int(str(i).strip().split(" ")[0].replace(",", "")))
            # print(change_num_list)
            return change_num_list

    @staticmethod
    def parser_fork(result):
        user_fork_repos = []
        fork_repo = parsel.Selector(result).xpath("//*[@id=\"user-repositories-list\"]/ul/*/div[1]/div[1]/h3/a/text()")\
            .getall()
        for i in fork_repo:
            user_fork_repos.append(i.strip())
        new_links = parsel.Selector(result).xpath('//*[@id="user-repositories-list"]/div[2]/div/a/@href') \
            .extract()
        while new_links and len(new_links) > 0:
            next_page = new_links[0]
            util = Util()
            headers = {"user-agent": util.get_user_agent(), "Authorization": "token " + util.get_api_token()}
            result = requests.get(url=next_page, headers=headers).text
            repo = parsel.Selector(result).xpath("//*[@id=\"user-repositories-list\"]/ul/*/div[1]/div[1]/h3/a/text()").getall()
            for i in repo:
                user_fork_repos.append(i.strip())
            new_links = parsel.Selector(result).xpath('//*[@id="user-repositories-list"]/div[2]/div/a[2]/@href') \
                .extract()
        return user_fork_repos
